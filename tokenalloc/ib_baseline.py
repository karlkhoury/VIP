"""
tokenalloc/ib_baseline.py
─────────────────────────
Baseline: variable-length variational feature encoding (VL-VFE) of Shao, Mao and Zhang,
"Learning task-oriented communication for edge inference: An information bottleneck
approach", IEEE JSAC 2022 (reference code: github.com/shaojiawei07/VL-VFE), adapted to
our three tasks and run inside our pipeline. Like our allocator it sits at the
transmitter and trades symbols for accuracy, but it decides HOW MANY SYMBOLS EACH TASK
TOKEN USES (from the noise level only), while the number of task tokens is fixed.

Kept from VL-VFE:
  * activation mu(sigma): a small network with nonnegative weights maps the noise level
    to one value per feature dimension, and an upper-triangular sum makes mu
    non-increasing along the dimensions. A threshold on mu therefore keeps a PREFIX of
    the dimensions, and a noisier channel never keeps fewer (variable length);
  * encoded feature x = tanh(mu * u);
  * sparsity-inducing KL term of the log-uniform prior on alpha^2 = sigma^2 / x^2,
    scaled by 0.1 / sigma and weighted by beta, off for a warm-up and then ramped in;
  * hard mask mu > threshold, with a straight-through estimator during training;
  * the decoder also receives a 16-value embedding of the noise level.

Adapted to our system:
  * the dimension that is switched on or off is one COMPLEX SYMBOL (real and imaginary
    part together) of a task token: each token still has L = 4 symbols, coded by the
    channel encoder of its task, and the first symbol of a token is always sent;
  * one mu network per task;
  * the number of task tokens is fixed at the equal split of the cap;
  * power: the sent symbols of each token are scaled to an average energy of one, as in
    our pipeline, so both systems are compared at the same SNR;
  * FinBERT and the task tokens stay frozen (cached features, as for the allocator); the
    channel encoders, decoders, pooling and heads start from the stage-1 pipeline and are
    retrained with the information-bottleneck loss. The pipeline itself is not modified,
    so the equal split is evaluated on exactly the same weights as everywhere else.
"""

import copy
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.channel import CHANNELS, Realization, complex_to_real, eval_generator, real_to_complex
from common.data import TASKS
from common.utils import sample_train_snr
from tokenalloc import policies as P
from tokenalloc.evaluate import _records
from tokenalloc.model import SYMBOLS_PER_TOKEN, PerTaskMLP
from tokenalloc.train import UNIFORM, cache_features, task_loss

NOISE_EMB = 16          # size of the noise-level embedding given to the decoder (as VL-VFE)


class GammaNet(nn.Module):
    """VL-VFE gamma function: 1 -> 16 -> 16 -> 16 -> L, nonnegative weights, tanh."""

    def __init__(self, n_out, width=16):
        super().__init__()
        sizes = [1, width, width, width, n_out]
        self.H = nn.ParameterList(nn.Parameter(torch.randn(o, i) * 0.1) for i, o in zip(sizes, sizes[1:]))
        self.b = nn.ParameterList(nn.Parameter(torch.randn(o) * 0.001) for o in sizes[1:])

    def forward(self, x):                                   # x [B, 1]
        for H, b in zip(self.H, self.b):
            x = torch.tanh(F.linear(x, torch.abs(H), b))
        return x


def kl_log_uniform(alpha_sq):
    """Approximate KL of the log-uniform prior (constants as in the VL-VFE code)."""
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    log_a = torch.log(alpha_sq)
    return -(k1 * torch.sigmoid(k2 + k3 * log_a) - 0.5 * F.softplus(-log_a) - k1)


def sub_realization(real: Realization, n_tokens: int) -> Realization:
    """The first n_tokens task-token slots of a realization: same h and noise per slot."""
    sub = copy.copy(real)
    sub.h, sub.n0 = real.h[:, :, :n_tokens], real.n0[:, :, :n_tokens]
    return sub


def noise_embedding():
    """VL-VFE noise-level embedding given to the decoder: 1 -> 16 -> 16 -> 16, ReLU."""
    return nn.Sequential(nn.Linear(1, NOISE_EMB), nn.ReLU(), nn.Linear(NOISE_EMB, NOISE_EMB),
                         nn.ReLU(), nn.Linear(NOISE_EMB, NOISE_EMB), nn.ReLU())


def snr_aware_decoder(pipeline):
    """Per-task channel decoder (8 + 16 -> 256 -> 768) that also reads the noise embedding,
    started from the pipeline decoder (the noise input starts at zero)."""
    L, H = SYMBOLS_PER_TOKEN, pipeline.encoder.hidden
    dec = PerTaskMLP(pipeline.T, 2 * L + NOISE_EMB, 256, H)
    with torch.no_grad():
        for new, old in zip(dec.nets, pipeline.ch_decoder.nets):
            new[0].weight.zero_()
            new[0].weight[:, :2 * L].copy_(old[0].weight)
            new[0].bias.copy_(old[0].bias)
            new[2].load_state_dict(old[2].state_dict())
    return dec


class IBSystem(nn.Module):
    def __init__(self, pipeline, threshold=1e-2):
        super().__init__()
        self.T, self.Jmax, self.L = pipeline.T, pipeline.Jmax, SYMBOLS_PER_TOKEN
        self.threshold = threshold
        H = pipeline.encoder.hidden
        # copies: the pipeline keeps its own weights for the no-allocator baselines
        self.ch_encoder = copy.deepcopy(pipeline.ch_encoder)
        self.pool = copy.deepcopy(pipeline.pool)
        self.heads = copy.deepcopy(pipeline.heads)
        self.gamma = nn.ModuleList(GammaNet(self.L) for _ in range(self.T))
        self.noise_net = noise_embedding()
        self.ch_decoder = snr_aware_decoder(pipeline)
        self.register_buffer("upper", torch.triu(torch.ones(self.L, self.L)))

    def activation(self, sigma):
        """mu [B, T, L], non-increasing along L, from the per-real noise std sigma [B]."""
        mu = torch.stack([g(sigma[:, None]) for g in self.gamma], dim=1)        # [B, T, L]
        return torch.clamp(F.linear(mu, self.upper), min=1e-4)

    def symbol_mask(self, mu):
        hard = (mu > self.threshold).float()
        hard[..., 0] = 1.0                                   # every sent token keeps one symbol
        return hard

    def forward(self, task_out, snr_db, realization: Realization, k):
        """k [B, T] task tokens per task (fixed split). Returns the same keys as transmit_decode."""
        z = task_out[:, :, :self.Jmax]                       # tokens beyond J_max are never sent
        real = sub_realization(realization, self.Jmax)
        B = z.shape[0]
        j = torch.arange(1, self.Jmax + 1, device=z.device)
        m_tok = (j <= k.unsqueeze(-1)).float()               # [B, T, Jmax]
        sigma = torch.sqrt(torch.pow(10.0, -snr_db / 10.0) / 2.0)   # per real dimension

        mu = self.activation(sigma)                          # [B, T, L]
        mu_r = torch.cat([mu, mu], dim=-1)[:, :, None, :]    # real and imaginary parts share mu
        x = torch.tanh(mu_r * self.ch_encoder(z))            # [B, T, Jmax, 2L]
        alpha_sq = sigma.view(B, 1, 1, 1) ** 2 / (x.pow(2) + 1e-4)
        kl = (kl_log_uniform(alpha_sq) * m_tok[..., None]).sum(dim=(1, 2, 3))
        kl = (kl * 0.1 / sigma).mean()

        hard = self.symbol_mask(mu)                          # [B, T, L]
        hard_r = torch.cat([hard, hard], dim=-1)[:, :, None, :]
        if self.training:                                    # forward: masked; backward: identity
            x = (x * hard_r - x).detach() + x
        else:
            x = x * hard_r
        xc = real_to_complex(x)                              # [B, T, Jmax, L]
        on = hard[:, :, None, :]
        energy = ((xc.real ** 2 + xc.imag ** 2) * on).sum(-1, keepdim=True) / on.sum(-1, keepdim=True)
        xc = xc / torch.sqrt(energy + 1e-8)                  # energy one per SENT symbol
        x_hat = real.apply(xc, snr_db) * on                  # unsent symbols carry nothing

        emb = self.noise_net(sigma[:, None])[:, None, None, :].expand(B, self.T, self.Jmax, NOISE_EMB)
        r = self.ch_decoder(torch.cat([complex_to_real(x_hat), emb], dim=-1))
        pooled = self.pool(r, m_tok)
        logits = [head(pooled[:, t]) for t, head in enumerate(self.heads)]
        per_token = hard.sum(-1)                             # [B, T] symbols per token
        return {"logits": logits, "k": k, "power": torch.ones_like(k), "kl": kl,
                "symbols": (k * per_token).sum(-1), "symbols_per_token": per_token}


# ── training ─────────────────────────────────────────────────────────────────

def _anneal(ep, warmup, ramp):
    """VL-VFE schedule: no KL during the warm-up, then a linear ramp to full weight."""
    return 0.0 if ep <= warmup else min(1.0, (ep - warmup) / max(ramp, 1))


@torch.no_grad()
def validate_ib(ib, feats, cfg, cap, batch_size, beta):
    """Fixed-noise validation of the training objective (CE + beta * KL) at full weight."""
    cls, task_out, labels = feats
    lo, hi = cfg["train"].get("snr_range", [-10, 10])
    channels = cfg["train"].get("channels", list(CHANNELS))
    draws = int(cfg["train"].get("val_draws", 1))
    tot = 0.0
    for bi, s in enumerate(range(0, cls.shape[0], batch_size)):
        t, y = task_out[s:s + batch_size], labels[s:s + batch_size]
        B = t.shape[0]
        k = P.equal_split(cap, B, max_tokens=ib.Jmax, device=t.device)
        w = torch.tensor(UNIFORM, device=t.device).expand(B, 3)
        for ch in channels:
            for r in range(draws):
                g = eval_generator(cfg["noise_seed"] + 7 + 1000 * r, ch, 0.0, bi)
                snr = (torch.rand(B, generator=g) * (hi - lo) + lo).to(t.device)
                real = Realization((B, task_out.shape[1], task_out.shape[2], SYMBOLS_PER_TOKEN), ch, g, t.device)
                out = ib(t, snr, real, k)
                loss = task_loss(out["logits"], y, w, {n: None for n in TASKS})[0]
                tot += (loss.item() + beta * out["kl"].item()) * B
    return tot / (cls.shape[0] * len(channels) * draws)


def train_ib(pipeline, loaders, class_w, cfg, seed, device, log):
    """Builds the IB baseline on the frozen pipeline, trains it and returns (ib, history)."""
    icfg, tr = cfg["ib"], cfg["train"]
    beta, cap, bs = float(icfg["beta"]), cfg["cap"], tr.get("batch_size", 16)
    epochs, warmup, ramp = icfg.get("epochs", 80), icfg.get("warmup", 5), icfg.get("ramp", 10)
    pyrng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    channels = tr.get("channels", list(CHANNELS))
    for p in pipeline.parameters():
        p.requires_grad = False
    ib = IBSystem(pipeline, icfg.get("threshold", 1e-2)).to(device)
    opt = torch.optim.AdamW(ib.parameters(), lr=icfg.get("lr", 1e-3), weight_decay=icfg.get("weight_decay", 5e-5))
    train_feats = cache_features(pipeline, loaders["train"], device)
    val_feats = cache_features(pipeline, loaders["val"], device)
    log(f"  cached features: train {tuple(train_feats[1].shape)}, val {tuple(val_feats[1].shape)}")
    # the kept epoch is chosen once the KL term has its full weight (VL-VFE also selects late)
    select_from = min(epochs, warmup + ramp)
    best, best_state, history = float("inf"), None, []
    N = train_feats[0].shape[0]
    for ep in range(1, epochs + 1):
        ib.train()
        a = _anneal(ep, warmup, ramp)
        perm = torch.randperm(N, generator=gen).to(device)
        tot, steps, syms = 0.0, 0, 0.0
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            t, y = train_feats[1][idx], train_feats[2][idx]
            B = t.shape[0]
            channel = pyrng.choice(channels)
            snr = sample_train_snr(B, tr.get("snr_range", [-10, 10]), device)
            real = Realization((B, t.shape[1], t.shape[2], SYMBOLS_PER_TOKEN), channel, None, device)
            k = P.equal_split(cap, B, max_tokens=ib.Jmax, device=device)
            out = ib(t, snr, real, k)
            w = torch.tensor(UNIFORM, device=device)
            loss = task_loss(out["logits"], y, w, class_w)[0] + beta * a * out["kl"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ib.parameters(), 1.0)
            opt.step()
            tot, steps, syms = tot + loss.item(), steps + 1, syms + out["symbols"].float().mean().item()
        ib.eval()
        val = validate_ib(ib, val_feats, cfg, cap, bs, beta)
        history.append({"phase": "ib", "epoch": ep, "train_loss": tot / steps, "avg_symbols": syms / steps,
                        "val_loss": val, "kl_weight": a})
        log(f"  [ib {ep}] train={tot / steps:.4f} symbols={syms / steps:.1f} kl_weight={a:.2f} val={val:.4f}")
        if ep >= select_from and val < best:
            best = val
            best_state = {k_: v.detach().cpu().clone() for k_, v in ib.state_dict().items()}
    if best_state is not None:
        ib.load_state_dict(best_state)
    ib.eval()
    return ib, history


# ── evaluation: same batches, SNRs and channel realizations as tokenalloc.evaluate ──

@torch.no_grad()
def evaluate_ib(ib, pipeline, loader, cfg, seed, device, split="test"):
    ib.eval()
    pipeline.eval()
    ev, cap = cfg["eval"], cfg["cap"]
    w_list = ev.get("priorities", [UNIFORM])[0]
    frames = []
    for bi, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        _, task_out = pipeline.encode(batch["input_ids"], batch["attention_mask"])
        B = task_out.shape[0]
        k = P.equal_split(cap, B, max_tokens=pipeline.Jmax, device=device)
        for ch in ev["channels"]:
            for s in ev["snrs"]:
                g = eval_generator(cfg["noise_seed"], ch, s, bi)
                real = pipeline.realization(B, ch, g, device)
                snr = torch.full((B,), float(s), device=device)
                base = {"pipeline": "tokenalloc", "seed": seed, "split": split, "channel": ch,
                        "snr_db": float(s), "cap": cap}
                for name in ev.get("policies", ["ib", "equal"]):
                    if name == "ib":
                        out = ib(task_out, snr, real, k)
                    elif name == "equal":
                        out = pipeline.transmit_decode(task_out, snr, real, k)
                    else:
                        raise ValueError(f"unknown policy {name!r} for the IB baseline")
                    frames.append(_records({**base, "method": name}, batch, out, w_list))
    return pd.concat(frames, ignore_index=True)

