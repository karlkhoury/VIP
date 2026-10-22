"""
tokenalloc/retrained_receiver.py
────────────────────────────────
Same-receiver control for the comparison with the information-bottleneck (IB) baseline.

The IB baseline (tokenalloc/ib_baseline.py) retrains copies of the channel encoders, the
channel decoders, the pooling layers and the heads for 80 epochs, and its decoder also reads
the noise level. Our allocator, in the main results, runs on the stage-1 pipeline as it is.
To compare the two DECISIONS (dropping task tokens vs trimming the symbols of each token)
on equally trained receivers, this module gives our side exactly the same receiver training:

  * the same starting point (copies of the stage-1 channel encoders, pooling and heads, and the
    same SNR-aware decoder, started from the stage-1 decoder);
  * the same optimizer, learning rate, weight decay, batch size, number of epochs and training
    SNRs, with FinBERT and the task tokens frozen (cached features);
  * the only difference: instead of trimming symbols, each sentence and task sends a random
    number of whole tokens (nested dropout, as in stage 1), so the receiver works with any
    counts the allocator may choose. All four symbols of a sent token are transmitted.

The trained receiver then replaces transmit_decode() of the pipeline, and the allocator is
trained and evaluated on it exactly as in the main results (tokenalloc/train.py, evaluate.py).
"""

import copy
import random

import torch
import torch.nn as nn

from common.channel import CHANNELS, Realization, complex_to_real, eval_generator, normalize_energy, real_to_complex
from common.data import TASKS
from common.utils import sample_train_snr
from tokenalloc.allocator import token_masks
from tokenalloc.ib_baseline import NOISE_EMB, noise_embedding, snr_aware_decoder, sub_realization
from tokenalloc.model import SYMBOLS_PER_TOKEN
from tokenalloc.train import UNIFORM, cache_features, task_loss


class RetrainedReceiver(nn.Module):
    def __init__(self, pipeline, slots):
        """slots: task-token slots handled per task (the most any task may send under the cap)."""
        super().__init__()
        self.T, self.L, self.tau = pipeline.T, SYMBOLS_PER_TOKEN, pipeline.tau
        self.slots = min(slots, pipeline.J)
        self.ch_encoder = copy.deepcopy(pipeline.ch_encoder)
        self.pool = copy.deepcopy(pipeline.pool)
        self.heads = copy.deepcopy(pipeline.heads)
        self.noise_net = noise_embedding()
        self.ch_decoder = snr_aware_decoder(pipeline)

    def transmit_decode(self, task_out, snr_db, realization: Realization, k_hard, c_soft=None, power_scores=None):
        """Same signature and outputs as TokenAllocSystem.transmit_decode (it replaces it)."""
        z = task_out[:, :, :self.slots]
        real = sub_realization(realization, self.slots)
        B = z.shape[0]
        if c_soft is None:
            j = torch.arange(1, self.slots + 1, device=z.device)
            m = (j <= k_hard.unsqueeze(-1)).float()
        else:                                                # straight-through masks for the allocator
            m = token_masks(c_soft, k_hard, self.slots, self.tau)
        x = normalize_energy(real_to_complex(self.ch_encoder(z)))   # energy one per symbol, every token
        x_hat = real.apply(x, snr_db)
        sigma = torch.sqrt(torch.pow(10.0, -snr_db / 10.0) / 2.0)
        emb = self.noise_net(sigma[:, None])[:, None, None, :].expand(B, self.T, self.slots, NOISE_EMB)
        r = self.ch_decoder(torch.cat([complex_to_real(x_hat), emb], dim=-1))
        pooled = self.pool(r, m)
        logits = [head(pooled[:, t]) for t, head in enumerate(self.heads)]
        return {"logits": logits, "k": k_hard, "power": torch.ones_like(k_hard),
                "symbols": k_hard.sum(-1) * self.L}


def attach(model, rx):
    """Make the pipeline use the retrained receiver (the allocator and the encoder are unchanged)."""
    model.rx = rx                                            # registered: saved, moved and frozen with the model
    model.transmit_decode = rx.transmit_decode
    return model


def _random_counts(B, T, slots, generator, device):
    return torch.randint(1, slots + 1, (B, T), generator=generator).float().to(device)


@torch.no_grad()
def validate_rx(rx, feats, cfg, batch_size):
    """Fixed-noise validation: task loss at random token counts and SNRs from fixed seeds."""
    cls, task_out, labels = feats
    lo, hi = cfg["train"].get("snr_range", [-10, 10])
    channels = cfg["train"].get("channels", list(CHANNELS))
    draws = int(cfg["train"].get("val_draws", 1))
    tot = 0.0
    for bi, s in enumerate(range(0, cls.shape[0], batch_size)):
        t, y = task_out[s:s + batch_size], labels[s:s + batch_size]
        B = t.shape[0]
        w = torch.tensor(UNIFORM, device=t.device).expand(B, 3)
        for ch in channels:
            for r in range(draws):
                g = eval_generator(cfg["noise_seed"] + 7 + 1000 * r, ch, 0.0, bi)
                snr = (torch.rand(B, generator=g) * (hi - lo) + lo).to(t.device)
                k = _random_counts(B, rx.T, rx.slots, g, t.device)
                real = Realization((B, t.shape[1], t.shape[2], SYMBOLS_PER_TOKEN), ch, g, t.device)
                out = rx.transmit_decode(t, snr, real, k)
                tot += task_loss(out["logits"], y, w, {n: None for n in TASKS})[0].item() * B
    return tot / (cls.shape[0] * len(channels) * draws)


def train_rx(pipeline, loaders, class_w, cfg, seed, device, log):
    """Trains the retrained receiver on the frozen pipeline; returns (rx, history)."""
    rcfg, tr = cfg["receiver"], cfg["train"]
    bs, epochs = tr.get("batch_size", 16), rcfg.get("epochs", 80)
    pyrng, gen = random.Random(seed), torch.Generator().manual_seed(seed)
    channels = tr.get("channels", list(CHANNELS))
    for p in pipeline.parameters():
        p.requires_grad = False
    rx = RetrainedReceiver(pipeline, rcfg["slots"]).to(device)
    opt = torch.optim.AdamW(rx.parameters(), lr=rcfg.get("lr", 1e-3), weight_decay=rcfg.get("weight_decay", 5e-5))
    train_feats = cache_features(pipeline, loaders["train"], device)
    val_feats = cache_features(pipeline, loaders["val"], device)
    log(f"  cached features: train {tuple(train_feats[1].shape)}, val {tuple(val_feats[1].shape)}; "
        f"receiver slots per task: {rx.slots}")
    best, best_state, history = float("inf"), None, []
    N = train_feats[0].shape[0]
    for ep in range(1, epochs + 1):
        rx.train()
        perm = torch.randperm(N, generator=gen).to(device)
        tot, steps = 0.0, 0
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            t, y = train_feats[1][idx], train_feats[2][idx]
            B = t.shape[0]
            channel = pyrng.choice(channels)
            snr = sample_train_snr(B, tr.get("snr_range", [-10, 10]), device)
            real = Realization((B, t.shape[1], t.shape[2], SYMBOLS_PER_TOKEN), channel, None, device)
            k = _random_counts(B, rx.T, rx.slots, gen, device)       # nested dropout over whole tokens
            out = rx.transmit_decode(t, snr, real, k)
            loss = task_loss(out["logits"], y, torch.tensor(UNIFORM, device=device), class_w)[0]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rx.parameters(), 1.0)
            opt.step()
            tot, steps = tot + loss.item(), steps + 1
        rx.eval()
        val = validate_rx(rx, val_feats, cfg, bs)
        history.append({"phase": "receiver", "epoch": ep, "train_loss": tot / steps, "val_loss": val})
        log(f"  [rx {ep}] train={tot / steps:.4f} val={val:.4f}")
        if val < best:
            best = val
            best_state = {k_: v.detach().cpu().clone() for k_, v in rx.state_dict().items()}
    rx.load_state_dict(best_state)
    rx.eval()
    return rx, history
