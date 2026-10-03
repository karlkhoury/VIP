"""
tokenalloc/train.py
───────────────────
Training (CLAUDE.md §5).

  Phase 1  warm-up, no allocator: nested dropout. Per sentence and task a random
           cutoff j in {1..4}; keep the first j tokens. Teaches token 1 to carry most.
  Phase 2  allocator on, warm-started at equal split. Nested dropout stays on for a
           fraction of batches. Priority w ~ Dirichlet per batch (or fixed).

Every batch: channel type drawn at random, SNR ~ U[-10, 10] dB per sentence.
Loss: sum_t w_t CE_t (class-weighted where configured) + lambda * sum_t c_soft_t.
The checkpoint with the lowest phase-2 validation loss is kept (phase-1 when there is
no phase 2, as in Exp. 1).
"""

import random

import numpy as np
import torch
import torch.nn.functional as F

from common.channel import CHANNELS, eval_generator
from common.data import TASKS
from common.utils import make_optimizer, sample_train_snr, set_trainable_layers

UNIFORM = [1 / 3, 1 / 3, 1 / 3]


def sample_priority(pcfg: dict, rng: np.random.Generator):
    if pcfg.get("mode", "dirichlet") == "fixed":
        w = np.asarray(pcfg["w"], dtype=np.float64)
        return (w / w.sum()).tolist()
    return rng.dirichlet([pcfg.get("alpha", 1.0)] * len(TASKS)).tolist()


def task_loss(logits, labels, w, class_w):
    """sum_t w_t CE_t, CE averaged over the batch. w is [3] or [B, 3]."""
    total = 0.0
    parts = {}
    for t, name in enumerate(TASKS):
        ce = F.cross_entropy(logits[t], labels[:, t], weight=class_w[name], reduction="none")
        wt = w[:, t] if w.dim() == 2 else w[t]
        total = total + (wt * ce).mean()
        parts[name] = ce.mean().item()
    return total, parts


def forward_batch(model, batch, channel, snr_db, w, cap, mode, generator=None):
    """
    mode 'nested'    random cutoff per sentence and task (no allocator)
    mode 'allocator' learned counts with straight-through gradients
    """
    cls, task_out = model.encode(batch["input_ids"], batch["attention_mask"])
    B = cls.shape[0]
    real = model.realization(B, channel, generator, device=cls.device)
    if mode == "nested":
        k = torch.randint(1, model.J + 1, (B, model.T), generator=generator).float().to(cls.device)
        out = model.transmit_decode(task_out, snr_db, real, k)
        out["c_soft"] = None
        return out
    a = model.allocate(cls, task_out, snr_db, channel, w, cap)
    out = model.transmit_decode(task_out, snr_db, real, a["k_hard"], a["c_soft"], a["power_scores"])
    out["c_soft"] = a["c_soft"]
    return out


@torch.no_grad()
def validate(model, loader, cfg, cap, device, mode):
    """Fixed-noise validation: every channel, SNR per sentence ~ U[-10,10] from a fixed seed."""
    model.eval()
    tot, n = 0.0, 0
    lo, hi = cfg["train"].get("snr_range", [-10, 10])
    for bi, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["labels"].shape[0]
        for ch in cfg["train"].get("channels", list(CHANNELS)):
            g = eval_generator(cfg["noise_seed"] + 7, ch, 0.0, bi)
            snr = (torch.rand(B, generator=g) * (hi - lo) + lo).to(device)
            w = torch.tensor(UNIFORM, device=device).expand(B, 3)
            out = forward_batch(model, batch, ch, snr, w, cap, mode, generator=g)
            loss, _ = task_loss(out["logits"], batch["labels"], w, {t: None for t in TASKS})
            if mode == "allocator":                     # same objective as training
                loss = loss + float(cfg["train"].get("lambda", 0.0)) * out["k"].sum(-1).float().mean()
            tot += loss.item() * B
            n += B
    model.train()
    return tot / max(n, 1)


def train_tokenalloc(model, loaders, class_w, cfg, seed, device, log):
    tr = cfg["train"]
    cap = cfg["cap"]
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    set_trainable_layers(model.encoder.bert, cfg["encoder"].get("unfreeze_last_n", -1))
    opt = make_optimizer(model, model.encoder.bert, tr.get("lr_encoder", 2e-5), tr.get("lr_other", 1e-3))
    lam = float(tr.get("lambda", 0.0))
    channels = tr.get("channels", list(CHANNELS))
    nested_frac = tr.get("nested_dropout_frac", 0.2)
    max_steps = tr.get("max_steps_per_epoch")            # smoke tests only
    best, best_state = float("inf"), None
    history = []

    phases = [("warmup", tr.get("epochs_warmup", 4)), ("allocator", tr.get("epochs_allocator", 8))]
    # checkpoint selection runs in the last phase that has epochs (warm-up only for Exp. 1)
    select_phase = "allocator" if tr.get("epochs_allocator", 8) > 0 else "warmup"
    for phase, n_epochs in phases:
        for ep in range(1, n_epochs + 1):
            model.train()
            sums, steps = {"loss": 0.0, "tokens": 0.0}, 0
            for step, batch in enumerate(loaders["train"]):
                if max_steps and step >= max_steps:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                B = batch["labels"].shape[0]
                channel = pyrng.choice(channels)
                snr = sample_train_snr(B, tr.get("snr_range", [-10, 10]), device)
                if phase == "warmup":
                    mode, w = "nested", torch.tensor(UNIFORM, device=device)
                else:
                    mode = "nested" if pyrng.random() < nested_frac else "allocator"
                    w = torch.tensor(sample_priority(tr.get("priority", {}), rng), device=device,
                                     dtype=torch.float32)
                out = forward_batch(model, batch, channel, snr, w.expand(B, 3), cap, mode)
                loss, parts = task_loss(out["logits"], batch["labels"], w, class_w)
                if out["c_soft"] is not None and lam > 0:
                    loss = loss + lam * out["c_soft"].sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sums["loss"] += loss.item()
                sums["tokens"] += out["k"].sum(-1).float().mean().item()
                steps += 1
            val_mode = "nested" if phase == "warmup" else "allocator"
            val = validate(model, loaders["val"], cfg, cap, device, val_mode)
            rec = {"phase": phase, "epoch": ep, "train_loss": sums["loss"] / max(steps, 1),
                   "avg_tokens": sums["tokens"] / max(steps, 1), "val_loss": val}
            history.append(rec)
            log(f"  [{phase} {ep}/{n_epochs}] train={rec['train_loss']:.4f} "
                f"tokens={rec['avg_tokens']:.2f} val={val:.4f}")
            if phase == select_phase and val < best:
                best = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ── Two-stage variant: frozen pipeline, allocator only ───────────────────────

@torch.no_grad()
def cache_features(model, loader, device):
    """Encoder outputs never change once the pipeline is frozen: compute them once."""
    model.eval()
    cls, task_out, labels = [], [], []
    for batch in loader:
        c, t = model.encode(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        cls.append(c)
        task_out.append(t)
        labels.append(batch["labels"].to(device))
    return torch.cat(cls), torch.cat(task_out), torch.cat(labels)


def _allocator_forward(model, cls, task_out, channel, snr, w, cap, generator=None):
    real = model.realization(cls.shape[0], channel, generator, device=cls.device)
    a = model.allocate(cls, task_out, snr, channel, w, cap)
    out = model.transmit_decode(task_out, snr, real, a["k_hard"], a["c_soft"], a["power_scores"])
    out["c_soft"] = a["c_soft"]
    return out


@torch.no_grad()
def validate_allocator(model, feats, cfg, cap, batch_size):
    """
    Fixed-noise validation objective of the allocator on cached validation features:
    task loss + lambda * tokens, the same objective it is trained on. (Selecting on the
    task loss alone always prefers the allocator that sends the most tokens.)
    """
    cls, task_out, labels = feats
    lam = float(cfg["train"].get("lambda", 0.0))
    lo, hi = cfg["train"].get("snr_range", [-10, 10])
    channels = cfg["train"].get("channels", list(CHANNELS))
    tot = 0.0
    for bi, s in enumerate(range(0, cls.shape[0], batch_size)):
        c, t, y = cls[s:s + batch_size], task_out[s:s + batch_size], labels[s:s + batch_size]
        B = c.shape[0]
        for ch in channels:
            g = eval_generator(cfg["noise_seed"] + 7, ch, 0.0, bi)
            snr = (torch.rand(B, generator=g) * (hi - lo) + lo).to(c.device)
            w = torch.tensor(UNIFORM, device=c.device).expand(B, 3)
            out = _allocator_forward(model, c, t, ch, snr, w, cap, generator=g)
            loss = task_loss(out["logits"], y, w, {n: None for n in TASKS})[0].item()
            tot += (loss + lam * out["k"].sum(-1).float().mean().item()) * B
    return tot / (cls.shape[0] * len(channels))


def train_allocator_frozen(model, loaders, class_w, cfg, seed, device, log):
    """
    Stage 2 of the two-stage schedule: every weight except the allocator is frozen
    (the pipeline comes from init_checkpoint, e.g. the Exp. 1 warm-up model), encoder
    outputs are cached, and only the allocator is trained with the same loss and the
    same straight-through estimator. The best allocator by validation loss is kept.
    """
    tr = cfg["train"]
    cap, lam, bs = cfg["cap"], float(tr.get("lambda", 0.0)), tr.get("batch_size", 16)
    rng, pyrng = np.random.default_rng(seed), random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    channels = tr.get("channels", list(CHANNELS))
    for p in model.parameters():
        p.requires_grad = False
    for p in model.allocator.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.allocator.parameters(), lr=tr.get("lr_allocator", 1e-3), weight_decay=0.01)
    train_feats = cache_features(model, loaders["train"], device)
    val_feats = cache_features(model, loaders["val"], device)
    log(f"  cached features: train {tuple(train_feats[1].shape)}, val {tuple(val_feats[1].shape)}")
    model.eval()                                          # frozen pipeline: no dropout
    best, best_state, history = validate_allocator(model, val_feats, cfg, cap, bs), None, []
    log(f"  [allocator 0] val={best:.4f} (equal-split start)")
    best_state = {k: v.detach().cpu().clone() for k, v in model.allocator.state_dict().items()}
    N = train_feats[0].shape[0]
    for ep in range(1, tr.get("epochs_allocator", 30) + 1):
        perm = torch.randperm(N, generator=gen).to(device)
        tot, steps, toks = 0.0, 0, 0.0
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            c, t, y = train_feats[0][idx], train_feats[1][idx], train_feats[2][idx]
            B = c.shape[0]
            channel = pyrng.choice(channels)
            snr = sample_train_snr(B, tr.get("snr_range", [-10, 10]), device)
            w = torch.tensor(sample_priority(tr.get("priority", {}), rng), device=device, dtype=torch.float32)
            out = _allocator_forward(model, c, t, channel, snr, w.expand(B, 3), cap)
            loss, _ = task_loss(out["logits"], y, w, class_w)
            if lam > 0:
                loss = loss + lam * out["c_soft"].sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.allocator.parameters(), 1.0)
            opt.step()
            tot, steps, toks = tot + loss.item(), steps + 1, toks + out["k"].sum(-1).float().mean().item()
        val = validate_allocator(model, val_feats, cfg, cap, bs)
        history.append({"phase": "allocator_frozen", "epoch": ep, "train_loss": tot / steps,
                        "avg_tokens": toks / steps, "val_loss": val})
        log(f"  [allocator {ep}] train={tot / steps:.4f} tokens={toks / steps:.2f} val={val:.4f}")
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.allocator.state_dict().items()}
    model.allocator.load_state_dict(best_state)
    return history
