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
The checkpoint with the lowest phase-2 validation loss is kept.
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
        for ch in CHANNELS:
            g = eval_generator(cfg["noise_seed"] + 7, ch, 0.0, bi)
            snr = (torch.rand(B, generator=g) * (hi - lo) + lo).to(device)
            w = torch.tensor(UNIFORM, device=device).expand(B, 3)
            out = forward_batch(model, batch, ch, snr, w, cap, mode, generator=g)
            loss, _ = task_loss(out["logits"], batch["labels"], w, {t: None for t in TASKS})
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
            if phase == "allocator" and val < best:
                best = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
