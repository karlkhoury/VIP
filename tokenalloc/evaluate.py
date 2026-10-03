"""
tokenalloc/evaluate.py
──────────────────────
Frozen-weight evaluation (CLAUDE.md §6). For every test batch the encoder runs
once; then for each channel and SNR one fixed channel Realization (shared by all
methods and all seeds) is drawn, and every policy is sent through it.

Per (batch, channel, SNR):
  1. loss curves: send k = j tokens for every task, j = 1..4. Because each task's
     receiver only sees its own tokens, this gives CE_t(j) per sentence and task.
     Logged as method 'uniform_k{j}' = Experiment 1 (saturation map).
  2. per priority w: the learned allocator, then each baseline, all through the
     same transmit_decode().

The SNR-only rule needs mean loss curves from the VALIDATION split, computed first
by mean_val_curves().
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common.channel import eval_generator
from tokenalloc import policies as P

TASK_KEYS = ("S", "F", "E")


def _records(meta, batch, out, w):
    """One row per sentence."""
    labels = batch["labels"].cpu().numpy()
    B = labels.shape[0]
    cols = dict(meta)
    cols["sentence_id"] = batch["sentence_id"].cpu().numpy()
    for i, key in enumerate(TASK_KEYS):
        logits = out["logits"][i]
        pred = logits.argmax(-1)
        ce = F.cross_entropy(logits, batch["labels"][:, i], reduction="none")
        cols[f"y_{key}"] = labels[:, i]
        cols[f"pred_{key}"] = pred.cpu().numpy()
        cols[f"correct_{key}"] = (pred == batch["labels"][:, i]).cpu().numpy().astype(np.int8)
        cols[f"ce_{key}"] = ce.cpu().numpy()
        cols[f"k_{key}"] = out["k"][:, i].cpu().numpy().astype(np.int8)
        cols[f"p_{key}"] = out["power"][:, i].cpu().numpy()
        cols[f"w_{key}"] = np.full(B, float(w[i]))
    cols["symbols"] = out["symbols"].cpu().numpy().astype(np.int16)
    return pd.DataFrame(cols)


def _curves(model, task_out, snr, real, labels):
    """CE and correctness for k = 1..4 on every task: ce [B, T, J]."""
    B = labels.shape[0]
    ces, outs = [], []
    for j in range(1, model.J + 1):
        k = torch.full((B, model.T), float(j), device=labels.device)
        out = model.transmit_decode(task_out, snr, real, k)
        ces.append(torch.stack([F.cross_entropy(out["logits"][t], labels[:, t], reduction="none")
                                for t in range(model.T)], dim=-1))
        outs.append(out)
    return torch.stack(ces, dim=-1), outs


@torch.no_grad()
def mean_val_curves(model, loader, channels, snrs, noise_seed, device):
    """Mean CE_t(j) over validation sentences and channels, per SNR -> {snr: [T, J]}."""
    model.eval()
    acc = {float(s): [] for s in snrs}
    for bi, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        _, task_out = model.encode(batch["input_ids"], batch["attention_mask"])
        B = task_out.shape[0]
        for ch in channels:
            for s in snrs:
                g = eval_generator(noise_seed + 1, ch, s, bi)          # val noise != test noise
                real = model.realization(B, ch, g, device)
                ce, _ = _curves(model, task_out, torch.full((B,), float(s), device=device),
                                real, batch["labels"])
                acc[float(s)].append(ce)
    return {s: torch.cat(v).mean(0) for s, v in acc.items()}


@torch.no_grad()
def evaluate(model, loader, cfg, seed, device, val_curves=None, split="test"):
    model.eval()
    ev = cfg["eval"]
    cap = cfg["cap"]
    lam = float(cfg["train"].get("lambda", 0.0))
    priorities = ev.get("priorities", [[1 / 3, 1 / 3, 1 / 3]])
    policies = ev.get("policies", ["allocator", "equal"])
    saturation = ev.get("saturation", True)
    frames = []
    for bi, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        cls, task_out = model.encode(batch["input_ids"], batch["attention_mask"])
        B = cls.shape[0]
        labels = batch["labels"]
        for ch in ev["channels"]:
            for s in ev["snrs"]:
                g = eval_generator(cfg["noise_seed"], ch, s, bi)
                real = model.realization(B, ch, g, device)
                snr = torch.full((B,), float(s), device=device)
                base = {"pipeline": "tokenalloc", "seed": seed, "split": split, "channel": ch,
                        "snr_db": float(s), "cap": cap}
                ce, uni = _curves(model, task_out, snr, real, labels)
                if saturation:
                    for j, out in enumerate(uni, start=1):
                        frames.append(_records({**base, "method": f"uniform_k{j}"}, batch, out, [1 / 3] * 3))
                for w_list in priorities:
                    w = torch.tensor(w_list, device=device, dtype=torch.float32).expand(B, 3)
                    alloc_k = None
                    rng = torch.Generator().manual_seed(int(cfg["noise_seed"]) * 1000 + bi)
                    for name in policies:
                        kind, arg = P.parse_policy(name)
                        extra = {}
                        if kind == "allocator":
                            a = model.allocate(cls, task_out, snr, ch, w, cap)
                            k, extra = a["k_hard"], {"power_scores": a["power_scores"]}
                            alloc_k = k
                        elif kind == "equal":
                            k = P.equal_split(cap, B, max_tokens=model.Jmax, device=device)
                        elif kind == "all":                  # no allocator, no cap: every token
                            k = torch.full((B, model.T), float(model.J), device=device)
                        elif kind == "fixed":
                            k = P.fixed_split(arg, B, device=device)
                        elif kind == "proportional":
                            k = P.proportional(w, cap, model.Jmax)
                        elif kind == "snr_only":
                            if val_curves is None:
                                raise ValueError("snr_only needs validation curves")
                            mean = val_curves[float(s)][:, :model.Jmax].unsqueeze(0).expand(B, -1, -1)
                            k = P.greedy_from_curves(mean, w, cap, lam)
                        elif kind == "random_matched":
                            if alloc_k is None:
                                raise ValueError("list 'allocator' before 'random_matched'")
                            k = P.random_matched(alloc_k.sum(-1), rng, max_tokens=model.Jmax)
                        elif kind == "oracle_greedy":
                            k = P.greedy_from_curves(ce[..., :model.Jmax], w, cap, lam)
                        elif kind == "oracle_exhaustive":
                            k = P.exhaustive_from_curves(ce[..., :model.Jmax], w, cap, lam)
                        else:
                            raise ValueError(f"unknown policy {name!r}")
                        out = model.transmit_decode(task_out, snr, real, k,
                                                    power_scores=extra.get("power_scores"))
                        frames.append(_records({**base, "method": name}, batch, out, w_list))
    return pd.concat(frames, ignore_index=True)
