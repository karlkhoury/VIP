"""
legacy/train.py
───────────────
Training and SNR sweep for the prior-paper baselines B0 / B1 / B3 (paper §IV-B):
one model per channel, AdamW, 12 epochs, batch 16, weighted CE, training SNR
~ U[-10, 10] dB per sentence. Loss = sum_t CE_t + recon_weight * MSE(H_hat, H)
(the reconstruction term is from the old code). Best validation loss is kept.

Evaluation uses the same fixed test split, SNR list and noise seeds as tokenalloc
and writes the same per-sentence log schema (k_* empty: legacy has no task tokens).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common.channel import Realization, eval_generator
from common.data import TASKS
from common.utils import make_optimizer, sample_train_snr, set_trainable_layers

METHOD_NAME = {"shared": "B0", "equal_split": "B1", "softmask": "B3"}
TASK_KEYS = ("S", "F", "E")


def legacy_loss(out, labels, class_w, recon_weight, attention_mask):
    loss = 0.0
    for t, name in enumerate(TASKS):
        loss = loss + F.cross_entropy(out["logits"][t], labels[:, t], weight=class_w[name])
    if recon_weight > 0 and out["recon"] is not None:
        H_hat, H = out["recon"]
        m = attention_mask.unsqueeze(-1).float()
        loss = loss + recon_weight * (((H_hat - H.detach()) ** 2) * m).sum() / (m.sum() * H.shape[-1])
    return loss


def run_model(model, batch, channel, snr, generator=None):
    real = Realization(model.symbol_shape(batch["input_ids"]), channel, generator,
                       device=batch["input_ids"].device)
    return model(batch["input_ids"], batch["attention_mask"], snr, real)


@torch.no_grad()
def validate(model, loader, channel, cfg, device):
    model.eval()
    lo, hi = cfg["train"].get("snr_range", [-10, 10])
    tot, n = 0.0, 0
    for bi, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["labels"].shape[0]
        g = eval_generator(cfg["noise_seed"] + 7, channel, 0.0, bi)
        snr = (torch.rand(B, generator=g) * (hi - lo) + lo).to(device)
        out = run_model(model, batch, channel, snr, g)
        tot += legacy_loss(out, batch["labels"], {t: None for t in TASKS}, 0.0,
                           batch["attention_mask"]).item() * B
        n += B
    model.train()
    return tot / max(n, 1)


def train_legacy(model, loaders, class_w, cfg, channel, device, log):
    tr = cfg["train"]
    set_trainable_layers(model.bert, cfg["encoder"].get("unfreeze_last_n", 2))
    opt = make_optimizer(model, model.bert, tr.get("lr_encoder", 2e-5), tr.get("lr_other", 2e-5))
    recon_w = tr.get("recon_weight", 0.1)
    max_steps = tr.get("max_steps_per_epoch")
    best, best_state, history = float("inf"), None, []
    for ep in range(1, tr.get("epochs", 12) + 1):
        model.train()
        tot, steps = 0.0, 0
        for step, batch in enumerate(loaders["train"]):
            if max_steps and step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            snr = sample_train_snr(batch["labels"].shape[0], tr.get("snr_range", [-10, 10]), device)
            out = run_model(model, batch, channel, snr)
            loss = legacy_loss(out, batch["labels"], class_w, recon_w, batch["attention_mask"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            steps += 1
        val = validate(model, loaders["val"], channel, cfg, device)
        history.append({"channel": channel, "epoch": ep, "train_loss": tot / max(steps, 1), "val_loss": val})
        log(f"  [{channel} {ep}] train={tot / max(steps, 1):.4f} val={val:.4f}")
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def evaluate_legacy(model, loader, cfg, channel, seed, device, split="test"):
    model.eval()
    method = cfg.get("method_name", METHOD_NAME[model.readout])
    frames = []
    for bi, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["labels"].shape[0]
        for s in cfg["eval"]["snrs"]:
            g = eval_generator(cfg["noise_seed"], channel, s, bi)
            out = run_model(model, batch, channel, torch.full((B,), float(s), device=device), g)
            cols = {"pipeline": "legacy", "method": method, "seed": seed, "split": split,
                    "channel": channel, "snr_db": float(s), "cap": np.nan,
                    "sentence_id": batch["sentence_id"].cpu().numpy()}
            for i, key in enumerate(TASK_KEYS):
                logits, y = out["logits"][i], batch["labels"][:, i]
                pred = logits.argmax(-1)
                cols.update({f"w_{key}": np.full(B, 1 / 3), f"y_{key}": y.cpu().numpy(),
                             f"pred_{key}": pred.cpu().numpy(),
                             f"correct_{key}": (pred == y).cpu().numpy().astype(np.int8),
                             f"ce_{key}": F.cross_entropy(logits, y, reduction="none").cpu().numpy(),
                             f"k_{key}": np.full(B, np.nan), f"p_{key}": np.ones(B)})
            cols["symbols"] = out["symbols"].cpu().numpy()
            frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)
