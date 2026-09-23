"""
common/utils.py
───────────────
Config loading, seeding, encoder/tokenizer loading, optimizer, per-sentence logs.
"""

import copy
import json
import os
import random

import numpy as np
import torch
import yaml

# ── Config ───────────────────────────────────────────────────────────────────

REQUIRED = ("name", "pipeline", "seeds", "data", "encoder", "train", "eval")


def load_config(path: str) -> dict:
    """Every experiment is launched by one YAML file (CLAUDE.md §10)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:                        # inherit shared settings from another file
        base_path = os.path.join(os.path.dirname(path), cfg.pop("base"))
        with open(base_path) as f:
            cfg = deep_update(yaml.safe_load(f), cfg)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if cfg["pipeline"] not in ("legacy", "tokenalloc"):
        raise ValueError(f"{path}: pipeline must be 'legacy' or 'tokenalloc'")
    for k in ("channels", "snrs"):
        if k not in cfg["eval"]:
            raise ValueError(f"{path}: eval.{k} is required")
    cfg.setdefault("noise_seed", 1234)
    cfg.setdefault("out_dir", os.path.join("runs", cfg["name"]))
    return cfg


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# ── Seeding ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Encoder checkpoint (CLAUDE.md §8) ────────────────────────────────────────

# ProsusAI/finbert was fine-tuned on Financial PhraseBank = our test set. Never use it.
FORBIDDEN_ENCODERS = {"ProsusAI/finbert"}


def load_encoder(enc_cfg: dict):
    """
    Returns (bert_model, tokenizer). `name: tiny-random` builds a small random BERT
    and a toy tokenizer offline, used only by the smoke tests.
    """
    name = enc_cfg["name"]
    if name in FORBIDDEN_ENCODERS:
        raise ValueError(f"{name} was fine-tuned on Financial PhraseBank (test-set leakage). "
                         f"Use yiyanghkust/finbert-pretrain or bert-base-uncased.")
    if name == "tiny-random":
        from common.tiny import tiny_bert_and_tokenizer
        return tiny_bert_and_tokenizer(hidden=enc_cfg.get("hidden", 64))
    from transformers import AutoModel, AutoTokenizer
    return AutoModel.from_pretrained(name), AutoTokenizer.from_pretrained(name)


def set_trainable_layers(bert, unfreeze_last_n: int):
    """-1 = train everything; n >= 0 = freeze all but the last n transformer layers."""
    for p in bert.parameters():
        p.requires_grad = unfreeze_last_n < 0
    if unfreeze_last_n > 0:
        for layer in bert.encoder.layer[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True


def make_optimizer(model, encoder_module, lr_encoder: float, lr_other: float, weight_decay=0.01):
    """AdamW with a small lr for the pretrained encoder and a larger one elsewhere."""
    enc_ids = {id(p) for p in encoder_module.parameters()}
    enc = [p for p in encoder_module.parameters() if p.requires_grad]
    other = [p for p in model.parameters() if id(p) not in enc_ids and p.requires_grad]
    groups = [g for g in ({"params": enc, "lr": lr_encoder}, {"params": other, "lr": lr_other})
              if g["params"]]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def sample_train_snr(batch_size: int, snr_range, device) -> torch.Tensor:
    """Training SNR: uniform in [lo, hi] dB, drawn per sentence."""
    lo, hi = snr_range
    return torch.empty(batch_size, device=device).uniform_(lo, hi)


# ── Per-sentence logs ────────────────────────────────────────────────────────

LOG_COLUMNS = [
    "pipeline", "method", "seed", "split", "channel", "snr_db", "cap",
    "w_S", "w_F", "w_E", "sentence_id",
    "y_S", "y_F", "y_E", "pred_S", "pred_F", "pred_E",
    "correct_S", "correct_F", "correct_E", "ce_S", "ce_F", "ce_E",
    "k_S", "k_F", "k_E", "symbols", "p_S", "p_F", "p_E",
]


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
