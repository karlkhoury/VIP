"""
common/data.py
──────────────
Financial PhraseBank multi-task data, shared by both pipelines.

Expected parquet columns (made by common/prepare_data.py):
    text | sentiment_label (0 neg, 1 neu, 2 pos) | fls_label (0 not, 1 non-specific,
    2 specific) | esg_label (0 E, 1 S, 2 G, 3 None)

One FIXED train/val/test split (split_seed) is used by every run, so every method
is tested on the same sentences. The old code had no test split (it swept SNR on
the validation set it also used for checkpoint selection).
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

TASKS = ("sentiment", "fls", "esg")
NUM_CLASSES = {"sentiment": 3, "fls": 3, "esg": 4}
LABEL_COLS = {t: f"{t}_label" for t in TASKS}


class PhraseBankDataset(Dataset):
    """Items are (sentence_id, text, y_sentiment, y_fls, y_esg)."""

    def __init__(self, parquet_path: str):
        df = pd.read_parquet(parquet_path).reset_index(drop=True)
        if "sentiment_label" not in df.columns and "label" in df.columns:
            df = df.rename(columns={"label": "sentiment_label"})
        missing = [c for c in ["text", *LABEL_COLS.values()] if c not in df.columns]
        if missing:
            raise ValueError(f"{parquet_path} is missing columns {missing}. "
                             f"Run: python -m common.prepare_data")
        self.df = df
        self.texts = df["text"].tolist()
        self.labels = df[[LABEL_COLS[t] for t in TASKS]].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        y = self.labels[idx]
        return idx, self.texts[idx], int(y[0]), int(y[1]), int(y[2])


def split_indices(n: int, fractions=(0.7, 0.1, 0.2), split_seed: int = 0):
    """Deterministic train/val/test index split, independent of the training seed."""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return {"train": perm[:n_train].tolist(),
            "val": perm[n_train:n_train + n_val].tolist(),
            "test": perm[n_train + n_val:].tolist()}


def make_collate(tokenizer, max_length: int = 128):
    def collate(batch):
        ids, texts, ys, yf, ye = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        return {
            "sentence_id": torch.tensor(ids, dtype=torch.long),
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor([ys, yf, ye], dtype=torch.long).T,   # [B, 3]
        }
    return collate


def make_loaders(data_cfg: dict, tokenizer, batch_size: int, seed: int):
    """Train loader shuffles with `seed`; val/test keep a fixed order (fixed noise)."""
    ds = PhraseBankDataset(data_cfg["path"])
    split = split_indices(len(ds), tuple(data_cfg.get("fractions", (0.7, 0.1, 0.2))),
                          data_cfg.get("split_seed", 0))
    collate = make_collate(tokenizer, data_cfg.get("max_length", 128))
    g = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(Subset(ds, split["train"]), batch_size=batch_size,
                            shuffle=True, generator=g, collate_fn=collate),
        "val": DataLoader(Subset(ds, split["val"]), batch_size=batch_size,
                          shuffle=False, collate_fn=collate),
        "test": DataLoader(Subset(ds, split["test"]), batch_size=batch_size,
                           shuffle=False, collate_fn=collate),
    }
    return ds, split, loaders


def class_weights(ds: PhraseBankDataset, train_idx, tasks_weighted=("esg",), device="cpu"):
    """Inverse-frequency CE weights (train split only) for the listed tasks, else None."""
    out = {}
    for i, t in enumerate(TASKS):
        if t not in tasks_weighted:
            out[t] = None
            continue
        counts = np.bincount(ds.labels[train_idx, i], minlength=NUM_CLASSES[t]).astype(np.float64)
        w = 1.0 / np.clip(counts, 1, None)
        w = w / w.sum() * NUM_CLASSES[t]
        out[t] = torch.tensor(w, dtype=torch.float32, device=device)
    return out


def print_label_distribution(ds: PhraseBankDataset):
    names = {"sentiment": ["neg", "neu", "pos"], "fls": ["not", "non-spec", "spec"],
             "esg": ["E", "S", "G", "None"]}
    for i, t in enumerate(TASKS):
        counts = np.bincount(ds.labels[:, i], minlength=NUM_CLASSES[t])
        parts = [f"{n}={c} ({100 * c / len(ds):.1f}%)" for n, c in zip(names[t], counts)]
        print(f"  {t:9s}: " + "  ".join(parts))
