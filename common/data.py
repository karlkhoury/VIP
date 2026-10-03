"""
common/data.py
──────────────
Financial PhraseBank multi-task data, shared by both pipelines.

Expected parquet columns (made by common/prepare_data.py):
    text | sentiment_label (0 neg, 1 neu, 2 pos) | fls_label (0 not, 1 non-specific,
    2 specific) | esg_label (0 E, 1 S, 2 G, 3 None)

One FIXED train/val/test split (split_seed) is used by every run, so every method
is tested on the same sentences. Templated near-duplicates (same words, different
numbers) are kept in one split. The old code had no test split (it swept SNR on
the validation set it also used for checkpoint selection).
"""

import re

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


def template_groups(texts) -> np.ndarray:
    """Group id per sentence; sentences that differ only in numbers/punctuation share one."""
    keys = [re.sub(r"[^a-z]", "", t.lower()) for t in texts]
    return pd.factorize(pd.Series(keys))[0]


def split_indices(n: int, fractions=(0.7, 0.1, 0.2), split_seed: int = 0, groups=None, strata=None):
    """
    Deterministic train/val/test index split, independent of the training seed.
    With `groups`, whole groups are assigned to one split (templated near-duplicates
    never straddle train and test). With `strata` (one label per sentence), each
    stratum is split by `fractions` on its own, so rare classes get their share in
    every split.
    """
    rng = np.random.default_rng(split_seed)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    strata = np.zeros(n, dtype=np.int64) if strata is None else np.asarray(strata)
    members = pd.Series(np.arange(n)).groupby(groups).apply(list)
    group_stratum = pd.Series(strata).groupby(groups).first()
    out = {"train": [], "val": [], "test": []}
    for s in np.unique(group_stratum.values):
        gs = rng.permutation(group_stratum.index[group_stratum.values == s].to_numpy())
        size = sum(len(members[g]) for g in gs)
        n_train, n_val = round(fractions[0] * size), round(fractions[1] * size)
        got = {"train": 0, "val": 0}
        for g in gs:
            name = "train" if got["train"] < n_train else ("val" if got["val"] < n_val else "test")
            out[name].extend(members[g])
            if name in got:
                got[name] += len(members[g])
    return {k: sorted(v) for k, v in out.items()}


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
    strata = ds.labels[:, 0] * 12 + ds.labels[:, 1] * 4 + ds.labels[:, 2]   # all label combinations
    split = split_indices(len(ds), tuple(data_cfg.get("fractions", (0.7, 0.1, 0.2))),
                          data_cfg.get("split_seed", 0), template_groups(ds.texts), strata)
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
    """
    Inverse-frequency CE weights (train split only) for the listed tasks, else None.
    Scaled so the AVERAGE weight over the training sentences is 1 ("balanced":
    w_c = n / (C * n_c)). Rare classes still count more inside a task, but the task's
    total loss stays on the same scale as an unweighted task. (Scaling to sum = C
    instead shrank the FLS/ESG losses ~5x against Sentiment, so the allocator and the
    priority weights saw Sentiment as the task that mattered most.)
    """
    out = {}
    for i, t in enumerate(TASKS):
        if t not in tasks_weighted:
            out[t] = None
            continue
        counts = np.bincount(ds.labels[train_idx, i], minlength=NUM_CLASSES[t]).astype(np.float64)
        w = 1.0 / np.clip(counts, 1, None)
        w = w / (w * counts).sum() * counts.sum()           # mean weight per sentence = 1
        out[t] = torch.tensor(w, dtype=torch.float32, device=device)
    return out


def print_label_distribution(ds: PhraseBankDataset):
    names = {"sentiment": ["neg", "neu", "pos"], "fls": ["not", "non-spec", "spec"],
             "esg": ["E", "S", "G", "None"]}
    for i, t in enumerate(TASKS):
        counts = np.bincount(ds.labels[:, i], minlength=NUM_CLASSES[t])
        parts = [f"{n}={c} ({100 * c / len(ds):.1f}%)" for n, c in zip(names[t], counts)]
        print(f"  {t:9s}: " + "  ".join(parts))
