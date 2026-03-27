"""
dataset.py  (updated for multi-task)
─────────────────────────────────────
Changes from original:
  • Loads 3 labels per sentence: sentiment, fls, esg
  • Backward compatible: if the parquet only has 'label' / 'sentiment_label'
    and no fls/esg columns, those labels default to -1 (ignored in loss)
  • tasks argument controls which labels are returned
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer


class PhraseBankTextDataset(Dataset):
    """
    Loads Financial PhraseBank (or multitask variant) from a parquet file.

    Expected columns (after running generate_labels.py):
        text              – the financial sentence
        sentiment_label   – 0=neg, 1=neu, 2=pos
        fls_label         – 0=not-FLS, 1=forward-looking   (-1 if missing)
        esg_label         – 0=E, 1=S, 2=G, 3=None          (-1 if missing)

    Also works with the OLD format where sentiment is stored as 'label'.
    """

    def __init__(self, parquet_path: str, tasks=("sentiment", "fls", "esg")):
        """
        Args:
            parquet_path : path to your parquet dataset file
            tasks        : tuple of task names to load labels for.
                           Any task not present in the file gets label -1.
                           Example: ("sentiment",)  ← single task mode
                                    ("sentiment", "fls", "esg")  ← full multitask
        """
        self.df    = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.tasks = tasks

        # ── Normalise column names ─────────────────────────────────────────
        # Support both old 'label' and new 'sentiment_label'
        if "sentiment_label" not in self.df.columns:
            if "label" in self.df.columns:
                self.df = self.df.rename(columns={"label": "sentiment_label"})
            else:
                raise ValueError("Dataset must have a 'label' or 'sentiment_label' column")

        # Fill missing task columns with -1
        for col in ("fls_label", "esg_label"):
            if col not in self.df.columns:
                self.df[col] = -1

        self.texts = self.df["text"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """Returns (text_string, sentiment_label, fls_label, esg_label)."""
        row = self.df.iloc[idx]
        return (
            self.texts[idx],
            int(row["sentiment_label"]),
            int(row["fls_label"]),
            int(row["esg_label"]),
        )


def collate_finbert(batch, tokenizer, max_length=128):
    """
    Collate function for DataLoader.

    Args:
        batch      : list of (text, sent_label, fls_label, esg_label)
        tokenizer  : HuggingFace tokenizer
        max_length : max token length

    Returns:
        input_ids       : [B, L]
        attention_mask  : [B, L]
        sentiment_labels: [B]  (LongTensor)
        fls_labels      : [B]  (LongTensor, -1 if not available)
        esg_labels      : [B]  (LongTensor, -1 if not available)
    """
    texts, sent_labels, fls_labels, esg_labels = zip(*batch)

    encoding = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return (
        encoding["input_ids"],
        encoding["attention_mask"],
        torch.tensor(sent_labels, dtype=torch.long),
        torch.tensor(fls_labels,  dtype=torch.long),
        torch.tensor(esg_labels,  dtype=torch.long),
    )


def make_collate_fn(tokenizer, max_length=128):
    """Helper to create a collate_fn with the tokenizer baked in."""
    def fn(batch):
        return collate_finbert(batch, tokenizer, max_length)
    return fn