"""
common/prepare_data.py
──────────────────────
Build data/phrasebank_multitask.parquet (run ONCE, needs internet + ideally a GPU).

  1. Financial PhraseBank sentences + human sentiment labels (0 neg, 1 neu, 2 pos).
     Source, in order of preference:
       --phrasebank-txt path/to/Sentences_50Agree.txt   (from FinancialPhraseBank-v1.0.zip)
       otherwise downloaded from the Hugging Face hub (takala/financial_phrasebank).
  2. FLS labels from yiyanghkust/finbert-fls   (0 Not FLS, 1 Non-specific, 2 Specific)
  3. ESG labels from yiyanghkust/finbert-esg   (0 E, 1 S, 2 G, 3 None)

Note: ESG labels come from FinBERT-ESG (not FinBERT-tone, as the prior paper's text
says). FLS/ESG are machine labels, so they carry a label-noise floor (CLAUDE.md §8).

Usage:
    python -m common.prepare_data --out data/phrasebank_multitask.parquet
"""

import argparse
import io
import os
import zipfile

import pandas as pd
import torch

SENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}
FLS_MAP = {"not fls": 0, "non-specific fls": 1, "specific fls": 2}
ESG_MAP = {"environmental": 0, "social": 1, "governance": 2, "none": 3}


def _parse_phrasebank_lines(lines):
    rows = []
    for line in lines:
        line = line.strip()
        if not line or "@" not in line:
            continue
        text, label = line.rsplit("@", 1)
        rows.append({"text": text.strip(), "sentiment_label": SENT_MAP[label.strip().lower()]})
    return pd.DataFrame(rows)


def load_phrasebank(txt_path=None, agree="50Agree"):
    if txt_path:
        with open(txt_path, encoding="latin-1") as f:
            return _parse_phrasebank_lines(f)
    from huggingface_hub import hf_hub_download
    zip_path = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip",
                               repo_type="dataset")
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(f"Sentences_{agree}.txt"))
        text = z.read(name).decode("latin-1")
    return _parse_phrasebank_lines(io.StringIO(text))


def label_with(model_name, sentences, mapping, default, batch_size):
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model_name, device=device,
                    truncation=True, max_length=512)
    out = []
    for pred in pipe(sentences, batch_size=batch_size):
        key = pred["label"].strip().lower()
        if key not in mapping:
            print(f"  WARNING: unexpected label {pred['label']!r} from {model_name}")
        out.append(mapping.get(key, default))
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phrasebank-txt", default=None)
    p.add_argument("--agree", default="50Agree", choices=["50Agree", "66Agree", "75Agree", "AllAgree"])
    p.add_argument("--out", default="data/phrasebank_multitask.parquet")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    df = load_phrasebank(args.phrasebank_txt, args.agree)
    n0 = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"PhraseBank {args.agree}: {n0} sentences, {len(df)} after removing duplicates")

    sentences = df["text"].tolist()
    print("Labelling FLS with yiyanghkust/finbert-fls ...")
    df["fls_label"] = label_with("yiyanghkust/finbert-fls", sentences, FLS_MAP, 0, args.batch_size)
    print("Labelling ESG with yiyanghkust/finbert-esg ...")
    df["esg_label"] = label_with("yiyanghkust/finbert-esg", sentences, ESG_MAP, 3, args.batch_size)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df)} rows -> {args.out}")
    for col in ("sentiment_label", "fls_label", "esg_label"):
        print(f"  {col}: {df[col].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
