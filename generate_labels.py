"""
generate_labels.py
──────────────────
Run this ONCE to generate FLS and ESG labels for your Financial PhraseBank data.
After this script finishes you get: phrasebank_multitask.parquet
That file has 4 columns: text | sentiment_label | fls_label | esg_label

FinBERT-FLS and FinBERT-ESG are used ONLY here as label generators.
They never appear in your training pipeline again.

FLS labels (3 classes — matches finbert-fls actual output):
    0 = Not FLS
    1 = Non-specific FLS  (vague: "we aim to improve performance")
    2 = Specific FLS      (concrete: "we expect 15% revenue growth next quarter")

ESG labels (4 classes — matches finbert-esg actual output):
    0 = Environmental
    1 = Social
    2 = Governance
    3 = None

Usage:
    python generate_labels.py --input your_phrasebank.parquet --output phrasebank_multitask.parquet
"""

import argparse
import pandas as pd
from transformers import pipeline
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="phrasebank.parquet")
    parser.add_argument("--output",     default="phrasebank_multitask.parquet")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} sentences from {args.input}")
    print(f"Columns: {df.columns.tolist()}")

    assert "text"  in df.columns, "Need a 'text' column"
    assert "label" in df.columns, "Need a 'label' column (sentiment: 0=neg,1=neu,2=pos)"

    sentences = df["text"].tolist()

    # ── Task 2: FLS labels ────────────────────────────────────────────────────
    # finbert-fls outputs EXACTLY 3 labels:
    #   "Specific FLS"     → concrete forward-looking statement
    #   "Non-specific FLS" → vague forward-looking statement
    #   "Not FLS"          → not forward-looking at all
    print("\nGenerating FLS labels with yiyanghkust/finbert-fls ...")
    fls_pipe = pipeline(
        "text-classification",
        model="yiyanghkust/finbert-fls",
        device=device,
        truncation=True,
        max_length=512,
    )
    fls_raw = fls_pipe(sentences, batch_size=args.batch_size)

    # ── FIXED mapping (old code wrongly assumed only 2 classes) ──────────────
    fls_map = {
        "not fls":          0,
        "non-specific fls": 1,
        "specific fls":     2,
    }
    fls_labels = []
    for pred in fls_raw:
        label_str = pred["label"].strip().lower()
        mapped    = fls_map.get(label_str, 0)
        if label_str not in fls_map:
            print(f"  WARNING: unexpected FLS label: '{pred['label']}'")
        fls_labels.append(mapped)

    print("\n  FLS distribution:")
    for cls, name in [(0,"Not FLS"), (1,"Non-specific FLS"), (2,"Specific FLS")]:
        cnt = fls_labels.count(cls)
        pct = 100 * cnt / len(fls_labels)
        bar = "█" * int(pct / 2)
        print(f"    {name:22s}: {cnt:5d}  ({pct:.1f}%)  {bar}")

    del fls_pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Task 3: ESG labels ────────────────────────────────────────────────────
    # finbert-esg outputs EXACTLY 4 labels:
    #   "Environmental", "Social", "Governance", "None"
    print("\nGenerating ESG labels with yiyanghkust/finbert-esg ...")
    esg_pipe = pipeline(
        "text-classification",
        model="yiyanghkust/finbert-esg",
        device=device,
        truncation=True,
        max_length=512,
    )
    esg_raw = esg_pipe(sentences, batch_size=args.batch_size)

    esg_map = {
        "environmental": 0,
        "social":        1,
        "governance":    2,
        "none":          3,
    }
    esg_labels = []
    for pred in esg_raw:
        label_str = pred["label"].strip().lower()
        mapped    = esg_map.get(label_str, 3)
        if label_str not in esg_map:
            print(f"  WARNING: unexpected ESG label: '{pred['label']}'")
        esg_labels.append(mapped)

    print("\n  ESG distribution:")
    for cls, name in [(0,"Environmental"), (1,"Social"), (2,"Governance"), (3,"None")]:
        cnt = esg_labels.count(cls)
        pct = 100 * cnt / len(esg_labels)
        bar = "█" * int(pct / 2)
        print(f"    {name:16s}: {cnt:5d}  ({pct:.1f}%)  {bar}")

    del esg_pipe

    # ── Save ──────────────────────────────────────────────────────────────────
    df["fls_label"] = fls_labels
    df["esg_label"] = esg_labels
    df = df.rename(columns={"label": "sentiment_label"})
    df.to_parquet(args.output, index=False)

    print(f"\nSaved to {args.output}")
    print(df[["text", "sentiment_label", "fls_label", "esg_label"]].head(5).to_string())

    # ── Sanity check: print examples of each class so you can verify ──────────
    print("\n── FLS Examples (verify these make sense) ──")
    for cls, name in [(0,"NOT FLS"), (1,"NON-SPECIFIC FLS"), (2,"SPECIFIC FLS")]:
        examples = df[df["fls_label"] == cls]["text"].head(3).tolist()
        print(f"\n  {name}:")
        for ex in examples:
            print(f"    · {ex[:100]}")

    print("\n── ESG Examples (verify these make sense) ──")
    for cls, name in [(0,"Environmental"), (1,"Social"), (2,"Governance"), (3,"None")]:
        examples = df[df["esg_label"] == cls]["text"].head(2).tolist()
        print(f"\n  {name}:")
        for ex in examples:
            print(f"    · {ex[:100]}")


if __name__ == "__main__":
    main()