"""
common/prepare_data.py
──────────────────────
Build data/phrasebank_multitask.parquet (run ONCE, needs internet + ideally a GPU).

  1. Financial PhraseBank sentences + human sentiment labels (0 neg, 1 neu, 2 pos),
     Sentences_50Agree (all sentences with a majority label). Source, in order of preference:
       --phrasebank-zip path/to/FinancialPhraseBank-v1.0.zip
       otherwise downloaded from the Hugging Face hub (takala/financial_phrasebank).
  2. FLS labels from yiyanghkust/finbert-fls   (0 Not FLS, 1 Non-specific, 2 Specific)
  3. ESG labels from yiyanghkust/finbert-esg   (0 E, 1 S, 2 G, 3 None)

Cleaning (every step is counted and printed):
  * text repair: the PhraseBank files carry UTF-8 letters that went through a DOS code
    page ("Sepp+ñl+ñ" = Seppälä, "+àland" = Åland). They are decoded back; the few
    doubly-broken sequences (apostrophes, euro signs) are replaced explicitly.
  * duplicates (same text after whitespace/case normalization): one copy kept if the
    sentiment labels agree, all copies dropped if they conflict.
  * template near-duplicates (same letters, only numbers/punctuation differ) whose
    sentiment labels conflict are dropped. The rest stay, and common/data.py keeps
    each template group inside one split so no template leaks from train to test.
  * fragments with fewer than 4 tokens ("Welcome !", "Status : Agreed") are dropped.
  * FLS/ESG labellers must return one of the known label names, else the run fails.

Extra columns for auditing: text_original, agree (highest annotator-agreement file the
sentence appears in: 0.5, 0.66, 0.75, 1.0), fls_conf, esg_conf (labeller probability of
the chosen label). They let later analysis check whether results hold on high-agreement
or high-confidence subsets. Nothing is filtered on them here.

Note: ESG labels come from FinBERT-ESG (not FinBERT-tone, as the prior paper's text
says). FLS/ESG are machine labels, so they carry a label-noise floor (CLAUDE.md §8).

Usage:
    python -m common.prepare_data --out data/phrasebank_multitask.parquet
"""

import argparse
import os
import re
import zipfile

import pandas as pd
import torch

SENT_MAP = {"negative": 0, "neutral": 1, "positive": 2}
FLS_MAP = {"not fls": 0, "non-specific fls": 1, "specific fls": 2}
ESG_MAP = {"environmental": 0, "social": 1, "governance": 2, "none": 3}
AGREE_LEVELS = (("AllAgree", 1.0), ("75Agree", 0.75), ("66Agree", 0.66), ("50Agree", 0.5))
MIN_TOKENS = 4

# ── Text repair ──────────────────────────────────────────────────────────────

_LEAD = {"+": 0xC3, "-": 0xC2}          # UTF-8 lead bytes that became '+' / '-'
_MOJIBAKE = re.compile(r"([+-])([\x80-\xff])")
# Doubly-broken sequences that the code-page repair cannot undo (checked by hand).
_REPLACE = [
    ("âEUR TM s", " 's"),           # companyâEUR TM s   -> company 's
    ("â x201a ¬ ", "EUR "),    # â x201a ¬ 59.3  -> EUR 59.3
    (" -¦ s ", " 's "),             # the company -¦ s shares -> the company 's shares
    ("â s ", " 's "),               # Myllykoskiâ s -> Myllykoski 's
    (" â ??", ""),                  # truncated tail "yearâ ??"
    ("â ??", ""),
    ("Asia â Pakistan", "Asia - Pakistan"),
    ("VarpaisjÃ rvi", "Varpaisjärvi"),
    ("Â ", ""),                     # stray Â left from a lost currency sign
]


def _fix_char(m):
    try:
        return bytes([_LEAD[m.group(1)], m.group(2).encode("cp850")[0]]).decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return m.group(0)


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", _MOJIBAKE.sub(_fix_char, s).replace(" ", " "))
    for bad, good in _REPLACE:
        s = s.replace(bad, good)
    return re.sub(r"\s+", " ", s).strip()


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def template_key(s: str) -> str:
    """Letters only: sentences that differ only in numbers/punctuation share a key
    (PhraseBank has templated news, e.g. '... increased from EUR 5.1 mn to EUR 6.3 mn')."""
    return re.sub(r"[^a-z]", "", s.lower())


# ── PhraseBank ───────────────────────────────────────────────────────────────

def _read(z, agree):
    name = next(n for n in z.namelist() if n.endswith(f"Sentences_{agree}.txt") and "__MACOSX" not in n)
    rows = []
    for line in z.read(name).decode("latin-1").splitlines():
        line = line.strip()
        if not line or "@" not in line:
            continue
        text, label = line.rsplit("@", 1)
        rows.append((text.strip(), SENT_MAP[label.strip().lower()]))
    return rows


def load_phrasebank(zip_path=None):
    if not zip_path:
        from huggingface_hub import hf_hub_download
        zip_path = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip",
                                   repo_type="dataset")
    with zipfile.ZipFile(zip_path) as z:
        base = _read(z, "50Agree")
        agree = {}
        for level, value in reversed(AGREE_LEVELS):          # higher levels overwrite
            for text, _ in _read(z, level):
                agree[text] = value
    return pd.DataFrame({"text_original": [t for t, _ in base],
                         "sentiment_label": [y for _, y in base],
                         "agree": [agree[t] for t, _ in base]})


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.copy()
    df["text"] = df["text_original"].map(clean_text)
    changed = int((df["text"] != df["text_original"].map(lambda s: re.sub(r"\s+", " ", s).strip())).sum())
    leftover = df["text"].str.contains(r"[ÂÃâ¦¬]|x201a", regex=True)
    if leftover.any():
        raise ValueError("unrepaired characters remain:\n" + "\n".join(df.loc[leftover, "text"].head(10)))
    print(f"  text repaired in {changed} sentences")

    key = df["text"].map(normalize)
    n_labels = df.groupby(key)["sentiment_label"].transform("nunique")
    conflict = n_labels > 1
    print(f"  duplicates with conflicting sentiment: {int(conflict.sum())} rows dropped")
    df = df[~conflict.values]
    key = key[~conflict.values]
    dup = key.duplicated()
    print(f"  exact duplicates: {int(dup.sum())} rows dropped")
    df = df[~dup.values]

    tkey = df["text"].map(template_key)
    t_conflict = df.groupby(tkey)["sentiment_label"].transform("nunique") > 1
    print(f"  template near-duplicates with conflicting sentiment: {int(t_conflict.sum())} rows dropped")
    df = df[~t_conflict.values]

    short = df["text"].str.split().str.len() < MIN_TOKENS
    print(f"  fragments (< {MIN_TOKENS} tokens): {int(short.sum())} rows dropped "
          f"{df.loc[short, 'text'].tolist()}")
    df = df[~short].reset_index(drop=True)
    print(f"  {n0} -> {len(df)} sentences")
    return df


# ── Machine labels ───────────────────────────────────────────────────────────

def label_with(model_name, sentences, mapping, batch_size):
    """Returns (label ids, probability of the chosen label). Fails on unknown label names."""
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-classification", model=model_name, device=device,
                    truncation=True, max_length=512)
    known = {v.strip().lower() for v in pipe.model.config.id2label.values()}
    if known != set(mapping):
        raise ValueError(f"{model_name} labels {sorted(known)} != expected {sorted(mapping)}")
    ids, conf = [], []
    for pred in pipe(sentences, batch_size=batch_size):
        ids.append(mapping[pred["label"].strip().lower()])
        conf.append(float(pred["score"]))
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ids, conf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phrasebank-zip", default=None)
    p.add_argument("--out", default="data/phrasebank_multitask.parquet")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    df = load_phrasebank(args.phrasebank_zip)
    print(f"PhraseBank 50Agree: {len(df)} sentences")
    df = clean(df)

    sentences = df["text"].tolist()
    print("Labelling FLS with yiyanghkust/finbert-fls ...")
    df["fls_label"], df["fls_conf"] = label_with("yiyanghkust/finbert-fls", sentences, FLS_MAP, args.batch_size)
    print("Labelling ESG with yiyanghkust/finbert-esg ...")
    df["esg_label"], df["esg_conf"] = label_with("yiyanghkust/finbert-esg", sentences, ESG_MAP, args.batch_size)

    df = df[["text", "sentiment_label", "fls_label", "esg_label",
             "agree", "fls_conf", "esg_conf", "text_original"]]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df)} rows -> {args.out}")
    for col in ("sentiment_label", "fls_label", "esg_label", "agree"):
        print(f"  {col}: {df[col].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
