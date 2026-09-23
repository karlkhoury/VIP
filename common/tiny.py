"""
common/tiny.py
──────────────
Offline stand-ins for smoke tests only: a small random BERT, a toy WordPiece
tokenizer, and a synthetic PhraseBank-shaped parquet. Never used for results.
"""

import os
import string
import tempfile

import numpy as np
import pandas as pd

WORDS = ("the company profit sales rose fell net quarter year expects will growth "
         "board shares emissions employees governance risk market revenue loss").split()


def tiny_bert_and_tokenizer(hidden: int = 64):
    from transformers import BertConfig, BertModel, BertTokenizerFast
    chars = list(string.ascii_lowercase + string.digits + string.punctuation)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS + chars + ["##" + c for c in chars]
    d = tempfile.mkdtemp(prefix="tiny_bert_")
    path = os.path.join(d, "vocab.txt")
    with open(path, "w") as f:
        f.write("\n".join(vocab))
    tok = BertTokenizerFast(vocab_file=path, do_lower_case=True)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=hidden, num_hidden_layers=2,
                     num_attention_heads=4, intermediate_size=2 * hidden, max_position_embeddings=256)
    return BertModel(cfg), tok


def synthetic_parquet(path: str, n: int = 96, seed: int = 0):
    rng = np.random.default_rng(seed)
    texts = [" ".join(rng.choice(WORDS, size=rng.integers(5, 15))) for _ in range(n)]
    df = pd.DataFrame({"text": texts,
                       "sentiment_label": rng.integers(0, 3, n),
                       "fls_label": rng.integers(0, 3, n),
                       "esg_label": rng.integers(0, 4, n)})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)
    return path
