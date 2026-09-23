"""
common/stats.py
───────────────
Paired bootstrap over (seed, channel, SNR, sentence) units (CLAUDE.md §7).

Two methods are paired on the same unit because they were evaluated on the same
sentence with the same fixed channel realization (common.channel.eval_generator).
"better" = CI of the mean difference excludes 0.
"""

import numpy as np
import pandas as pd

UNIT = ["seed", "channel", "snr_db", "sentence_id"]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000,
                     alpha: float = 0.05, seed: int = 0):
    """Mean of (a - b) with a percentile bootstrap CI. a, b are aligned per unit."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(n_resamples)
    chunk = max(1, 2_000_000 // max(n, 1))          # bound memory of the index matrix
    for i in range(0, n_resamples, chunk):
        m = min(chunk, n_resamples - i)
        idx = rng.integers(0, n, size=(m, n))
        means[i:i + m] = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "significant": bool(lo > 0 or hi < 0), "n_units": int(n)}


def compare_methods(df: pd.DataFrame, method_a: str, method_b: str, metric: str,
                    group_cols=("channel",), **kw) -> pd.DataFrame:
    """
    Paired bootstrap of metric(method_a) - metric(method_b), per group (default per
    channel). Rows are joined on the unit key plus the priority, so both methods
    must have been evaluated on the same units.
    """
    key = UNIT + ["w_S", "w_F", "w_E"]
    a = df[df.method == method_a][key + [metric]]
    b = df[df.method == method_b][key + [metric]]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    rows = []
    for g, sub in m.groupby(list(group_cols)) if group_cols else [((), m)]:
        r = paired_bootstrap(sub[f"{metric}_a"].values, sub[f"{metric}_b"].values, **kw)
        r.update(dict(zip(group_cols, g if isinstance(g, tuple) else (g,))))
        r.update({"a": method_a, "b": method_b, "metric": metric})
        rows.append(r)
    return pd.DataFrame(rows)
