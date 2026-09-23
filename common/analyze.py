"""
common/analyze.py
─────────────────
Tables, paired-bootstrap CIs and figures from per-sentence logs (CLAUDE.md §6-7).

    python -m common.analyze runs/exp2_main_cap8 runs/legacy_B1 --ref allocator --out runs/report_cap8

Writes to --out:
    summary.csv            mean accuracy per method x channel x task (+ avg symbols)
    bootstrap.csv          ref - method, paired over (seed, channel, SNR, sentence)
    acc_vs_snr_<ch>.png    accuracy per task vs SNR
    symbols_vs_snr.png     average symbols sent vs SNR
    allocation_<ch>.png    mean tokens per task vs SNR, and histogram of totals (allocator)
    saturation.png         accuracy vs tokens per task (Experiment 1)
    priority.png           response to w_E (Experiment 3), when several priorities exist
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common.stats import compare_methods  # noqa: E402

TASK_KEYS = ("S", "F", "E")
TASK_NAMES = {"S": "Sentiment", "F": "FLS", "E": "ESG"}
TASK_COLORS = {"S": "#2a78d6", "F": "#1baf7a", "E": "#eb6834"}       # matches the design figure
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(paths):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "**", "per_sentence.csv.gz"), recursive=True))
    if not files:
        raise SystemExit(f"no per_sentence.csv.gz under {paths}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def method_colors(methods):
    """Colour follows the method name (fixed order of first appearance), never its rank."""
    return {m: SERIES[i % len(SERIES)] for i, m in enumerate(methods)}


def _style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _line(ax, x, y, color, label):
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1, label=label)


def _legend_below(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06 + 0.04 * ((len(labels) - 1) // 5), 1, 1))


def summary(df):
    cols = [f"correct_{k}" for k in TASK_KEYS] + ["symbols"]
    out = df.groupby(["method", "channel"])[cols].mean().reset_index()
    for k in TASK_KEYS:
        out[f"correct_{k}"] *= 100
    return out.rename(columns={f"correct_{k}": f"acc_{TASK_NAMES[k]}" for k in TASK_KEYS})


def plot_acc_vs_snr(df, methods, out):
    colors = method_colors(methods)
    for ch, sub in df.groupby("channel"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
        for ax, k in zip(axes, TASK_KEYS):
            g = sub.groupby(["method", "snr_db"])[f"correct_{k}"].mean().mul(100).reset_index()
            for m in methods:
                r = g[g.method == m]
                if len(r):
                    _line(ax, r.snr_db, r[f"correct_{k}"], colors[m], m)
            ax.set_title(TASK_NAMES[k], color=INK, loc="left", fontsize=11)
            _style(ax, "SNR (dB)", "accuracy (%)")
        fig.suptitle(f"Accuracy vs SNR, {ch}", color=INK, x=0.01, ha="left")
        _legend_below(fig, axes[-1])
        fig.savefig(os.path.join(out, f"acc_vs_snr_{ch}.png"), dpi=160)
        plt.close(fig)


def plot_symbols(df, methods, out):
    colors = method_colors(methods)
    chans = sorted(df.channel.unique())
    fig, axes = plt.subplots(1, len(chans), figsize=(4.3 * len(chans), 3.6), sharey=True, squeeze=False)
    for ax, ch in zip(axes[0], chans):
        g = df[df.channel == ch].groupby(["method", "snr_db"]).symbols.mean().reset_index()
        for m in methods:
            r = g[g.method == m]
            if len(r):
                _line(ax, r.snr_db, r.symbols, colors[m], m)
        ax.set_title(ch, color=INK, loc="left", fontsize=11)
        _style(ax, "SNR (dB)", "avg symbols / sentence")
    _legend_below(fig, axes[0][-1])
    fig.savefig(os.path.join(out, "symbols_vs_snr.png"), dpi=160)
    plt.close(fig)


def plot_allocation(df, out, method="allocator"):
    a = df[df.method == method]
    if a.empty:
        return
    for ch, sub in a.groupby("channel"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
        g = sub.groupby("snr_db")[[f"k_{k}" for k in TASK_KEYS]].mean()
        for k in TASK_KEYS:
            _line(ax1, g.index, g[f"k_{k}"], TASK_COLORS[k], TASK_NAMES[k])
        ax1.set_ylim(0.8, 4.2)
        ax1.set_title("Mean task tokens per task", color=INK, loc="left", fontsize=11)
        _style(ax1, "SNR (dB)", "task tokens")
        ax1.legend(frameon=False, fontsize=8)
        tot = sub.assign(total=sub[[f"k_{k}" for k in TASK_KEYS]].sum(1))
        h = pd.crosstab(tot.snr_db, tot.total, normalize="index")
        im = ax2.imshow(h.T.values, aspect="auto", origin="lower", cmap="Blues", vmin=0, vmax=1)
        ax2.set_xticks(range(len(h.index)), [f"{s:g}" for s in h.index])
        ax2.set_yticks(range(len(h.columns)), [str(int(c)) for c in h.columns])
        ax2.set_title("Share of sentences by total task tokens", color=INK, loc="left", fontsize=11)
        _style(ax2, "SNR (dB)", "total task tokens")
        ax2.grid(False)
        fig.colorbar(im, ax=ax2, fraction=0.04)
        fig.suptitle(f"Allocation, {ch}", color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"allocation_{ch}.png"), dpi=160)
        plt.close(fig)


def plot_saturation(df, out):
    s = df[df.method.str.startswith("uniform_k")].copy()
    if s.empty:
        return
    s["k"] = s.method.str[-1].astype(int)
    snrs = sorted(s.snr_db.unique())
    fig, axes = plt.subplots(1, len(snrs), figsize=(3.6 * len(snrs), 3.4), sharey=True, squeeze=False)
    for ax, snr in zip(axes[0], snrs):
        g = s[s.snr_db == snr].groupby("k")[[f"correct_{k}" for k in TASK_KEYS]].mean() * 100
        for k in TASK_KEYS:
            _line(ax, g.index, g[f"correct_{k}"], TASK_COLORS[k], TASK_NAMES[k])
        ax.set_xticks([1, 2, 3, 4])
        ax.set_title(f"{snr:g} dB", color=INK, loc="left", fontsize=11)
        _style(ax, "task tokens per task", "accuracy (%)")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Saturation map (all channels pooled)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "saturation.png"), dpi=160)
    plt.close(fig)


def plot_priority(df, methods, out):
    if df.w_E.round(3).nunique() < 3:
        return
    colors = method_colors(methods)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for m in methods:
        g = df[df.method == m].groupby(df.w_E.round(3))
        acc = g[[f"correct_{k}" for k in TASK_KEYS]].mean() * 100
        for ax, k in zip(axes[:3], TASK_KEYS):
            _line(ax, acc.index, acc[f"correct_{k}"], colors[m], m)
        if df[df.method == m].k_E.notna().any():
            _line(axes[3], acc.index, g.k_E.mean(), colors[m], m)
    for ax, k in zip(axes[:3], TASK_KEYS):
        ax.set_title(TASK_NAMES[k], color=INK, loc="left", fontsize=11)
        _style(ax, "ESG priority w_E", "accuracy (%)")
    axes[3].set_title("ESG task tokens", color=INK, loc="left", fontsize=11)
    _style(axes[3], "ESG priority w_E", "mean k_E")
    _legend_below(fig, axes[3])
    fig.savefig(os.path.join(out, "priority.png"), dpi=160)
    plt.close(fig)


def bootstrap_table(df, ref, n_resamples):
    rows = []
    others = [m for m in df.method.unique() if m != ref and not m.startswith("uniform_k")]
    for m in others:
        for metric in [f"correct_{k}" for k in TASK_KEYS] + ["symbols"]:
            r = compare_methods(df, ref, m, metric, n_resamples=n_resamples)
            if len(r):
                rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run folders (searched recursively)")
    p.add_argument("--out", default="runs/report")
    p.add_argument("--ref", default="allocator", help="method compared against all others")
    p.add_argument("--resamples", type=int, default=10_000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load_runs(args.runs)
    methods = [m for m in dict.fromkeys(df.method) if not m.startswith("uniform_k")]

    s = summary(df)
    s.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print(s.round(2).to_string(index=False))
    main_df = df[~df.method.str.startswith("uniform_k")]
    plot_acc_vs_snr(main_df, methods, args.out)
    plot_symbols(main_df, methods, args.out)
    plot_allocation(df, args.out)
    plot_saturation(df, args.out)
    plot_priority(main_df, methods, args.out)
    if args.ref in methods:
        b = bootstrap_table(main_df, args.ref, args.resamples)
        b.to_csv(os.path.join(args.out, "bootstrap.csv"), index=False)
        print(b.round(4).to_string(index=False))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
