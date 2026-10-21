"""
paper/make_figures.py
─────────────────────
Draws the two result figures of the paper from the numbers printed by
paper_results.ipynb (Colab, stage 2 "v2", 3 seeds, test set). Run from the repository root:

    python paper/make_figures.py

Writes paper/figures/benchmarks_snr.pdf (vector, full width) and paper/figures/caps.pdf
(vector, single column).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
COL = {"AWGN": "#2a78d6", "Rayleigh": "#eb6834"}
INK = "#2b2b2b"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.3,
                     "lines.markersize": 3.2, "pdf.fonttype": 42})

# ── the three benchmarks per SNR: J = 32, K = 16, lambda = 0.005, test set ───
SNR = [-15, -10, -5, -2, 0, 2, 5, 10, 15, 20]
BENCH = {
    "AWGN": {"alloc_sym": [63.5, 63.5, 58.9, 45.1, 34.6, 27.0, 20.1, 14.1, 12.2, 12.1],
             "all_acc": [74.7, 87.9, 89.2, 89.2, 89.2, 89.1, 89.3, 89.2, 89.2, 89.2],
             "equal_acc": [59.3, 78.0, 88.0, 88.8, 88.9, 89.1, 89.2, 89.2, 89.1, 89.2],
             "alloc_acc": [58.8, 78.4, 87.8, 88.5, 88.7, 88.9, 89.2, 89.3, 89.3, 89.3]},
    "Rayleigh": {"alloc_sym": [62.4, 62.9, 63.4, 62.7, 58.6, 51.7, 40.8, 28.5, 23.0, 17.0],
                 "all_acc": [69.4, 84.6, 88.3, 88.8, 88.7, 88.8, 88.7, 88.5, 88.5, 88.5],
                 "equal_acc": [52.5, 66.7, 80.7, 85.8, 87.4, 88.4, 88.6, 88.7, 88.7, 88.8],
                 "alloc_acc": [51.9, 66.3, 81.1, 85.7, 87.3, 87.8, 88.1, 88.5, 88.6, 88.7]},
}

# ── information-bottleneck baseline (Shao et al.), K = 16, beta chosen on validation, test set.
# None = not run yet on that channel (the curve is then left out).
IB = {
    "AWGN": {"sym": [64.00, 62.33, 62.33, 54.00, 54.00, 48.33, 41.33, 32.33, 30.33, 25.00],
             "acc": [63.42, 80.04, 88.60, 89.19, 89.50, 89.49, 89.73, 89.79, 89.86, 89.44]},
    "Rayleigh": None,
}

# ── every cap, lambda = 0.005, test set: symbols saved (%) and accuracy difference [95% CI]
CAPS = [8, 16, 32, 64]
CAP = {
    "AWGN": {"saved": [26.34, 45.12, 66.09, 82.53],
             "d": [-0.06, -0.05, -0.85, -1.52], "lo": [-0.18, -0.16, -0.97, -1.66], "hi": [0.06, 0.06, -0.73, -1.38]},
    "Rayleigh": {"saved": [0.81, 26.41, 48.59, 72.30],
                 "d": [0.27, -0.24, -0.58, -1.96], "lo": [0.11, -0.35, -0.71, -2.12], "hi": [0.42, -0.12, -0.46, -1.81]},
}


def benchmarks_snr():
    """Full-width figure: symbols per SNR (both channels), then accuracy per SNR for each channel.
    Colour = channel, line style = system (the two no-allocator systems are drawn in black/grey)."""
    eq = dict(color=INK, ls="--", marker="s", mfc="white", ms=4.4, lw=1.0, zorder=2)
    al = dict(marker="o", ms=3.0, lw=1.3, zorder=3)
    ib = dict(ls="-.", marker="^", mfc="white", ms=3.6, lw=1.1, zorder=3)
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(7.16, 2.05))
    a0.plot(SNR, [384] * len(SNR), color="#8a8a85", ls=":", lw=1.2)
    a0.plot(SNR, [64] * len(SNR), color=INK, ls="--", lw=1.0)
    for ch, c in COL.items():
        a0.plot(SNR, BENCH[ch]["alloc_sym"], color=c, **al)
        if IB[ch]:
            a0.plot(SNR, IB[ch]["sym"], color=c, **ib)
    a0.set_yscale("log")
    a0.yaxis.set_major_formatter(ScalarFormatter())
    a0.yaxis.set_minor_formatter(NullFormatter())
    a0.set_yticks([12, 24, 48, 96, 192, 384])
    a0.set_ylabel("Symbols per sentence")
    a0.set_title("Symbols sent")
    for ax, ch in ((a1, "AWGN"), (a2, "Rayleigh")):
        d, c = BENCH[ch], COL[ch]
        ax.plot(SNR, d["all_acc"], color="#8a8a85", ls=":", lw=1.2)
        ax.plot(SNR, d["equal_acc"], **eq)
        ax.plot(SNR, d["alloc_acc"], color=c, **al)
        if IB[ch]:
            ax.plot(SNR, IB[ch]["acc"], color=c, **ib)
        ax.set_title(f"Mean accuracy, {ch}")
        ax.set_ylabel("Mean accuracy (%)")
    for ax in (a0, a1, a2):
        ax.set_xlabel("SNR (dB)")
        ax.set_xticks([-15, -10, -5, 0, 5, 10, 15, 20])
    handles = [Line2D([], [], color=COL["AWGN"], lw=2.0, label="AWGN"),
               Line2D([], [], color=COL["Rayleigh"], lw=2.0, label="Rayleigh"),
               Line2D([], [], color="#8a8a85", ls=":", lw=1.2, label="All tokens, no cap (384 symbols)"),
               Line2D([], [], label="Equal split, $K{=}16$ (64 symbols)", **eq),
               Line2D([], [], color=INK, label="Allocator, $K{=}16$", **al)]
    if any(IB.values()):
        handles.append(Line2D([], [], color=INK, label="Information bottleneck [Shao], $K{=}16$", **ib))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.6, handlelength=2.6)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.2)
    fig.savefig(os.path.join(OUT, "benchmarks_snr.pdf"), bbox_inches="tight")
    plt.close(fig)


def caps():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.85))
    x = range(len(CAPS))
    w = 0.38
    for i, (ch, c) in enumerate(COL.items()):
        d = CAP[ch]
        xs = [k + (i - 0.5) * w for k in x]
        a1.bar(xs, d["saved"], w, color=c, label=ch)
        err = [[m - lo for m, lo in zip(d["d"], d["lo"])], [hi - m for m, hi in zip(d["d"], d["hi"])]]
        a2.bar(xs, d["d"], w, color=c, yerr=err, error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": INK})
    a2.axhline(0, color=INK, lw=0.7)
    a1.set_ylabel("Symbols saved (%)")
    a2.set_ylabel(r"$\Delta$ accuracy (points)")
    for ax in (a1, a2):
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(k) for k in CAPS])
        ax.set_xlabel("Cap $K$ (task tokens)")
        ax.grid(axis="x", visible=False)
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8)
    fig.savefig(os.path.join(OUT, "caps.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    benchmarks_snr()
    caps()
    print("wrote", os.path.join(OUT, "benchmarks_snr.pdf"), "and", os.path.join(OUT, "caps.pdf"))
