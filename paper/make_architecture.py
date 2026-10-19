"""
paper/make_architecture.py
──────────────────────────
Draws the pipeline figure of the paper (Fig. 1) as a vector PDF, with the size of the
data leaving every block written above it. Run from the repository root:

    python paper/make_architecture.py

Writes paper/figures/architecture.pdf (and architecture_preview.png for a quick look).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "pdf.fonttype": 42})

INK, MUTED, DIM = "#1f2328", "#57606a", "#0b5cad"
EDGE, FILL = "#b9bec6", "#f6f7f9"
KIND = {  # pill colour per kind of block
    "pre": ("#ece4fb", "#5b2bb5", "pretrained, fine-tuned"),
    "new": ("#dcf5e3", "#17733a", "trained from scratch"),
    "fixed": ("#e8eaed", "#4a5058", "no training"),
}
ALLOC_EDGE, ALLOC_FILL = "#3b6fd9", "#eef3fd"
CH_EDGE, CH_FILL, CQI = "#e3a33b", "#fff7ea", "#c8640a"

W, H = 7.16, 3.40                 # inches: IEEE two-column text width
BW, BH = 1.22, 0.74               # block width / height
GAP = (W - 2 * 0.06 - 5 * BW) / 4
X = [0.06 + i * (BW + GAP) for i in range(5)]
Y_TX, Y_RX, Y_CH, CH_H = 2.34, 0.42, 1.44, 0.62


def block(ax, x, y, title, lines, kind, size=None, edge=EDGE, fill=FILL, lw=0.7, h=BH):
    ax.add_patch(FancyBboxPatch((x, y), BW, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=fill, ec=edge, lw=lw))
    ax.text(x + BW / 2, y + h - 0.11, title, ha="center", va="center", fontsize=6.6,
            fontweight="bold", color=INK)
    for i, line in enumerate(lines):
        ax.text(x + BW / 2, y + h - 0.25 - i * 0.115, line, ha="center", va="center",
                fontsize=5.3, color=MUTED)
    if kind:
        bg, fg, label = KIND[kind]
        ax.add_patch(FancyBboxPatch((x + 0.16, y + 0.045), BW - 0.32, 0.105,
                                    boxstyle="round,pad=0,rounding_size=0.05", fc=bg, ec="none"))
        ax.text(x + BW / 2, y + 0.098, label, ha="center", va="center", fontsize=4.9,
                color=fg, fontweight="bold")
    if size:   # size of the data this block outputs, written above the block
        ax.text(x + BW / 2, y + h + 0.075, size, ha="center", va="center", fontsize=5.4, color=DIM)


def arrow(ax, p, q, color=MUTED, ls="-", lw=0.7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, color=color,
                                 lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def main():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(0.06, H - 0.07, "TRANSMITTER", fontsize=6, fontweight="bold", color="#2f6fb5", va="center")
    ax.text(0.06, Y_RX + BH + 0.32, "RECEIVER", fontsize=6, fontweight="bold", color="#2f8a4a", va="center")

    # ── transmitter ───────────────────────────────────────────────────────────
    block(ax, X[0], Y_TX, "1  Tokenizer", ["sentence $\\to$ $n$ word pieces,", "adds the $TJ$ task tokens"],
          "fixed", size="$2 + TJ + n$ token ids")
    block(ax, X[1], Y_TX, "2  FinBERT encoder", ["task tokens read the words", "and their own task's tokens"],
          "pre", size="CLS: 768,  task tokens: $TJ \\times 768$")
    block(ax, X[2], Y_TX, "3  Allocator", ["CLS + SNR + priority $\\mathbf{w}$", "$\\to$ tokens per task $k_t$"],
          "new", size="in: $768 + 32 + 3$,  out: $k_S, k_F, k_E$",
          edge=ALLOC_EDGE, fill=ALLOC_FILL, lw=1.1)
    block(ax, X[3], Y_TX, "4  Channel encoder", ["one per task, per kept token:", "$768 \\to 256 \\to 8$ reals"],
          "new", size="4 complex symbols per token")
    block(ax, X[4], Y_TX, "5  Frame", ["header with the counts $k_t$", "+ 4 symbols per kept token"],
          "fixed", size="$N = 4(k_S{+}k_F{+}k_E)$ symbols")

    # what the allocator decision does, written under the allocator
    for i, line in enumerate(["keeps only the first $k_t$ task", "tokens of each task; the rest",
                              "are never encoded or sent"]):
        ax.text(X[2] + BW / 2 - 0.12, Y_TX - 0.08 - i * 0.095, line, ha="center", va="center",
                fontsize=5.1, color=ALLOC_EDGE, style="italic")

    # ── channel ───────────────────────────────────────────────────────────────
    block(ax, X[4], Y_CH, "6  Wireless channel", ["$y = h x + n$,  $N$ complex uses",
                                                "AWGN or Rayleigh fading"],
          None, edge=CH_EDGE, fill=CH_FILL, lw=1.0, h=CH_H)

    # ── receiver ──────────────────────────────────────────────────────────────
    block(ax, X[4], Y_RX, "7  Equalize", ["zero forcing $\\hat{x} = y / h$", "($h$ known at the receiver)"],
          "fixed")
    ax.text(X[4] + BW / 2 - 0.06, Y_RX + BH + 0.09, "$N$ symbols", ha="right", va="center",
            fontsize=5.4, color=DIM)
    block(ax, X[3], Y_RX, "8  Read header", ["uses the counts $k_t$ to split", "the symbols by task and token"],
          "fixed", size="$k_S{+}k_F{+}k_E$ blocks of 4 symbols")
    block(ax, X[2], Y_RX, "9  Channel decoder", ["one per task, per token:", "$8 \\to 256 \\to 768$"],
          "new", size="$k_t \\times 768$ for task $t$")
    block(ax, X[1], Y_RX, "10  Combine", ["attention pooling per task,", "ignores tokens not sent"],
          "new", size="one $768$ vector per task")
    block(ax, X[0], Y_RX, "11  Task heads", ["one linear layer per task:", "Sentiment, FLS, ESG"],
          "new", size="$768 \\to 3$, $3$, $4$ classes")

    # ── arrows ────────────────────────────────────────────────────────────────
    mid_tx, mid_rx = Y_TX + BH / 2, Y_RX + BH / 2
    for i in range(4):
        arrow(ax, (X[i] + BW, mid_tx), (X[i + 1], mid_tx))
        arrow(ax, (X[i + 1], mid_rx), (X[i] + BW, mid_rx))
    cx = X[4] + BW / 2
    arrow(ax, (cx, Y_TX), (cx, Y_CH + CH_H))
    arrow(ax, (cx, Y_CH), (cx, Y_RX + BH))

    # CQI feedback: channel -> allocator (dashed), entering the allocator on its right side
    fx = X[2] + BW - 0.10
    fy = Y_CH + CH_H / 2
    ax.plot([X[4], fx], [fy, fy], color=CQI, lw=0.8, ls=(0, (3, 2)))
    arrow(ax, (fx, fy), (fx, Y_TX), color=CQI, ls=(0, (3, 2)), lw=0.8)
    tx = (fx + X[4]) / 2 + 0.01
    ax.text(tx, fy + 0.06, "CQI: the receiver reports the average SNR",
            ha="center", va="bottom", fontsize=5.1, color=CQI, fontweight="bold")
    ax.text(tx, fy - 0.05, "never the fading of individual symbols",
            ha="center", va="top", fontsize=5.0, color=CQI, style="italic")

    # legend
    lx, ly = 0.06, 0.13
    ax.text(lx, ly, "Legend:", fontsize=5.4, fontweight="bold", color=INK, va="center")
    lx += 0.42
    for kind in ("pre", "new", "fixed"):
        bg, fg, label = KIND[kind]
        w = 0.06 * len(label) * 0.62 + 0.12
        ax.add_patch(FancyBboxPatch((lx, ly - 0.05), w, 0.10, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=bg, ec="none"))
        ax.text(lx + w / 2, ly, label, ha="center", va="center", fontsize=4.9, color=fg, fontweight="bold")
        lx += w + 0.08
    ax.text(lx + 0.05, ly, "$T{=}3$ tasks, $J$ task tokens per task, $k_t$ tokens sent by task $t$. "
            "Blue: size of the data leaving each block.", fontsize=5.0, color=MUTED, va="center")

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "architecture.pdf"))
    fig.savefig(os.path.join(OUT, "architecture_preview.png"), dpi=220)
    plt.close(fig)
    print("wrote", os.path.join(OUT, "architecture.pdf"))


if __name__ == "__main__":
    main()
