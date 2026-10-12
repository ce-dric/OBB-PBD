#!/usr/bin/env python3
"""
Generate paper/figures/fig1_architecture.{png,pdf}.

Regenerate from the repo root with
    uv run --no-project --with matplotlib python scripts/make_fig1_architecture.py

History (older notes condensed; see git log for the full text):
  2026-08-08  footer text corrected: the HBB block uses corner coordinates, the OBB block
              center-and-size, so the first four slots are not "unchanged".
  2026-08-24/26  in-image annotations that duplicated the caption removed.
  2026-09-29  the figure only showed the
              token blocks; an end-to-end architecture diagram was added.
              Panel (a) now shows the full inference path (image -> MoonViT -> projector ->
              language model -> one parallel pass over the six geometry slots -> codec
              decode -> oriented box), which parts are adapted, and the training losses;
              panel (b) keeps the block comparison. Also fixes the </box> token that was
              clipped at the right edge of the previous version.
"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(REPO_ROOT, "paper", "figures", "fig1_architecture.png")
OUT_PDF = os.path.join(REPO_ROOT, "paper", "figures", "fig1_architecture.pdf")

FIG_W, FIG_H = 14.0, 10.6

COL_PROMPT = "#c8c2b4"
COL_IMAGE_TOK = "#9fb7a3"
COL_TRIGGER = "#3a3a3a"
COL_BLUE = "#2f6fed"
COL_ORANGE = "#e8630a"
COL_MODULE = "#eef2f8"
COL_MODULE_EDGE = "#44546a"
COL_ADAPT = "#7a3db8"
COL_TEXT_DARK = "#1a1a1a"
COL_TEXT_GRAY = "#555555"


def box(ax, x, y, w, h, label, face, text=COL_TEXT_DARK, size=11, edge=None, lw=0, weight="normal",
        style="round,pad=0.02,rounding_size=0.06", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw, facecolor=face,
                                edgecolor=edge or face, linestyle=ls))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size, color=text,
                fontweight=weight, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color=COL_TEXT_GRAY, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=lw, linestyle=ls))


def panel_a(ax, top):
    ax.text(0.3, top, "(a) End-to-end OBB-PBD grounding",
            fontsize=13.5, fontweight="bold", color=COL_TEXT_DARK)
    y_img, y_txt = top - 1.55, top - 3.05
    h = 0.95

    # inputs
    box(ax, 0.3, y_img, 1.9, h, "UAV image\n(1024x576)", "#d9d4c7")
    box(ax, 0.3, y_txt, 1.9, h, "Referring\nexpression", "#e9e4d6")
    ax.text(1.25, y_txt - 0.15, '"the bottom-most car"', ha="center", va="top", fontsize=9.5,
            style="italic", color=COL_TEXT_GRAY)

    # vision path
    box(ax, 2.7, y_img, 2.1, h, "MoonViT\nvision encoder", COL_MODULE, edge=COL_MODULE_EDGE, lw=1.2, size=10)
    box(ax, 5.2, y_img, 1.7, h, "MLP\nprojector", COL_MODULE, edge=COL_MODULE_EDGE, lw=1.2)
    arrow(ax, 2.2, y_img + h / 2, 2.7, y_img + h / 2)
    arrow(ax, 4.8, y_img + h / 2, 5.2, y_img + h / 2)
    ax.text(3.75, y_img - 0.12, "LoRA on last 6 blocks", ha="center", va="top", fontsize=9, color=COL_ADAPT)
    ax.text(6.05, y_img - 0.12, "fully fine-tuned", ha="center", va="top", fontsize=9, color=COL_ADAPT)

    # text path
    box(ax, 2.7, y_txt, 2.1, h, "Tokenizer +\nembedding", COL_MODULE, edge=COL_MODULE_EDGE, lw=1.2)
    arrow(ax, 2.2, y_txt + h / 2, 2.7, y_txt + h / 2)

    # input sequence
    seq_x, seq_y, seq_h = 7.4, top - 2.45, 0.75
    ax.text(seq_x + 6.2, seq_y + seq_h + 0.12, "input sequence", ha="right", fontsize=10, color=COL_TEXT_GRAY)
    x = seq_x
    for w, label, face, tc in ((0.95, "image\ntokens", COL_IMAGE_TOK, COL_TEXT_DARK),
                               (0.95, "prompt\ntokens", COL_PROMPT, COL_TEXT_DARK),
                               (0.7, "<box>", COL_TRIGGER, "white")):
        box(ax, x, seq_y, w, seq_h, label, face, text=tc, size=8)
        x += w + 0.06
    mask_x0 = x
    for _ in range(6):
        box(ax, x, seq_y, 0.5, seq_h, "[m]", "#b9c9ef", text=COL_TEXT_DARK, size=8)
        x += 0.54
    mask_x1 = x - 0.04
    ax.text((mask_x0 + mask_x1) / 2, seq_y - 0.1, "6 mask slots", ha="center", va="top", fontsize=9,
            color=COL_TEXT_GRAY)
    arrow(ax, 6.9, y_img + h / 2, seq_x + 0.45, seq_y + seq_h)
    arrow(ax, 4.8, y_txt + h / 2, seq_x + 1.45, seq_y)

    # language model
    lm_y = top - 5.0
    box(ax, 7.4, lm_y, 6.2, 1.0, "", COL_MODULE, edge=COL_MODULE_EDGE, lw=1.2)
    ax.text(10.5, lm_y + 0.74, "Qwen2.5-3B language model", ha="center", va="center", fontsize=12,
            fontweight="bold", color=COL_TEXT_DARK)
    ax.text(10.5, lm_y + 0.3, "LoRA r=16; every mask slot sees the\nfull image + prompt prefix",
            ha="center", va="center", fontsize=8.8, color=COL_ADAPT)
    arrow(ax, 10.5, seq_y - 0.35, 10.5, lm_y + 1.0)
    ax.text(10.65, (seq_y - 0.35 + lm_y + 1.0) / 2, "one forward pass", fontsize=9.5, va="center",
            color=COL_TEXT_DARK, fontweight="bold")

    # outputs
    out_y = top - 6.55
    labels = ["cx", "cy", "w", "h", "sin2θ", "cos2θ"]
    colors = [COL_BLUE] * 4 + [COL_ORANGE] * 2
    x = 7.4
    for label, color in zip(labels, colors):
        box(ax, x, out_y, 0.78, 0.7, label, color, text="white", size=9)
        x += 0.84
    ax.text(7.4, out_y - 0.12, "each slot: 1001-way classification",
            ha="left", va="top", fontsize=8.8, color=COL_TEXT_GRAY)
    arrow(ax, 10.5, lm_y, 10.5, out_y + 0.7)

    # decode -> oriented box
    arrow(ax, 12.45, out_y + 0.35, 12.85, out_y + 0.35)
    ax.text(13.3, out_y + 1.0, "decode:\nθ = ½ atan2(s, c)", ha="center", fontsize=9, va="bottom",
            color=COL_TEXT_DARK)
    ax.add_patch(Polygon([[12.95, out_y + 0.12], [13.75, out_y + 0.42], [13.62, out_y + 0.78],
                          [12.82, out_y + 0.48]], closed=True, fill=False, edgecolor="#1f9e3a",
                         linewidth=2.4))
    ax.text(13.3, out_y - 0.12, "oriented\nbox", ha="center", va="top", fontsize=9, color=COL_TEXT_DARK)

    # training losses
    ty = lm_y - 0.35
    box(ax, 0.3, ty, 6.6, 1.35, "", "white", edge=COL_TEXT_GRAY, lw=1.1, ls="--")
    ax.text(0.45, ty + 1.2, "Training only", fontsize=9.5, fontweight="bold", color=COL_TEXT_GRAY, va="top")
    ax.text(0.45, ty + 0.8,
            r"$\mathcal{L} = \mathcal{L}_{\mathrm{CE}}"
            r" + \lambda_{s}\mathcal{L}_{\mathrm{spatial}}"
            r" + \lambda_{c}\mathcal{L}_{\mathrm{circle}}"
            r" + \lambda_{d}\mathcal{L}_{\mathrm{direction}}$",
            fontsize=11, va="top", color=COL_TEXT_DARK)
    ax.text(0.45, ty + 0.25, "auxiliary terms use each slot's soft-argmax",
            fontsize=8.8, va="top", color=COL_TEXT_GRAY)
    arrow(ax, 7.35, out_y + 0.35, 6.2, ty, ls="--")


def block_row(ax, y, title, labels, colors, block_size, n_future):
    h = 0.72
    ax.text(0.3, y + h + 0.12, title, fontsize=12, fontweight="bold", color=COL_TEXT_DARK)
    x = 0.3
    box(ax, x, y, 2.0, h, "image +\nprompt tokens", COL_PROMPT, size=9)
    x += 2.12
    box(ax, x, y, 0.9, h, "<box>", COL_TRIGGER, text="white", size=9.5)
    x += 1.02
    start = x
    for label, color in zip(labels, colors):
        box(ax, x, y, 0.9, h, label, color, text="white", size=10.5)
        x += 0.97
    end = x - 0.07
    box(ax, x + 0.05, y, 1.1, h, "</box>", COL_TRIGGER, text="white", size=9.5)
    ax.text(end + 1.45, y + h / 2,
            f"{len(labels)} slots, 1 parallel forward pass\n(block_size={block_size}, n_future_tokens={n_future})",
            ha="left", va="center", fontsize=9, color=COL_TEXT_GRAY)


def main():
    # Data coordinates span FIG_W x FIG_H; the canvas is drawn smaller so that point-sized text
    # stays legible when the figure is scaled to the journal's text width.
    fig, ax = plt.subplots(figsize=(FIG_W * 0.8, FIG_H * 0.8))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    panel_a(ax, top=10.3)

    ax.plot([0.3, FIG_W - 0.3], [2.95, 2.95], color="#cccccc", lw=1)
    ax.text(0.3, 2.45, "(b) Coordinate block: LocateAnything (HBB) vs. this work (OBB)",
            fontsize=13.5, fontweight="bold", color=COL_TEXT_DARK)
    block_row(ax, 1.25, "LocateAnything: four-coordinate HBB block", ["x1", "y1", "x2", "y2"],
              [COL_BLUE] * 4, 6, 4)
    block_row(ax, 0.0, "This work: six-slot OBB-PBD block", ["cx", "cy", "w", "h", "sin2θ", "cos2θ"],
              [COL_BLUE] * 4 + [COL_ORANGE] * 2, 8, 6)

    fig.savefig(OUT_PNG, dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_PDF, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    print("wrote", OUT_PNG)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
