# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Diagram: where one Muon matrix lives and moves in a training step, FSDP2 +
DistMuon against FlexShard + DistMuon. Four ranks; the matrix's owner is rank 2."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
P, G, GR, U = (
    "#2b6cb0",
    "#bee3f8",
    "#dd6b20",
    "#38a169",
)  # parameter, gathered parameter, gradient, Newton-Schulz / update
SQ, GAP, CELL_W, CELL_GAP = 0.9, 0.3, 4 * 0.9 + 3 * 0.3, 1.0
OWNER = 2
fig, ax = plt.subplots(figsize=(7.6, 3.7), dpi=220)


def cell_x(col):
    return col * (CELL_W + CELL_GAP)


def square(x, y, fill=None, stripe=None, label=None):
    """fill: whole-matrix color; stripe: (rank, color) for a Shard(0) quarter."""
    ax.add_patch(
        plt.Rectangle((x, y), SQ, SQ, fc="white", ec="#4a5568", lw=0.8, zorder=2)
    )
    if fill:
        ax.add_patch(
            plt.Rectangle((x, y), SQ, SQ, fc=fill, ec="#4a5568", lw=0.8, zorder=3)
        )
    if stripe:
        r, color = stripe
        ax.add_patch(
            plt.Rectangle(
                (x, y + SQ - (r + 1) * SQ / 4),
                SQ,
                SQ / 4,
                fc=color,
                ec="none",
                zorder=3,
            )
        )
    if label:
        ax.text(
            x + SQ / 2,
            y + SQ / 2,
            label,
            ha="center",
            va="center",
            fontsize=6.3,
            color="white",
            fontweight="bold",
            zorder=4,
        )


def arrows(col, y, to_owner, both=False):
    for r in range(4):
        if r == OWNER:
            continue
        x_r = cell_x(col) + r * (SQ + GAP) + SQ / 2
        x_o = cell_x(col) + OWNER * (SQ + GAP) + SQ / 2
        a, b = (
            ((x_r, y + SQ + 0.05), (x_o, y + SQ + 0.05))
            if to_owner
            else ((x_o, y + SQ + 0.05), (x_r, y + SQ + 0.05))
        )
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                connectionstyle=f"arc3,rad={-0.45 if a[0] < b[0] else 0.45}",
                arrowstyle=("<|-|>" if both else "-|>")
                + ",head_length=3,head_width=1.8",
                color="#4a5568",
                lw=0.7,
                zorder=5,
            )
        )


def row(y, cells, texts):
    for col, spec in enumerate(cells):
        for r in range(4):
            square(cell_x(col) + r * (SQ + GAP), y, **spec[r])
        for r in range(4):
            ax.text(
                cell_x(col) + r * (SQ + GAP) + SQ / 2,
                y - 0.14,
                f"r{r}",
                ha="center",
                va="top",
                fontsize=6,
                color="#718096",
            )
        ax.text(
            cell_x(col) + CELL_W / 2,
            y - 0.5,
            texts[col],
            ha="center",
            va="top",
            fontsize=6.6,
            linespacing=1.22,
        )


COLS = ["Between steps (storage)", "Forward", "Backward", "Optimizer step"]
for col, name in enumerate(COLS):
    ax.text(
        cell_x(col) + CELL_W / 2,
        2.75,
        name,
        ha="center",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color="#1a202c",
    )

y1, y2 = 1.0, -2.6
row(
    y1,
    [
        [dict(stripe=(r, P)) for r in range(4)],
        [dict(fill=G) for r in range(4)],
        [dict(stripe=(r, GR)) for r in range(4)],
        [
            dict(fill=U, label="NS") if r == OWNER else dict(stripe=(r, U))
            for r in range(4)
        ],
    ],
    [
        "Shard(0): each rank\nkeeps 1/4 of the rows",
        "all-gather (BF16):\nevery rank holds\nthe matrix",
        "reduce-scatter (BF16):\neach rank holds its\nrows of the gradient",
        "all-to-all to the owner\n(FP32), Newton-Schulz on\nthe owner, all-to-all back",
    ],
)
arrows(3, y1, to_owner=True, both=True)
row(
    y2,
    [
        [dict(fill=P) if r == OWNER else {} for r in range(4)],
        [dict(fill=G) for r in range(4)],
        [dict(fill=GR) if r == OWNER else {} for r in range(4)],
        [dict(fill=U, label="NS") if r == OWNER else {} for r in range(4)],
    ],
    [
        "owner storage: rank 2\nkeeps the whole matrix;\nother ranks keep none of it",
        "broadcast from the\nowner (BF16): every\nrank holds the matrix",
        "reduce to the owner\n(BF16): rank 2 holds\nthe whole gradient",
        "Newton-Schulz on the\nowner, locally:\nno communication",
    ],
)
arrows(1, y2, to_owner=False)
arrows(2, y2, to_owner=True)
ax.text(
    -0.35,
    y1 + SQ / 2,
    "FSDP2 +\nDistMuon",
    ha="right",
    va="center",
    fontsize=8.2,
    fontweight="bold",
    color="#9b2c2c",
)
ax.text(
    -0.35,
    y2 + SQ / 2,
    "FlexShard +\nDistMuon",
    ha="right",
    va="center",
    fontsize=8.2,
    fontweight="bold",
    color="#2c5282",
)
ax.axhline(-0.95, color="#cbd5e0", lw=0.7, xmin=0.02, xmax=0.98)
ax.set_xlim(-2.0, cell_x(3) + CELL_W + 0.2)
ax.set_ylim(y2 - 1.55, 3.25)
ax.set_aspect("equal")
ax.axis("off")
ax.legend(
    handles=[
        Patch(color=P, label="parameter and momentum (FP32)"),
        Patch(color=G, label="gathered parameter (BF16, temporary)"),
        Patch(color=GR, label="reduced gradient"),
        Patch(color=U, label="Newton-Schulz (NS) and its update"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.47, -0.07),
    ncol=4,
    frameon=False,
    fontsize=6.8,
)
fig.savefig("fig4_dataflow.png", bbox_inches="tight")
