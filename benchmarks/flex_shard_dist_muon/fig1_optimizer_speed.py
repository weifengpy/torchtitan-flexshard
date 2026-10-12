# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Figure 4: optimizer step speed of each backend as the cluster grows (pipeline
parallelism 4 fixed). Only 256 GPUs (data parallelism 64) was run; the lines are
dist_muon_roofline.py's projection (ROOFLINE_DISTMUON_DSV3.md). Speed is relative
to FSDP2 + DistMuon's measured optimizer step on 256 GPUs. Higher is better."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})
dp = [8, 16, 32, 64, 128, 256, 512]
flex = [
    1972,
    989,
    531,
    303,
    188,
    131,
    112,
]  # slowest rank (model v2: progressive all-to-alls)
fsdp = [2110, 1209, 774, 558, 446, 399, 395]  # slowest rank
BASE = 526.6  # FSDP2 + DistMuon's slowest traced rank on 256 GPUs (rank 15; ranks 31 and 58: 525.8, 525.4 ms)
FLEX_SLOWEST = (
    301.3  # FlexShard's slowest traced rank (rank 57; ranks 0 and 52: 297.2, 288.9 ms)
)
RED, BLUE, INK = "#c53030", "#2b6cb0", "#2d3748"
speed = lambda ms: 1000.0 / ms  # optimizer steps per second

fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=220)
ax.axvspan(64 / 1.18, 64 * 1.18, color="#fefcbf", zorder=0, lw=0)
ax.text(
    64,
    12.0,
    "measured:\nDSV3 256-GPU\nbenchmark",
    ha="center",
    va="top",
    fontsize=6.7,
    color="#744210",
)
ax.text(
    256,
    12.0,
    "other sizes: model projection",
    ha="center",
    va="top",
    fontsize=6.7,
    color="#718096",
    style="italic",
)
ax.plot(dp, [speed(t) for t in fsdp], "-o", color=RED, ms=4.5, lw=1.8)
ax.plot(dp, [speed(t) for t in flex], "-s", color=BLUE, ms=4.5, lw=1.8)
ax.scatter(
    [64],
    [speed(BASE)],
    marker="*",
    s=110,
    color=RED,
    edgecolor="black",
    lw=0.4,
    zorder=5,
)
ax.scatter(
    [64],
    [speed(FLEX_SLOWEST)],
    marker="*",
    s=110,
    color=BLUE,
    edgecolor="black",
    lw=0.4,
    zorder=5,
)
ax.text(
    512 * 1.1,
    speed(112),
    "FlexShard\n+ DistMuon",
    fontsize=7,
    color=BLUE,
    va="center",
    fontweight="bold",
)
ax.text(
    512 * 1.1,
    speed(395),
    "FSDP2\n+ DistMuon",
    fontsize=7,
    color=RED,
    va="center",
    fontweight="bold",
)

x64 = 64 * 1.27
ax.annotate(
    "",
    xy=(x64, speed(FLEX_SLOWEST)),
    xytext=(x64, speed(BASE)),
    arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9),
)
ax.text(
    x64 * 1.06,
    3.1,
    "measured:\nFlexShard\n1.75x faster",
    fontsize=6.8,
    color=INK,
    va="center",
    ha="left",
)
x512 = 512 * 0.9
ax.annotate(
    "",
    xy=(x512, speed(112) * 0.97),
    xytext=(x512, speed(395) * 1.03),
    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1, linestyle=(0, (1.2, 1.6))),
)
ax.text(
    x512 / 1.07,
    4.6,
    "estimated:\nFlexShard\n3.5x faster",
    fontsize=6.8,
    color=INK,
    va="center",
    ha="right",
)
ax.text(
    0.015,
    0.03,
    "FSDP2 + DistMuon's optimizer stops speeding up past 1,024 GPUs: its step\n"
    "floors near 400 ms. FlexShard's keeps falling, to 112 ms on 2,048 GPUs.",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=6.7,
    color=INK,
    linespacing=1.35,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e0", lw=0.7),
)

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xticks(dp)
ax.set_xticklabels([f"{d}\n{4 * d:,} GPUs" for d in dp], fontsize=7.4)
ax.set_yticks([0.5, 1, 2, 4, 8])
ax.set_yticklabels(["0.5", "1", "2", "4", "8"])
ax.set_xlim(6.5, 1100)
ax.set_ylim(0.22, 12.6)
ax.set_xlabel("data parallelism and GPUs (pipeline parallelism 4 throughout)")
ax.set_ylabel("optimizer steps per second\n(1 ÷ optimizer step time)")
ax.set_title(
    "Optimizer step speed of each backend's slowest rank as the cluster grows (higher is better)",
    fontsize=8.6,
    loc="left",
)
ax.grid(True, which="major", color="#edf2f7", lw=0.6)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
handles = [
    Line2D(
        [],
        [],
        color=RED,
        marker="o",
        ms=4.5,
        lw=1.8,
        label="FSDP2 + DistMuon, current runtime (model projection)",
    ),
    Line2D(
        [],
        [],
        color=BLUE,
        marker="s",
        ms=4.5,
        lw=1.8,
        label="FlexShard + DistMuon (model projection)",
    ),
]
star = dict(
    marker="*", ms=10.5, ls="none", markeredgecolor="black", markeredgewidth=0.4
)
stars = (
    Line2D([], [], markerfacecolor=RED, **star),
    Line2D([], [], markerfacecolor=BLUE, **star),
)
labels = [h.get_label() for h in handles] + [
    "measured, DSV3 256-GPU benchmark (slowest traced rank)"
]
fig.legend(
    handles=handles + [stars],
    labels=labels,
    handler_map={tuple: HandlerTuple(ndivide=None, pad=1.0)},
    handlelength=3.0,
    fontsize=6.7,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=2,
    frameon=False,
)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("fig1_optimizer_speed.png", bbox_inches="tight")
print("FSDP2", [f"{speed(t):.3f}" for t in fsdp])
print("FlexShard", [f"{speed(t):.3f}" for t in flex])
