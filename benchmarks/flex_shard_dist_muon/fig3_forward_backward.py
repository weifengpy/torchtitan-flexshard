# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Figure: forward-backward costs the same in both backends; only the optimizer
differs. Rank 0, step 4, DSV3 256-GPU benchmark, two jobs per backend (each pair
bitwise identical in numerics). Compute-stream kernel time by class inside the
pipeline window (fb_kernels.py), the compute stream's idle time by what ran
meanwhile (fb_actions.py), and the optimizer's GPU window (step_gpu_phases.py)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
JOBS = [
    "FSDP2 + DistMuon, job 1",
    "FSDP2 + DistMuon, job 2 (rerun)",
    "FlexShard + DistMuon, job 1",
    "FlexShard + DistMuon, job 2",
]
COLORS = ["#c53030", "#fc8181", "#2b6cb0", "#90cdf4"]
ROWS = [  # (label, [FSDP2 job 1, FSDP2 job 2, FlexShard job 1, FlexShard job 2]) in ms
    ("MoE all-to-all", [27101, 27115, 26930, 27070]),
    ("waiting for pipeline receives", [14353.8, 14340.9, 14292.3, 14181.8]),
    ("matmuls", [14456, 14483, 14640, 14368]),
    ("element-wise", [4811, 4810, 4830, 4817]),
    ("attention", [3581, 3571, 3589, 3560]),
    ("copies, cat and split", [2890, 2889, 2880, 2874]),
    ("other idle", [1238.5, 1266.1, 1248.9, 1273.3]),
    ("reductions", [249, 249, 249, 248]),
    (
        "exposed data-parallel collectives\n(gathers and gradient reduces)",
        [44.9, 45.1, 50.3, 51.4],
    ),
    ("optimizer step", [496.8, 498.6, 297.2, 291.5]),
]
fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=220)
bar_h, group_gap = 0.17, 0.42
yticks, ylabels = [], []
y = 0.0
for i, (label, vals) in enumerate(ROWS):
    if label == "optimizer step":
        y -= 0.35
        ax.axhline(y + bar_h * 2 + 0.17, color="#a0aec0", lw=0.8)
    for j, v in enumerate(vals):
        ax.barh(y - j * bar_h, v, height=bar_h * 0.92, color=COLORS[j], left=10)
    fs = sum(vals[:2]) / 2
    fl = sum(vals[2:]) / 2
    d = fl - fs
    txt = f"{d:+,.0f} ms" if abs(d) >= 1 else f"{d:+.1f} ms"
    bold = label in (
        "optimizer step",
        "exposed data-parallel collectives\n(gathers and gradient reduces)",
    )
    ax.text(
        60000,
        y - 1.5 * bar_h,
        txt,
        va="center",
        ha="right",
        fontsize=7.6,
        fontweight="bold" if bold else None,
        color="#2c5282" if d < 0 else "#9b2c2c",
    )
    yticks.append(y - 1.5 * bar_h)
    ylabels.append(label)
    y -= 4 * bar_h + group_gap
ax.set_xscale("log")
ax.set_xlim(10, 70000)
ax.set_xticks([10, 30, 100, 300, 1000, 3000, 10000, 30000])
ax.set_xticklabels(["10", "30", "100", "300", "1,000", "3,000", "10,000", "30,000"])
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=7.6)
ax.tick_params(axis="y", length=0)
ax.text(
    60000,
    0.42,
    "FlexShard − FSDP2\n(means of two jobs)",
    ha="right",
    va="bottom",
    fontsize=7,
    color="#4a5568",
)
ax.text(
    12,
    yticks[-1] + 0.62,
    "optimizer, for contrast",
    fontsize=7,
    color="#4a5568",
    style="italic",
)
ax.text(
    12,
    0.42,
    "forward and backward (pipeline window)",
    fontsize=7,
    color="#4a5568",
    style="italic",
    va="bottom",
)
ax.set_xlabel(
    "GPU time per step on rank 0 (ms, log scale), step 4, DSV3 256-GPU benchmark"
)
ax.set_title(
    "Forward and backward cost the same in both backends; only the optimizer differs",
    fontsize=8.6,
    loc="left",
)
ax.grid(True, axis="x", which="major", color="#edf2f7", lw=0.6)
ax.set_ylim(yticks[-1] - 2.5 * bar_h - 0.12, 0.95)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(
    handles=[Patch(color=c, label=l) for c, l in zip(COLORS, JOBS)],
    loc="upper center",
    bbox_to_anchor=(0.4, -0.1),
    ncol=2,
    frameon=False,
    fontsize=7,
)
fig.tight_layout()
fig.savefig("fig3_forward_backward.png", bbox_inches="tight")
