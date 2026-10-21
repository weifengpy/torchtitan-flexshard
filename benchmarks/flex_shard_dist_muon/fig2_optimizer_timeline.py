# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Figure: the optimizer step on the GPU, step 4 of the DSV3 256-GPU benchmark.
FSDP2 + DistMuon on FlexShard's layout (data-parallel ranks 0-2 of pipeline
rank 0, one host and one clock) against FlexShard + DistMuon (same package,
rank 0). Input: extract_optim_kernels.py's JSON files."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})
COLORS = {"gemm": "#2b6cb0", "other": "#90cdf4", "nccl": "#e53e3e"}


def load(name):
    d = json.load(open(f"{name}.json"))
    return d["kernels"], d["ranges"]


fsdp = {r: load(f"fsdp2_strict_r{r}") for r in range(3)}
flex = load("flex_r0")
origin_fsdp = min(
    k[0] for k in fsdp[0][0]
)  # rank 0's first optimizer kernel; ranks 0-2 share a host clock
origin_flex = min(k[0] for k in flex[0])


def merge(kernels, cats, origin, gap_us=15.0):
    out = []
    for s, e, c, *_ in sorted(k for k in kernels if k[2] in cats):
        s, e = (s - origin) / 1e3, (e - origin) / 1e3
        if out and out[-1][2] == c and s - out[-1][1] <= gap_us / 1e3:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e, c])
    return out


def part_spans(kernels, origin):
    spans = {}
    for s, e, c, idx, _ in kernels:
        lo, hi = spans.get(idx, (s, e))
        spans[idx] = (min(lo, s), max(hi, e))
    muon = [
        spans[i] for i in sorted(spans) if i % 2 == 0
    ]  # even ranges: DistMuon steps
    return [((lo - origin) / 1e3, (hi - origin) / 1e3) for lo, hi in muon]


fig, ax = plt.subplots(figsize=(7.4, 4.1), dpi=220)
lane_h, gap = 0.36, 0.08
Y = {0: 0.0, 1: -1.22, 2: -2.44}
rows = [(Y[r], fsdp[r][0], origin_fsdp, True, f"rank {r}") for r in (0, 1, 2)]
sep_y, flex_title_y, flex_y = -3.3, -3.42, -4.05
rows.append((flex_y, flex[0], origin_flex, False, "rank 0"))
bottom, top = flex_y - lane_h - gap - 0.42, lane_h / 2 + 0.12

PART_NAMES = [
    "model part 1\n(3 dense layers)",
    "model part 2\n(4 MoE layers)",
    "model part 3\n(4 MoE layers)",
    "model part 4\n(4 MoE layers)",
]
band_colors = ["#f7fafc", "#e2e8f0"]
fsdp_parts = part_spans(fsdp[0][0], origin_fsdp)
flex_parts = part_spans(flex[0], origin_flex)
for i, (lo, hi) in enumerate(fsdp_parts):
    ax.fill_between(
        [lo, hi], sep_y + 0.12, top, color=band_colors[i % 2], zorder=0, lw=0
    )
    ax.text(
        (lo + hi) / 2,
        top + 0.04,
        PART_NAMES[i],
        ha="center",
        va="bottom",
        fontsize=7,
        color="#4a5568",
    )
for i, (lo, hi) in enumerate(flex_parts):
    ax.fill_between(
        [lo, hi],
        bottom + 0.2,
        flex_y + lane_h / 2 + 0.06,
        color=band_colors[i % 2],
        zorder=0,
        lw=0,
    )
    ax.text(
        (lo + hi) / 2,
        bottom + 0.24,
        f"part {i + 1}",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#4a5568",
    )

for y0, kernels, origin, has_nccl, _ in rows:
    for s0, e0, c in merge(kernels, ("gemm", "other"), origin):
        ax.add_patch(
            plt.Rectangle(
                (s0, y0 - lane_h / 2),
                max(e0 - s0, 0.15),
                lane_h,
                color=COLORS[c],
                lw=0,
                zorder=2,
            )
        )
    yn = y0 - lane_h - gap
    if has_nccl:
        for s0, e0, c in merge(kernels, ("nccl",), origin):
            ax.add_patch(
                plt.Rectangle(
                    (s0, yn - lane_h / 2),
                    max(e0 - s0, 0.15),
                    lane_h,
                    color=COLORS[c],
                    lw=0,
                    zorder=2,
                )
            )
    else:
        ax.text(
            306,
            yn,
            "no collectives in the optimizer",
            va="center",
            fontsize=7,
            color="#718096",
            style="italic",
        )

ytick, ylab = [], []
for y0, *_rest in rows:
    ytick += [y0, y0 - lane_h - gap]
    ylab += [f"{_rest[-1]}  compute", "NCCL"]
ax.set_yticks(ytick)
ax.set_yticklabels(ylab, fontsize=7.3)
ax.tick_params(axis="y", length=0)
ax.axhline(sep_y, color="#a0aec0", lw=0.8)
ax.set_title(
    "FSDP2 + DistMuon on FlexShard's layout (data-parallel ranks 0-2 of pipeline rank 0)",
    loc="left",
    fontsize=8,
    fontweight="bold",
    color="#9b2c2c",
    pad=25,
)
ax.text(
    0,
    flex_title_y,
    "FlexShard + DistMuon (rank 0)",
    ha="left",
    va="top",
    fontsize=8,
    fontweight="bold",
    color="#2c5282",
)

end_fsdp = max(hi for _, hi in fsdp_parts)
end_flex = max(hi for _, hi in flex_parts)
ax.text(
    end_fsdp + 5,
    Y[0],
    f"{end_fsdp:.0f} ms",
    va="center",
    fontsize=8,
    fontweight="bold",
    color="#9b2c2c",
)
ax.text(
    end_flex + 5,
    flex_y,
    f"{end_flex:.0f} ms",
    va="center",
    fontsize=8,
    fontweight="bold",
    color="#2c5282",
)
ax.text(
    6,
    Y[1] - 0.02,
    "ranks 1 and 2 own nothing in part 1:\n193 ms inside NCCL, waiting for the owners",
    va="center",
    ha="left",
    fontsize=7,
    color="#742a2a",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#feb2b2", lw=0.6),
    zorder=5,
)
ax.set_xlim(0, 545)
ax.set_ylim(bottom, top + 0.62)
ax.set_xlabel("GPU time from the first optimizer kernel (ms), step 4")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.legend(
    handles=[
        Patch(color=COLORS["gemm"], label="Newton-Schulz matmuls"),
        Patch(color=COLORS["other"], label="other optimizer compute"),
        Patch(color=COLORS["nccl"], label="NCCL all-to-all (mostly waiting)"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.42, -0.13),
    ncol=3,
    frameon=False,
    fontsize=7.4,
)
fig.tight_layout()
fig.savefig("fig2_optimizer_timeline.png", bbox_inches="tight")
print(
    "parts FSDP2",
    [f"{hi - lo:.1f}" for lo, hi in fsdp_parts],
    "FlexShard",
    [f"{hi - lo:.1f}" for lo, hi in flex_parts],
)
