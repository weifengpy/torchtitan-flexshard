# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schematic: why per-bucket exchanges cost the sum over buckets of the slowest
owner, while owner storage costs only the slowest rank's own total.
Toy example: 4 ranks, 3 buckets, made-up work units."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update(
    {"font.size": 8.5, "font.family": "DejaVu Sans", "hatch.linewidth": 0.6}
)
work = {1: [6, 1, 1, 0], 2: [1, 5, 2, 1], 3: [1, 0, 1, 4]}  # work[bucket][rank]
BC = {1: "#2b6cb0", 2: "#319795", 3: "#805ad5"}
XCH, WAIT = "#dd6b20", "#fed7d7"
x_len = 0.4
ranks = range(4)
fig, (a, b) = plt.subplots(
    2, 1, figsize=(7.4, 3.9), dpi=220, sharex=True, gridspec_kw={"hspace": 0.55}
)


def unit(ax, x, r, w, color, text=None):
    if w <= 0:
        return
    ax.add_patch(plt.Rectangle((x, r - 0.32), w, 0.64, fc=color, ec="white", lw=0.8))
    if text and w >= 0.9:
        ax.text(x + w / 2, r, text, ha="center", va="center", color="white", fontsize=7)


# FSDP2 + DistMuon: buckets in lockstep, each closed by its slowest owner.
t = 0.0
for bk in (1, 2, 3):
    for r in ranks:
        a.add_patch(
            plt.Rectangle((t, r - 0.32), x_len, 0.64, fc=XCH, ec="white", lw=0.5)
        )
    t += x_len
    m = max(work[bk])
    for r in ranks:
        w = work[bk][r]
        unit(a, t, r, w, BC[bk], f"b{bk}")
        if w < m:
            a.add_patch(
                plt.Rectangle(
                    (t + w, r - 0.32),
                    m - w,
                    0.64,
                    fc=WAIT,
                    ec="#e53e3e",
                    lw=0.4,
                    hatch="////",
                )
            )
    a.text(
        t + m / 2,
        -0.75,
        f"bucket {bk}: slowest owner {m}",
        ha="center",
        va="center",
        fontsize=7,
        color=BC[bk],
    )
    t += m
    for r in ranks:
        a.add_patch(
            plt.Rectangle((t, r - 0.32), x_len, 0.64, fc=XCH, ec="white", lw=0.5)
        )
    t += x_len
end_fsdp = t
a.axvline(end_fsdp, color="#9b2c2c", lw=1, ls="--")
a.text(
    end_fsdp + 0.15,
    1.5,
    r"$\Sigma_b\,\mathrm{max}_r\, w(r, b)$" + "\n= 6 + 5 + 4 = 15,\nplus 6 exchanges",
    va="center",
    fontsize=7.2,
    color="#9b2c2c",
)
a.set_title(
    "FSDP2 + DistMuon: every bucket's exchange waits for that bucket's slowest owner",
    loc="left",
    fontsize=8.5,
    color="#9b2c2c",
    fontweight="bold",
)

# FlexShard + DistMuon: each rank runs what it owns, back to back.
totals = []
end_hint = max(sum(work[bk][r] for bk in work) for r in ranks)
for r in ranks:
    t = 0.0
    for bk in (1, 2, 3):
        unit(b, t, r, work[bk][r], BC[bk], f"b{bk}")
        t += work[bk][r]
    totals.append(t)
    b.text(
        t + 0.12 if t < end_hint else t + 0.3,
        r,
        f"{t:g}",
        va="center",
        fontsize=7,
        color="#2c5282",
    )
end_flex = max(totals)
b.axvline(end_flex, color="#2c5282", lw=1, ls="--")
b.text(
    end_flex + 0.15,
    1.5,
    r"$\mathrm{max}_r\,\Sigma_b\, w(r, b)$" + "\n= 8: the slowest rank's\nown total",
    va="center",
    fontsize=7.2,
    color="#2c5282",
)
b.set_title(
    "FlexShard + DistMuon: each rank runs only the matrices it owns; nothing to wait for",
    loc="left",
    fontsize=8.5,
    color="#2c5282",
    fontweight="bold",
)

for ax in (a, b):
    ax.set_yticks(list(ranks))
    ax.set_yticklabels([f"rank {r}" for r in ranks], fontsize=7.5)
    ax.set_ylim(3.6, -1.05)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
a.set_ylim(3.6, -1.05)
b.set_ylim(3.6, -0.55)
b.set_xlim(0, 21.5)
b.set_xlabel(
    "time, in units of Newton-Schulz work (made-up example: w(r, b) is rank r's work in bucket b)"
)
fig.legend(
    handles=[
        Patch(color=BC[1], label="bucket 1"),
        Patch(color=BC[2], label="bucket 2"),
        Patch(color=BC[3], label="bucket 3"),
        Patch(color=XCH, label="all-to-all transfer"),
        Patch(
            fc=WAIT, ec="#e53e3e", hatch="////", label="waiting inside the all-to-all"
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=5,
    frameon=False,
    fontsize=7.2,
)
fig.savefig("fig5_toy_sync.png", bbox_inches="tight")
print("FSDP2", end_fsdp, "FlexShard", end_flex)
