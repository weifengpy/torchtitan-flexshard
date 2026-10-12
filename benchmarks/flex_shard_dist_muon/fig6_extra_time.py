# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Figure 5: FSDP2 + DistMuon's extra optimizer time over FlexShard, by cause, from
dist_muon_roofline.py's model, with FlexShard's whole optimizer step for scale."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})
dp = [8, 16, 32, 64, 128, 256, 512]
flex = [1972, 989, 531, 303, 188, 131, 112]  # model v2: progressive all-to-alls
exchange = [4, 20, 37, 37, 37, 47, 71]
sync = [60, 134, 136, 143, 143, 132, 132]
coupling = [74, 66, 70, 75, 78, 89, 80]
extra = [
    138,
    220,
    243,
    255,
    258,
    268,
    283,
]  # FSDP2 - FlexShard; parts rounded to sum exactly
ORANGE, RED, LIGHT, BLUE = "#dd6b20", "#c53030", "#feb2b2", "#2b6cb0"
fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=220)
x = list(range(len(dp)))
ax.bar(
    x,
    exchange,
    color=ORANGE,
    width=0.62,
    label="exchange: transfers that compute can't hide",
)
ax.bar(
    x,
    sync,
    bottom=exchange,
    color=RED,
    width=0.62,
    label="per-model-part sync: every rank waits for the part's slowest owner",
)
ax.bar(
    x,
    coupling,
    bottom=[e + s for e, s in zip(exchange, sync)],
    color=LIGHT,
    width=0.62,
    label="bucket coupling: at most one bucket ahead of the slowest owner",
)
for i, t in enumerate(extra):
    ax.text(i, t - 6, f"+{t} ms", ha="center", va="top", fontsize=6.8, color="#742a2a")
ax.scatter(
    x[3:],
    flex[3:],
    marker="_",
    s=330,
    lw=2.4,
    color=BLUE,
    zorder=4,
    label="FlexShard's whole optimizer step, for scale",
)
for i in x[3:]:
    ax.text(i + 0.36, flex[i], f"{flex[i]}", va="center", fontsize=6.6, color=BLUE)
ax.text(
    0.99,
    0.98,
    "FlexShard's whole step at data parallelism\n8 / 16 / 32: 1,972 / 989 / 531 ms (off scale)",
    transform=ax.transAxes,
    fontsize=6.4,
    color=BLUE,
    va="top",
    ha="right",
)
ax.set_xticks(x)
ax.set_xticklabels([f"{d}\n{4 * d:,} GPUs" for d in dp], fontsize=7.4)
ax.set_ylim(0, 520)
ax.set_xlabel(
    "data parallelism and GPUs (pipeline parallelism 4 throughout; all values from the model)"
)
ax.set_ylabel("extra optimizer time (ms)")
ax.set_title(
    "Extra time FSDP2 + DistMuon's optimizer step takes over FlexShard's, by cause (model)",
    fontsize=8.3,
    loc="left",
)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(fontsize=6.5, loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("fig6_extra_time.png", bbox_inches="tight")
