# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compute-stream kernel time inside the pipeline window (first to last PP action)
of one profiled step, by kernel class, for comparing backends' forward-backward.

    python fb_kernels.py TRACE.json.gz OUT.json
"""

import gzip
import json
import re
import sys
from collections import defaultdict

src, out = sys.argv[1], sys.argv[2]
with gzip.open(src, "rt") as f:
    events = [
        e for e in json.load(f)["traceEvents"] if e.get("ph") == "X" and "dur" in e
    ]
ACTION = re.compile(r"^PP:(\d+)([A-Z_]+?)(\d*)$")
acts = [
    (e["ts"], e["ts"] + e["dur"])
    for e in events
    if e.get("cat") == "gpu_user_annotation" and ACTION.match(e["name"])
]
lo, hi = min(a for a, _ in acts), max(b for _, b in acts)
COMPUTE = 7


def klass(name):
    n = name.lower()
    if "sendrecv" in n:
        return "MoE all-to-all (SendRecv)"
    if "nccl" in n:
        return "other NCCL"
    if any(k in n for k in ("gemm", "nvjet", "cutlass", "sm90_xmma", "cublas")):
        return "matmul"
    if "flash" in n or "fmha" in n or "attention" in n or "sdpa" in n:
        return "attention"
    if "copy" in n or "memcpy" in n or "cat" in n or "split" in n:
        return "copy/cat/split"
    if "reduce" in n or "sum" in n:
        return "reductions"
    return "elementwise/other"


by_class = defaultdict(float)
by_name = defaultdict(float)
for e in events:
    if (
        e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset")
        or e.get("tid") != COMPUTE
    ):
        continue
    s, t = max(e["ts"], lo), min(e["ts"] + e["dur"], hi)
    if t <= s:
        continue
    by_class[klass(e["name"])] += (t - s) / 1e3
    by_name[re.sub(r"<.*", "", e["name"])[:90]] += (t - s) / 1e3
json.dump(
    {"window_ms": (hi - lo) / 1e3, "by_class": by_class, "by_name": by_name},
    open(out, "w"),
)
print(
    f"{out}: window {(hi - lo) / 1e3:.1f} ms; "
    + ", ".join(
        f"{k} {v:.0f}" for k, v in sorted(by_class.items(), key=lambda x: -x[1])
    )
)
