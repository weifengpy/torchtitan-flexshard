# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kernels launched inside one profiled step's optimizer (every Optimizer.step#...
CPU range except the container's), as compact JSON for plotting.

    python extract_optim_kernels.py TRACE.json.gz OUT.json

Each kernel: [start_us, end_us, category, optimizer_range_index, stream]. Categories
as optim_gpu_bubbles.py: nccl, gemm (Newton-Schulz matmuls), other.
"""

import bisect
import gzip
import json
import sys

src, out = sys.argv[1], sys.argv[2]
with gzip.open(src, "rt") as f:
    events = [
        e for e in json.load(f)["traceEvents"] if e.get("ph") == "X" and "dur" in e
    ]
gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
by_corr = {
    e["args"].get("correlation"): e for e in gpu if "correlation" in e.get("args", {})
}
opt = sorted(
    (
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e["name"].startswith("Optimizer.step#")
        and "Container" not in e["name"]
    ),
    key=lambda e: e["ts"],
)
tid = opt[0]["tid"]
launches = sorted(
    (
        e
        for e in events
        if e.get("tid") == tid
        and e.get("cat") in ("cuda_runtime", "cuda_driver")
        and "correlation" in e.get("args", {})
    ),
    key=lambda e: e["ts"],
)
starts = [l["ts"] for l in launches]


def category(name):
    n = name.lower()
    if "nccl" in n:
        return "nccl"
    if any(k in n for k in ("gemm", "xmma", "cutlass", "nvjet", "sm90")):
        return "gemm"
    return "other"


kernels = []
for index, r in enumerate(opt):
    lo, hi = r["ts"], r["ts"] + r["dur"]
    for l in launches[bisect.bisect_left(starts, lo) :]:
        if l["ts"] > hi:
            break
        k = by_corr.get(l["args"]["correlation"])
        if k is not None:
            kernels.append(
                [k["ts"], k["ts"] + k["dur"], category(k["name"]), index, k.get("tid")]
            )
ranges = [[r["name"], r["ts"], r["ts"] + r["dur"]] for r in opt]
json.dump({"trace": src, "ranges": ranges, "kernels": kernels}, open(out, "w"))
lo = min(k[0] for k in kernels)
hi = max(k[1] for k in kernels)
print(
    f"{out}: {len(kernels)} kernels, {len(opt)} optimizer ranges, window {(hi - lo) / 1e3:.1f} ms"
)
