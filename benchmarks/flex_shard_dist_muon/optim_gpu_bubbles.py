# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU busy time and idle gaps inside the optimizer window of one profiled step.

    python optim_gpu_bubbles.py TRACE.json.gz [TOP]

Window: the first to the last GPU kernel launched from an Optimizer.step#... CPU
range (the container's own range excluded), as step_gpu_phases.py. Busy: the
union of all GPU kernels, memcpys and memsets in the window, on any stream.
Gaps: idle stretches of that union. For each large gap: the kernel before it,
the one after it, and when the latter was launched relative to the gap's start
(a launch after the gap starts means the GPU waited on the CPU).
"""

import bisect
import gzip
import json
import sys
from collections import Counter

top = int(sys.argv[2]) if len(sys.argv) > 2 else 8
with gzip.open(sys.argv[1], "rt") as f:
    events = [
        e for e in json.load(f)["traceEvents"] if e.get("ph") == "X" and "dur" in e
    ]
gpu = sorted(
    (e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
    key=lambda e: e["ts"],
)
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
launch_of = {e["args"]["correlation"]: e for e in launches}
opt_kernels = []
for r in opt:
    lo, hi = r["ts"], r["ts"] + r["dur"]
    i = bisect.bisect_left([l["ts"] for l in launches], lo)
    for l in launches[i:]:
        if l["ts"] > hi:
            break
        k = by_corr.get(l["args"]["correlation"])
        if k is not None:
            opt_kernels.append(k)
w_lo = min(k["ts"] for k in opt_kernels)
w_hi = max(k["ts"] + k["dur"] for k in opt_kernels)
inside = [e for e in gpu if e["ts"] < w_hi and e["ts"] + e["dur"] > w_lo]
busy, cur = [], None
for e in inside:
    s, t = max(e["ts"], w_lo), min(e["ts"] + e["dur"], w_hi)
    if cur and s <= cur[1]:
        if t > cur[1]:
            cur[1], cur[3] = t, e
    else:
        if cur:
            busy.append(cur)
        cur = [s, t, e, e]
busy.append(cur)
busy_us = sum(t - s for s, t, *_ in busy)
gaps = [
    (busy[i + 1][0] - busy[i][1], busy[i][3], busy[i + 1][2])
    for i in range(len(busy) - 1)
]
window = w_hi - w_lo
print(
    f"optimizer window {window / 1e3:.1f} ms; GPU busy {busy_us / 1e3:.1f} ms "
    f"({100 * busy_us / window:.0f}%), idle {(window - busy_us) / 1e3:.1f} ms in {len(gaps)} gaps"
)
buckets = Counter()
for g, *_ in gaps:
    key = (
        "<10us"
        if g < 10
        else "10-100us"
        if g < 100
        else "0.1-1ms"
        if g < 1000
        else ">1ms"
    )
    buckets[key] += g
print(
    "idle by gap size (ms): "
    + ", ".join(
        f"{k} {buckets[k] / 1e3:.1f}" for k in ("<10us", "10-100us", "0.1-1ms", ">1ms")
    )
)
kinds = Counter()
for e in opt_kernels:
    n = e["name"].lower()
    kinds[
        (
            "nccl"
            if "nccl" in n
            else (
                "gemm"
                if any(x in n for x in ("gemm", "xmma", "cutlass", "nvjet", "sm90"))
                else "other"
            )
        )
    ] += e["dur"]
print(
    "optimizer kernel time (ms): "
    + ", ".join(f"{k} {v / 1e3:.1f}" for k, v in kinds.most_common())
    + f"; {len(opt_kernels)} kernels"
)
print("largest gaps:")
for g, before, after in sorted(gaps, key=lambda x: -x[0])[:top]:
    l = launch_of.get(after.get("args", {}).get("correlation"))
    gap_start = before["ts"] + before["dur"]
    when = (
        f"launched {(l['ts'] + l['dur'] - gap_start) / 1e3:+.2f} ms vs gap start"
        if l
        else "launch not found"
    )
    print(
        f"  {g / 1e3:6.2f} ms at +{(gap_start - w_lo) / 1e3:7.1f} ms: "
        f"after {before['name'][:45]!r} -> {after['name'][:45]!r}, {when}"
    )
