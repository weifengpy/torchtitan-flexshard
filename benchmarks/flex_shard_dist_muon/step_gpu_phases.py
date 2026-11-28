# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One profiled step on the GPU, in phases: before the first pipeline action, the
pipeline (first action to last action), from the last action to the first
optimizer kernel, the optimizer, and after it; plus the duration distribution
of the MoE all-to-all kernels (SendRecv on the compute stream) by forward and
backward actions.

    python step_gpu_phases.py TRACE.json.gz
"""

import gzip
import json
import re
import statistics
import sys
from bisect import bisect_right
from collections import defaultdict

with gzip.open(sys.argv[1], "rt") as f:
    events = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]
step = next(
    e
    for e in events
    if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("ProfilerStep#")
)
step_lo, step_hi = step["ts"], step["ts"] + step["dur"]
ACTION = re.compile(r"^PP:(\d+)([A-Z_]+?)(\d*)$")
actions = sorted(
    (e["ts"], e["ts"] + e["dur"], ACTION.match(e["name"]).group(2))
    for e in events
    if e.get("cat") == "gpu_user_annotation" and ACTION.match(e["name"])
)
pipe_lo, pipe_hi = actions[0][0], max(a[1] for a in actions)
opt_ranges = [
    e
    for e in events
    if e.get("cat") == "user_annotation"
    and e["name"].startswith("Optimizer.step#")
    and "Container" not in e["name"]
]
main_tid = opt_ranges[0]["tid"]
launch_ts = {
    e["args"]["correlation"]: e["ts"]
    for e in events
    if e.get("cat") == "cuda_runtime"
    and e.get("tid") == main_tid
    and "correlation" in e.get("args", {})
}
opt_kernels = []
for e in events:
    if e.get("cat") != "kernel":
        continue
    t = launch_ts.get(e.get("args", {}).get("correlation"))
    if t is not None and any(r["ts"] <= t <= r["ts"] + r["dur"] for r in opt_ranges):
        opt_kernels.append(e)
opt_lo = min(e["ts"] for e in opt_kernels)
opt_hi = max(e["ts"] + e["dur"] for e in opt_kernels)
print(f"GPU step {(step_hi - step_lo) / 1e3:.1f} ms:")
for name, lo, hi in (
    ("before the first pipeline action", step_lo, pipe_lo),
    ("pipeline: first to last action", pipe_lo, pipe_hi),
    ("last action to first optimizer kernel", pipe_hi, opt_lo),
    ("optimizer kernels, first to last", opt_lo, opt_hi),
    ("after the optimizer", opt_hi, step_hi),
):
    print(f"  {name:40s} {(hi - lo) / 1e3:10.1f} ms")

compute = 7
starts = [a[0] for a in actions]
a2a = defaultdict(list)
for e in events:
    if (
        e.get("cat") == "kernel"
        and e.get("tid") == compute
        and "SendRecv" in e["name"]
        and pipe_lo <= e["ts"] <= pipe_hi
    ):
        i = bisect_right(starts, e["ts"]) - 1
        kind = (
            actions[i][2] if i >= 0 and actions[i][1] >= e["ts"] else "outside actions"
        )
        a2a[kind].append(e["dur"] / 1e3)
print(
    "MoE all-to-all kernels on the compute stream, by action: count, total, median, p90, max ms"
)
for kind, durs in sorted(a2a.items()):
    durs.sort()
    print(
        f"  {kind:16s} {len(durs):5d} {sum(durs):9.1f} {statistics.median(durs):7.2f} "
        f"{durs[int(0.9 * (len(durs) - 1))]:7.2f} {durs[-1]:8.2f}"
    )
