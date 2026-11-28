# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forward-backward of one profiled step, from a torch.profiler trace: pipeline
actions (GPU-side ``PP:<stage><ACTION><microbatch>`` annotations) by type, the
timeline's milestones, FlexShard/FSDP annotations, NCCL kernels by name, and the
compute stream's idle time attributed to what ran meanwhile on other streams.

    python fb_actions.py TRACE.json.gz [OUT.json]

OUT.json keeps every action's GPU start and end (relative to the first action),
for comparing two runs action by action.
"""

import gzip
import json
import re
import statistics
import sys
from bisect import bisect_left
from collections import defaultdict

path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else None
with gzip.open(path, "rt") as f:
    events = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]

ACTION = re.compile(r"^PP:(\d+)([A-Z_]+?)(\d*)$")
gpu_actions = []
for e in events:
    if e.get("cat") == "gpu_user_annotation":
        m = ACTION.match(e["name"])
        if m:
            gpu_actions.append(
                (
                    e["ts"],
                    e["ts"] + e["dur"],
                    int(m.group(1)),
                    m.group(2),
                    m.group(3),
                    e["name"],
                )
            )
gpu_actions.sort()
t0 = gpu_actions[0][0]
t_end = max(a[1] for a in gpu_actions)
print(
    f"pipeline actions on the GPU: {len(gpu_actions)}, first start to last end {(t_end - t0) / 1e3:.1f} ms"
)

by_type = defaultdict(list)
for start, end, stage, kind, mb, name in gpu_actions:
    by_type[kind].append(end - start)
print("by action type (GPU span): count, total ms, median ms")
for kind, spans in sorted(by_type.items(), key=lambda kv: -sum(kv[1])):
    print(
        f"  {kind:12s} {len(spans):5d} {sum(spans) / 1e3:10.1f} {statistics.median(spans) / 1e3:8.2f}"
    )

first = {}
for start, end, stage, kind, mb, name in gpu_actions:
    first.setdefault(kind, start)
last_f = max(a[1] for a in gpu_actions if a[3] == "F")
last_b = max(a[1] for a in gpu_actions if a[3] == "B")
print(
    "milestones (ms from the first action): "
    + ", ".join(
        f"first {k} {(v - t0) / 1e3:.1f}"
        for k, v in sorted(first.items(), key=lambda kv: kv[1])
    )
    + f", last F end {(last_f - t0) / 1e3:.1f}, last B end {(last_b - t0) / 1e3:.1f}"
)

ann = defaultdict(lambda: [0, 0.0])
for e in events:
    if e.get("cat") == "gpu_user_annotation" and e["name"].startswith(
        ("FlexShard::", "FSDP::")
    ):
        key = re.sub(r" \(.*\)$", "", e["name"])
        ann[key][0] += 1
        ann[key][1] += e["dur"]
print("FlexShard/FSDP GPU annotations: count, total GPU span ms")
for key, (count, dur) in sorted(ann.items(), key=lambda kv: -kv[1][1])[:20]:
    print(f"  {key[:60]:60s} {count:5d} {dur / 1e3:10.1f}")

kernels = [
    e
    for e in events
    if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    and e["ts"] < t_end
    and e["ts"] + e["dur"] > t0
]
stream_busy = defaultdict(float)
for e in kernels:
    stream_busy[e["tid"]] += e["dur"]
compute = max(
    stream_busy,
    key=lambda tid: sum(
        e["dur"] for e in kernels if e["tid"] == tid and "nccl" not in e["name"].lower()
    ),
)
nccl = defaultdict(lambda: [0, 0.0])
for e in kernels:
    if "nccl" in e["name"].lower():
        key = re.sub(r"\(.*", "", e["name"])
        nccl[(key, e["tid"])][0] += 1
        nccl[(key, e["tid"])][1] += e["dur"]
print(f"NCCL kernels (count, total ms) by name and stream; compute stream is {compute}")
for (key, tid), (count, dur) in sorted(nccl.items(), key=lambda kv: -kv[1][1])[:16]:
    print(f"  {key[:55]:55s} s{tid:<4} {count:6d} {dur / 1e3:10.1f}")


def category(name):
    n = name.lower()
    if "nccl" in n:
        if "sendrecv" in n:
            return "pipeline/all-to-all SendRecv"
        if "allgather" in n or "broadcast" in n:
            return "all-gather/broadcast"
        if "reducescatter" in n or "reduce_" in n or "_reduce" in n:
            return "reduce"
        return "other NCCL"
    if "memcpy" in n or "copy" in n:
        return "copy"
    return "compute"


comp = sorted((e for e in kernels if e["tid"] == compute), key=lambda e: e["ts"])
comp_busy = 0.0
gaps, cursor = [], t0
for e in comp:
    lo, hi = max(e["ts"], t0), min(e["ts"] + e["dur"], t_end)
    if lo > cursor:
        gaps.append((cursor, lo))
    if hi > cursor:
        comp_busy += hi - max(lo, cursor)
        cursor = hi
if t_end > cursor:
    gaps.append((cursor, t_end))
others = sorted((e for e in kernels if e["tid"] != compute), key=lambda e: e["ts"])
other_starts = [e["ts"] for e in others]
max_dur = max((e["dur"] for e in others), default=0)
attributed = defaultdict(float)
for lo, hi in gaps:
    overlap = defaultdict(float)
    i = bisect_left(other_starts, lo - max_dur)
    while i < len(others) and others[i]["ts"] < hi:
        e = others[i]
        a, z = max(e["ts"], lo), min(e["ts"] + e["dur"], hi)
        if z > a:
            overlap[category(e["name"])] += z - a
        i += 1
    if overlap:
        attributed[max(overlap, key=overlap.get)] += hi - lo
    else:
        attributed["nothing on the GPU (CPU-bound or idle)"] += hi - lo
total = t_end - t0
print(
    f"compute stream: busy {comp_busy / 1e3:.1f} ms, idle {(total - comp_busy) / 1e3:.1f} ms "
    f"of {total / 1e3:.1f} ms; idle by what else ran (ms):"
)
for key, value in sorted(attributed.items(), key=lambda kv: -kv[1]):
    print(f"  {key:45s} {value / 1e3:10.1f}")

if out_path:
    with open(out_path, "w") as f:
        json.dump(
            {
                "actions": [
                    [(s - t0) / 1e3, (e - t0) / 1e3, name]
                    for s, e, _st, _k, _mb, name in gpu_actions
                ],
                "compute_stream": compute,
            },
            f,
        )
