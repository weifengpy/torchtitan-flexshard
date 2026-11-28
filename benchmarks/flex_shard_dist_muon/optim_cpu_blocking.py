# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""List the longest CUDA runtime calls inside each optimizer step's CPU range.

    python optim_cpu_blocking.py TRACE.json.gz [--match DistMuon] [--top 8]

For each matching ``Optimizer.step#...`` CPU range: its duration, the longest
CUDA runtime calls inside it, and the innermost CPU op holding each call.
"""

import argparse
import bisect
import gzip
import json

parser = argparse.ArgumentParser()
parser.add_argument("trace")
parser.add_argument("--match", default="DistMuon")
parser.add_argument("--top", type=int, default=8)
args = parser.parse_args()
with gzip.open(args.trace) as f:
    events = json.load(f)["traceEvents"]
complete = [e for e in events if e.get("ph") == "X" and "dur" in e]
steps = sorted(
    (
        e
        for e in complete
        if e["name"].startswith("Optimizer.step#")
        and args.match in e["name"]
        and e.get("cat") in ("user_annotation", "cpu_op")
    ),
    key=lambda e: e["ts"],
)
by_tid = {}
for e in complete:
    if e.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver"):
        by_tid.setdefault(e["tid"], []).append(e)
for tid_events in by_tid.values():
    tid_events.sort(key=lambda e: e["ts"])
for step in steps:
    t0, t1 = step["ts"], step["ts"] + step["dur"]
    tid_events = by_tid.get(step["tid"], [])
    starts = [e["ts"] for e in tid_events]
    lo, hi = bisect.bisect_left(starts, t0), bisect.bisect_right(starts, t1)
    inside = tid_events[lo:hi]
    ops = [e for e in inside if e.get("cat") == "cpu_op"]
    runtime = [e for e in inside if e.get("cat") != "cpu_op"]
    total = {}
    for e in runtime:
        total[e["name"]] = total.get(e["name"], 0) + e["dur"]
    print(
        f"{step['name']} at {(t0 - steps[0]['ts']) / 1e3:.1f} ms: {step['dur'] / 1e3:.1f} ms"
    )
    print(
        "   runtime totals:",
        {
            k: round(v / 1e3, 1)
            for k, v in sorted(total.items(), key=lambda kv: -kv[1])[:6]
        },
    )
    for e in sorted(runtime, key=lambda x: -x["dur"])[: args.top]:
        holders = [
            o
            for o in ops
            if o["ts"] <= e["ts"] and e["ts"] + e["dur"] <= o["ts"] + o["dur"]
        ]
        inner = min(holders, key=lambda o: o["dur"])["name"] if holders else "-"
        print(
            f"     {e['name']:28s} {e['dur'] / 1e3:8.2f} ms at +{(e['ts'] - t0) / 1e3:8.2f} ms  in {inner}"
        )
