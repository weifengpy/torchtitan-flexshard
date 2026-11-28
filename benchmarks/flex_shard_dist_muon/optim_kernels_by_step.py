# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GPU kernel time launched from inside each CPU range Optimizer.step#<NAME>.step
(CPU annotations only), by category, plus the compiled graphs called there.

    python optim_kernels_by_step.py TRACE.json.gz [OPTIMIZER_NAME_SUBSTRING]
"""

import gzip
import json
import sys
from collections import Counter, defaultdict

path = sys.argv[1]
wanted = sys.argv[2] if len(sys.argv) > 2 else "Muon"
with gzip.open(path, "rt") as f:
    events = [e for e in json.load(f)["traceEvents"] if e.get("ph") == "X"]
steps = sorted(
    (
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e["name"].startswith("Optimizer.step#")
        and wanted in e["name"]
    ),
    key=lambda e: e["ts"],
)
tid = steps[0]["tid"]
kernels = {
    e["args"]["correlation"]: e
    for e in events
    if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    and "correlation" in e.get("args", {})
}


def category(name):
    n = name.lower()
    if "nccl" in n:
        return "nccl"
    if any(k in n for k in ("gemm", "xmma", "cutlass", "cublas", "sm90", "nvjet")):
        return "gemm"
    if "memcpy" in n or "copy" in n:
        return "copy"
    return "other"


totals = defaultdict(float)
for step in steps:
    lo, hi = step["ts"], step["ts"] + step["dur"]
    by_cat, graphs, first, last = defaultdict(float), Counter(), None, None
    for e in events:
        if e.get("tid") != tid or not (lo <= e["ts"] <= hi):
            continue
        if e.get("cat") == "user_annotation" and e["name"].startswith(
            "## Call CompiledFxGraph"
        ):
            graphs[e["name"].split()[3][:10]] += 1
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            kernel = kernels.get(e.get("args", {}).get("correlation"))
            if kernel is not None:
                by_cat[category(kernel["name"])] += kernel["dur"]
                first = kernel["ts"] if first is None else min(first, kernel["ts"])
                last = (
                    kernel["ts"] + kernel["dur"]
                    if last is None
                    else max(last, kernel["ts"] + kernel["dur"])
                )
    for k, v in by_cat.items():
        totals[k] += v
    span = (last - first) / 1e3 if first is not None else 0.0
    print(
        f"{step['name'][15:40]:25s} CPU {step['dur'] / 1e3:6.1f} ms, GPU span {span:6.1f} ms: "
        + ", ".join(
            f"{k} {v / 1e3:.1f}"
            for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])
        )
        + (f"; compiled graphs {dict(graphs)}" if graphs else "")
    )
print(
    "total: "
    + ", ".join(
        f"{k} {v / 1e3:.1f} ms"
        for k, v in sorted(totals.items(), key=lambda kv: -kv[1])
    )
)
