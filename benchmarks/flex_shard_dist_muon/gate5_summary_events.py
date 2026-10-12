# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gate5_summary.py's numbers from pre-extracted phase events (grep -H output of
every rank's structured log), which is fast where reading a network file system file by file
isn't: per phase, the slowest rank per step, then the median over steps 3 and
5-10 (step 4 is profiled).

    python gate5_summary_events.py EVENTS.txt [...]
"""

import json
import re
import statistics
import sys
from collections import defaultdict

PHASES = ("step", "forward_backward", "optim_step")
for path in sys.argv[1:]:
    starts = {}
    per_rank = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        file_part, _, payload = line.partition(".jsonl:")
        rank = int(re.search(r"global_rank_(\d+)", file_part).group(1))
        event = json.loads(payload)
        name = event["log_type_name"]
        for phase in PHASES:
            if name == f"{phase}_start":
                starts[(rank, phase)] = event["time_us"]
            elif name == f"{phase}_end" and (rank, phase) in starts:
                per_rank[rank][phase].append(
                    (event["time_us"] - starts.pop((rank, phase))) / 1e3
                )
    print(f"== {path} ({len(per_rank)} ranks)")
    for phase in PHASES:
        steps = min(len(w[phase]) for w in per_rank.values())
        slowest = [max(w[phase][s] for w in per_rank.values()) for s in range(steps)]
        keep = [t for s, t in enumerate(slowest, start=1) if s >= 3 and s != 4]
        print(
            f"   {phase}: median {statistics.median(keep) / 1e3:.2f} s (steps 3, 5-10, slowest rank); per step "
            + " ".join(f"{t / 1e3:.2f}" for t in slowest)
        )
