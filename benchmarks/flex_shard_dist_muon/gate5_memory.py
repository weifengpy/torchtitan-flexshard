# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Every rank's device memory at step 10 of gate-5 jobs, from the hosts'
stdout logs in the job's output directory: max, median and the top ranks.

    python gate5_memory.py OUTPUT_DIR [OUTPUT_DIR ...]
"""

import glob
import os
import re
import statistics
import sys

PATTERN = re.compile(
    r"rank (\d+) step 10: device used ([0-9.]+) GiB, PyTorch reserved ([0-9.]+) GiB, outside PyTorch ([0-9.]+) GiB"
)
for run_dir in sys.argv[1:]:
    used = {}
    for path in glob.glob(os.path.join(run_dir, "logs", "*", "*_stdout.log")):
        with open(path, errors="replace") as f:
            for line in f:
                m = PATTERN.search(line)
                if m:
                    used[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
    if not used:
        print(f"== {os.path.basename(run_dir)}: no memory lines")
        continue
    values = sorted(used.items(), key=lambda kv: -kv[1][0])
    device = [v[0] for v in used.values()]
    outside = [v[2] for v in used.values()]
    print(
        f"== {os.path.basename(run_dir.rstrip('/'))}: {len(used)} ranks; device used max {max(device):.2f} GiB, "
        f"median {statistics.median(device):.2f}; outside PyTorch median {statistics.median(outside):.2f} GiB"
    )
    print("   top: " + ", ".join(f"rank {r} {v[0]:.2f}" for r, v in values[:4]))
