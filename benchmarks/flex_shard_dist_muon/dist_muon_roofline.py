# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Roofline model of DistMuon's optimizer step for DeepSeek V3 671B (Kimi's
Muon units: whole matrices, per MLA head, per expert), FlexShard + DistMuon
without optimizer communication against FSDP2 + DistMuon with its all-to-all,
as data parallelism grows.

Method as deepseek-v3.ezyang.com/studies/03-roofline.html: speed-of-light
rates, compute time = FLOPs / rate (or bytes / HBM bandwidth when memory-bound),
communication time = bytes per link / link bandwidth. Calibrated on gate 5
(FSDP2 + DistMuon on FlexShard's layout): 52 ms of matmuls per
rank per MoE model part -> 660 TFLOP/s for Newton-Schulz GEMMs.

Layout: gate 5's pipeline rank 0 (pipeline parallelism 4 with 4 virtual
stages: model parts of layers [0-2], [15-18], [31-34], [47-50]; 3 dense and 12
MoE blocks), every block parameter owned whole on FlexShard's owner
(make_bucketed_owned_full_param_placement_fn(carry_loads=True), 8 ranks per
host), routed experts local (256 experts per layer over the data-parallel
ranks). DistMuon's buckets as the MLA recipe: one per dense layer, one per MoE
pair plus its routed experts' local bucket.

    PYTHONPATH=<flex_shard>/src python dist_muon_roofline.py
"""

import math
import os
from unittest.mock import MagicMock

os.environ["LOCAL_WORLD_SIZE"] = "8"

import torch
import torch.nn as nn
from flex_shard.custom_placements.owned import (
    make_bucketed_owned_full_param_placement_fn,
)

# H100 SXM rates.
GEMM = (
    660e12  # Newton-Schulz GEMMs, measured at gate 5 (SOL in ezyang's study: 757.7e12)
)
GEMM_WIDE = 700e12  # 7168-wide matrices (dense FFN, wo): rank 0's dense w13 in both gate-5 traces, 64.5 ms for 45.25 TFLOP
HBM = 2.8e12  # bytes/s, achievable
IB = 50e9  # bytes/s per GPU (400 Gb/s NIC), as the study
NVL = 360e9  # bytes/s per GPU per direction (450 * 0.8)
ELEMENTWISE = 0.037e-9  # s per owned element: momentum, casts, norms, copies, update (measured, FlexShard rank 0)
SHARD_ELEMENTWISE = 0.012e-9  # s per storage-shard element for FSDP2's pack and finalize on the transfer stream
A2A_LATENCY = 60e-6  # s per all-to-all
# All-to-all rule. "progressive": each transfer starts once both of its ranks have joined,
# each rank's NIC and NVLink ports serve their transfers in arrival order, and a rank
# leaves the all-to-all when its own last transfer finishes. "barrier" (the first
# version): every rank leaves together, the busiest rank's whole transfer after the last
# rank joins; it overstated FSDP2 + DistMuon's step by 19% at gate 5.
A2A_RULE = os.environ.get("A2A_RULE", "progressive")
# Pipeline rank 0's model parts (layer lists), overridable for other pipeline layouts.
PARTS_ENV = os.environ.get("PARTS")
RANKS_PER_HOST = 8
NS_STEPS = 5

D, Q_LORA, KV_LORA, HEADS, NOPE, ROPE, V = 7168, 1536, 512, 128, 128, 64, 128
DENSE_F, MOE_F, EXPERTS = 18432, 2048, 256

# Block parameters in module order: (name, shape, Muon units as (rows, columns, count)).
ATTENTION = [
    ("attention.wq_a", (Q_LORA, D), [(Q_LORA, D, 1)]),
    ("attention.q_norm", (Q_LORA,), []),
    (
        "attention.wq_b",
        (HEADS * (NOPE + ROPE), Q_LORA),
        [(NOPE, Q_LORA, HEADS), (ROPE, Q_LORA, HEADS)],
    ),
    ("attention.wkv_a", (KV_LORA + ROPE, D), [(KV_LORA, D, 1), (ROPE, D, 1)]),
    ("attention.kv_norm", (KV_LORA,), []),
    (
        "attention.wkv_b",
        (HEADS * (NOPE + V), KV_LORA),
        [(NOPE, KV_LORA, HEADS), (V, KV_LORA, HEADS)],
    ),
    ("attention.wo", (D, HEADS * V), [(D, HEADS * V, 1)]),
    ("attention_norm", (D,), []),
    ("ffn_norm", (D,), []),
]
DENSE = [
    ("feed_forward.w13", (2, DENSE_F, D), [(DENSE_F, D, 2)]),
    ("feed_forward.w2", (D, DENSE_F), [(D, DENSE_F, 1)]),
]
MOE = [
    ("moe.router.gate", (EXPERTS, D), [(EXPERTS, D, 1)]),
    ("moe.shared_experts.w13", (2, MOE_F, D), [(MOE_F, D, 2)]),
    ("moe.shared_experts.w2", (D, MOE_F), [(D, MOE_F, 1)]),
]
EXPERT_UNITS = [(MOE_F, D, 2), (D, MOE_F, 1)]  # one routed expert: gate and up, down
PARTS = [[0, 1, 2], [15, 16, 17, 18], [31, 32, 33, 34], [47, 48, 49, 50]]
if PARTS_ENV:  # e.g. "0-2,7-10,15-18" for pipeline parallelism 2's rank 0
    PARTS = [
        list(range(int(a), int(b) + 1))
        for a, b in (x.split("-") for x in PARTS_ENV.split(","))
    ]


def ns_time(rows: int, columns: int, count: int) -> float:
    """Newton-Schulz on `count` matrices: max(compute, HBM traffic)."""
    m, n = sorted((rows, columns))
    flops = NS_STEPS * (4 * m * m * n + 2 * m**3)
    # Per step in BF16: read X, write and read A, A @ A, B, write X'.
    bytes_ = NS_STEPS * 2 * (3 * m * n + 5 * m * m)
    return count * max(flops / gemm_rate(m), bytes_ / HBM)


def gemm_rate(m: int) -> float:
    return GEMM_WIDE if m >= D else GEMM


def unit_time(units) -> float:
    return sum(ns_time(*u) for u in units)


def block_params(layer: int):
    return ATTENTION + (DENSE if layer < 3 else MOE)


def numel(shape) -> int:
    return math.prod(shape)


def placement(N: int) -> dict[str, int]:
    """FlexShard's owner for every block parameter of pipeline rank 0's parts."""
    mesh = MagicMock()
    mesh.mesh = torch.arange(N)
    mesh.size.return_value = N
    fn = make_bucketed_owned_full_param_placement_fn(carry_loads=True)
    owners = {}
    for part in PARTS:
        for layer in part:
            named = [
                (
                    f"layers.{layer}.{name}",
                    nn.Parameter(torch.empty(shape, device="meta")),
                )
                for name, shape, _ in block_params(layer)
            ]
            for fqn, (p,) in fn(named, mesh).items():
                owners[fqn] = p.segments_by_fqn[fqn][0].owner_rank
    return owners


def experts_on_rank(N: int, rank: int) -> float:
    """Routed experts per MoE layer on a rank: whole experts while N <= 256,
    then expert projections (3 units per expert) spread over the ranks."""
    if N <= EXPERTS:
        return EXPERTS / N
    units = 3 * EXPERTS
    return (units // N + (rank < units % N)) / 3


def buckets(owners: dict[str, int]):
    """DistMuon's buckets per model part: (kind, layers, units by owner)."""
    out = []
    for part in PARTS:
        plans = []
        dense = [l for l in part if l < 3]
        moe = [l for l in part if l >= 3]
        groups = [[l] for l in dense] + [moe[i : i + 2] for i in range(0, len(moe), 2)]
        for layers in groups:
            units = []  # (owner, elements, ns seconds)
            for layer in layers:
                for name, shape, mu in block_params(layer):
                    if mu:
                        units.append(
                            (
                                owners[f"layers.{layer}.{name}"],
                                numel(shape),
                                unit_time(mu),
                            )
                        )
            plans.append(("R", layers, units))
            if layers[0] >= 3:
                plans.append(("L", layers, None))
        out.append(plans)
    return out


def hosts_apart(a: int, b: int) -> bool:
    return a // RANKS_PER_HOST != b // RANKS_PER_HOST


def a2a_seconds(units, N: int) -> float:
    """One all-to-all moving each unit between its owner and every rank's row
    shard, FP32: the slowest rank's NIC and NVLink traffic, concurrently."""
    send_ib, recv_ib, send_nvl, recv_nvl = [0.0] * N, [0.0] * N, [0.0] * N, [0.0] * N
    for owner, elements, _ in units:
        share = 4.0 * elements / N
        for r in range(N):
            if r == owner:
                continue
            if hosts_apart(r, owner):
                send_ib[r] += share
                recv_ib[owner] += share
            else:
                send_nvl[r] += share
                recv_nvl[owner] += share
    return A2A_LATENCY + max(
        max(max(send_ib[r], recv_ib[r]) / IB, max(send_nvl[r], recv_nvl[r]) / NVL)
        for r in range(N)
    )


def a2a_pairs(units, N: int, direction: str) -> dict[tuple[int, int], float]:
    """FP32 bytes per (sender, receiver) for one all-to-all: "in" sends every rank's
    row shard of each unit to its owner, "out" sends each rank its rows of the update.
    """
    pairs: dict[tuple[int, int], float] = {}
    for owner, elements, _ in units:
        share = 4.0 * elements / N
        for r in range(N):
            if r == owner:
                continue
            key = (r, owner) if direction == "in" else (owner, r)
            pairs[key] = pairs.get(key, 0.0) + share
    return pairs


def a2a_progressive(join, pairs, N: int) -> list[float]:
    """Per-rank exit times of one all-to-all under the progressive rule. A transfer
    starts when both ranks have joined; every (rank, link, direction) port serves its
    transfers in arrival order (ties in NCCL's rotation order); a transfer ends when
    both its send and receive ports have served it; a rank exits after its last one."""
    items = []
    for (src, dst), nbytes in pairs.items():
        ib = hosts_apart(src, dst)
        items.append(
            (
                max(join[src], join[dst]),
                (dst - src) % N,
                src,
                dst,
                nbytes / (IB if ib else NVL),
                ib,
            )
        )
    items.sort()
    send_free: dict = {}
    recv_free: dict = {}
    end = list(join)
    for t0, _, src, dst, seconds, ib in items:
        ts = max(send_free.get((src, ib), 0.0), t0) + seconds
        td = max(recv_free.get((dst, ib), 0.0), t0) + seconds
        send_free[(src, ib)] = ts
        recv_free[(dst, ib)] = td
        finish = max(ts, td)
        end[src] = max(end[src], finish)
        end[dst] = max(end[dst], finish)
    return [e + A2A_LATENCY for e in end]


def simulate(N: int, plans_by_part, schedule: str):
    """Per-rank GPU time of DistMuon's step over pipeline rank 0's 4 parts.

    schedule "flex": no communication, each rank computes what it owns.
    "fsdp2_2buf": DistMuon's runtime today (2 buffers, local buckets stop the
    prefetch, the caller waits for the previous bucket's return), one DistMuon
    per model part, collectives finishing when the last rank joins.
    "fsdp2_ahead": every exchange issued at the start of each part (rejected
    change 2), for the bound.
    """
    C = [0.0] * N
    T = [0.0] * N
    if schedule == "fsdp2_ahead_merged":
        plans_by_part = [[p for plans in plans_by_part for p in plans]]
        schedule = "fsdp2_ahead"
    for plans in plans_by_part:
        R = [p for p in plans if p[0] == "R"]
        own = {
            id(p): [
                sum(t for o, _, t in p[2] if o == r)
                + ELEMENTWISE * sum(e for o, e, _ in p[2] if o == r)
                for r in range(N)
            ]
            for p in R
        }
        shard = {id(p): SHARD_ELEMENTWISE * sum(e for _, e, _ in p[2]) / N for p in R}
        local = {
            id(p): [
                2
                * experts_on_rank(N, r)
                * (unit_time(EXPERT_UNITS) + ELEMENTWISE * 3 * MOE_F * D)
                for r in range(N)
            ]
            for p in plans
            if p[0] == "L"
        }
        if schedule == "flex":
            for p in plans:
                for r in range(N):
                    C[r] += own[id(p)][r] if p[0] == "R" else local[id(p)][r]
            continue
        a2a = {id(p): a2a_seconds(p[2], N) for p in R}
        pairs = {(id(p), d): a2a_pairs(p[2], N, d) for p in R for d in ("in", "out")}
        ready, done_compute, done = {}, {}, {}

        def collective(start_times, p, direction):
            if A2A_RULE == "barrier":
                return [max(start_times) + a2a[id(p)]] * N
            return a2a_progressive(start_times, pairs[(id(p), direction)], N)

        def pre(p):
            starts = [T[r] + shard[id(p)] for r in range(N)]
            end = collective(starts, p, "in")
            for r in range(N):
                T[r] = end[r]
            ready[id(p)] = end

        def ret(p):
            starts = [max(T[r], done_compute[id(p)][r]) for r in range(N)]
            end = collective(starts, p, "out")
            for r in range(N):
                T[r] = end[r] + shard[id(p)]
            done[id(p)] = list(T)

        def compute(p):
            done_compute[id(p)] = [0.0] * N
            for r in range(N):
                C[r] = max(C[r], ready[id(p)][r]) + own[id(p)][r]
                done_compute[id(p)][r] = C[r]

        started = False
        if schedule == "fsdp2_ahead":
            for r in range(N):
                T[r] = max(T[r], C[r])
            for p in R:
                pre(p)
            for p in plans:
                if p[0] == "L":
                    for r in range(N):
                        C[r] += local[id(p)][r]
                    continue
                compute(p)
                ret(p)
            for r in range(N):
                C[r] = max(C[r], T[r])
            continue
        previous, prefetched, nxt, idx_r = None, [], 0, 0
        for idx, p in enumerate(plans):
            if p[0] == "L":
                for r in range(N):
                    C[r] += local[id(p)][r]
                nxt = idx + 1
                continue
            if not started:
                for r in range(N):
                    T[r] = max(T[r], C[r])
                started = True
            prev = previous
            if prefetched:
                prefetched.pop(0)
                if prev is not None:
                    ret(prev)
            else:
                if prev is not None and idx_r % 2 == (idx_r - 1) % 2:
                    ret(prev)
                    pre(p)
                else:
                    pre(p)
                    if prev is not None:
                        ret(prev)
                nxt = idx + 1
            compute(p)
            if prev is not None:
                for r in range(N):
                    C[r] = max(C[r], done[id(prev)][r])
            previous = p
            idx_r += 1
            while len(prefetched) < 1 and nxt < len(plans) and plans[nxt][0] == "R":
                pre(plans[nxt])
                prefetched.append(plans[nxt])
                nxt += 1
        ret(previous)
        for r in range(N):
            C[r] = max(C[r], done[id(previous)][r])
    return C


def per_element_table():
    print(
        "Per-element roofline of each Muon unit class (FSDP2's all-to-all moves 8 bytes per element: FP32 to the owner and back)"
    )
    print(
        f"{'unit':34s} {'m x n':>13s} {'FLOP/elem':>10s} {'AI FLOP/B':>10s} {'bound':>7s} "
        f"{'NS ns/elem':>10s} {'a2a ns/elem IB':>15s} {'a2a/NS':>7s}"
    )
    classes = [
        ("dense w13 gate or up, dense w2", DENSE_F, D),
        ("wo", D, HEADS * V),
        ("shared or routed expert matrix", MOE_F, D),
        ("wq_a", Q_LORA, D),
        ("wkv_a latent", KV_LORA, D),
        ("router", EXPERTS, D),
        ("wq_b head: q_nope", NOPE, Q_LORA),
        ("wkv_b head: k_nope or v", NOPE, KV_LORA),
        ("wq_b head: q_rope / wkv_a rope", ROPE, Q_LORA),
    ]
    for name, rows, cols in classes:
        m, n = sorted((rows, cols))
        flops = NS_STEPS * (4 * m * m * n + 2 * m**3)
        bytes_ = NS_STEPS * 2 * (3 * m * n + 5 * m * m)
        t = max(flops / gemm_rate(m), bytes_ / HBM) / (m * n) + ELEMENTWISE
        a2a = 8 / IB
        bound = "compute" if flops / gemm_rate(m) > bytes_ / HBM else "memory"
        print(
            f"{name:34s} {f'{m}x{n}':>13s} {flops / (m * n):10.0f} {flops / bytes_:10.0f} {bound:>7s} "
            f"{t * 1e9:10.3f} {a2a * 1e9:15.3f} {a2a / t:7.1f}"
        )
    print(
        f"(ridge point {GEMM / HBM:.0f} FLOP/B; NS time includes {ELEMENTWISE * 1e9:.3f} ns/elem "
        f"of element-wise work; over NVLink the all-to-all costs {8 / NVL * 1e9:.3f} ns/elem)"
    )


def main():
    per_element_table()
    print()
    ne = sum(numel(s) for l in sum(PARTS, []) for _, s, mu in block_params(l) if mu)
    print(
        f"Pipeline rank 0: {ne / 1e9:.2f}B Muon elements outside the routed experts, "
        f"{12 * EXPERTS * 3 * MOE_F * D / 1e9:.1f}B in routed experts (local in both backends)"
    )
    ne_ns = sum(unit_time(mu) for l in sum(PARTS, []) for _, _, mu in block_params(l))
    ex_ns = 12 * EXPERTS * unit_time(EXPERT_UNITS)
    print(
        f"Newton-Schulz on one GPU: {ne_ns:.2f} s outside the experts, {ex_ns:.2f} s for the experts"
    )
    print()
    print(
        "Optimizer time per step, slowest rank (ms); the FSDP2 gap to FlexShard split into: "
        "exchange (bytes and their exposed edges), "
        "per-model-part sync (one DistMuon per part), bucket-by-bucket coupling (2 buffers)"
    )
    print(
        f"{'DP':>4s} {'GPUs':>5s} | {'NS/rank avg':>11s} {'largest unit':>12s} {'a2a/rank':>8s} {'a2a bytes':>9s} | "
        f"{'FlexShard':>9s} {'FSDP2':>6s} {'gap':>5s} = {'exchange':>8s} + {'part sync':>9s} + "
        f"{'coupling':>8s} | {'FSDP2/Flex':>10s}"
    )
    for N in (8, 16, 32, 64, 128, 256, 512):
        owners = placement(N)
        plans = buckets(owners)
        flex = simulate(N, plans, "flex")
        two = simulate(N, plans, "fsdp2_2buf")
        ahead = simulate(N, plans, "fsdp2_ahead")
        merged = simulate(N, plans, "fsdp2_ahead_merged")
        units = [u for part in plans for p in part if p[0] == "R" for u in p[2]]
        largest = max(t for _, _, t in units)
        avg_ns = (ne_ns + ex_ns) / N
        bytes_rank = 2 * 4.0 * ne / N * (N - 1) / N
        remote = (N - RANKS_PER_HOST) / (N - 1) if N > RANKS_PER_HOST else 0.0
        a2a_avg = bytes_rank * remote / IB + bytes_rank * (1 - remote) / NVL
        f, t2, a, m = (max(x) * 1e3 for x in (flex, two, ahead, merged))
        print(
            f"{N:4d} {4 * N:5d} | {avg_ns * 1e3:11.0f} {largest * 1e3:12.0f} {a2a_avg * 1e3:8.1f} {bytes_rank / 1e9:7.2f}GB | "
            f"{f:9.0f} {t2:6.0f} {t2 - f:5.0f} = {m - f:8.0f} + {a - m:9.0f} + {t2 - a:8.0f} | {t2 / f:10.2f}"
        )
        if N == 64:
            gate5 = (N, plans, flex, two)


def per_part(N, plans):
    out = {}
    for schedule in ("flex", "fsdp2_2buf"):
        ends, prev = [], [0.0] * N
        for k in range(len(plans)):
            C = simulate(N, plans[: k + 1], schedule)
            ends.append(max(C) - max(prev))
            prev = C
        out[schedule] = ends
    return out


def per_part_ranks(N, plans, ranks):
    """Per-model-part spans on given ranks: (end of part k) - (end of part k - 1)."""
    out = {}
    for schedule in ("flex", "fsdp2_2buf"):
        ends, prev = {r: [] for r in ranks}, [0.0] * N
        for k in range(len(plans)):
            C = simulate(N, plans[: k + 1], schedule)
            for r in ranks:
                ends[r].append(C[r] - prev[r])
            prev = C
        out[schedule] = ends
    return out


if __name__ == "__main__":
    main()
    N = 64
    plans = buckets(placement(N))
    parts = per_part(N, plans)
    print()
    print(
        "Gate 5 check (data parallelism 64): per-model-part time, slowest rank (ms); "
        "measured FSDP2 strict on rank 0: 194 / 84 / 109 / 110"
    )
    for schedule, ends in parts.items():
        print(
            f"  {schedule:10s} "
            + " / ".join(f"{e * 1e3:.0f}" for e in ends)
            + f"  (total {sum(ends) * 1e3:.0f})"
        )
    print(
        f"Per rank (all-to-all rule: {A2A_RULE}); measured FSDP2 strict ranks 0 / 1 / 2: 194 / 84 / 109 / 110, "
        "194 / 85 / 107 / 114, 194 / 84 / 109 / 110; measured FlexShard rank 0: 75 / 77 / 72 / 73"
    )
    for schedule, by_rank in per_part_ranks(N, plans, (0, 1, 2)).items():
        for r, ends in by_rank.items():
            print(
                f"  {schedule:10s} rank {r}: "
                + " / ".join(f"{e * 1e3:.0f}" for e in ends)
                + f"  (total {sum(ends) * 1e3:.0f})"
            )
