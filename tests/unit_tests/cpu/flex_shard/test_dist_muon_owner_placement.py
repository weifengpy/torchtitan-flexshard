# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.flex_shard.dist_muon import (
    _balance_loads_across_partitions,
    _estimate_muon_compute_cost,
    _host_interleaved_order,
)

# DeepSeek V3 671B's whole Muon matrices of one MoE layer.
_MOE_LAYER_MATRICES = {
    "attention.wo": (7168, 16384),
    "attention.wq_a": (1536, 7168),
    "moe.shared_experts.w1": (2048, 7168),
    "moe.shared_experts.w3": (2048, 7168),
    "moe.shared_experts.w2": (7168, 2048),
    "moe.router.gate": (256, 7168),
}


def _moe_stage_owners(participants: tuple[int, ...], ranks_per_host: int) -> dict:
    """Owners of a 4-layer MoE stage's whole matrices, planned as DistMuon
    plans them: two buckets of two layers, with loads carried across them."""
    cumulative_loads = (0,) * len(participants)
    owners = {}
    for layers in ((0, 1), (2, 3)):
        names, loads = [], []
        for layer in layers:
            for name, shape in _MOE_LAYER_MATRICES.items():
                fqn = f"layers.{layer}.{name}"
                names.append(fqn)
                loads.append(
                    (
                        _estimate_muon_compute_cost(torch.Size(shape), 5),
                        shape[0] * shape[1] * 2,
                        fqn,
                    )
                )
        partitions, cumulative_loads = _balance_loads_across_partitions(
            loads,
            initial_cumulative_primary_loads=cumulative_loads,
            partition_order=_host_interleaved_order(participants, ranks_per_host),
        )
        owners.update(
            (fqn, participants[partition])
            for fqn, partition in zip(names, partitions, strict=True)
        )
    return owners


class TestDistMuonOwnerPlacement(unittest.TestCase):
    def test_host_interleaved_order(self):
        positions = _host_interleaved_order(tuple(range(16)), 8)
        self.assertEqual(
            sorted(range(16), key=positions.__getitem__),
            [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15],
        )
        # One host, or one rank per host, keeps rank order.
        self.assertEqual(_host_interleaved_order(tuple(range(8)), 8), tuple(range(8)))
        self.assertEqual(_host_interleaved_order(tuple(range(8)), 1), tuple(range(8)))
        # A group of global ranks, such as one pipeline rank's data-parallel
        # group: ranks 64, 72 take positions before 65, 73.
        self.assertEqual(_host_interleaved_order((64, 65, 72, 73), 8), (0, 2, 1, 3))

    def test_largest_matrices_land_on_different_hosts(self):
        # 64 data-parallel ranks on 8 hosts. Rank order (one host of 64 ranks,
        # the previous behavior) puts the stage's 4 wo on 2 hosts; host-first
        # order puts them on 4.
        participants = tuple(range(64, 128))
        rank_order = _moe_stage_owners(participants, 64)
        host_first = _moe_stage_owners(participants, 8)
        wo = [f"layers.{layer}.attention.wo" for layer in range(4)]
        self.assertEqual(len({rank_order[fqn] // 8 for fqn in wo}), 2)
        self.assertEqual(len({host_first[fqn] // 8 for fqn in wo}), 4)
        # Every matrix still has one owner, and owners never share a matrix.
        self.assertEqual(set(host_first), set(rank_order))
        self.assertEqual(len(set(host_first.values())), len(host_first))


if __name__ == "__main__":
    unittest.main()
