# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for DistMuon DP x TP layouts and pipeline group creation."""

from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.optim.optimizer import (
    DistMuon as ConfigurableDistMuon,
    OptimizersContainer,
)
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.flex_shard.dist_muon import (
    _adjust_muon_learning_rate,
    DistMuon,
)
from torchtitan.distributed.parallelism_context import ParallelismContext


def _flattened_group(mesh: DeviceMesh) -> DeviceMesh:
    """Return one process group over all of ``mesh``'s ranks, in storage order."""
    ranks = mesh.mesh.flatten().tolist()
    return DeviceMesh.from_group(
        dist.new_group(ranks), mesh.device_type, mesh=torch.tensor(ranks)
    )


class TestDistMuonTensorParallelCPU(DTensorTestBase):
    mesh_shape = (2, 2)

    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_owned_rejects_omitted_sharded_axis(self):
        mesh = init_device_mesh(
            self.device_type, self.mesh_shape, mesh_dim_names=("dp_shard", "tp")
        )
        parameter = torch.nn.Parameter(
            distribute_tensor(torch.ones(4, 4), mesh, (Shard(0), Shard(1)))
        )
        with self.assertRaisesRegex(ValueError, "non-replicated axes:.*tp"):
            DistMuon(
                [{"params": [parameter], "param_names": ["matrix"]}],
                compute_sharding_by_fqn={
                    "matrix": ComputeLayout(
                        shardings_by_mesh_axis={"dp_shard": Owned()}
                    )
                },
                bucket_configs=(BucketConfig(patterns=("matrix",)),),
            )

    @with_comms
    def test_mixed_block_row_scales_follow_sequential_and_strided_storage(self):
        mesh = init_device_mesh(
            self.device_type, self.mesh_shape, mesh_dim_names=("dp_shard", "tp")
        )
        block_sizes = (4, 2)
        reference_ratio = _adjust_muon_learning_rate(1.0, None, (6, 3))
        expected = torch.tensor(
            [
                _adjust_muon_learning_rate(1.0, None, (rows, 3)) / reference_ratio
                for rows in block_sizes * 2
                for _ in range(rows)
            ]
        ).reshape(12, 1)
        values = torch.arange(36).reshape(12, 3).float().div_(41)
        gradient = values.mul(0.37).add(0.2).sin()
        lr = 0.03
        transport_mesh = _flattened_group(mesh)
        for storage_placements in (
            (Shard(0), Shard(0)),
            (_StridedShard(0, split_factor=2), Shard(0)),
        ):
            parameter = torch.nn.Parameter(
                distribute_tensor(values.clone(), mesh, storage_placements)
            )
            optimizer = DistMuon(
                [{"params": [parameter], "param_names": ["matrix"]}],
                compute_sharding_by_fqn={
                    "matrix": ComputeLayout(
                        shardings_by_mesh_axis={
                            "dp_shard": BlockShard(0, block_sizes),
                            "tp": BlockShard(0, block_sizes),
                        },
                        compute_dtype=torch.bfloat16,
                    )
                },
                bucket_configs=(BucketConfig(patterns=("matrix",)),),
                redistribution_meshes=(transport_mesh,),
                lr=lr,
                weight_decay=0,
                momentum=0,
                nesterov=False,
                ns_steps=2,
            )
            expected_local = distribute_tensor(
                expected.clone(), mesh, storage_placements
            ).to_local()
            torch.testing.assert_close(
                optimizer._direction_row_scale_by_fqn["matrix"],
                expected_local,
                rtol=0,
                atol=0,
            )
            parameter.grad = distribute_tensor(
                gradient.clone(), mesh, storage_placements
            )
            reference_parameters = [
                torch.nn.Parameter(block.clone())
                for block in values.split(block_sizes * 2)
            ]
            reference_optimizer = torch.optim.Muon(
                reference_parameters,
                lr=lr,
                weight_decay=0,
                momentum=0,
                nesterov=False,
                ns_steps=2,
            )
            for reference, grad in zip(
                reference_parameters, gradient.split(block_sizes * 2), strict=True
            ):
                reference.grad = grad.clone()
            optimizer.step()
            reference_optimizer.step()
            expected_parameter = torch.cat(
                [reference.detach() for reference in reference_parameters]
            )
            # Redistribution scales directions in BF16 before the FP32 add;
            # plain Muon applies its scalar alpha without that extra rounding.
            # Compare normalized updates, not near-equal resident weights.
            torch.testing.assert_close(
                (parameter.full_tensor() - values) / lr,
                (expected_parameter - values) / lr,
                rtol=0,
                atol=torch.finfo(torch.bfloat16).eps,
            )


class TestDistMuonUnevenPipelineCPU(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @property
    def device_type(self):
        return "cpu"

    @with_comms
    def test_uneven_pipeline_group_creation_and_muon_parity(self):
        """Stages with different buckets and storage meshes create every group.

        Stage 0 holds only a dense matrix on (dp_shard, tp); stage 1 holds only
        experts on (edp_shard, ep), so neither needs the other's groups.
        ParallelismContext creates the combined (dp_shard, tp) groups of both
        stages on every rank.
        """
        with patch("torchtitan.distributed.parallelism_context.device_type", "cpu"):
            parallelism_context = ParallelismContext(
                dp_replicate=1,
                dp_shard=2,
                cp=1,
                tp=2,
                pp=2,
                ep=2,
                world_size=self.world_size,
                enable_sequence_parallel=False,
            )
            dense_mesh = parallelism_context.get_mesh(["dp_shard", "tp"])
            expert_mesh = parallelism_context.get_mesh(["edp_shard", "ep"])
        name = "dense" if self.rank < 4 else "experts"
        mesh = dense_mesh if name == "dense" else expert_mesh
        value = (
            torch.arange(16).reshape(4, 4).float().div_(32)
            if name == "dense"
            else torch.arange(24).reshape(4, 3, 2).float().div_(32)
        )
        placements = (Shard(0), Shard(1))
        parameter = torch.nn.Parameter(
            distribute_tensor(value.clone(), mesh, placements)
        )
        model = torch.nn.Module()
        model.register_parameter(name, parameter)
        muon = ConfigurableDistMuon.Config(
            pattern=".*",
            compute_sharding_by_fqn={
                "dense": ComputeLayout(
                    {"dp_shard": Owned(), "tp": Owned()}, compute_dtype=torch.bfloat16
                ),
                "experts": ComputeLayout(
                    {"edp_shard": Shard(0), "ep": Shard(0)},
                    compute_dtype=torch.bfloat16,
                ),
            },
            bucket_configs=[
                BucketConfig(patterns=("dense",)),
                BucketConfig(patterns=("experts",)),
            ],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )
        with patch("torchtitan.distributed.parallelism_context.device_type", "cpu"):
            optimizer = OptimizersContainer.Config(optimizers=[muon]).build(
                model_parts=[model], parallelism_context=parallelism_context
            )
        # Nonmember ranks can skip native first-use group creation without an
        # immediate Gloo hang. The next WORLD group exposes the desynchronization.
        dist.barrier()
        group = dist.new_group(
            ranks=list(range(self.world_size)), timeout=timedelta(seconds=5)
        )
        total = torch.ones(1)
        dist.all_reduce(total, group=group)
        self.assertEqual(total.item(), self.world_size)
        reference_parameters = [
            torch.nn.Parameter(matrix.clone())
            for matrix in (value.unsqueeze(0) if name == "dense" else value)
        ]
        reference = torch.optim.Muon(
            reference_parameters,
            lr=muon.lr,
            weight_decay=muon.weight_decay,
            momentum=muon.momentum,
            nesterov=muon.nesterov,
            ns_steps=muon.ns_steps,
        )
        for step in range(2):
            gradient = value.mul(0.37).add(0.2 + step).sin()
            parameter.grad = distribute_tensor(gradient.clone(), mesh, placements)
            for ref, grad in zip(
                reference_parameters,
                gradient.unsqueeze(0) if name == "dense" else gradient,
                strict=True,
            ):
                ref.grad = grad.clone()
            optimizer.step()
            reference.step()
            expected = (
                reference_parameters[0].detach()
                if name == "dense"
                else torch.stack([ref.detach() for ref in reference_parameters])
            )
            torch.testing.assert_close(
                parameter.full_tensor(), expected, rtol=0, atol=0
            )
        dist.destroy_process_group(group)
