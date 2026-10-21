# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""torchtitan's configurable DistMuon, a wrapper around FlexShard's
``flex_shard.dist_muon.DistMuon``, and its path for FlexShard models, which
runs DistMuon on each rank's own storage with no collectives.

FlexShard (meta-pytorch/flex_shard) is an optional dependency, so only DistMuon
users import this module; the optimizer container reaches it only through their
DistMuon configs."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast, Literal, overload

import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import Replicate, Shard
from torch.optim import Optimizer

try:
    # pyrefly: ignore [missing-import]
    from flex_shard.dist_muon import (
        BlockShard,
        BucketConfig,
        ComputeLayout,
        DistMuon as FlexShardDistMuon,
        Owned,
    )
except ModuleNotFoundError as error:
    # flex_shard imports torchao.
    if error.name not in ("flex_shard", "torchao"):
        raise
    raise ModuleNotFoundError(
        "DistMuon requires the optional flex_shard package (meta-pytorch/flex_shard "
        "main, at 34dd58b or later) and torchao. Install them with: "
        "pip install torchao && pip install --no-deps "
        '"git+https://github.com/meta-pytorch/flex_shard.git@main"',
        name=error.name,
    ) from error

from .optimizer import BaseOptimizer

__all__ = [
    "BlockShard",
    "BucketConfig",
    "ComputeLayout",
    "DistMuon",
    "Owned",
]


class DistMuon(FlexShardDistMuon, BaseOptimizer):
    """Configurable TorchTitan wrapper around FlexShard's ``DistMuon``."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseOptimizer.Config):
        compute_sharding_by_fqn: Mapping[str, ComputeLayout]
        bucket_configs: Sequence[BucketConfig]
        lr: float = 1e-3
        weight_decay: float = 0.1
        momentum: float = 0.95
        nesterov: bool = True
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
        eps: float = 1e-7
        ns_steps: int = 5
        adjust_lr_fn: Literal[
            "original", "match_rms_adamw", "spectral_unclamped"
        ] | None = None
        # Ranks per host, for placing whole-matrix owners on different hosts
        # first. None reads torchrun's LOCAL_WORLD_SIZE.
        ranks_per_host: int | None = None
        owner_by_fqn: Mapping[str, int] = field(default_factory=dict)
        """Fixed owners of ``Owned`` parameters, as ranks on their
        redistribution mesh axis (e.g. ``dp_shard``); DistMuon balances the
        other ``Owned`` parameters around them. Entries for parameters on other
        pipeline stages are ignored."""

    def __init__(
        self,
        config: Config,
        *,
        params: Iterable[dict[str, Any]] | Iterable[Tensor],
    ) -> None:
        super().__init__(
            cast(Iterable[dict[str, Any]], params),
            compute_sharding_by_fqn=config.compute_sharding_by_fqn,
            bucket_configs=config.bucket_configs,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=config.nesterov,
            ns_coefficients=config.ns_coefficients,
            eps=config.eps,
            ns_steps=config.ns_steps,
            adjust_lr_fn=config.adjust_lr_fn,
            ranks_per_host=config.ranks_per_host,
            owner_by_fqn=config.owner_by_fqn,
        )


class _LocalDistMuon(DistMuon):
    """DistMuon on DTensor proxies of one rank's FlexShard storage, on a
    one-rank mesh (see ``_build_flex_shard_dist_muon``). The container drives
    it as any optimizer, so its step first gives the proxies their parameters'
    grads, and its ``zero_grad`` clears the parameters' grads too."""

    # flex_shard.dist_muon.LocalDistMuonBinding, set once it's built.
    binding: Any

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        self.binding.copy_grads_to_proxies()
        return super().step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        self.binding.zero_real_grads(set_to_none)


def _local_compute_layout(compute_layout: ComputeLayout, axis: str) -> ComputeLayout:
    """The compute layout on a one-rank mesh axis that gives DistMuon the matrix
    views it gives the data-parallel layout: blocks for ``BlockShard`` and
    ``Owned`` with blocks, the whole local tensor otherwise."""
    shardings = [
        sharding
        for sharding in compute_layout.shardings_by_mesh_axis.values()
        if not isinstance(sharding, Replicate)
    ]
    blocks = []
    for sharding in shardings:
        if isinstance(sharding, BlockShard):
            blocks.append(sharding)
        elif isinstance(sharding, Owned) and sharding.block_sizes is not None:
            blocks.append(BlockShard(dim=0, block_sizes=sharding.block_sizes))
    if blocks:
        sharding = blocks[0]
    elif any(isinstance(sharding, Owned) for sharding in shardings):
        sharding = Owned()
    else:
        sharding = Shard(0)
    return ComputeLayout(shardings_by_mesh_axis={axis: sharding})


def _build_flex_shard_dist_muon(
    config: BaseOptimizer.Config,
    model: nn.Module,
    param_group: dict[str, Any],
) -> Optimizer | None:
    """DistMuon over a FlexShard model's parameters, with no collectives in
    its step.

    Each Muon parameter must be stored where DistMuon computes it: whole on
    an owner (``parallelism.flex_shard_owned_params``), or in row shards that
    hold whole units, e.g. whole heads, gate or up matrices, or experts, which
    DistMuon's own plan for FSDP2's layout shows (``_NoRedistributionTransition``).
    torchtitan's DistMuon then runs on each rank's local storage on a one-rank
    mesh, through FlexShard's ``build_local_dist_muon``: the same matrix views
    and batches per parameter as on the data-parallel mesh, so the same update.
    A whole matrix computed alone gives the same result on any owner.

    Returns None on a rank that stores none of the group's parameters, e.g. a
    model part of dense blocks with every block parameter owned, at more ranks
    than it has Muon parameters: the rank has no Muon work there, and the step
    runs no collectives that other ranks would wait on.
    """
    assert isinstance(config, DistMuon.Config)
    # pyrefly: ignore [missing-import]
    from flex_shard import get_placements

    # pyrefly: ignore [missing-import]
    from flex_shard.custom_placements.owned import BucketedOwned

    # pyrefly: ignore [missing-import]
    from flex_shard.dist_muon import build_local_dist_muon

    # pyrefly: ignore [missing-import]
    from flex_shard.dist_muon.optimizer import (
        _NoRedistributionTransition,
        _resolve_storage_to_compute_transition,
    )
    from torchtitan.distributed.flex_shard_fsdp import as_fsdp2_dtensor

    named_params = list(
        zip(param_group["param_names"], param_group["params"], strict=True)
    )
    for fqn, param in named_params:
        (placement,) = get_placements(param)
        if isinstance(placement, BucketedOwned):
            continue
        transition = _resolve_storage_to_compute_transition(
            fqn, as_fsdp2_dtensor(param), config.compute_sharding_by_fqn[fqn]
        ).storage_to_compute_transition
        if not isinstance(transition, _NoRedistributionTransition):
            raise ValueError(
                f"DistMuon with FlexShard computes each parameter where it is "
                f"stored, but {fqn!r}'s row shards split its compute units over "
                "ranks; list it in parallelism.flex_shard_owned_params, or use a "
                "data-parallel degree that keeps its units whole."
            )

    def optimizer_factory(
        param_groups: list[dict[str, Any]], _layouts: Mapping[str, Any]
    ) -> Optimizer:
        (group,) = param_groups
        (axis,) = group["params"][0].device_mesh.mesh_dim_names
        local_config = dataclasses.replace(
            config,
            compute_sharding_by_fqn={
                fqn: _local_compute_layout(config.compute_sharding_by_fqn[fqn], axis)
                for fqn in group["param_names"]
            },
            # FlexShard already stores each parameter on its owner.
            owner_by_fqn={},
        )
        return _LocalDistMuon(local_config, params=param_groups)

    binding = build_local_dist_muon(
        model, named_params, optimizer_factory=optimizer_factory
    )
    if binding is None:
        return None
    optimizer = cast(_LocalDistMuon, binding.optimizer)
    optimizer.binding = binding
    return optimizer
