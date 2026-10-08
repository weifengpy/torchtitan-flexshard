# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data parallelism with FlexShard in place of FSDP2's ``fully_shard``.

``apply_flex_shard_to_decoder`` gives FlexShard (meta-pytorch/flex_shard) one
``fsdp2_compatible`` bucket for each group that ``apply_fsdp_to_decoder`` passes
to ``fully_shard``, with the same placements and reshard policy. The buckets'
collectives then match FSDP2's byte for byte, so training is bitwise identical.
With expert parallelism, ``fully_shard`` splits an MoE block into one group per
mesh, and the block gets one bucket per group. FlexShard is imported only when
``parallelism.fsdp_backend='flex_shard'``.
"""

import logging
import sys
from collections.abc import Iterable
from typing import cast, TYPE_CHECKING

import torch
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan.distributed.fsdp import (
    get_fsdp_reshard_after_forward_policy,
    linear_param_shard_placements,
    routed_expert_param_placements,
)

if TYPE_CHECKING:
    from torchtitan.models.common.decoder import Decoder
    from torchtitan.models.common.moe import MoE

__all__ = [
    "apply_flex_shard_to_decoder",
    "as_fsdp2_dtensor",
    "grads_for_norm",
]

logger = logging.getLogger(__name__)

# The edp_shard meshes of FlexShard's routed-expert buckets, mapped to the sparse
# storage mesh on which FSDP2 lays out their parameters, with the ep axis.
_EXPERT_STORAGE_MESHES: dict[DeviceMesh, DeviceMesh] = {}


def apply_flex_shard_to_decoder(
    model: "Decoder",
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
) -> None:
    """Shard a decoder with FlexShard as ``apply_fsdp_to_decoder`` does with FSDP2.

    Each ``fully_shard`` group becomes a bucket naming the same modules: the
    embedding (with the norm and output projection when weights are tied), the
    norm with the output projection, each transformer block, and the parameters
    left to the root group. Gradients are summed without division, as
    ``disable_fsdp_gradient_division`` makes FSDP2 do.

    With expert parallelism, ``fully_shard`` splits an MoE block into two
    groups: the routed experts on ``edp_mesh``'s ``edp_shard`` axis, and the
    rest of the block on ``dp_mesh``. The block then gets a bucket for each.

    Args:
        model: The decoder to shard.
        dp_mesh: The 1D ``dp_shard`` mesh to shard over.
        param_dtype: The dtype of the unsharded parameters.
        reduce_dtype: The dtype of gradient reduction.
        pp_enabled: Whether pipeline parallelism is enabled.
        reshard_after_forward_policy: "default", "always" or "never", as for
            ``apply_fsdp_to_decoder``.
        ep_degree: The expert-parallel degree.
        edp_mesh: With ``ep_degree > 1``, the sparse storage mesh, with the
            ``edp_shard`` and ``ep`` axes.
    """
    # flex_shard is an optional dependency, needed only for this backend.
    # pyrefly: ignore [missing-import]
    from flex_shard import BucketSpec, flex_shard, MixedPrecisionPolicy

    # pyrefly: ignore [missing-import]
    from flex_shard.custom_placements.shard import Shard as FlexShardShard

    if dp_mesh.mesh_dim_names != ("dp_shard",):
        raise ValueError(
            "FlexShard shards over a 1D mesh with the dp_shard axis, but got "
            f"mesh axes {dp_mesh.mesh_dim_names}."
        )
    expert_mesh = None
    if ep_degree > 1:
        if edp_mesh is None or edp_mesh.mesh_dim_names != ("edp_shard", "ep"):
            raise ValueError(
                "With expert parallelism, FlexShard shards routed experts over the "
                "edp_shard axis of a mesh with the edp_shard and ep axes, but got "
                f"{None if edp_mesh is None else edp_mesh.mesh_dim_names}."
            )
        expert_mesh = edp_mesh["edp_shard"]
        _EXPERT_STORAGE_MESHES[expert_mesh] = edp_mesh
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    # Shard dims other than 0, which FSDP2 gets from shard_placement_fn.
    shard_dims: dict[nn.Parameter, int] = {}

    def placement_fn(named_params, mesh):
        del mesh
        return {
            fqn: (FlexShardShard(shard_dims.get(param, 0)),)
            for fqn, param in named_params
        }

    buckets: list[BucketSpec] = []
    bucketed_params: set[nn.Parameter] = set()

    def add_bucket(
        patterns: list[str], reshard_after_forward: bool, mesh: DeviceMesh = dp_mesh
    ) -> None:
        buckets.append(
            BucketSpec(
                patterns,
                placement_fn=placement_fn,
                mesh=mesh,
                mp_policy=mp_policy,
                gradient_divide_factor=1.0,
                reshard_after_forward=reshard_after_forward,
                fsdp2_compatible=True,
            )
        )

    def add_module_bucket(module_fqns: list[str], reshard_after_forward: bool) -> None:
        for fqn in module_fqns:
            bucketed_params.update(model.get_submodule(fqn).parameters())
        add_bucket(module_fqns, reshard_after_forward)

    if model.enable_weight_tying:
        add_module_bucket(
            [
                name
                for name in ("tok_embeddings", "norm", "lm_head")
                if getattr(model, name) is not None
            ],
            reshard_after_forward_policy == "always",
        )
    else:
        if model.tok_embeddings is not None:
            add_module_bucket(["tok_embeddings"], reshard_after_forward)
        if model.norm is not None and model.lm_head is not None:
            add_module_bucket(
                ["norm", "lm_head"], reshard_after_forward_policy == "always"
            )
    param_fqns = {param: fqn for fqn, param in model.named_parameters()}
    for layer_id, transformer_block in model.layers.items():
        block_fqn = f"layers.{layer_id}"
        placements = linear_param_shard_placements(transformer_block)
        moe = None
        if getattr(transformer_block, "moe_enabled", False):
            moe = cast("MoE", transformer_block.moe)
            placements.update(
                routed_expert_param_placements(
                    moe.routed_experts,
                    num_experts=moe.num_experts,
                    expert_sharding_size=(
                        dp_mesh.size()
                        if expert_mesh is None
                        else expert_mesh.size() * ep_degree
                    ),
                )
            )
        shard_dims.update((param, p.dim) for param, p in placements.items())
        if moe is None or expert_mesh is None:
            add_module_bucket([block_fqn], reshard_after_forward)
            continue
        # Parameter-name buckets hook the deepest module holding their params:
        # the block for the dense bucket, the routed experts for the other.
        # Name them as flex_shard does, with any activation-checkpoint wrapper.
        experts = set(moe.routed_experts.parameters())
        dense_fqns, expert_fqns = [], []
        for param in transformer_block.parameters():
            fqns = expert_fqns if param in experts else dense_fqns
            fqns.append(param_fqns[param])
        bucketed_params.update(transformer_block.parameters())
        add_bucket(dense_fqns, reshard_after_forward)
        add_bucket(expert_fqns, reshard_after_forward, mesh=expert_mesh)
    # fully_shard(model) puts the remaining parameters in the root group, which
    # FSDP2 does not reshard after forward.
    root_fqns = [
        fqn for fqn, param in model.named_parameters() if param not in bucketed_params
    ]
    if root_fqns:
        add_bucket(root_fqns, reshard_after_forward=False)

    flex_shard(model, buckets=buckets)
    logger.info("Applied FlexShard to the model")


def as_fsdp2_dtensor(param: torch.Tensor) -> torch.Tensor:
    """Return a FlexShard parameter as the DTensor FSDP2 would give it.

    FlexShard stores a parameter as its plain local shard. The returned DTensor
    views that shard, on its bucket's mesh with its ``Shard`` placement, so code
    written for FSDP2's DTensor parameters (e.g. parallelism-agnostic
    initialization) behaves the same. Other tensors are returned unchanged.
    FlexShard parameters exist only once ``flex_shard`` is imported.
    """
    return _as_fsdp2_dtensor(param, param)


def grads_for_norm(parameters: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """Return the parameters' grads, with FlexShard's local shards as DTensors.

    Viewed as the DTensors FSDP2 would produce, FlexShard grads reduce their
    norm over their meshes as FSDP2's grads do.
    """
    return [
        _as_fsdp2_dtensor(param.grad, param)
        for param in parameters
        if param.grad is not None
    ]


def _as_fsdp2_dtensor(tensor: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """View ``tensor``, laid out as ``param``'s local shard, as a DTensor if
    ``param`` is a FlexShard parameter."""
    flex_shard = sys.modules.get("flex_shard")
    if flex_shard is None or not flex_shard.is_flex_shard_param(param):
        return tensor
    global_shape = flex_shard.get_global_shape(param)
    mesh = flex_shard.get_mesh(param)
    placements = tuple(Shard(p.dim) for p in flex_shard.get_placements(param))
    if (storage_mesh := _EXPERT_STORAGE_MESHES.get(mesh)) is not None:
        # Expert parallelism shards the experts' dim 0 over ep; FlexShard's
        # global shape is one ep rank's experts. As in FSDP2, the edp_shard
        # placement shards within each ep shard of the same dim.
        ep = storage_mesh["ep"].size()
        (fsdp_placement,) = placements
        placements = (
            _StridedShard(0, split_factor=ep)
            if fsdp_placement.dim == 0
            else fsdp_placement,
            Shard(0),
        )
        global_shape = torch.Size((global_shape[0] * ep, *global_shape[1:]))
        mesh = storage_mesh
    return DTensor.from_local(
        tensor,
        mesh,
        placements,
        run_check=False,
        shape=global_shape,
        stride=make_contiguous_strides_for(global_shape),
    )
