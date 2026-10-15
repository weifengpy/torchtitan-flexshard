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

import contextlib
import logging
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, cast, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.distributed.fsdp import (
    get_fsdp_reshard_after_forward_policy,
    linear_param_shard_placements,
    routed_expert_param_placements,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types, spmd_axes

if TYPE_CHECKING:
    from torchtitan.models.common.decoder import Decoder
    from torchtitan.models.common.moe import MoE

__all__ = [
    "apply_flex_shard_to_decoder",
    "as_flex_shard_state_dict",
    "as_fsdp2_dtensor",
    "as_fsdp2_state_dict",
    "enable_pipelining",
    "flex_shard_owner_map",
    "fsdp2_dtensor_params",
    "grads_for_norm",
    "owned_placement_fn",
    "relayout_owned_buckets",
]

logger = logging.getLogger(__name__)

# The edp_shard meshes of FlexShard's routed-expert buckets, mapped to the sparse
# storage mesh on which FSDP2 lays out their parameters, with the ep axis.
_EXPERT_STORAGE_MESHES: dict[DeviceMesh, DeviceMesh] = {}
# Under tensor parallelism, the dp_shard meshes of the other buckets, mapped to
# the dense storage mesh on which FSDP2 lays out their parameters, with tp.
_DENSE_STORAGE_MESHES: dict[DeviceMesh, DeviceMesh] = {}


def apply_flex_shard_to_decoder(
    model: "Decoder",
    storage_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
    extra_blocks: Sequence[tuple[str, nn.Module]] = (),
    owned_param_patterns: Sequence[str] = (),
    blockwise_fp8: bool = False,
    blockwise_fp8_dense_reference: bool = False,
    separate_reduce_scatter_group: bool = False,
) -> None:
    """Shard a decoder with FlexShard as ``apply_fsdp_to_decoder`` does with FSDP2.

    Each ``fully_shard`` group becomes a bucket naming the same modules: the
    embedding (with the norm and output projection when weights are tied), the
    norm with the output projection, each transformer block, and the parameters
    left to the root group. Gradients are summed without division, as
    ``disable_fsdp_gradient_division`` makes FSDP2 do.

    Buckets shard over ``storage_mesh``'s ``dp_shard`` axis. With context
    parallelism they shard over ``dp_shard`` and ``cp``, flattened as
    ``fully_shard`` flattens several shard axes. With tensor parallelism,
    parameters are TP-local shards: they declare their TP layout, and the
    parameters typed ``R`` on tp, whose grads FSDP2 all-reduces over TP before
    its reduce-scatter, declare the TP group as their partial-grad group.

    With expert parallelism, ``fully_shard`` splits an MoE block into two
    groups: the routed experts on ``edp_mesh``'s ``edp_shard`` axis, and the
    rest of the block on the dense mesh. The block then gets a bucket for each.
    The routed experts first declare which experts each ep rank holds, so that
    FlexShard's checkpoint layouts cover every ep rank's experts.

    Args:
        model: The decoder to shard.
        storage_mesh: The dense storage mesh from ``resolve_fsdp_mesh``, with the
            ``dp_shard`` axis, and ``cp`` or ``tp`` when context or tensor
            parallelism is on.
        param_dtype: The dtype of the unsharded parameters.
        reduce_dtype: The dtype of gradient reduction.
        pp_enabled: Whether pipeline parallelism is enabled.
        reshard_after_forward_policy: "default", "always" or "never", as for
            ``apply_fsdp_to_decoder``.
        ep_degree: The expert-parallel degree.
        edp_mesh: With ``ep_degree > 1``, the sparse storage mesh, with the
            ``edp_shard`` and ``ep`` axes.
        extra_blocks: Transformer blocks outside ``model.layers``, by FQN, each
            sharded like a block after the last one, e.g. DeepSeek V3's MTP
            layers, which ``apply_fsdp_to_mtp_decoder`` appends to the layers.
        owned_param_patterns: Regex patterns of a block's parameter FQNs that
            go in a bucket of their own, to be stored whole on owner ranks
            after initialization (``parallelism.flex_shard_owned_params``,
            ``relayout_owned_buckets``). Until then the bucket shards like the
            others, so initialization matches FSDP2's. Patterns covering the
            whole block outside the routed experts leave the block one bucket.
        blockwise_fp8: Whether ``convert_blockwise_fp8_linears`` converted
            linears, which then all-gather blockwise FP8. A block holding them
            gets a mixed bucket in place of its dense ``fsdp2_compatible`` one.
        blockwise_fp8_dense_reference: As ``blockwise_fp8``, but the converted
            linears all-gather bf16 and are quantized after the gather, with
            the same layout: the A/B reference for ``blockwise_fp8``.
        separate_reduce_scatter_group: Whether gradient reduce-scatters run on a
            process group of their own
            (``FlexShardModule.set_separate_reduce_scatter_group``).
    """
    # flex_shard is an optional dependency, needed only for this backend.
    # pyrefly: ignore [missing-import]
    from flex_shard import BucketSpec, flex_shard, MixedPrecisionPolicy

    # pyrefly: ignore [missing-import]
    from flex_shard.custom_placements.shard import Shard as FlexShardShard

    mesh_axis_names = storage_mesh.mesh_dim_names
    if mesh_axis_names == ("dp_shard",):
        dp_mesh = storage_mesh
    elif mesh_axis_names == ("dp_shard", "cp"):
        # The flattened mesh fully_shard creates for these shard axes.
        dp_mesh = storage_mesh["dp_shard", "cp"]._flatten("dp_shard_cp")
    elif mesh_axis_names == ("dp_shard", "tp"):
        dp_mesh = storage_mesh["dp_shard"]
        _DENSE_STORAGE_MESHES[dp_mesh] = storage_mesh
    elif mesh_axis_names == ("dp_shard", "cp", "tp"):
        dp_mesh = storage_mesh["dp_shard", "cp"]._flatten("dp_shard_cp")
        # As FSDP2's DTensorSpec with several shard axes: the flattened mesh
        # and tp.
        _DENSE_STORAGE_MESHES[dp_mesh] = DeviceMesh._concatenate(
            [dp_mesh, storage_mesh["tp"]]
        )
    else:
        raise ValueError(
            "FlexShard shards over the dp_shard axis, with cp, tp or both under "
            f"context or tensor parallelism, but got mesh axes {mesh_axis_names}."
        )
    if "tp" in mesh_axis_names:
        _declare_spmd_global_layouts(model, storage_mesh, routed_experts=False)
        _declare_tensor_parallel_partial_grads(model, storage_mesh.get_group("tp"))
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
        _declare_spmd_global_layouts(model, edp_mesh, routed_experts=True)
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

    fp8_weights: set[nn.Parameter] = set()
    mixed_placement_fn = expert_mixed_placement_fn = None
    if blockwise_fp8 or blockwise_fp8_dense_reference:
        from torchtitan.distributed import flex_shard_blockwise_fp8

        fp8_weights = flex_shard_blockwise_fp8.blockwise_fp8_weights(model)
        mixed_placement_fn = flex_shard_blockwise_fp8.make_mixed_placement_fn(
            model, placement_fn, dense_reference=blockwise_fp8_dense_reference
        )
        # FP8 routed experts (parallelism.blockwise_fp8_routed_experts) keep
        # block-row units: their initialization draws whole rows.
        expert_mixed_placement_fn = flex_shard_blockwise_fp8.make_mixed_placement_fn(
            model,
            placement_fn,
            dense_reference=blockwise_fp8_dense_reference,
            row_blocks=True,
        )

    buckets: list[BucketSpec] = []
    bucketed_params: set[nn.Parameter] = set()

    def add_bucket(
        patterns: list[str],
        reshard_after_forward: bool,
        mesh: DeviceMesh = dp_mesh,
        *,
        mixed: bool = False,
        experts: bool = False,
        defer_post_backward: bool = False,
        post_reduce_hook: Callable[[list[tuple[str, nn.Parameter]]], None]
        | None = None,
    ) -> int:
        if mixed:
            bucket_placement_fn = (
                expert_mixed_placement_fn if experts else mixed_placement_fn
            )
        else:
            bucket_placement_fn = placement_fn
        buckets.append(
            BucketSpec(
                patterns,
                placement_fn=bucket_placement_fn,
                mesh=mesh,
                mp_policy=mp_policy,
                gradient_divide_factor=1.0,
                reshard_after_forward=reshard_after_forward,
                post_reduce_hook=post_reduce_hook,
                defer_post_backward=defer_post_backward,
                # fsdp2_compatible requires Shard placements.
                fsdp2_compatible=not mixed,
            )
        )
        return len(buckets) - 1

    def add_module_bucket(
        module_fqns: list[str], reshard_after_forward: bool, *, mixed: bool = False
    ) -> int:
        for fqn in module_fqns:
            bucketed_params.update(model.get_submodule(fqn).parameters())
        return add_bucket(module_fqns, reshard_after_forward, mixed=mixed)

    # Bucket indices, for the explicit prefetch under expert parallelism.
    embedding_bucket = head_bucket = None
    block_buckets: list[list[int]] = []
    expert_buckets: list[int] = []
    if model.enable_weight_tying:
        embedding_bucket = head_bucket = add_module_bucket(
            [
                name
                for name in ("tok_embeddings", "norm", "lm_head")
                if getattr(model, name) is not None
            ],
            reshard_after_forward_policy == "always",
        )
    else:
        if model.tok_embeddings is not None:
            embedding_bucket = add_module_bucket(
                ["tok_embeddings"], reshard_after_forward
            )
        if model.norm is not None and model.lm_head is not None:
            head_bucket = add_module_bucket(
                ["norm", "lm_head"], reshard_after_forward_policy == "always"
            )
    param_fqns = {param: fqn for fqn, param in model.named_parameters()}
    owned_res = [re.compile(pattern) for pattern in owned_param_patterns]

    def is_owned(param: nn.Parameter) -> bool:
        fqn = canonical_fqn(param_fqns[param])
        return any(pattern.search(fqn) for pattern in owned_res)

    blocks = [
        (f"layers.{layer_id}", transformer_block)
        for layer_id, transformer_block in model.layers.items()
    ]
    for block_fqn, transformer_block in (*blocks, *extra_blocks):
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
        owned = {param for param in transformer_block.parameters() if is_owned(param)}
        mixed = not fp8_weights.isdisjoint(transformer_block.parameters())
        if not owned and (moe is None or expert_mesh is None):
            block_buckets.append(
                [add_module_bucket([block_fqn], reshard_after_forward, mixed=mixed)]
            )
            continue
        # Parameter-name buckets hook the deepest module holding their params:
        # the block for the dense and owned buckets, the routed experts for the
        # expert bucket. Name them as flex_shard does, with any
        # activation-checkpoint wrapper.
        experts = (
            set(moe.routed_experts.parameters())
            if moe is not None and expert_mesh is not None
            else set()
        )
        if experts & owned:
            raise ValueError(
                "parallelism.flex_shard_owned_params can't name routed experts, "
                "which FlexShard shards over the expert mesh."
            )
        dense_fqns, owned_fqns, expert_fqns = [], [], []
        for param in transformer_block.parameters():
            if param in experts:
                expert_fqns.append(param_fqns[param])
            elif param in owned:
                owned_fqns.append(param_fqns[param])
            else:
                dense_fqns.append(param_fqns[param])
        bucketed_params.update(transformer_block.parameters())
        # With expert parallelism, the block's end issues its expert
        # reduce-scatter, then its dense one, as FSDP2 does for a module's two
        # parameter groups: the expert bucket defers its post-backward, and the
        # dense bucket finishes it before reducing its own grads. With the
        # experts-first backward prefetch, both NCCL streams then carry expert,
        # then dense collectives at the same time. Hooked on the routed experts,
        # the expert reduce-scatter would start mid-block instead, beside the
        # next block's dense gather.
        reduce_experts_at_block_end = bool(expert_fqns and dense_fqns and ep_degree > 1)
        # When the owned patterns cover the whole block, its owned bucket is its
        # only bucket outside the routed experts: one collective per block.
        buckets_of_block = []
        if dense_fqns:
            buckets_of_block.append(
                add_bucket(
                    dense_fqns,
                    reshard_after_forward,
                    mixed=mixed,
                    post_reduce_hook=(
                        _finish_deferred_backward_of(model, expert_fqns[0])
                        if reduce_experts_at_block_end
                        else None
                    ),
                )
            )
        if owned_fqns:
            buckets_of_block.append(add_bucket(owned_fqns, reshard_after_forward))
        if expert_fqns:
            expert_bucket = add_bucket(
                expert_fqns,
                reshard_after_forward,
                mesh=expert_mesh,
                mixed=not fp8_weights.isdisjoint(experts),
                experts=True,
                defer_post_backward=reduce_experts_at_block_end,
            )
            buckets_of_block.append(expert_bucket)
            expert_buckets.append(expert_bucket)
        block_buckets.append(buckets_of_block)
    # fully_shard(model) puts the remaining parameters in the root group, which
    # FSDP2 does not reshard after forward.
    root_fqns = [
        fqn for fqn, param in model.named_parameters() if param not in bucketed_params
    ]
    if root_fqns:
        add_bucket(root_fqns, reshard_after_forward=False)

    if owned_res:
        # Owner storage serves optimizers that compute on local storage, such as
        # DistMuon through build_local_dist_muon, which reads each parameter's
        # checkpoint coordinates from before flex_shard().
        # pyrefly: ignore [missing-import]
        from flex_shard.dist_muon import capture_flex_shard_muon_canonical_shards

        capture_flex_shard_muon_canonical_shards(model)
    flex_shard(model, buckets=buckets)
    if separate_reduce_scatter_group:
        model.set_separate_reduce_scatter_group()
    if blockwise_fp8 or blockwise_fp8_dense_reference:
        flex_shard_blockwise_fp8.set_global_layouts(model)
    if ep_degree > 1:
        _set_explicit_prefetch(
            # One bucket storage per BucketSpec, all of which name parameters.
            # pyrefly: ignore [bad-argument-type]
            model.sharded_bucket_storages,
            embedding_bucket=embedding_bucket,
            block_buckets=block_buckets,
            head_bucket=head_bucket,
            expert_buckets=expert_buckets,
        )
        # A MoE layer is two buckets, its routed experts and the rest. With one
        # pending reduce-grad, the dense bucket's reduce-scatter first waits for
        # the expert reduce-scatter issued just before it, in the same layer, and
        # so do the compute stream and the next backward all-gathers. FSDP2 holds
        # both parameter groups in one module per layer and waits for the
        # previous layer's reduce-scatter instead; two pending match that.
        model.set_max_pending_reduce_grads(2)
    logger.info("Applied FlexShard to the model")


def _declare_spmd_global_layouts(
    model: nn.Module, mesh: DeviceMesh, *, routed_experts: bool
) -> None:
    """Declare where each parameter sits in its full parameter, from the SPMD
    layouts ``Module._parallelize`` records in each module's ``_sharding_config``.

    ``_parallelize`` leaves routed experts as each ep rank's plain local shard
    of the experts' dim 0, and TP-sharded parameters as each tp rank's shard.
    Declared before ``flex_shard``, the layout composes with FlexShard's own
    split, so checkpoints describe the full parameters. ``routed_experts``
    selects the layouts on the sparse mesh, with the ep axis, or the others,
    on the dense storage mesh. Axes absent from ``mesh`` count as size 1.
    """
    # pyrefly: ignore [missing-import]
    from flex_shard.layout_adapters.spmd_types import spmd_types_to_global_layout

    layouts = {}
    for module_fqn, module in model.named_modules():
        sharding_config = getattr(module, "_sharding_config", None)
        if sharding_config is None:
            continue
        prefix = f"{module_fqn}." if module_fqn else ""
        for name, layout in sharding_config.state_shardings.items():
            if (MeshAxisName.EP in spmd_axes(layout)) == routed_experts:
                layouts[f"{prefix}{name}"] = layout
    spmd_types_to_global_layout(model, layouts, mesh)


def _declare_tensor_parallel_partial_grads(
    model: nn.Module, tp_group: dist.ProcessGroup
) -> None:
    """Declare the TP group on the parameters typed ``R`` on tp.

    Under sequence parallelism, these are the norm weights: each tp rank's grad
    covers its own tokens. FSDP2 types such a grad ``P`` and all-reduces it over
    TP before its reduce-scatter; FlexShard does the same for a declared group.
    """
    # pyrefly: ignore [missing-import]
    from flex_shard import set_partial_grad_group

    for module in model.modules():
        sharding_config = getattr(module, "_sharding_config", None)
        if sharding_config is None:
            continue
        for name, layout in sharding_config.state_shardings.items():
            param = module._parameters.get(name)
            if (
                param is not None
                and _per_axis_types(layout).get(MeshAxisName.TP) is spmd.R
            ):
                set_partial_grad_group(param, tp_group)


def _finish_deferred_backward_of(
    model: nn.Module, param_fqn: str
) -> Callable[[list[tuple[str, nn.Parameter]]], None]:
    """A ``BucketSpec.post_reduce_hook`` that finishes the deferred post-backward
    of the bucket holding ``param_fqn``: its reduce-scatter, then its reshard.

    flex_shard runs the hook after the hooked bucket takes its grads and before
    its own reduce-scatter, so the deferred bucket's reduce-scatter goes first.
    The parameter is looked up when the hook runs, since flex_shard registers
    the unsharded parameter during backward.
    """

    def hook(named_params: list[tuple[str, nn.Parameter]]) -> None:
        # flex_shard() turns the model into a FlexShardModule.
        # pyrefly: ignore [missing-attribute]
        model.finish_deferred_backward(model.get_parameter(param_fqn))

    return hook


def _set_explicit_prefetch(
    storages: list,
    *,
    embedding_bucket: int | None,
    block_buckets: list[list[int]],
    head_bucket: int | None,
    expert_buckets: list[int],
) -> None:
    """Set the explicit prefetch ``apply_fsdp_to_decoder`` sets with expert
    parallelism, whose device-to-host syncs keep the CPU from issuing the next
    all-gathers early.

    In forward, the embedding prefetches the first block, and each block the
    next, the last one the norm with the output projection; in backward, the
    output projection prefetches the last block, and each block the previous
    one, the first one the embedding. A block prefetches from its first bucket,
    which hooks the block, and prefetches all its buckets. Its expert bucket,
    whose group FSDP2 unshards with the block, prefetches nothing.

    In backward, a block's expert bucket is gathered first, as FSDP2 orders a
    module's groups. Gathering the dense bucket first left each dense gather
    running beside the current block's expert reduce-scatter, on the same
    network cards.
    """
    experts = set(expert_buckets)

    def block(idx: int) -> list:
        return [storages[bucket] for bucket in block_buckets[idx]]

    def block_backward(idx: int) -> list:
        ordered = sorted(block_buckets[idx], key=lambda bucket: bucket not in experts)
        return [storages[bucket] for bucket in ordered]

    for bucket in expert_buckets:
        storages[bucket].set_buckets_to_forward_prefetch([])
        storages[bucket].set_buckets_to_backward_prefetch([])
    if embedding_bucket is not None and block_buckets:
        storages[embedding_bucket].set_buckets_to_forward_prefetch(block(0))
    if head_bucket is not None and block_buckets:
        storages[head_bucket].set_buckets_to_backward_prefetch(block_backward(-1))
    for idx, buckets in enumerate(block_buckets):
        first = storages[buckets[0]]
        if idx + 1 < len(block_buckets):
            first.set_buckets_to_forward_prefetch(block(idx + 1))
        elif head_bucket is not None:
            first.set_buckets_to_forward_prefetch([storages[head_bucket]])
        else:
            first.set_buckets_to_forward_prefetch([])
        if idx > 0:
            first.set_buckets_to_backward_prefetch(block_backward(idx - 1))
        elif embedding_bucket is not None:
            first.set_buckets_to_backward_prefetch([storages[embedding_bucket]])


def as_fsdp2_dtensor(param: torch.Tensor) -> torch.Tensor:
    """Return a FlexShard parameter as the DTensor FSDP2 would give it.

    FlexShard stores a parameter as its plain local shard. The returned DTensor
    views that shard, on its bucket's mesh with its ``Shard`` placement, so code
    written for FSDP2's DTensor parameters (e.g. parallelism-agnostic
    initialization) behaves the same. Other tensors are returned unchanged.
    FlexShard parameters exist only once ``flex_shard`` is imported.
    """
    return _as_fsdp2_dtensor(param, param)


@contextlib.contextmanager
def fsdp2_dtensor_params(module: nn.Module) -> Iterator[None]:
    """Expose ``module``'s own FlexShard parameters as the DTensors FSDP2 would
    give them while the context is active, e.g. for ``reset_parameters``.

    The DTensors view the local shards, so in-place updates reach them.
    """
    flex_shard = sys.modules.get("flex_shard")
    swapped: dict[str, nn.Parameter] = {}
    if flex_shard is not None:
        for name, param in module.named_parameters(recurse=False):
            if flex_shard.is_flex_shard_param(param):
                swapped[name] = param
                module._parameters[name] = nn.Parameter(
                    _as_fsdp2_dtensor(param, param),
                    requires_grad=param.requires_grad,
                )
    try:
        yield
    finally:
        for name, param in swapped.items():
            module._parameters[name] = param


def as_fsdp2_state_dict(
    model_parts: Iterable[nn.Module], state_dict: dict[str, Any]
) -> dict[str, Any]:
    """Return ``state_dict`` with its FlexShard parameters as the DTensors FSDP2
    would give them, for code written for FSDP2's state dicts, e.g. the Hugging
    Face conversions of state dict adapters.

    The DTensors view the state dict's tensors, so in-place updates such as
    ``dcp.load`` reach them. Other entries are unchanged.

    A blockwise FP8 parameter's local shard is a range of its matrix's rows,
    packed across its bucket's parameters, which no DTensor placement describes.
    It is given instead as a new DTensor in FSDP2's layout, Shard on the matrix
    rows, filled from every rank's rows: a copy, not a view, whose loaded values
    ``as_flex_shard_state_dict`` maps back to the local rows.
    """
    converted = dict(state_dict)
    for key, param, row_layout in _flex_shard_params(model_parts, state_dict):
        if row_layout is None:
            converted[key] = _as_fsdp2_dtensor(state_dict[key], param)
        else:
            converted[key] = _rows_as_fsdp2_dtensor(state_dict[key], param, row_layout)
    return converted


def as_flex_shard_state_dict(
    model_parts: Iterable[nn.Module], state_dict: dict[str, Any]
) -> dict[str, Any]:
    """Return ``state_dict`` with its FlexShard parameters' DTensors as the
    parameters' local shards, inverting ``as_fsdp2_state_dict`` for
    ``load_state_dict``."""
    converted = dict(state_dict)
    for key, param, row_layout in _flex_shard_params(model_parts, state_dict):
        value = state_dict[key]
        assert isinstance(value, DTensor), f"expected a DTensor for {key}"
        if row_layout is not None:
            full = value.full_tensor()
            local = full.new_empty(param.shape)
            for global_box, local_box in _layout_boxes(row_layout):
                local[local_box] = full[global_box]
            converted[key] = local
            continue
        view = cast(DTensor, _as_fsdp2_dtensor(param.detach(), param))
        value = value.redistribute(view.device_mesh, view.placements)
        converted[key] = value.to_local()
    return converted


def _flex_shard_params(
    model_parts: Iterable[nn.Module], state_dict: dict[str, Any]
) -> Iterator[tuple[str, nn.Parameter, Any]]:
    """Yield the FlexShard parameters of ``model_parts`` with their keys in
    ``state_dict`` and, for blockwise FP8 parameters, the ``GlobalLayout`` of
    their local rows (``None`` for the others)."""
    flex_shard = sys.modules.get("flex_shard")
    if flex_shard is None:
        return
    for model in model_parts:
        row_layouts: dict[str, Any] | None = None
        for fqn, param in model.named_parameters():
            key = canonical_fqn(fqn)
            if key not in state_dict or not flex_shard.is_flex_shard_param(param):
                continue
            row_layout = None
            if _has_row_range_shard(param):
                if row_layouts is None:
                    row_layouts = {
                        canonical_fqn(name): layout
                        for name, layout in flex_shard.get_flex_shard_global_layouts(
                            model
                        ).items()
                    }
                row_layout = row_layouts[key]
            yield key, param, row_layout


def _has_row_range_shard(param: torch.Tensor) -> bool:
    """Whether ``param``'s local shard is blockwise FP8 matrix rows."""
    # pyrefly: ignore [missing-import]
    from flex_shard.custom_placements import Fp8BucketedBlockShard

    placements = sys.modules["flex_shard"].get_placements(param)
    return any(isinstance(p, Fp8BucketedBlockShard) for p in placements)


def _layout_boxes(layout: Any) -> Iterator[tuple[tuple[slice, ...], tuple[slice, ...]]]:
    """Yield (full-parameter region, local-shard region) slices of ``layout``."""
    for global_offset, local_offset, size in zip(
        layout.global_offsets, layout.local_offsets, layout.local_sizes, strict=True
    ):
        yield (
            tuple(slice(o, o + n) for o, n in zip(global_offset, size, strict=True)),
            tuple(slice(o, o + n) for o, n in zip(local_offset, size, strict=True)),
        )


def _rows_as_fsdp2_dtensor(
    tensor: torch.Tensor, param: torch.Tensor, layout: Any
) -> DTensor:
    """A blockwise FP8 parameter laid out as FSDP2 lays out linear weights: Shard
    on the matrix rows, so a stacked [N, F, D] weight keeps N whole.

    Every rank holds a disjoint set of the full parameter's elements, so summing
    zero-filled full copies over the mesh assembles it exactly.
    """
    flex_shard = sys.modules["flex_shard"]
    mesh = flex_shard.get_mesh(param)
    if (
        mesh.ndim != 1
        or mesh in _DENSE_STORAGE_MESHES
        or mesh in _EXPERT_STORAGE_MESHES
    ):
        raise NotImplementedError(
            "FSDP2 state dicts of blockwise FP8 parameters need a 1D mesh without "
            "tensor or expert parallelism storage meshes."
        )
    full = tensor.new_zeros(layout.global_shape)
    for global_box, local_box in _layout_boxes(layout):
        full[global_box] = tensor[local_box]
    dist.all_reduce(full, group=mesh.get_group())
    sharded = distribute_tensor(full, mesh, [Shard(full.ndim - 2)], src_data_rank=None)
    # Copy the local rows out of ``full``, so that a view doesn't keep the whole
    # parameter alive on every rank until the load ends.
    return DTensor.from_local(
        sharded.to_local().clone(),
        mesh,
        sharded.placements,
        run_check=False,
        shape=sharded.shape,
        stride=sharded.stride(),
    )


class _FSDPModuleOrFlexShardMeta(type):
    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, FSDPModule) or hasattr(
            instance, "bucket_storage_of"
        )


class _FSDPModuleOrFlexShard(metaclass=_FSDPModuleOrFlexShardMeta):
    """Matches FSDP2 and FlexShard modules in ``isinstance`` checks."""


def enable_pipelining() -> None:
    """Let ``torch.distributed.pipelining`` drive FlexShard modules as it
    drives FSDP2's.

    PyTorch's pipeline stages and schedules treat a stage module as data
    parallel only if it is an ``FSDPModule``. A FlexShard module has every
    method they call: ``set_manual_backward_finalization``,
    ``set_requires_gradient_sync``, ``set_reshard_after_backward``,
    ``finalize_backward(async_op=True)``, ``unshard(async_op=True)``, with
    which multi-stage schedules gather a stage ahead of its forward, and
    ``reshard``. Their ``isinstance`` checks therefore match FlexShard modules
    too. ``defer_reduce_grad_wait`` also checks that no two FSDP2 stages share
    a communication context, reading FSDP2's state. FlexShard keeps its
    contexts on each stage's root module, so its stages can't share one, and
    the check covers FSDP2's stages only.

    TODO: remove once ``torch.distributed.pipelining`` checks for the methods
    it calls rather than for ``FSDPModule``.
    """
    from torch.distributed.pipelining import schedules, stage as stage_module

    if schedules.FSDPModule is not _FSDPModuleOrFlexShard:
        # pyrefly: ignore [bad-assignment]
        stage_module.FSDPModule = _FSDPModuleOrFlexShard
        # pyrefly: ignore [bad-assignment]
        schedules.FSDPModule = _FSDPModuleOrFlexShard
        runtime = schedules._PipelineScheduleRuntime
        validate = runtime._validate_deferred_gradient_reduction

        def validate_fsdp2_stages(self: schedules._PipelineScheduleRuntime) -> None:
            schedules.FSDPModule = FSDPModule
            try:
                validate(self)
            finally:
                # pyrefly: ignore [bad-assignment]
                schedules.FSDPModule = _FSDPModuleOrFlexShard

        runtime._validate_deferred_gradient_reduction = validate_fsdp2_stages


def grads_for_norm(parameters: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    """Return the parameters' grads, with FlexShard's local shards as DTensors.

    Viewed as the DTensors FSDP2 would produce, FlexShard grads reduce their
    norm over their meshes as FSDP2's grads do. A grad stored whole on its
    owner (``parallelism.flex_shard_owned_params``), and empty on the other
    ranks, has no DTensor placement; as a ``Shard(0)`` DTensor on its bucket's
    mesh, a norm still sums each rank's local elements and reduces over the
    mesh, so each element counts once. These views serve norms only.
    """
    return [
        _owner_grad_for_norm(param)
        if _is_owner_stored(param)
        else _as_fsdp2_dtensor(param.grad, param)
        for param in parameters
        if param.grad is not None
    ]


def _is_owner_stored(param: torch.Tensor) -> bool:
    flex_shard = sys.modules.get("flex_shard")
    if flex_shard is None or not flex_shard.is_flex_shard_param(param):
        return False
    # pyrefly: ignore [missing-import]
    from flex_shard.custom_placements.owned import BucketedOwned

    return isinstance(flex_shard.get_placements(param)[0], BucketedOwned)


def _owner_grad_for_norm(param: torch.Tensor) -> torch.Tensor:
    flex_shard = sys.modules["flex_shard"]
    global_shape = flex_shard.get_global_shape(param)
    return DTensor.from_local(
        param.grad,
        flex_shard.get_mesh(param),
        (Shard(0),),
        run_check=False,
        shape=global_shape,
        stride=make_contiguous_strides_for(global_shape),
    )


def owned_placement_fn() -> Callable:
    """The placement of owned buckets: each parameter whole on one rank of
    its bucket's mesh, owners balanced over hosts first, then over ranks
    within a host, carrying each rank's load from bucket to bucket. Share one
    across a rank's model parts, so virtual stages balance together too."""
    # pyrefly: ignore [missing-import]
    from flex_shard.custom_placements.owned import (
        make_bucketed_owned_full_param_placement_fn,
    )

    return make_bucketed_owned_full_param_placement_fn(carry_loads=True)


def flex_shard_owner_map(
    model_parts: Sequence[nn.Module],
    owned_param_patterns: Sequence[str],
) -> dict[str, int]:
    """The owner FlexShard gives each parameter ``owned_param_patterns`` names,
    by canonical FQN, as its rank on the data-parallel mesh: each block's owned
    parameters one bucket, as ``apply_flex_shard_to_decoder`` builds them
    (decoder layers, then MTP layers), placed in model-part order by one
    ``owned_placement_fn``, as ``relayout_owned_buckets`` places them.

    The placement depends only on the parameters' global sizes, their order,
    and the mesh's ranks and hosts, so another backend's model parts with the
    same blocks, e.g. FSDP2's on the same data-parallel mesh, get the owners a
    FlexShard run of the same job has."""
    patterns = [re.compile(pattern) for pattern in owned_param_patterns]
    placement_fn = owned_placement_fn()
    owners: dict[str, int] = {}
    for model in model_parts:
        fqns = {param: canonical_fqn(name) for name, param in model.named_parameters()}
        layers = getattr(model, "layers", None)
        mtp_layers = getattr(model, "mtp_layers", None)
        blocks = [
            *(layers.values() if layers is not None else ()),
            *(mtp_layers.children() if mtp_layers is not None else ()),
        ]
        for block in blocks:
            named_params = [
                (fqns[param], param)
                for param in block.parameters()
                if any(pattern.search(fqns[param]) for pattern in patterns)
            ]
            if not named_params:
                continue
            meshes = {
                param.device_mesh
                for _fqn, param in named_params
                if isinstance(param, DTensor)
            }
            if len(meshes) != 1 or next(iter(meshes)).ndim != 1:
                raise ValueError(
                    "flex_shard_owner_map needs each block's owned parameters "
                    "on one one-dimensional data-parallel mesh, as DTensors"
                )
            # Full parameters, as FlexShard places them: DTensor sizes are global.
            for fqn, (placement,) in placement_fn(
                named_params, next(iter(meshes))
            ).items():
                (segment,) = placement.segments_by_fqn[fqn]
                owners[fqn] = segment.owner_rank
    return owners


def relayout_owned_buckets(
    model: nn.Module,
    owned_param_patterns: Sequence[str],
    placement_fn: Callable | None = None,
) -> None:
    """Move the buckets ``apply_flex_shard_to_decoder`` made for
    ``owned_param_patterns`` to whole-parameter owners (``owned_placement_fn``,
    or ``placement_fn``), once, after initialization or loading. Values move
    bitwise."""
    # pyrefly: ignore [missing-import]
    from flex_shard import relayout_bucket

    patterns = [re.compile(pattern) for pattern in owned_param_patterns]
    if placement_fn is None:
        placement_fn = owned_placement_fn()
    # pyrefly: ignore [missing-attribute]
    for storage in list(model.sharded_bucket_storages):
        owned = [
            any(pattern.search(canonical_fqn(fqn)) for pattern in patterns)
            for fqn in storage.param_infos
        ]
        if not any(owned):
            continue
        assert all(owned), list(storage.param_infos)
        relayout_bucket(model, storage, placement_fn)


def _as_fsdp2_dtensor(tensor: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """View ``tensor``, laid out as ``param``'s local shard, as a DTensor if
    ``param`` is a FlexShard parameter."""
    flex_shard = sys.modules.get("flex_shard")
    if flex_shard is None or not flex_shard.is_flex_shard_param(param):
        return tensor
    global_shape = flex_shard.get_global_shape(param)
    mesh = flex_shard.get_mesh(param)
    # A blockwise FP8 parameter's local shard is a row range, not a Shard
    # chunk. Viewed as Shard(0), its grad still reduces its norm correctly.
    placements = tuple(
        Shard(getattr(p, "dim", 0)) for p in flex_shard.get_placements(param)
    )
    if (storage_mesh := _DENSE_STORAGE_MESHES.get(mesh)) is not None:
        # Under TP, the global shape is the TP-local shape. As FSDP2 does, lay
        # the parameter out on its data-parallel mesh and tp, with Shard on tp
        # for the dim its TP layout splits; the data-parallel placement then
        # shards within each TP shard of that dim.
        (fsdp_placement,) = placements
        tp_placement: Shard | Replicate = Replicate()
        if (outer_layout := flex_shard.get_outer_layout(param)) is not None:
            for dim, (size, local_size) in enumerate(
                zip(outer_layout.global_shape, global_shape, strict=True)
            ):
                if size != local_size:
                    tp_placement = Shard(dim)
            global_shape = torch.Size(outer_layout.global_shape)
        if isinstance(tp_placement, Shard) and tp_placement.dim == fsdp_placement.dim:
            fsdp_placement = _StridedShard(
                fsdp_placement.dim, split_factor=storage_mesh.size(-1)
            )
        placements = (fsdp_placement, tp_placement)
        mesh = storage_mesh
    elif (storage_mesh := _EXPERT_STORAGE_MESHES.get(mesh)) is not None:
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
