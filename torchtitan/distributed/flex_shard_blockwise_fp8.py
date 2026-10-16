# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blockwise FP8 all-gather and GEMMs for FlexShard's dense linears.

With ``parallelism.flex_shard_blockwise_fp8``, the linears that
``BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS`` names all-gather 128x128-block FP8
weights and scales instead of bf16, and run FP8 GEMMs. Each stays in its
block's bucket: a ``MixedBucketPlacement`` gathers the FP8 weights and the
block's ``Shard(0)`` parameters in one collective, so the number of all-gathers
is unchanged. Routed experts, embeddings, norms, the router and the output
projection stay in higher precision. With
``parallelism.flex_shard_blockwise_fp8_dense_reference``, the same buckets
all-gather the weights' bf16 shards and quantize after the gather instead, an
A/B reference that should train bitwise identically. Imported only with these
options, as FlexShard is an optional dependency.
"""

import fnmatch

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate

# pyrefly: ignore [missing-import]
from flex_shard import (
    get_flex_shard_global_layouts,
    get_mesh,
    GlobalLayout,
    PlacementFn,
)

# pyrefly: ignore [missing-import]
from flex_shard.blockwise_fp8 import (
    BlockwiseFp8Weight,
    convert_to_flex_shard_float8_blockwise_linear,
    FlexShardFloat8BlockwiseLinear,
)

# pyrefly: ignore [missing-import]
from flex_shard.custom_placements.dense_all_gather_reference import (
    make_dense_all_gather_reference_placement_fn,
)

# pyrefly: ignore [missing-import]
from flex_shard.custom_placements.fp8_bucketed_block_shard import (
    make_fp8_bucketed_block_placement_fn,
)

# pyrefly: ignore [missing-import]
from flex_shard.custom_placements.mixed_bucket import MixedBucketPlacement

from torchtitan.protocols.module import Module

# DeepSeek V3's attention and dense feed-forward linears, and its shared experts.
# ``w13`` stacks the gate and up projections as one ``[2, F, D]`` weight.
BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS: tuple[str, ...] = (
    "layers.*.attention.wq_a",
    "layers.*.attention.wq_b",
    "layers.*.attention.wkv_a",
    "layers.*.attention.wkv_b",
    "layers.*.attention.wo",
    "layers.*.feed_forward.w13",
    "layers.*.feed_forward.w2",
    "layers.*.moe.shared_experts.w13",
    "layers.*.moe.shared_experts.w2",
)
_FP8_DTYPE = torch.float8_e4m3fn


class FlexShardBlockwiseFp8Linear(FlexShardFloat8BlockwiseLinear, Module):
    """FlexShard's blockwise FP8 linear, with the Module protocol state of the
    linear it replaces."""

    # Where each parameter's local shard sits in the full parameter, set by
    # set_global_layouts once FlexShard has sharded it.
    _global_layouts: dict[str, GlobalLayout] | None = None

    @classmethod
    def from_float(
        cls, mod: nn.Linear, use_triton: bool = False
    ) -> "FlexShardBlockwiseFp8Linear":
        if getattr(mod, "_parallelized", False):
            raise RuntimeError(
                "Blockwise FP8 conversion must run before Module._parallelize(), "
                "so the replacement receives the original module's sharding."
            )
        new_mod = super().from_float(mod, use_triton=use_triton)
        new_mod._param_init = getattr(mod, "_param_init", None)
        new_mod._sharding_config = getattr(mod, "_sharding_config", None)
        return new_mod

    def _init_param(self, name: str, param: nn.Parameter) -> None:
        layout = None if self._global_layouts is None else self._global_layouts.get(name)
        if layout is None or self._param_init is None:
            super()._init_param(name, param)
            return
        # The FP8 placement's local shards are row ranges, which no DTensor
        # placement describes. Initialize the full parameter as a replicated
        # DTensor, which draws what FSDP2's shards of it draw, and keep this
        # rank's regions.
        mesh = get_mesh(param)
        full = DTensor.from_local(
            param.new_empty(layout.global_shape),
            mesh,
            [Replicate()] * mesh.ndim,
            run_check=False,
        )
        self._param_init[name](full)
        full_local = full.to_local()
        with torch.no_grad():
            for global_offset, local_offset, size in zip(
                layout.global_offsets,
                layout.local_offsets,
                layout.local_sizes,
                strict=True,
            ):
                param[_box(local_offset, size)].copy_(
                    full_local[_box(global_offset, size)]
                )


def _box(offset: tuple[int, ...], size: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(
        slice(start, start + length) for start, length in zip(offset, size, strict=True)
    )


def convert_blockwise_fp8_linears(model: nn.Module, *, use_triton: bool) -> None:
    """Replace the linears ``BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS`` names with
    ``FlexShardBlockwiseFp8Linear``, whose GEMMs run TorchAO's Triton kernels
    with ``use_triton`` and cuBLAS otherwise. Call before
    ``Module._parallelize()``."""
    convert_to_flex_shard_float8_blockwise_linear(
        model,
        filter_fn=lambda fqn, _module: any(
            fnmatch.fnmatch(fqn, pattern)
            for pattern in BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS
        ),
        use_triton=use_triton,
        linear_cls=FlexShardBlockwiseFp8Linear,
    )
    if not blockwise_fp8_weights(model):
        raise ValueError(
            "parallelism.flex_shard_blockwise_fp8 found no linears matching "
            f"{BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS}."
        )


def blockwise_fp8_weights(model: nn.Module) -> set[nn.Parameter]:
    """Return the weights of the model's ``FlexShardBlockwiseFp8Linear`` modules."""
    return {
        module.weight
        for module in model.modules()
        if isinstance(module, FlexShardBlockwiseFp8Linear)
    }


def make_mixed_placement_fn(
    model: nn.Module, shard_placement_fn: PlacementFn, *, dense_reference: bool
) -> PlacementFn:
    """Place a bucket's FP8 linear weights and its other parameters in one
    ``MixedBucketPlacement``.

    ``shard_placement_fn`` gives the other parameters' placements, which must be
    ``Shard(0)``: a mixed bucket has no member for other shard dims. With
    ``dense_reference``, the FP8 weights all-gather bf16 and are quantized
    after the gather, with the same layout.
    """
    fp8_weights = blockwise_fp8_weights(model)
    make_fp8_placement_fn = (
        make_dense_all_gather_reference_placement_fn
        if dense_reference
        else make_fp8_bucketed_block_placement_fn
    )
    fp8_placement_fn = make_fp8_placement_fn(
        weight_factory=BlockwiseFp8Weight, fp8_dtype=_FP8_DTYPE
    )

    def placement_fn(named_params, mesh):
        mixed_bucket = MixedBucketPlacement({})
        fp8_params = [(fqn, param) for fqn, param in named_params if param in fp8_weights]
        shard_params = [
            (fqn, param) for fqn, param in named_params if param not in fp8_weights
        ]
        placements = {}
        for fqn, (placement,) in shard_placement_fn(shard_params, mesh).items():
            if placement.dim != 0:
                raise ValueError(
                    f"A blockwise FP8 mixed bucket shards its other parameters on "
                    f"dim 0, but {fqn!r} shards on dim {placement.dim}."
                )
            placements[fqn] = (mixed_bucket.shard0,)
        (fp8_placement,) = fp8_placement_fn(fp8_params, mesh)[fp8_params[0][0]]
        fp8_member = mixed_bucket.fp8_bucketed_block_shard(fp8_placement)
        placements.update((fqn, (fp8_member,)) for fqn, _ in fp8_params)
        return placements

    return placement_fn


def set_global_layouts(model: nn.Module) -> None:
    """Give each ``FlexShardBlockwiseFp8Linear`` its parameters' layouts, for
    initialization. Call once ``flex_shard`` has sharded the model."""
    layouts = get_flex_shard_global_layouts(model)
    for fqn, module in model.named_modules():
        if isinstance(module, FlexShardBlockwiseFp8Linear):
            module._global_layouts = {
                name: layouts[f"{fqn}.{name}"]
                for name, _ in module.named_parameters(recurse=False)
            }
