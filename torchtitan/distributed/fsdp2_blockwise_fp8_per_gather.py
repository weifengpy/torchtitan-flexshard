# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP2 with blockwise FP8 weights quantized once per all-gather: the strongest
FSDP2 baseline for FlexShard's FP8 all-gather where FSDP2's even split cuts
128-row blocks.

With ``parallelism.fsdp2_blockwise_fp8_quantize_per_gather``, each FP8 linear's
weight is a ``Fsdp2BlockwiseFp8GatherWeight``. Its FSDP2 all-gather extension
sends the bf16 shard (a rank can't quantize its rows alone when they split
128-row blocks), quantizes the all-gathered weight once into a
``BlockwiseFp8Weight`` in the post-all-gather hook, and has FSDP2 free the bf16
all-gather buffers right after. The forward, the recomputed forward and the
input gradient of one all-gather share that FP8 weight: 128x128 blocks have the
same scales in either layout. The linears are FlexShard's FP8 linears, so both
backends run the same GEMMs. Imported only with this option, as FlexShard is an
optional dependency.
"""

import fnmatch
import math
from typing import Any

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
import triton

# pyrefly: ignore [missing-import]
from flex_shard.blockwise_fp8 import BlockwiseFp8Weight
from torch.utils._python_dispatch import return_and_correct_aliasing
from torchao.prototype.blockwise_fp8_training import kernels as torchao_kernels

from torchtitan.distributed.blockwise_fp8_targets import (
    BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS,
)
from torchtitan.distributed.flex_shard_blockwise_fp8 import (
    FlexShardBlockwiseFp8Linear,
)

_BLOCK_SIZE = 128
_FP8_DTYPE = torch.float8_e4m3fn

# What FSDP2 runs to shard, pad, copy and materialize a parameter keeps the
# wrapper, as for TorchAO's float8 FSDP2 weight wrapper.
_OPS_TO_PRESERVE_SUBCLASS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}


class _GatheredBlockwiseFp8Weight(BlockwiseFp8Weight):
    """The all-gathered FP8 weight FSDP2 installs on the module. FSDP2 views it
    with ``torch.as_strided`` at its own contiguous size and wraps it in a
    parameter (``detach``), and both have to keep the type."""

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten.detach.default:
            tensor = args[0]
            return cls(
                tensor._data.detach(),
                tensor._recip_scale.detach(),
                tensor._block_size,
                tensor._orig_dtype,
                requires_grad=False,
            )
        if func == torch.ops.aten.alias.default or (
            func == torch.ops.aten.as_strided.default
            and _is_identity_as_strided(*args, **kwargs)
        ):
            tensor = args[0]
            alias = torch.ops.aten.alias.default
            result = cls(
                alias(tensor._data),
                alias(tensor._recip_scale),
                tensor._block_size,
                tensor._orig_dtype,
                requires_grad=False,
            )
            return return_and_correct_aliasing(func, args, kwargs, result)
        return super().__torch_dispatch__(func, types, args, kwargs)


def _is_identity_as_strided(tensor, size, stride, storage_offset=None) -> bool:
    return (
        tuple(size) == tuple(tensor.shape)
        and tuple(stride) == tuple(tensor.stride())
        and (storage_offset or 0) == tensor.storage_offset()
    )


def _quantize_into(
    weight: torch.Tensor, data_t: torch.Tensor, scale_t: torch.Tensor
) -> None:
    """TorchAO's transposed-RHS weight quantization kernel, writing into ``data_t``
    and ``scale_t``: the column-major layouts its wrapper allocates, i.e. the
    row-major weight and scales seen through ``.t()``."""
    rows, cols = weight.shape
    grid = (triton.cdiv(rows, _BLOCK_SIZE), triton.cdiv(cols, _BLOCK_SIZE))
    torchao_kernels.triton_fp8_blockwise_weight_quant_transposed_rhs_kernel[grid](
        weight,
        weight.stride(0),
        weight.stride(1),
        data_t,
        data_t.stride(0),
        data_t.stride(1),
        scale_t,
        scale_t.stride(0),
        scale_t.stride(1),
        rows,
        cols,
        BLOCK_SIZE=_BLOCK_SIZE,
        EPS=torchao_kernels.EPS,
        FP8_MAX=torch.finfo(_FP8_DTYPE).max,
    )


class Fsdp2BlockwiseFp8GatherWeight(torch.Tensor):
    """An FP8 linear's sharded weight under FSDP2: all-gathers the bf16 shard and
    quantizes the gathered weight once, into the module's ``BlockwiseFp8Weight``.

    Wraps the high-precision shard (``_tensor``) for every other op, as TorchAO's
    ``WeightWithDynamicFloat8CastTensor`` does.
    """

    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == torch.ops.aten.detach.default:
            return cls(args[0]._tensor)
        args, kwargs = pytree.tree_map_only(
            cls, lambda t: t._tensor, (args, kwargs or {})
        )
        out = func(*args, **kwargs)
        if func not in _OPS_TO_PRESERVE_SUBCLASS:
            return out
        return pytree.tree_map_only(torch.Tensor, cls, out)

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __tensor_flatten__(self):
        return ["_tensor"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        _ = flatten_spec, outer_size, outer_stride
        return Fsdp2BlockwiseFp8GatherWeight(inner_tensors["_tensor"])

    def __repr__(self) -> str:
        return f"Fsdp2BlockwiseFp8GatherWeight({self._tensor})"

    def fsdp_pre_all_gather(
        self,
        mesh: Any,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
        module: nn.Module,
        mp_policy: Any,
    ) -> tuple[tuple[torch.Tensor, ...], Any]:
        _ = mesh, outer_stride, module
        # The shard's rows split 128-row blocks, so it can't be quantized alone.
        return (self._tensor.to(mp_policy.param_dtype),), (tuple(outer_size),)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ):
        (data,) = all_gather_outputs
        (outer_size,) = metadata
        rows, cols = math.prod(outer_size[:-1]), outer_size[-1]
        # FSDP2 reassembles a Shard(i > 0) weight before this hook, so the
        # gathered data is the weight in its own row order.
        if data.numel() != rows * cols:
            # TODO: crop FSDP2's padding for weights it splits unevenly.
            raise ValueError(
                "fsdp2_blockwise_fp8_quantize_per_gather needs FSDP2 to split each "
                f"FP8 weight evenly; {tuple(outer_size)} was padded to "
                f"{data.numel()} elements."
            )
        matrix = data.view(rows, cols)
        if out is not None:
            # From the second all-gather on, FSDP2 has re-allocated the FP8 data
            # and scales of the weight it installed; quantize into them.
            _quantize_into(
                matrix,
                out._data.view(rows, cols).t(),
                out._recip_scale.t(),
            )
            return
        data_t, scale_t = (
            torchao_kernels.triton_fp8_blockwise_weight_quant_transposed_rhs(
                matrix, _BLOCK_SIZE, _FP8_DTYPE
            )
        )
        fp8_data, recip_scale = data_t.t(), scale_t.t()
        weight = _GatheredBlockwiseFp8Weight(
            fp8_data.view(outer_size),
            recip_scale,
            _BLOCK_SIZE,
            param_dtype,
            requires_grad=False,
        )
        return weight, (fp8_data, recip_scale)

    def fsdp_should_release_all_gather_outputs_after_post_all_gather(self) -> bool:
        return True


def convert_fsdp2_blockwise_fp8_linears_quantize_per_gather(model: nn.Module) -> None:
    """Replace the linears ``BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS`` names with
    FlexShard's FP8 linear on a ``Fsdp2BlockwiseFp8GatherWeight``. Call before
    ``Module._parallelize()`` and FSDP2."""
    num_converted = 0

    def convert_children(parent: nn.Module, prefix: str = "") -> None:
        nonlocal num_converted
        for name, child in list(parent.named_children()):
            fqn = f"{prefix}.{name}" if prefix else name
            if isinstance(child, FlexShardBlockwiseFp8Linear):
                continue
            if isinstance(child, nn.Linear) and any(
                fnmatch.fnmatch(fqn, pattern)
                for pattern in BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS
            ):
                if child.bias is not None:
                    raise ValueError(
                        f"{fqn}: FP8 linears with a bias aren't supported."
                    )
                rows = math.prod(child.weight.shape[:-1])
                cols = child.weight.shape[-1]
                if rows % _BLOCK_SIZE or cols % _BLOCK_SIZE:
                    raise ValueError(
                        f"{fqn}: fsdp2_blockwise_fp8_quantize_per_gather needs whole "
                        f"{_BLOCK_SIZE}-blocks, got a [{rows}, {cols}] weight matrix."
                    )
                new_mod = FlexShardBlockwiseFp8Linear.from_float(child)
                # FSDP2 shards a stacked [N, F, D] weight on its rows, not on N,
                # for the linears with num_linears > 1 (linear_param_shard_placements).
                new_mod.num_linears = getattr(child, "num_linears", 1)
                new_mod.weight = nn.Parameter(
                    Fsdp2BlockwiseFp8GatherWeight(child.weight.data),
                    requires_grad=child.weight.requires_grad,
                )
                setattr(parent, name, new_mod)
                num_converted += 1
            else:
                convert_children(child, fqn)

    convert_children(model)
    if num_converted == 0:
        raise ValueError(
            "parallelism.fsdp2_blockwise_fp8_quantize_per_gather found no linears "
            f"matching {BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS}."
        )
