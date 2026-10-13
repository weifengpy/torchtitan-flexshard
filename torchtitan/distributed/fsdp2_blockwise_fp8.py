# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blockwise FP8 GEMMs on FSDP2's bf16 all-gather: the FSDP2 baseline for
FlexShard's blockwise FP8 all-gather.

With ``parallelism.fsdp2_blockwise_fp8``, the linears that
``BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS`` names run TorchAO's 128x128-block FP8
GEMMs on the bf16 weights FSDP2 all-gathers. As TorchAO's
``Float8BlockwiseLinear`` does, they quantize the weight on every call:
forward, the recomputed forward under activation checkpointing, and backward.
FSDP2's even ``Shard(0)`` split cuts 128-row blocks across ranks, so the
weights can't be quantized before the all-gather without communicating amax.

The quantization and GEMM kernels are FlexShard's (TorchAO's Triton
quantization, cuBLAS ``scaled_mm``), so the arms differ in the all-gather and
in how often the weight is quantized.
"""

import fnmatch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.prototype.blockwise_fp8_training.linear import (
    Float8BlockwiseLinear,
    fp8_blockwise_mm,
)

from torchtitan.distributed.blockwise_fp8_targets import (
    BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS,
)
from torchtitan.protocols.module import Module


class Fsdp2BlockwiseFp8Linear(Float8BlockwiseLinear, Module):
    """TorchAO's ``Float8BlockwiseLinear`` for stacked and unaligned weights.

    A stacked ``[*stack, out, in]`` weight runs one GEMM on its flattened
    ``[prod(stack) * out, in]`` matrix and returns ``[..., *stack, out]``, as
    ``torchtitan.models.common.linear.Linear`` does. TorchAO's quantization
    kernels need both matrix dims divisible by the block size, so the weight
    (and the input, for its in dim) is zero-padded to whole blocks, as
    FlexShard's FP8 linear pads, and the output sliced back.
    """

    @classmethod
    def from_float(cls, mod: nn.Linear) -> "Fsdp2BlockwiseFp8Linear":
        if getattr(mod, "_parallelized", False):
            raise RuntimeError(
                "Blockwise FP8 conversion must run before Module._parallelize(), "
                "so the replacement receives the original module's sharding."
            )
        if mod.bias is not None:
            raise ValueError("Fsdp2BlockwiseFp8Linear doesn't support a bias.")
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        # FSDP2 shards a stacked [N, F, D] weight on its rows, not on N, for the
        # linears with num_linears > 1 (linear_param_shard_placements).
        new_mod.num_linears = getattr(mod, "num_linears", 1)
        new_mod.train(mod.training)
        new_mod._param_init = getattr(mod, "_param_init", None)
        new_mod._sharding_config = getattr(mod, "_sharding_config", None)
        return new_mod

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # ``input``, as torchtitan's Linear names it: its sharding config is keyed
        # by the forward's argument names.
        x = input
        # [out] for a 2D weight, [*stack, out] for a stacked one.
        out_shape = self.weight.shape[:-1]
        weight = self.weight.flatten(0, -2)
        out_dim, in_dim = weight.shape
        pad_rows = -out_dim % self.block_size
        pad_cols = -in_dim % self.block_size
        if pad_rows or pad_cols:
            weight = F.pad(weight, (0, pad_cols, 0, pad_rows))
        if pad_cols:
            x = F.pad(x, (0, pad_cols))
        out = fp8_blockwise_mm.apply(
            x, weight, self.block_size, self.dtype, self.use_triton
        )
        if pad_rows:
            out = out[..., :out_dim]
        if len(out_shape) > 1:
            out = out.unflatten(-1, out_shape)
        return out


def convert_fsdp2_blockwise_fp8_linears(model: nn.Module) -> None:
    """Replace the linears ``BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS`` names with
    ``Fsdp2BlockwiseFp8Linear``. Call before ``Module._parallelize()``."""
    num_converted = 0

    def convert_children(parent: nn.Module, prefix: str = "") -> None:
        nonlocal num_converted
        for name, child in list(parent.named_children()):
            fqn = f"{prefix}.{name}" if prefix else name
            if isinstance(child, Fsdp2BlockwiseFp8Linear):
                continue
            if isinstance(child, nn.Linear) and any(
                fnmatch.fnmatch(fqn, pattern)
                for pattern in BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS
            ):
                setattr(parent, name, Fsdp2BlockwiseFp8Linear.from_float(child))
                num_converted += 1
            else:
                convert_children(child, fqn)

    convert_children(model)
    if num_converted == 0:
        raise ValueError(
            "parallelism.fsdp2_blockwise_fp8 found no linears matching "
            f"{BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS}."
        )


def num_fsdp2_blockwise_fp8_linears(model: nn.Module) -> int:
    return sum(isinstance(m, Fsdp2BlockwiseFp8Linear) for m in model.modules())

