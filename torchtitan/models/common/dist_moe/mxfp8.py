# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 prepared-weight lifecycle for Dist-MoE grouped projections."""

from __future__ import annotations

import functools
import math
import sys
from dataclasses import dataclass, field

import torch

from torchtitan.quantization._fsdp_tensor import (
    _ShardedFSDPTensor,
    _UnshardedFSDPTensor,
)

from . import _dist_moe as dist_moe
from .routed_experts import DistMoeRoutedExperts


__all__ = ["MXFP8DistMoeRoutedExperts"]


@dataclass(frozen=True, slots=True)
class _DistMoeMXFP8Operands:
    """Independent MXFP8 tensors owned by one FSDP unshard lifetime."""

    qdata: torch.Tensor
    fprop_scale: torch.Tensor
    dgrad_scale: torch.Tensor

    def prepared(self, source: torch.Tensor) -> dist_moe.PreparedWeight:
        """Return the annex facade consumed by one Dist-MoE invocation."""
        return dist_moe.PreparedWeight._create(
            source=source,
            format=dist_moe.BlockScaledFormat.MXFP8_E4M3,
            fprop_data=self.qdata,
            fprop_scale=self.fprop_scale,
            dgrad_data=self.qdata,
            dgrad_scale=self.dgrad_scale,
        )


def _prepare_mxfp8_weight(
    weight_EOI: torch.Tensor,
    out: _DistMoeMXFP8Operands | None = None,
) -> _DistMoeMXFP8Operands:
    """Allocate or refill the annex's grouped 32x32 MXFP8 weight operands."""
    prepared_out = None if out is None else out.prepared(weight_EOI)
    prepared = dist_moe.prepare_block_scaled_weight(
        weight_EOI,
        dist_moe.BlockScaledConfig(),
        out=prepared_out,
    )
    if prepared.dgrad_data is not prepared.fprop_data:
        raise RuntimeError("MXFP8 Dist-MoE FPROP and DGRAD must share qdata")
    if prepared.dgrad_scale is None:
        raise RuntimeError("MXFP8 Dist-MoE preparation returned incomplete operands")
    return _DistMoeMXFP8Operands(
        qdata=prepared.fprop_data,
        fprop_scale=prepared.fprop_scale,
        dgrad_scale=prepared.dgrad_scale,
    )


def _grouped_compute_view(
    tensor: torch.Tensor, rows_per_group: int | None, *, gate_up: bool
) -> torch.Tensor:
    """Return ``tensor``, a W13 ``[E, 2, F, D]`` or W2 ``[E, D, F]`` weight or
    their 2-D storage with ``rows_per_group`` rows per expert, as the grouped
    ``[E, out, in]`` view Dist-MoE computes with."""
    if rows_per_group is not None:
        return tensor.view(-1, rows_per_group, tensor.shape[-1])
    return tensor.flatten(1, 2) if gate_up else tensor


class _DistMoeW13ShardedTensor(_ShardedFSDPTensor):
    """Persistent W13 parameter that prepares grouped MXFP8 compute operands."""

    # Rows per expert of a weight stored 2-D (GroupedLinear rows_on_dim0), set by
    # the subclass _rows_on_dim0_type creates; None for the stacked layout.
    _rows_per_group: int | None = None

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeMXFP8Operands | None = None,
    ) -> _DistMoeMXFP8Operands:
        return _prepare_mxfp8_weight(
            _grouped_compute_view(logical_tensor, self._rows_per_group, gate_up=True),
            out,
        )


class _DistMoeW2ShardedTensor(_ShardedFSDPTensor):
    """Persistent W2 parameter that prepares grouped MXFP8 compute operands."""

    _rows_per_group: int | None = None

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeMXFP8Operands | None = None,
    ) -> _DistMoeMXFP8Operands:
        return _prepare_mxfp8_weight(
            _grouped_compute_view(logical_tensor, self._rows_per_group, gate_up=False),
            out,
        )


@functools.cache
def _rows_on_dim0_type(base: type, rows_per_group: int) -> type:
    """``base`` for a weight stored 2-D with ``rows_per_group`` rows per expert.

    The row count rides on the type, not the instance, because FSDP re-wraps
    each shard with ``type(template)``. The type is a module attribute, so
    pickling finds it by name.
    """
    name = f"{base.__name__}Rows{rows_per_group}"
    cls = type(
        name, (base,), {"_rows_per_group": rows_per_group, "__module__": __name__}
    )
    setattr(sys.modules[__name__], name, cls)
    return cls


def _dynamic_prepared_weight(
    logical_weight: torch.Tensor,
    *,
    gate_up: bool,
    rows_per_group: int | None = None,
) -> dist_moe.PreparedWeight:
    """Prepare a weight when FSDP does not own its unshard lifetime."""
    source = _grouped_compute_view(logical_weight, rows_per_group, gate_up=gate_up)
    storage = (
        logical_weight._tensor
        if isinstance(logical_weight, _ShardedFSDPTensor)
        else logical_weight
    )
    compute_weight = _grouped_compute_view(storage, rows_per_group, gate_up=gate_up)
    with torch.no_grad():
        operands = _prepare_mxfp8_weight(compute_weight)
    return operands.prepared(source)


class MXFP8DistMoeRoutedExperts(DistMoeRoutedExperts):
    """Dist-MoE routed experts using native asynchronous MXFP8 kernels.

    The module preserves the inherited W13/W2 parameter and checkpoint layout,
    while TorchTitan's shared FSDP tensor lifecycle prepares the qdata and scale
    layouts required by Dist-MoE for each unshard lifetime. Without FSDP, the
    same layouts are prepared directly from the live parameters before use.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DistMoeRoutedExperts.Config):
        """Configure native MXFP8 execution for Dist-MoE routed experts.

        Args:
            block_scaled_config: Annex MXFP8 kernel policy. Its ``pipeline``
                selects separate staged kernels or the fused Mega pipeline;
                ``fast_math`` enables approximate sigmoid math in fused SwiGLU;
                and ``kernel_config`` is an expert-only CuTe tuning override.
        """

        block_scaled_config: dist_moe.BlockScaledConfig = field(
            default_factory=lambda: dist_moe.BlockScaledConfig()
        )

        def __post_init__(self) -> None:
            """Validate the common expert and MXFP8-specific policies."""
            DistMoeRoutedExperts.Config.__post_init__(self)
            if (
                self.block_scaled_config.format
                is not dist_moe.BlockScaledFormat.MXFP8_E4M3
            ):
                raise ValueError(
                    "MXFP8DistMoeRoutedExperts requires the MXFP8_E4M3 format"
                )
            if self.bf16_grouped_gemm_preset is not None:
                raise ValueError(
                    "bf16_grouped_gemm_preset is not used by MXFP8 Dist-MoE"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.block_scaled_config = config.block_scaled_config
        w13_type, w2_type = _DistMoeW13ShardedTensor, _DistMoeW2ShardedTensor
        if self.w13.rows_on_dim0:
            w13_type = _rows_on_dim0_type(w13_type, self._rows_per_group(self.w13))
        if self.w2.rows_on_dim0:
            w2_type = _rows_on_dim0_type(w2_type, self._rows_per_group(self.w2))
        self.w13.weight = torch.nn.Parameter(
            w13_type(self.w13.weight.data),
            requires_grad=self.w13.weight.requires_grad,
        )
        self.w2.weight = torch.nn.Parameter(
            w2_type(self.w2.weight.data),
            requires_grad=self.w2.weight.requires_grad,
        )

    @staticmethod
    def _rows_per_group(linear) -> int | None:
        """Rows per expert of ``linear``'s weight stored 2-D, else None."""
        return math.prod(linear._output_shape) if linear.rows_on_dim0 else None

    def _weight_operands(
        self,
    ) -> tuple[
        torch.Tensor | dist_moe.PreparedWeight,
        torch.Tensor | dist_moe.PreparedWeight,
    ]:
        """Return prepared W13 and W2 operands for this unshard lifetime."""
        w13_E2FD = self.w13.weight
        w2_EDF = self.w2.weight
        w13_rows = self._rows_per_group(self.w13)
        w2_rows = self._rows_per_group(self.w2)
        w13_EFD = _grouped_compute_view(w13_E2FD, w13_rows, gate_up=True)
        w2_source = _grouped_compute_view(w2_EDF, w2_rows, gate_up=False)
        w13_operand = (
            w13_E2FD.operands.prepared(w13_EFD)
            if isinstance(w13_E2FD, _UnshardedFSDPTensor)
            else _dynamic_prepared_weight(
                w13_E2FD, gate_up=True, rows_per_group=w13_rows
            )
        )
        w2_operand = (
            w2_EDF.operands.prepared(w2_source)
            if isinstance(w2_EDF, _UnshardedFSDPTensor)
            else _dynamic_prepared_weight(w2_EDF, gate_up=False, rows_per_group=w2_rows)
        )
        return w13_operand, w2_operand
