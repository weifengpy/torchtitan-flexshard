# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurable linear modules.

``Linear`` uses diamond inheritance (``nn.Linear`` + ``Module``) so that:
- The module hierarchy stays flat (no extra wrapper layer).
- Standard ``nn.Linear`` parameter and state-dict behavior is retained.
- The ``Module`` protocol is satisfied and ``build()`` is inherited
  from ``Configurable.Config``.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_group
from torchtitan.protocols.module import Module


class Linear(nn.Linear, Module):
    """Configurable linear that can store multiple stacked projections.

    A single projection keeps the standard ``[out_features, in_features]``
    parameter shape. Multiple projections use
    ``[num_linears, out_features, in_features]``, keeping each projection
    contiguous for blockwise weight quantization, and return
    ``[..., num_linears, out_features]``.

    The local projection is the remat region ``<fqn>.linear``, so every
    subclass, including quantized and LoRA ones that override ``_linear``,
    declares the same region. Tensor-parallel subclasses declare their
    collectives as sibling regions. Callers must not wrap a ``Linear`` call in
    another region: a saved outer region cannot contain a recomputed one.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        num_linears: int = 1
        bias: bool = False

    def __init__(self, config: Config):
        super().__init__(
            config.in_features,
            config.num_linears * config.out_features,
            bias=config.bias,
        )
        self.out_features = config.out_features
        self.num_linears = config.num_linears
        if config.num_linears > 1:
            self.weight = nn.Parameter(
                self.weight.detach().unflatten(
                    0, (config.num_linears, config.out_features)
                ),
                requires_grad=self.weight.requires_grad,
            )
            if self.bias is not None:
                self.bias = nn.Parameter(
                    self.bias.detach().unflatten(
                        0, (config.num_linears, config.out_features)
                    ),
                    requires_grad=self.bias.requires_grad,
                )

    def reset_parameters(self) -> None:
        # Flattening handles both ordinary and stacked projections while
        # keeping fan-in equal to in_features.
        nn.init.kaiming_uniform_(self.weight.flatten(0, -2), a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def _flatten_weight_and_bias(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Flatten stacked parameters for one linear operation."""
        weight = self.weight.flatten(0, -2)
        bias = None if self.bias is None else self.bias.flatten()
        return weight, bias

    def _unflatten_output(self, output: torch.Tensor) -> torch.Tensor:
        """Restore the logical stacked output dimensions after a linear operation."""
        if self.num_linears == 1:
            return output
        return output.unflatten(-1, self.weight.shape[:-1])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, bias = self._flatten_weight_and_bias()
        output = remat.region(
            self._linear,
            self.remat_region_name("linear"),
            recompute=self.remat_should_recompute("linear"),
        )(input, weight, bias)
        return self._unflatten_output(output)

    def extra_repr(self) -> str:
        result = nn.Linear.extra_repr(self)
        if self.num_linears > 1:
            result += f", num_linears={self.num_linears}"
        return result

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply local projection compute without outer communication.

        LoRA and quantized subclasses override this method so column- and
        row-parallel ``forward`` methods continue to own their collectives.
        Explicit operands let those boundaries adjust an operand's SPMD type
        before invoking the selected local compute implementation.
        """
        return F.linear(input, weight, bias)


def maybe_gather_tp_input(module: Module, x: torch.Tensor) -> torch.Tensor:
    """Redistribute a TP input to ``Replicate`` in the ``<fqn>.tp_gather`` region.

    ``<fqn>`` is ``module``'s remat name. This is an all-gather under sequence
    parallelism, and otherwise a forward no-op whose backward all-reduces.
    Modules whose projections share one input call this once at their common
    boundary. Returns ``x`` unchanged without TP.
    """
    tp_group = spmd_mesh_group(MeshAxisName.TP)
    if tp_group is None:
        return x
    return remat.region(
        spmd.redistribute,
        module.remat_region_name("tp_gather"),
        recompute=module.remat_should_recompute("tp_gather"),
    )(
        x,
        tp_group,
        src=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
        dst=spmd.R,
        backward_options={"op_dtype": x.dtype},
    )


class ColumnParallelLinear(Linear):
    """Prepare an input for a column-parallel Linear.

    This is a ``Linear`` rather than a wrapper around one, so its parameter
    FQNs remain unchanged. The same module handles both tensor-parallel modes.
    With sequence parallelism, ``Shard(0) -> Replicate`` is an input all-gather.
    Without sequence parallelism, ``Invariant -> Replicate`` is a forward no-op
    whose backward performs the required all-reduce.

    The redistribution is the remat region ``<fqn>.tp_gather``, a sibling of
    the projection region ``<fqn>.linear``. When the projection is saved and
    the redistribution is recomputed, the projection does not retain the
    gathered input for its weight gradient; replay re-gathers it from the
    sequence shard.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(maybe_gather_tp_input(self, input))


class SharedExpertRowParallelLinear(Linear):
    """Row-parallel shared-expert projection with a conditional reduction.

    With sequence parallelism, the output is reduce-scattered from Partial to
    Shard(0). Otherwise it remains Partial so the MoE can combine routed and
    shared partials before one all-reduce.

    Like ``RowParallelLinear``, the projection and the reduction are the
    separately controlled remat regions ``<fqn>.linear`` and
    ``<fqn>.tp_reduce``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        weight, bias = self._flatten_weight_and_bias()
        linear_fn = remat.region(
            self._linear,
            self.remat_region_name("linear"),
            recompute=self.remat_should_recompute("linear"),
        )
        if bias is not None and tp_group is not None:
            bias = spmd.convert(
                bias,
                tp_group,
                src=spmd.I,
                dst=spmd.P,
                expert_mode=True,
            )
            # The selected local compute may be native, LoRA, or quantized.
            # TODO: Remove this suppression once spmd_types recognizes the
            # rowwise F.linear type combination [V, V, P] -> P.
            with spmd.no_typecheck():
                output = self._unflatten_output(linear_fn(input, weight, bias))
            if spmd.is_type_checking():
                spmd.assert_local_type_like(
                    output,
                    input,
                    {tp_group: spmd.P},  # pyrefly: ignore [bad-argument-type]
                )
        else:
            output = self._unflatten_output(linear_fn(input, weight, bias))
        if tp_group is None or not spmd_dense_sp_enabled():
            return output
        # A recomputed reduction after a saved projection keeps the TP-times
        # larger partial output for replay; save both to avoid it.
        return remat.region(
            spmd.redistribute,
            self.remat_region_name("tp_reduce"),
            recompute=self.remat_should_recompute("tp_reduce"),
        )(
            output,
            tp_group,
            src=spmd.P,
            dst=spmd.S(0),
            backward_options={"op_dtype": output.dtype},
        )


class RowParallelLinear(Linear):
    """Reduce the partial output of an independently configured Linear.

    This is a ``Linear`` rather than a wrapper around one, so its parameter
    FQNs remain unchanged. ``Partial -> Shard(0)`` is a reduce-scatter, while
    ``Partial -> Invariant`` is an all-reduce without it. Dense SP state selects
    between the two. An invariant bias is converted to a partial contribution
    before local compute so the reduction adds it exactly once.

    The projection and the reduction are the separately controlled remat
    regions ``<fqn>.linear`` and ``<fqn>.tp_reduce``. Saving only the
    projection keeps its TP-times larger partial output for the replayed
    reduction; saving only the reduction skips the collective during replay.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group(MeshAxisName.TP)
        weight, bias = self._flatten_weight_and_bias()
        linear_fn = remat.region(
            self._linear,
            self.remat_region_name("linear"),
            recompute=self.remat_should_recompute("linear"),
        )
        if bias is not None and tp_group is not None:
            bias = spmd.convert(
                bias,
                tp_group,
                src=spmd.I,
                dst=spmd.P,
                expert_mode=True,
            )
            # The selected local compute may be native, LoRA, or quantized.
            # Its row-sharded operands and bias jointly produce a partial output.
            # TODO: Remove this suppression once spmd_types recognizes the
            # rowwise F.linear type combination [V, V, P] -> P.
            with spmd.no_typecheck():
                output = self._unflatten_output(linear_fn(input, weight, bias))
            if spmd.is_type_checking():
                spmd.assert_local_type_like(
                    output,
                    input,
                    {tp_group: spmd.P},  # pyrefly: ignore [bad-argument-type]
                )
        else:
            output = self._unflatten_output(linear_fn(input, weight, bias))
        if tp_group is None:
            return output

        # A recomputed reduction after a saved projection keeps the TP-times
        # larger partial output for replay; save both to avoid it.
        return remat.region(
            spmd.redistribute,
            self.remat_region_name("tp_reduce"),
            recompute=self.remat_should_recompute("tp_reduce"),
        )(
            output,
            tp_group,
            src=spmd.P,
            dst=spmd.S(0) if spmd_dense_sp_enabled() else spmd.I,
            backward_options={"op_dtype": output.dtype},
        )


class GroupedLinear(Module):
    """A collection of linears selected by cumulative group offsets.

    Like :class:`Linear`, ``num_linears`` retains a projection axis in parameter
    storage. For example, a fused gate/up projection stores ``[E, 2, F, D]``
    and returns ``[R, 2, F]`` while grouped GEMM consumes its zero-copy
    ``[E, 2F, D]`` view. With ``rows_on_dim0``, it stores the same rows as a
    2-D ``[E * 2F, D]`` weight instead.

    The grouped matmul is the remat region ``<fqn>.grouped_mm``, so every
    subclass, including quantized and LoRA ones that override ``_grouped_mm``,
    declares the same region.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Configure a grouped linear.

        Attributes:
            group_size: Number of independently selected linear weights.
            in_features: Input features for each linear.
            out_features: Output features for each linear.
            num_linears: Number of stacked linears per element in the group.
                Values greater than one retain a projection axis before
                ``out_features``.
            rows_on_dim0: Store the weight 2-D, every group's output rows
                stacked on dimension 0, so FSDP shards it on dimension 0, the
                only dimension FSDP2 lets it shard unevenly, e.g. 4 experts
                over 6 ranks. Forward views it as ``[E, num_linears * out, in]``.
            init_seed: With ``rows_on_dim0``, part of the seed each group's
                block is drawn from (see ``_init_param``), so that grouped
                linears draw differently. ``split_gate_up_and_store_experts_2d``
                gives each its own.
        """

        group_size: int
        in_features: int
        out_features: int
        num_linears: int = 1
        rows_on_dim0: bool = False
        init_seed: int = 0

    def __init__(self, config: Config):
        super().__init__()
        self.group_size = config.group_size
        self.in_features = config.in_features
        self.out_features = config.out_features
        self.num_linears = config.num_linears
        self.rows_on_dim0 = config.rows_on_dim0
        self.init_seed = config.init_seed
        # With rows_on_dim0, the index of this weight's first group among the
        # groups that draw from ``init_seed``, e.g. this expert-parallel rank's
        # first expert, so that every group draws the block it would draw with
        # all groups in one weight.
        self.group_offset = 0
        output_shape = self._output_shape
        self.weight = nn.Parameter(
            torch.empty(config.group_size * math.prod(output_shape), config.in_features)
            if config.rows_on_dim0
            else torch.empty(config.group_size, *output_shape, config.in_features)
        )

    @property
    def _output_shape(self) -> tuple[int, ...]:
        if self.num_linears == 1:
            return (self.out_features,)
        return (self.num_linears, self.out_features)

    def _init_param(self, name: str, param: nn.Parameter) -> None:
        """With ``rows_on_dim0``, initialize each group's ``[*output, in]``
        block through the ``[1, *output, in]`` view the initializers expect.

        A shard is a run of rows that may start or end inside a block, so each
        block a rank holds rows of is drawn on its own from a seed of its own,
        and the rank keeps its rows. The values don't depend on the sharding,
        and FSDP2 and FlexShard draw the same ones.
        """
        if not self.rows_on_dim0 or name not in (self._param_init or {}):
            super()._init_param(name, param)
            return
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._utils import (
            compute_local_shape_and_global_offset,
        )

        from torchtitan.distributed.flex_shard_fsdp import as_fsdp2_dtensor

        init = self._param_init[name]
        weight = as_fsdp2_dtensor(param)
        if isinstance(weight, DTensor):
            local = weight.to_local()
            _, global_offset = compute_local_shape_and_global_offset(
                weight.shape, weight.device_mesh, weight.placements
            )
            first_row = global_offset[0]
        else:
            local, first_row = weight, 0
        self._init_rows(local, first_row, init)

    def _init_rows(self, local: torch.Tensor, first_row: int, init: Callable) -> None:
        """Initialize ``local``, the rows of the 2-D weight from ``first_row``
        on (see ``_init_param``)."""
        if local.numel() == 0:
            return
        rows_per_group = math.prod(self._output_shape)
        end_row = first_row + local.shape[0]
        # The training seed, which every rank of a model replica shares.
        seed = torch.initial_seed() * 1_000_003 + self.init_seed * 4_099
        devices = [local.device.index] if local.device.type == "cuda" else []
        with torch.no_grad():
            for group in range(
                first_row // rows_per_group, (end_row - 1) // rows_per_group + 1
            ):
                block = torch.empty(
                    1,
                    *self._output_shape,
                    self.in_features,
                    dtype=local.dtype,
                    device=local.device,
                )
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed((seed + self.group_offset + group) % 2**62)
                    init(block)
                start = max(first_row, group * rows_per_group)
                end = min(end_row, (group + 1) * rows_per_group)
                block_start = group * rows_per_group
                local[start - first_row : end - first_row].copy_(
                    block.view(rows_per_group, self.in_features)[
                        start - block_start : end - block_start
                    ]
                )

    def grouped_view(self, weight: torch.Tensor) -> torch.Tensor:
        """Return ``weight``, this module's weight or a tensor of its layout, as
        ``[E, num_linears * out, in]``, a view in both layouts."""
        if self.rows_on_dim0:
            return weight.view(-1, math.prod(self._output_shape), self.in_features)
        return weight.flatten(1, -2)

    def forward(self, input_RI: torch.Tensor, offsets_E: torch.Tensor) -> torch.Tensor:
        """Apply each grouped linear to rows selected by cumulative offsets.

        Args:
            input_RI: Input rows grouped by the selected linear.
            offsets_E: Exclusive cumulative row end for each group.

        Returns:
            Output rows with an optional ``num_linears`` axis before the output
            feature axis.
        """
        output_shape = self._output_shape
        weight_EOI = self.grouped_view(self.weight)
        output_RO = remat.region(
            self._grouped_mm,
            self.remat_region_name("grouped_mm"),
            recompute=self.remat_should_recompute("grouped_mm"),
        )(input_RI=input_RI, weight_EOI=weight_EOI, offsets_E=offsets_E)
        return output_RO.reshape(*output_RO.shape[:-1], *output_shape)

    def _grouped_mm(
        self,
        *,
        input_RI: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        """Execute ``input_RI @ weight_EOI.transpose(-2, -1)`` by expert."""
        return torch._grouped_mm(
            input_RI,
            weight_EOI.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        )


__all__ = [
    "ColumnParallelLinear",
    "GroupedLinear",
    "Linear",
    "RowParallelLinear",
    "maybe_gather_tp_input",
]
