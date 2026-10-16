# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blockwise FP8 GEMMs for routed experts, under either FSDP backend.

With ``parallelism.blockwise_fp8_routed_experts``, the routed experts' grouped
linears run flex_shard's ``blockwise_fp8_grouped_mm``: an FP8 forward and input
gradient with 1x128 activation and 128x128 weight blocks, on TorchAO's DeepGEMM
backend, and a BF16 weight gradient. Under FlexShard the experts' weights
all-gather as FP8 blocks (``flex_shard_blockwise_fp8``), which the GEMM takes as
gathered; under FSDP2 they all-gather in bf16 and the GEMM quantizes them on
every call. Both quantize the same blocks, so both compute the same bytes.
The routed experts' token dispatcher pads each expert's token group to a whole
128-row block, so the GEMM needs no padding of its own, whatever the routing.
flex_shard is imported only with this option.
"""

import math

import torch
import torch_remat as remat
from torch import nn

from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    TorchAOTokenDispatcher,
)

BLOCKWISE_FP8_GROUPED_LINEAR_MODULE_PATTERNS: tuple[str, ...] = (
    "layers.*.moe.routed_experts.w13",
    "layers.*.moe.routed_experts.w2",
)

# The FP8 grouped GEMM's block rows, to which the token dispatcher pads each
# expert's token group.
_TOKEN_GROUP_ALIGNMENT = 128


class BlockwiseFp8GroupedLinear(GroupedLinear):
    """A ``GroupedLinear`` whose grouped GEMM is blockwise FP8.

    ``groups_aligned``: every group's row count is a multiple of 128, as the
    padding token dispatcher makes them, so the GEMM skips padding of its own
    and reading the group sizes on the host.
    """

    groups_aligned: bool = False
    # Where each parameter's local shard sits in the full parameter, set by
    # flex_shard_blockwise_fp8.set_global_layouts once FlexShard has sharded
    # the experts' FP8 weights.
    _global_layouts = None

    def forward(self, input_RI: torch.Tensor, offsets_E: torch.Tensor) -> torch.Tensor:
        # pyrefly: ignore [missing-import]
        from flex_shard.blockwise_fp8 import BlockwiseFp8Weight

        weight = self.weight
        if isinstance(weight, BlockwiseFp8Weight):
            if weight.ndim != 2:
                raise ValueError(
                    "FP8-gathered routed experts need their weights stored 2-D "
                    "(GroupedLinear rows_on_dim0)."
                )
        else:
            weight = weight.reshape(-1, self.in_features)
        output_RO = remat.region(
            self._blockwise_fp8_grouped_mm,
            self.remat_region_name("grouped_mm"),
            recompute=self.remat_should_recompute("grouped_mm"),
        )(input_RI=input_RI, weight=weight, offsets_E=offsets_E)
        return output_RO.reshape(*output_RO.shape[:-1], *self._output_shape)

    def _blockwise_fp8_grouped_mm(
        self,
        *,
        input_RI: torch.Tensor,
        weight: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        # pyrefly: ignore [missing-import]
        from flex_shard.blockwise_fp8 import blockwise_fp8_grouped_mm

        return blockwise_fp8_grouped_mm(
            input_RI,
            weight,
            offsets_E,
            num_groups=offsets_E.numel(),
            groups_aligned=self.groups_aligned,
        )

    def _init_param(self, name: str, param: nn.Parameter) -> None:
        layout = (
            None if self._global_layouts is None else self._global_layouts.get(name)
        )
        if layout is None or name not in (self._param_init or {}):
            super()._init_param(name, param)
            return
        # FlexShard's FP8 placement keeps whole 128-row blocks, which no
        # DTensor placement describes. Its boxes are full-width row ranges of
        # the full weight: initialize each from the same per-group draws
        # FSDP2's shards get (GroupedLinear._init_rows).
        init = self._param_init[name]
        for global_offset, local_offset, size in zip(
            layout.global_offsets,
            layout.local_offsets,
            layout.local_sizes,
            strict=True,
        ):
            if global_offset[-1] != 0 or size[-1] != self.in_features:
                raise ValueError(
                    "FP8 routed experts need row-block shards, got a box of "
                    f"{size} at {global_offset}."
                )
            rows = math.prod(size[:-1])
            self._init_rows(
                param.data.view(-1, self.in_features)[
                    local_offset[0] : local_offset[0] + rows
                ],
                global_offset[0],
                init,
            )


def convert_blockwise_fp8_grouped_linears(model: nn.Module) -> None:
    """Turn the grouped linears ``BLOCKWISE_FP8_GROUPED_LINEAR_MODULE_PATTERNS``
    names into ``BlockwiseFp8GroupedLinear``, keeping their parameters and
    state, and make their routed experts' token dispatcher pad each expert's
    token group to 128 rows, so the GEMM's groups are whole blocks for any
    routing. Call before ``Module._parallelize()``."""
    import fnmatch

    converted = 0
    for fqn, module in model.named_modules():
        if type(module) is GroupedLinear and any(
            fnmatch.fnmatch(fqn, pattern)
            for pattern in BLOCKWISE_FP8_GROUPED_LINEAR_MODULE_PATTERNS
        ):
            module.__class__ = BlockwiseFp8GroupedLinear
            module.groups_aligned = True
            converted += 1
    if not converted:
        raise ValueError(
            "parallelism.blockwise_fp8_routed_experts found no grouped linears "
            f"matching {BLOCKWISE_FP8_GROUPED_LINEAR_MODULE_PATTERNS}."
        )
    for module in model.modules():
        if isinstance(module, RoutedExperts) and isinstance(
            module.w13, BlockwiseFp8GroupedLinear
        ):
            _pad_token_groups(module.token_dispatcher)


def _pad_token_groups(dispatcher: nn.Module) -> None:
    """Make an all-to-all token dispatcher pad each local expert's token group
    to ``_TOKEN_GROUP_ALIGNMENT`` rows: ``TorchAOTokenDispatcher`` is the
    all-to-all dispatcher plus that padding, done without host syncs."""
    if isinstance(dispatcher, TorchAOTokenDispatcher):
        if dispatcher.pad_multiple % _TOKEN_GROUP_ALIGNMENT:
            raise ValueError(
                "FP8 routed experts need token groups padded to a multiple of "
                f"{_TOKEN_GROUP_ALIGNMENT} rows, but the dispatcher pads to "
                f"{dispatcher.pad_multiple}."
            )
        return
    if type(dispatcher) is not AllToAllTokenDispatcher:
        raise ValueError(
            "FP8 routed experts need the all-to-all token dispatcher, got "
            f"{type(dispatcher).__name__}."
        )
    dispatcher.__class__ = TorchAOTokenDispatcher
    dispatcher.pad_multiple = _TOKEN_GROUP_ALIGNMENT
