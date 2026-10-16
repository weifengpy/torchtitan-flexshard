# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torch import nn

from torchtitan.distributed.blockwise_fp8_grouped import (
    BlockwiseFp8GroupedLinear,
    convert_blockwise_fp8_grouped_linears,
)
from torchtitan.models.common.config_utils import (
    make_routed_experts_config,
    split_gate_up_and_store_experts_2d,
)
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.token_dispatcher import TorchAOTokenDispatcher


def _grouped_linear(num_linears: int) -> GroupedLinear:
    return GroupedLinear.Config(
        group_size=2,
        in_features=256,
        out_features=128,
        num_linears=num_linears,
        rows_on_dim0=True,
    ).build()


def _model() -> nn.Module:
    model = nn.Module()
    model.layers = nn.ModuleDict({"3": nn.Module()})
    model.layers["3"].moe = nn.Module()
    model.layers["3"].moe.routed_experts = nn.Module()
    model.layers["3"].moe.routed_experts.w13 = _grouped_linear(2)
    model.layers["3"].moe.routed_experts.w2 = _grouped_linear(1)
    model.other = _grouped_linear(1)
    return model


def test_converts_routed_expert_grouped_linears_in_place():
    model = _model()
    experts = model.layers["3"].moe.routed_experts
    w13_weight = experts.w13.weight
    convert_blockwise_fp8_grouped_linears(model)

    for name in ("w13", "w2"):
        module = getattr(experts, name)
        assert type(module) is BlockwiseFp8GroupedLinear
        assert module.groups_aligned
        assert module.rows_on_dim0
    assert experts.w13.weight is w13_weight
    assert type(model.other) is GroupedLinear


def test_conversion_without_routed_experts_raises():
    with pytest.raises(ValueError, match="found no grouped linears"):
        convert_blockwise_fp8_grouped_linears(nn.Module())


def test_routed_experts_dispatcher_pads_token_groups():
    """Their token dispatcher pads each expert's group to 128 rows, so the FP8
    GEMM's groups are whole blocks whatever the routing."""
    config = make_routed_experts_config(
        dim=256, hidden_dim=128, num_experts=4, top_k=2, param_init={}
    )
    split_gate_up_and_store_experts_2d(config)
    model = nn.Module()
    model.layers = nn.ModuleDict({"3": nn.Module()})
    model.layers["3"].moe = nn.Module()
    model.layers["3"].moe.routed_experts = config.build()
    convert_blockwise_fp8_grouped_linears(model)

    experts = model.layers["3"].moe.routed_experts
    assert type(experts.token_dispatcher) is TorchAOTokenDispatcher
    assert experts.token_dispatcher.pad_multiple == 128
    assert experts.w13.groups_aligned and experts.w2.groups_aligned
