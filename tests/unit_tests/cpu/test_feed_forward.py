# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from torchtitan.models.common.activation import SiTUGLU
from torchtitan.models.common.config_utils import (
    fused_gate_up_param_init,
    make_routed_experts_config,
    split_gate_up_and_store_experts_2d,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear


def _fill(value: float) -> Callable[[torch.Tensor], None]:
    def init(tensor: torch.Tensor) -> None:
        torch.nn.init.constant_(tensor, value)

    return init


def test_feed_forward_uses_one_physical_gate_up_linear():
    config = FeedForward.Config(
        w13=Linear.Config(
            in_features=4,
            out_features=8,
            num_linears=2,
            param_init=fused_gate_up_param_init(
                {"weight": _fill(1.0)},
                {"weight": _fill(5.0)},
            ),
        ),
        w2=Linear.Config(
            in_features=8,
            out_features=4,
            bias=True,
            param_init={"weight": _fill(3.0), "bias": _fill(4.0)},
        ),
    )
    feed_forward = config.build()
    feed_forward.init_states()

    assert set(feed_forward._modules) == {"w13", "w2"}
    assert set(feed_forward.state_dict()) == {
        "w13.weight",
        "w2.weight",
        "w2.bias",
    }
    assert feed_forward.state_dict()["w13.weight"].data_ptr() == (
        feed_forward.w13.weight.data_ptr()
    )

    w13_2HD = feed_forward.w13.weight
    torch.testing.assert_close(w13_2HD[0], torch.ones_like(w13_2HD[0]))
    torch.testing.assert_close(w13_2HD[1], 5 * torch.ones_like(w13_2HD[1]))


def test_feed_forward_requires_two_w13_projections():
    config = FeedForward.Config(
        w13=Linear.Config(in_features=4, out_features=8),
        w2=Linear.Config(in_features=8, out_features=4),
    )

    with pytest.raises(ValueError, match="w13 requires num_linears=2"):
        config.build()


def test_split_gate_up_matches_fused_feed_forward():
    fused_config = FeedForward.Config(
        w13=Linear.Config(
            in_features=4,
            out_features=8,
            num_linears=2,
            param_init=fused_gate_up_param_init(
                {"weight": _fill(1.0)},
                {"weight": _fill(5.0)},
            ),
        ),
        w2=Linear.Config(
            in_features=8,
            out_features=4,
            param_init={"weight": _fill(3.0)},
        ),
    )
    split_config = copy.deepcopy(fused_config)
    split_gate_up_and_store_experts_2d(split_config)
    assert split_config.w13 is None
    fused = fused_config.build()
    fused.init_states()
    split = split_config.build()
    split.init_states()

    assert set(split._modules) == {"w1", "w3", "w2"}
    torch.testing.assert_close(split.w1.weight, fused.w13.weight[0])
    torch.testing.assert_close(split.w3.weight, fused.w13.weight[1])

    with torch.no_grad():
        fused.w13.weight.normal_()
        fused.w2.weight.normal_()
        split.w1.weight.copy_(fused.w13.weight[0])
        split.w3.weight.copy_(fused.w13.weight[1])
        split.w2.weight.copy_(fused.w2.weight)
    x = torch.randn(3, 4)
    torch.testing.assert_close(split(x), fused(x))


def test_feed_forward_takes_w13_or_w1_and_w3():
    def linear(out_features: int, num_linears: int = 1) -> Linear.Config:
        return Linear.Config(
            in_features=4, out_features=out_features, num_linears=num_linears
        )

    with pytest.raises(ValueError, match="w13, or w1 and w3, not both"):
        FeedForward.Config(
            w13=linear(8, 2), w1=linear(8), w3=linear(8), w2=linear(4)
        ).build()
    with pytest.raises(ValueError, match="requires w13, or w1 and w3"):
        FeedForward.Config(w1=linear(8), w2=linear(4)).build()


def test_dim0_layout_stores_routed_experts_2d():
    routed_experts = make_routed_experts_config(
        dim=4, hidden_dim=8, num_experts=3, top_k=1, param_init={}
    )
    split_gate_up_and_store_experts_2d(routed_experts)
    assert routed_experts.w13.rows_on_dim0
    assert routed_experts.w2.rows_on_dim0
    assert routed_experts.w13.init_seed != routed_experts.w2.init_seed


def test_feed_forward_loads_native_checkpoint_and_matches_reference():
    config = FeedForward.Config(
        w13=Linear.Config(in_features=4, out_features=8, num_linears=2),
        w2=Linear.Config(in_features=8, out_features=4),
    )
    feed_forward = config.build()
    w1_HD = torch.randn(8, 4)
    w3_HD = torch.randn(8, 4)
    state_dict = {
        "w13.weight": torch.stack((w1_HD, w3_HD)),
        "w2.weight": torch.randn(4, 8),
    }
    feed_forward.load_state_dict(state_dict)

    x_TD = torch.randn(3, 4, requires_grad=True)
    reference_x_TD = x_TD.detach().clone().requires_grad_()
    w1_HD = w1_HD.detach().clone().requires_grad_()
    w2_DH = state_dict["w2.weight"].detach().clone().requires_grad_()
    w3_HD = w3_HD.detach().clone().requires_grad_()
    expected_TD = F.linear(
        F.silu(F.linear(reference_x_TD, w1_HD)) * F.linear(reference_x_TD, w3_HD),
        w2_DH,
    )
    actual_TD = feed_forward(x_TD)
    torch.testing.assert_close(actual_TD, expected_TD)

    grad_TD = torch.randn_like(actual_TD)
    actual_TD.backward(grad_TD)
    expected_TD.backward(grad_TD)
    torch.testing.assert_close(x_TD.grad, reference_x_TD.grad)
    w13_grad_2HD = feed_forward.w13.weight.grad
    torch.testing.assert_close(w13_grad_2HD[0], w1_HD.grad)
    torch.testing.assert_close(w13_grad_2HD[1], w3_HD.grad)
    torch.testing.assert_close(feed_forward.w2.weight.grad, w2_DH.grad)


def test_feed_forward_uses_configured_activation():
    activation_fn = SiTUGLU.Config(beta=4.0, linear_beta=25.0)
    config = FeedForward.Config(
        w13=Linear.Config(in_features=4, out_features=8, num_linears=2),
        w2=Linear.Config(in_features=8, out_features=4),
        activation_fn=activation_fn,
    )
    feed_forward = config.build()
    feed_forward.load_state_dict(
        {
            "w13.weight": torch.randn(2, 8, 4),
            "w2.weight": torch.randn(4, 8),
        }
    )

    x_TD = torch.randn(3, 4)
    gate_up_T2F = feed_forward.w13(x_TD)
    expected_TD = feed_forward.w2(activation_fn.build()(gate_up_T2F))
    torch.testing.assert_close(feed_forward(x_TD), expected_TD)
