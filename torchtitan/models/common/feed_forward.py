# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch_remat as remat

from torchtitan.models.common.activation import BinaryActivationFn, SwiGLU
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module

# Shape suffix legend:
#   T = token dimensions, D = model dimension, F = feed-forward hidden dimension

__all__ = ["FeedForward", "compute_ffn_hidden_dim"]


def compute_ffn_hidden_dim(
    dim: int,
    *,
    multiple_of: int = 1,
    ffn_dim_multiplier: float | None = None,
) -> int:
    """Compute the SwiGLU hidden dimension for Llama3/4-style models.

    This applies the 2/3 scaling, optional multiplier, and rounds up to multiple_of.
    """
    hidden_dim = int(2 * 4 * dim / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class FeedForward(Module):
    """SwiGLU feed-forward with one physical gate-and-up projection.

    ``w13`` stores the gate and up projections as ``[2, F, D]``. Config takes the
    **final** hidden_dim (no internal 2/3 scaling). Use
    compute_ffn_hidden_dim() for Llama3/4-style dim computation.

    Given ``w1`` and ``w3`` instead of ``w13``, the gate and up projections are
    separate ``[F, D]`` linears, so FSDP can shard them on dimension 0, the only
    dimension FSDP2 lets it shard unevenly (``split_gate_up_and_store_experts_2d``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w13: Linear.Config | None = None
        w2: Linear.Config
        w1: Linear.Config | None = None
        w3: Linear.Config | None = None
        activation_fn: BinaryActivationFn.Config = field(default_factory=SwiGLU.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.split_gate_up = config.w13 is None
        if self.split_gate_up:
            if config.w1 is None or config.w3 is None:
                raise ValueError("FeedForward requires w13, or w1 and w3")
            if config.w1.num_linears != 1 or config.w3.num_linears != 1:
                raise ValueError("FeedForward w1 and w3 require num_linears=1")
            self.w1 = config.w1.build()
            self.w3 = config.w3.build()
        else:
            if config.w1 is not None or config.w3 is not None:
                raise ValueError("FeedForward takes w13, or w1 and w3, not both")
            if config.w13.num_linears != 2:
                raise ValueError(
                    "FeedForward w13 requires num_linears=2 for gate/up projections"
                )
            self.w13 = config.w13.build()
        self.w2 = config.w2.build()
        self.activation_fn = config.activation_fn.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # w13 (or w1 and w3) and w2 declare their own remat regions.
        if self.split_gate_up:
            gate_up_T2F = torch.stack((self.w1(x), self.w3(x)), dim=-2)
        else:
            gate_up_T2F = self.w13(x)
        remat.recompute_needs_tensor(gate_up_T2F)
        return self.w2(self.activation_fn(gate_up_T2F))
