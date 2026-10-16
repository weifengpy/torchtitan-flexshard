# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The linears that run blockwise FP8 GEMMs, shared by the FlexShard and FSDP2
blockwise FP8 options so both convert the same modules."""

# DeepSeek V3's attention and dense feed-forward linears, and its shared experts.
# Llama 3 shares the ``attention.wo`` and ``feed_forward`` names and adds its fused
# ``attention.qkv_linear.wqkv``. ``w13`` stacks the gate and up projections as one
# ``[2, F, D]`` weight; ``split_gate_up_and_store_experts_2d`` splits it into w1
# and w3.
BLOCKWISE_FP8_LINEAR_MODULE_PATTERNS: tuple[str, ...] = (
    "layers.*.attention.wq_a",
    "layers.*.attention.wq_b",
    "layers.*.attention.wkv_a",
    "layers.*.attention.wkv_b",
    "layers.*.attention.qkv_linear.wqkv",
    "layers.*.attention.wo",
    "layers.*.feed_forward.w13",
    "layers.*.feed_forward.w1",
    "layers.*.feed_forward.w3",
    "layers.*.feed_forward.w2",
    "layers.*.moe.shared_experts.w13",
    "layers.*.moe.shared_experts.w1",
    "layers.*.moe.shared_experts.w3",
    "layers.*.moe.shared_experts.w2",
)
