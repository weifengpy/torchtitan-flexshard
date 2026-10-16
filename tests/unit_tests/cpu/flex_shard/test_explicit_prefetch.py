# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.distributed.flex_shard_fsdp import _set_explicit_prefetch


class _Storage:
    """Records the buckets a bucket storage is told to prefetch, by name."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.forward: list[str] | None = None
        self.backward: list[str] | None = None

    def set_buckets_to_forward_prefetch(self, buckets: list["_Storage"]) -> None:
        self.forward = [bucket.name for bucket in buckets]

    def set_buckets_to_backward_prefetch(self, buckets: list["_Storage"]) -> None:
        self.backward = [bucket.name for bucket in buckets]


@pytest.mark.parametrize("backward_experts_first", [False, True])
def test_explicit_prefetch_backward_order(backward_experts_first: bool) -> None:
    names = ["embedding", "dense0", "experts0", "dense1", "experts1", "head"]
    storages = [_Storage(name) for name in names]
    _set_explicit_prefetch(
        storages,
        embedding_bucket=0,
        block_buckets=[[1, 2], [3, 4]],
        head_bucket=5,
        expert_buckets=[2, 4],
        backward_experts_first=backward_experts_first,
    )
    # Backward: the block's buckets, the expert bucket first if asked.
    if backward_experts_first:
        assert storages[5].backward == ["experts1", "dense1"]
        assert storages[3].backward == ["experts0", "dense0"]
    else:
        assert storages[5].backward == ["dense1", "experts1"]
        assert storages[3].backward == ["dense0", "experts0"]
    assert storages[1].backward == ["embedding"]
    # Forward keeps the block's bucket order either way.
    assert storages[0].forward == ["dense0", "experts0"]
    assert storages[1].forward == ["dense1", "experts1"]
    assert storages[3].forward == ["head"]
    # Expert buckets prefetch nothing.
    assert storages[2].forward == [] and storages[2].backward == []
    assert storages[4].forward == [] and storages[4].backward == []
