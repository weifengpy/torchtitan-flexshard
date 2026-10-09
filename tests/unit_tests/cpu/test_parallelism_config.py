# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.config.parallelism import FSDPSymmMemScope, ParallelismConfig
from torchtitan.distributed.context_parallel import PTRRFlexAttentionCPLoadBalancer


def test_parallelism_config_default_load_balancer() -> None:
    assert ParallelismConfig().context_parallel_load_balancer is None


def test_parallelism_config_accepts_ptrr_when_cp_disabled() -> None:
    config = ParallelismConfig(
        context_parallel_degree=1,
        context_parallel_load_balancer=(PTRRFlexAttentionCPLoadBalancer.Config()),
    )
    assert isinstance(
        config.context_parallel_load_balancer,
        PTRRFlexAttentionCPLoadBalancer.Config,
    )


def test_parallelism_config_default_schedule() -> None:
    assert ParallelismConfig().pipeline_parallel_schedule == "1F1B"


def test_parallelism_config_accepts_interleaved_1f1b() -> None:
    config = ParallelismConfig(pipeline_parallel_schedule="Interleaved1F1B")
    assert config.pipeline_parallel_schedule == "Interleaved1F1B"


def test_parallelism_config_accepts_pipeline_schedule_multi() -> None:
    config = ParallelismConfig(pipeline_parallel_schedule="PipelineScheduleMulti")
    assert config.pipeline_parallel_schedule == "PipelineScheduleMulti"


@pytest.mark.parametrize("schedule", ["foo", "Interleved1F1B", ""])
def test_parallelism_config_rejects_invalid_schedule(schedule: str) -> None:
    with pytest.raises(
        ValueError,
        match=rf"pipeline_parallel_schedule {schedule!r}",
    ):
        ParallelismConfig(pipeline_parallel_schedule=schedule)


def test_parallelism_config_rejects_unknown_schedule_when_pp_disabled() -> None:
    with pytest.raises(ValueError, match=r"pipeline_parallel_schedule 'foo'"):
        ParallelismConfig(
            pipeline_parallel_degree=1,
            pipeline_parallel_schedule="foo",
        )


def test_parallelism_config_disables_fsdp_symm_mem_by_default() -> None:
    assert ParallelismConfig().fsdp_symm_mem_scope is None


@pytest.mark.parametrize("scope", ["all", "dense"])
def test_parallelism_config_accepts_fsdp_symm_mem_scopes(
    scope: FSDPSymmMemScope, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda: (9, 0))

    config = ParallelismConfig(fsdp_symm_mem_scope=scope)

    assert config.fsdp_symm_mem_scope == scope


def test_parallelism_config_rejects_unknown_fsdp_symm_mem_scope() -> None:
    with pytest.raises(
        ValueError,
        match=r"fsdp_symm_mem_scope must be one of: .* \(got 'sparse'\)",
    ):
        ParallelismConfig(
            fsdp_symm_mem_scope="sparse"  # pyrefly: ignore [bad-argument-type]
        )


def test_parallelism_config_defaults_to_fsdp2() -> None:
    assert ParallelismConfig().fsdp_backend == "fsdp2"


def test_parallelism_config_rejects_unknown_fsdp_backend() -> None:
    with pytest.raises(
        ValueError, match=r"fsdp_backend must be one of .* \(got 'fsdp1'\)"
    ):
        ParallelismConfig(fsdp_backend="fsdp1")  # pyrefly: ignore[bad-argument-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data_parallel_replicate_degree": 2},
        {"pipeline_parallel_degree": 2},
        {"fsdp_symm_mem_scope": "all"},
    ],
)
def test_flex_shard_rejects_unsupported_parallelism(kwargs: dict) -> None:
    (name,) = kwargs
    with pytest.raises(ValueError, match=rf"does not support parallelism\.{name}="):
        ParallelismConfig(fsdp_backend="flex_shard", **kwargs)


def test_flex_shard_supports_expert_parallelism() -> None:
    ParallelismConfig(fsdp_backend="flex_shard", expert_parallel_degree=2)


def test_flex_shard_supports_context_parallelism() -> None:
    ParallelismConfig(fsdp_backend="flex_shard", context_parallel_degree=2)


def test_flex_shard_supports_tensor_parallelism() -> None:
    ParallelismConfig(fsdp_backend="flex_shard", tensor_parallel_degree=2)


def test_flex_shard_supports_tensor_and_context_parallelism_together() -> None:
    ParallelismConfig(
        fsdp_backend="flex_shard",
        tensor_parallel_degree=2,
        context_parallel_degree=2,
    )
