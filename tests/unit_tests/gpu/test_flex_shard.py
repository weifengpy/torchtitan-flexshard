# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``parallelism.fsdp_backend='flex_shard'`` against FSDP2, bit for bit."""

from itertools import product

import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config import TrainingConfig
from torchtitan.config.configs import DebugConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.fsdp import get_fsdp_group, set_requires_gradient_sync
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)
from torchtitan.models.llama3 import build_model_config

flex_shard = pytest.importorskip("flex_shard")

pytestmark = pytest.mark.multi_gpu

_SEQ_LEN = 128


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


class TestFlexShardDecoder(DTensorTestBase):
    @property
    def world_size(self) -> int:
        # Sums of four values make the reduction order observable.
        return 4

    def _build(
        self,
        fsdp_backend: str,
        reshard_after_forward: str,
        parallelism_context: ParallelismContext,
        build_config=build_model_config,
    ):
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=self.world_size,
            expert_parallel_degree=parallelism_context.ep,
            fsdp_backend=fsdp_backend,
            fsdp_reshard_after_forward=reshard_after_forward,
        )
        model_config = build_config("debugmodel", seq_len=_SEQ_LEN)
        model_config.set_sharding_(parallelism)
        with parallelism_context.activate_spmd(), torch.device("meta"):
            model = model_config.build()
        model = model.parallelize(
            parallelism_context=parallelism_context,
            training=TrainingConfig(),
            parallelism=parallelism,
            local_compile_regions=[],
            ac_config=None,
            dump_folder="",
        )
        dist_utils.set_determinism(
            parallelism_context,
            torch.device(self.device_type),
            DebugConfig(seed=0),
            distinct_seed_mesh_axes=[],
        )
        with parallelism_context.activate_spmd():
            model.to_empty(device=self.device_type)
            with torch.no_grad():
                model.init_weights(buffer_device=None)
        return model, model_config, parallelism, parallelism_context

    def _train(
        self,
        model,
        model_config,
        parallelism,
        parallelism_context,
        microbatches,
        chunked_loss,
    ):
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
        loss_fn = None
        if chunked_loss:
            # As the trainer sets it up: the model returns the hidden states,
            # and the loss applies lm_head to each chunk of them.
            loss_fn = ChunkedLossWrapper(ChunkedLossWrapper.Config(num_chunks=4))
            loss_fn.set_lm_head(model.lm_head, get_fsdp_group(model, model.lm_head))
            model._skip_lm_head = True
        # One packed sequence per microbatch, as the data loader produces.
        positions = torch.arange(_SEQ_LEN, device=self.device_type)
        history = []
        for step in range(3):
            losses = []
            for microbatch in range(microbatches):
                if microbatches > 1:
                    set_requires_gradient_sync(model, microbatch == microbatches - 1)
                generator = torch.Generator().manual_seed(
                    1000 * step + 10 * microbatch + self.rank
                )
                tokens, labels = (
                    torch.randint(
                        model_config.vocab_size, (_SEQ_LEN,), generator=generator
                    ).to(self.device_type)
                    for _ in range(2)
                )
                inputs, labels, model_kwargs = model.preprocess_inputs(
                    {"input": tokens, "labels": labels, "positions": positions},
                    parallelism_context=parallelism_context,
                    parallelism=parallelism,
                    max_context_length=_SEQ_LEN,
                )
                if "aux_loss_denominators" in model_kwargs:
                    # The trainer's global routed-token count: no padding, so
                    # every token of every microbatch on every rank.
                    model_kwargs["aux_loss_denominators"] = torch.tensor(
                        [_SEQ_LEN * self.world_size * microbatches],
                        device=self.device_type,
                    )
                with parallelism_context.activate_spmd():
                    output = model(inputs, **model_kwargs)
                    if loss_fn is None:
                        loss = F.cross_entropy(output.float(), labels)
                    else:
                        loss, _ = loss_fn(output, labels)
                    with spmd.no_typecheck():
                        loss.backward()
                losses.append(loss.detach())
            grad_norm = dist_utils.clip_grad_norm_(
                list(model.parameters()),
                max_norm=1.0,
                foreach=True,
                ep_enabled=parallelism_context.ep_enabled,
            )
            grads = {
                fqn: _local(param.grad).clone()
                for fqn, param in model.named_parameters()
                if param.grad is not None
            }
            optim.step()
            optim.zero_grad()
            history.append((losses, grad_norm, grads, self._local_params(model)))
        return history

    def _local_params(self, model) -> dict[str, torch.Tensor]:
        return {
            fqn: _local(param).detach().clone()
            for fqn, param in model.named_parameters()
        }

    def _assert_equal_tensors(self, expected, actual, context: str) -> None:
        self.assertEqual(list(expected), list(actual), msg=context)
        for fqn, expected_tensor in expected.items():
            self.assertTrue(
                torch.equal(expected_tensor, actual[fqn]), msg=f"{context} {fqn}"
            )

    def _check_matches_fsdp2(
        self,
        parallelism_context,
        configs,
        build_config=build_model_config,
        chunked_loss=False,
    ) -> None:
        for reshard_after_forward, microbatches in configs:
            context = (
                f"reshard_after_forward={reshard_after_forward} "
                f"microbatches={microbatches} chunked_loss={chunked_loss}"
            )
            fsdp2, *fsdp2_setup = self._build(
                "fsdp2", reshard_after_forward, parallelism_context, build_config
            )
            flex, *flex_setup = self._build(
                "flex_shard", reshard_after_forward, parallelism_context, build_config
            )
            for fqn, param in flex.named_parameters():
                self.assertTrue(flex_shard.is_flex_shard_param(param), msg=fqn)
            self._assert_equal_tensors(
                self._local_params(fsdp2), self._local_params(flex), f"{context} init"
            )
            expected_history = self._train(
                fsdp2, *fsdp2_setup, microbatches, chunked_loss
            )
            actual_history = self._train(flex, *flex_setup, microbatches, chunked_loss)
            for step, (expected, actual) in enumerate(
                zip(expected_history, actual_history, strict=True)
            ):
                step_context = f"{context} step {step}"
                for expected_loss, actual_loss in zip(
                    expected[0], actual[0], strict=True
                ):
                    self.assertTrue(
                        torch.equal(expected_loss, actual_loss),
                        msg=f"{step_context} loss",
                    )
                self.assertTrue(
                    torch.equal(expected[1], actual[1]), msg=f"{step_context} grad norm"
                )
                self._assert_equal_tensors(
                    expected[2], actual[2], f"{step_context} grad"
                )
                self._assert_equal_tensors(
                    expected[3], actual[3], f"{step_context} param"
                )

    @with_comms
    def test_matches_fsdp2(self):
        # One context, as in training: both backends shard over its dp_shard mesh.
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        self._check_matches_fsdp2(
            parallelism_context, product(("default", "always"), (1, 2))
        )

    @with_comms
    def test_matches_fsdp2_with_expert_parallelism(self):
        # Routed experts shard over edp_shard (2 ranks per ep shard), the rest
        # of each MoE block over dp_shard.
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=2,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        self._check_matches_fsdp2(
            parallelism_context,
            [("default", 1), ("default", 2)],
            build_deepseek_v3_model_config,
        )

    @with_comms
    def test_matches_fsdp2_with_chunked_loss(self):
        # ChunkedLossWrapper keeps the norm and lm_head group unsharded across
        # its chunks and reduce-scatters lm_head's gradient at the last one,
        # the norm's in the decoder backward; under gradient accumulation it
        # reduce-scatters them in every microbatch.
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        self._check_matches_fsdp2(
            parallelism_context,
            product(("default", "always", "never"), (1, 2)),
            chunked_loss=True,
        )
