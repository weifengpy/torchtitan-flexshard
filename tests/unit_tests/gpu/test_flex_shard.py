# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``parallelism.fsdp_backend='flex_shard'`` against FSDP2, bit for bit."""

import os
from itertools import product
from typing import Any, cast

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import (
    ChunkedLossWrapper,
    cross_entropy_loss,
    CrossEntropyLoss,
)
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.configs import DebugConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import ContextParallelTransform
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.fsdp import get_fsdp_group, set_requires_gradient_sync
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.attention.attention import FlexInnerAttention
from torchtitan.models.common.attention.cp_attention import (
    KVAllGatherCPFlexInnerAttention,
)
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)
from torchtitan.models.llama3 import build_model_config

flex_shard = pytest.importorskip("flex_shard")

pytestmark = pytest.mark.multi_gpu

_SEQ_LEN = 128


def _seq_len(parallelism_context: ParallelismContext) -> int:
    # CP's FlexAttention needs one 128-token block per cp rank.
    return _SEQ_LEN * parallelism_context.cp


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
            data_parallel_shard_degree=parallelism_context.dp_shard,
            context_parallel_degree=parallelism_context.cp,
            tensor_parallel_degree=parallelism_context.tp,
            enable_sequence_parallel=parallelism_context.enable_sequence_parallel,
            expert_parallel_degree=parallelism_context.ep,
            fsdp_backend=fsdp_backend,
            fsdp_reshard_after_forward=reshard_after_forward,
        )
        model_config = build_config("debugmodel", seq_len=_seq_len(parallelism_context))
        if parallelism_context.cp > 1:
            # As CP recipes do: attention all-gathers keys and values over cp.
            model_config = ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: KVAllGatherCPFlexInnerAttention
                }
            ).transform(model_config)
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
        *,
        optim=None,
        steps=range(3),
    ):
        if optim is None:
            optim = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
        loss_fn = None
        if chunked_loss:
            # As the trainer sets it up: the model returns the hidden states,
            # and the loss applies lm_head to each chunk of them.
            loss_fn = ChunkedLossWrapper(
                ChunkedLossWrapper.Config(
                    num_chunks=4,
                    loss_fn=CrossEntropyLoss.Config(
                        global_vocab_size=model_config.vocab_size
                    ),
                )
            )
            loss_fn.set_lm_head(model.lm_head, get_fsdp_group(model, model.lm_head))
            model._skip_lm_head = True
        # One packed sequence per microbatch, as the data loader produces. The
        # data-parallel ranks get different data; their CP and TP peers share it.
        seq_len = _seq_len(parallelism_context)
        positions = torch.arange(seq_len, device=self.device_type)
        dp_mesh = parallelism_context.get_mesh("dp_shard")
        dp_rank, dp_size = dp_mesh.get_local_rank(), dp_mesh.size()
        history = []
        for step in steps:
            losses = []
            for microbatch in range(microbatches):
                if microbatches > 1:
                    set_requires_gradient_sync(model, microbatch == microbatches - 1)
                generator = torch.Generator().manual_seed(
                    1000 * step + 10 * microbatch + dp_rank
                )
                tokens, labels = (
                    torch.randint(
                        model_config.vocab_size, (seq_len,), generator=generator
                    ).to(self.device_type)
                    for _ in range(2)
                )
                # CP shards the inputs over the active SPMD mesh's cp axis.
                with parallelism_context.activate_spmd():
                    inputs, labels, model_kwargs = model.preprocess_inputs(
                        {"input": tokens, "labels": labels, "positions": positions},
                        parallelism_context=parallelism_context,
                        parallelism=parallelism,
                        max_context_length=seq_len,
                    )
                if "aux_loss_denominators" in model_kwargs:
                    # The trainer's global routed-token count: no padding, so
                    # every token of every microbatch on every data-parallel
                    # rank.
                    model_kwargs["aux_loss_denominators"] = torch.tensor(
                        [seq_len * dp_size * microbatches],
                        device=self.device_type,
                    )
                with parallelism_context.activate_spmd():
                    output = model(inputs, **model_kwargs)
                    if loss_fn is None:
                        # Vocab-parallel under TP, as the trainer's loss.
                        loss = (
                            cross_entropy_loss(
                                output,
                                labels,
                                global_vocab_size=model_config.vocab_size,
                            )
                            / labels.numel()
                        )
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
            self._assert_equal_histories(expected_history, actual_history, context)

    def _assert_equal_histories(
        self, expected_history, actual_history, context: str
    ) -> None:
        for step, (expected, actual) in enumerate(
            zip(expected_history, actual_history, strict=True)
        ):
            step_context = f"{context} step {step}"
            for expected_loss, actual_loss in zip(expected[0], actual[0], strict=True):
                self.assertTrue(
                    torch.equal(expected_loss, actual_loss),
                    msg=f"{step_context} loss",
                )
            self.assertTrue(
                torch.equal(expected[1], actual[1]), msg=f"{step_context} grad norm"
            )
            self._assert_equal_tensors(expected[2], actual[2], f"{step_context} grad")
            self._assert_equal_tensors(expected[3], actual[3], f"{step_context} param")

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

    def _context(
        self, *, ep: int = 1, cp: int = 1, tp: int = 1, sp: bool = False
    ) -> ParallelismContext:
        return ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size // (cp * tp),
            cp=cp,
            tp=tp,
            pp=1,
            ep=ep,
            world_size=self.world_size,
            enable_sequence_parallel=sp,
        )

    def _optimizers(self, model) -> OptimizersContainer:
        return OptimizersContainer(
            OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=".*", lr=1e-3)]
            ),
            model_parts=[model],
        )

    def _checkpointer(
        self, model, optimizers, folder: str, **config
    ) -> CheckpointManager:
        return CheckpointManager(
            CheckpointManager.Config(
                folder=folder, interval=1, keep_latest_k=0, **config
            ),
            dataloader=None,
            model_parts=[model],
            optimizers=optimizers,
            lr_schedulers=cast(Any, None),
            ema=None,
            states={},
            sd_adapter=None,
        )

    def _local_state(self, model, optimizers) -> dict[str, torch.Tensor]:
        """The model's and optimizers' local tensors, keyed as checkpointed."""
        state = self._local_params(model)
        for key, value in optimizers.state_dict().items():
            if isinstance(value, torch.Tensor):
                state[f"optimizer.{key}"] = _local(value).detach().clone()
        return state

    def _assert_equal_checkpoints(
        self, expected_dir: str, actual_dir: str, context: str
    ) -> None:
        """Check that two DCP checkpoints hold the same keys, the same chunks of
        the same full tensors, and the same values."""
        expected_metadata, actual_metadata = (
            dcp.FileSystemReader(path).read_metadata().state_dict_metadata
            for path in (expected_dir, actual_dir)
        )
        self.assertEqual(
            sorted(expected_metadata), sorted(actual_metadata), msg=context
        )
        for key, expected in expected_metadata.items():
            actual = actual_metadata[key]
            self.assertIs(type(actual), type(expected), msg=f"{context} {key}")
            if isinstance(expected, TensorStorageMetadata):
                self.assertEqual(
                    (actual.size, actual.properties.dtype, _chunks(actual)),
                    (expected.size, expected.properties.dtype, _chunks(expected)),
                    msg=f"{context} {key}",
                )
        expected_state, actual_state = (
            _load_full_checkpoint(path, f"{path}.pt")
            for path in (expected_dir, actual_dir)
        )
        for key, expected in expected_state.items():
            if isinstance(expected, torch.Tensor):
                self.assertTrue(
                    torch.equal(expected, actual_state[key]), msg=f"{context} {key}"
                )
            else:
                self.assertEqual(expected, actual_state[key], msg=f"{context} {key}")

    def _check_checkpoints_match_fsdp2(
        self, parallelism_context, build_config=build_model_config
    ) -> None:
        checkpoint_dirs = {}
        for backend in ("fsdp2", "flex_shard"):
            model, *setup = self._build(
                backend, "default", parallelism_context, build_config
            )
            optimizers = self._optimizers(model)
            self._train(model, *setup, 1, False, optim=optimizers, steps=range(2))
            for kind, config in (
                ("sync", {}),
                ("async", {"async_mode": "async"}),
                ("export", {"export_dtype": "bfloat16"}),
            ):
                folder = os.path.join(self.temp_dir, backend, kind)
                checkpointer = self._checkpointer(model, optimizers, folder, **config)
                checkpointer.save(2, last_step=kind == "export")
                checkpointer.close()
                checkpoint_dirs[backend, kind] = os.path.join(folder, "step-2")
        # Rank 0 compares and shares the outcome, so that a mismatch fails
        # every rank at once instead of leaving the others in a collective.
        outcome: list[Exception | None] = [None]
        if self.rank == 0:
            try:
                for kind, fsdp2_kind in (
                    ("sync", "sync"),
                    ("async", "sync"),
                    ("export", "export"),
                ):
                    self._assert_equal_checkpoints(
                        checkpoint_dirs["fsdp2", fsdp2_kind],
                        checkpoint_dirs["flex_shard", kind],
                        kind,
                    )
            except Exception as error:
                outcome = [error]
        dist.broadcast_object_list(outcome)
        if outcome[0] is not None:
            raise outcome[0]

    def _check_resumes_across_backends(
        self, parallelism_context, build_config=build_model_config
    ) -> None:
        model, *setup = self._build(
            "fsdp2", "default", parallelism_context, build_config
        )
        reference = self._train(
            model, *setup, 1, False, optim=self._optimizers(model), steps=range(6)
        )
        for save_backend, load_backend in (
            ("fsdp2", "flex_shard"),
            ("flex_shard", "fsdp2"),
            ("flex_shard", "flex_shard"),
        ):
            context = f"{save_backend} to {load_backend}"
            folder = os.path.join(self.temp_dir, f"{save_backend}_{load_backend}")
            model, *setup = self._build(
                save_backend, "default", parallelism_context, build_config
            )
            optimizers = self._optimizers(model)
            self._train(model, *setup, 1, False, optim=optimizers, steps=range(3))
            checkpointer = self._checkpointer(model, optimizers, folder)
            checkpointer.save(3)
            checkpointer.close()

            model, *setup = self._build(
                load_backend, "default", parallelism_context, build_config
            )
            optimizers = self._optimizers(model)
            checkpointer = self._checkpointer(model, optimizers, folder)
            self.assertTrue(checkpointer.load(3))
            resumed = self._train(
                model, *setup, 1, False, optim=optimizers, steps=range(3, 6)
            )
            self._assert_equal_histories(reference[3:], resumed, context)
            # The load replaced the tensors of ModelWrapper's cached state dict.
            checkpointer.save(6)
            checkpointer.close()
            expected = self._local_state(model, optimizers)

            model, *_ = self._build(
                load_backend, "default", parallelism_context, build_config
            )
            optimizers = self._optimizers(model)
            checkpointer = self._checkpointer(model, optimizers, folder)
            self.assertTrue(checkpointer.load(6))
            checkpointer.close()
            self._assert_equal_tensors(
                expected,
                self._local_state(model, optimizers),
                f"{context}, saved after the resume",
            )

    @with_comms
    @with_temp_dir
    def test_checkpoints_match_fsdp2(self):
        # FlexShard declares its local shards as the same chunks of the same
        # full tensors as FSDP2's DTensors: in full checkpoints, saved in sync
        # or threaded async mode, and in bf16 model-only exports. Llama 3
        # shards its stacked linears on dim 1.
        self._check_checkpoints_match_fsdp2(self._context(ep=1))

    @with_comms
    @with_temp_dir
    def test_checkpoints_match_fsdp2_with_expert_parallelism(self):
        # The routed experts' chunks cover every ep rank's experts.
        self._check_checkpoints_match_fsdp2(
            self._context(ep=2), build_deepseek_v3_model_config
        )

    @with_comms
    @with_temp_dir
    def test_resumes_across_backends(self):
        # Train 3 steps under one backend, save, and resume under the other, or
        # under FlexShard again, into a model that has not stepped: the next 3
        # steps match those of an uninterrupted run bit for bit. The resumed run
        # then saves again, and a load of that checkpoint restores its state.
        self._check_resumes_across_backends(self._context(ep=1))

    @with_comms
    def test_matches_fsdp2_with_tensor_parallelism(self):
        # Parameters are TP-local shards, sharded over dp_shard. With sequence
        # parallelism, the norm weights' grads are partial over TP, and both
        # backends all-reduce them over TP before the reduce-scatter.
        parallelism_context = self._context(tp=2, sp=True)
        self._check_matches_fsdp2(
            parallelism_context, product(("default", "always"), (1, 2))
        )
        self._check_matches_fsdp2(
            parallelism_context, [("default", 1), ("default", 2)], chunked_loss=True
        )

    @with_comms
    def test_matches_fsdp2_with_tensor_parallelism_without_sequence_parallelism(
        self,
    ):
        # The norm weights are then typed I on tp: no TP all-reduce.
        self._check_matches_fsdp2(
            self._context(tp=2), product(("default", "always"), (1, 2))
        )

    @with_comms
    @with_temp_dir
    def test_checkpoints_match_fsdp2_with_tensor_parallelism(self):
        self._check_checkpoints_match_fsdp2(self._context(tp=2, sp=True))

    @with_comms
    @with_temp_dir
    def test_resumes_across_backends_with_tensor_parallelism(self):
        self._check_resumes_across_backends(self._context(tp=2, sp=True))

    @with_comms
    def test_matches_fsdp2_with_context_parallelism(self):
        # Parameters shard over dp_shard x cp, flattened as fully_shard does,
        # and the reduce-scatter sums the grads of every sequence chunk.
        self._check_matches_fsdp2(
            self._context(cp=2), product(("default", "always"), (1, 2))
        )

    @with_comms
    @with_temp_dir
    def test_checkpoints_match_fsdp2_with_context_parallelism(self):
        self._check_checkpoints_match_fsdp2(self._context(cp=2))

    @with_comms
    @with_temp_dir
    def test_resumes_across_backends_with_context_parallelism(self):
        self._check_resumes_across_backends(self._context(cp=2))

    @with_comms
    @with_temp_dir
    def test_resumes_across_backends_with_expert_parallelism(self):
        self._check_resumes_across_backends(
            self._context(ep=2), build_deepseek_v3_model_config
        )


def _chunks(metadata: TensorStorageMetadata) -> list[tuple[tuple[int, ...], ...]]:
    """A checkpointed tensor's non-empty chunks, as (offsets, sizes) pairs."""
    return sorted(
        (tuple(chunk.offsets), tuple(chunk.sizes))
        for chunk in metadata.chunks
        if all(chunk.sizes)
    )


def _load_full_checkpoint(checkpoint_dir: str, torch_save_path: str) -> dict[str, Any]:
    """Load every full tensor of a DCP checkpoint in this process."""
    dcp_to_torch_save(checkpoint_dir, torch_save_path)
    return torch.load(torch_save_path, weights_only=False)
