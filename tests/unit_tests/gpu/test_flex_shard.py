# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``parallelism.fsdp_backend='flex_shard'`` against FSDP2, bit for bit."""

import functools
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
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.configs import DebugConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import ContextParallelTransform
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.cuda_graph import (
    cuda_graph_teardown,
    NUM_CUDA_GRAPH_WARMUP_STEPS,
    wrap_fwd_bwd_with_cuda_graph,
)
from torchtitan.distributed.flex_shard_fsdp import (
    as_flex_shard_state_dict,
    as_fsdp2_state_dict,
    relayout_owned_buckets,
)
from torchtitan.distributed.fsdp import get_fsdp_group, set_requires_gradient_sync
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.attention.attention import FlexInnerAttention
from torchtitan.models.common.attention.cp_attention import (
    KVAllGatherCPFlexInnerAttention,
)
from torchtitan.models.deepseek_v3 import (
    build_model_config as build_deepseek_v3_model_config,
)
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from torchtitan.models.llama3 import build_model_config

flex_shard = pytest.importorskip("flex_shard")

pytestmark = pytest.mark.multi_gpu

_SEQ_LEN = 128


def _seq_len(parallelism_context: ParallelismContext) -> int:
    # CP's FlexAttention needs one 128-token block per cp rank.
    return _SEQ_LEN * parallelism_context.cp


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _box(offset: tuple[int, ...], size: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(o, o + n) for o, n in zip(offset, size, strict=True))


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
        owned_params: tuple[str, ...] = (),
        **parallelism_options,
    ):
        parallelism = ParallelismConfig(
            data_parallel_shard_degree=parallelism_context.dp_shard,
            context_parallel_degree=parallelism_context.cp,
            tensor_parallel_degree=parallelism_context.tp,
            enable_sequence_parallel=parallelism_context.enable_sequence_parallel,
            expert_parallel_degree=parallelism_context.ep,
            fsdp_backend=fsdp_backend,
            fsdp_reshard_after_forward=reshard_after_forward,
            flex_shard_owned_params=list(owned_params),
            **parallelism_options,
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
            if owned_params:
                # As the trainer does after initialization.
                relayout_owned_buckets(model, owned_params)
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
        spmd_typechecking=False,
        defer_gradient_reduction=True,
        cuda_graphs=False,
        max_norm=1.0,
    ):
        if optim is None:
            optim = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
        # As the trainer's loss: vocab-parallel under TP, normalized by the
        # global token count, with one objective per MTP layer.
        mtp_layers = getattr(model, "mtp_layers", None)
        num_objectives = 1 + (0 if mtp_layers is None else len(mtp_layers))
        loss_fn = (
            CrossEntropyLoss(
                CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size)
            )
            if num_objectives == 1
            else MTPLoss(MTPLoss.Config(global_vocab_size=model_config.vocab_size))
        )
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

        def forward_backward(prepared_microbatches) -> torch.Tensor:
            losses = []
            for microbatch, (
                inputs,
                labels,
                model_kwargs,
                global_loss_token_counts,
            ) in enumerate(prepared_microbatches):
                # As parallelism.fsdp_defer_gradient_reduction: only the last
                # microbatch's backward reduces.
                if microbatches > 1 and defer_gradient_reduction:
                    set_requires_gradient_sync(model, microbatch == microbatches - 1)
                with parallelism_context.activate_spmd(typechecking=spmd_typechecking):
                    output = model(inputs, **model_kwargs)
                    loss, _ = loss_fn(
                        output,
                        labels,
                        global_loss_token_counts=global_loss_token_counts,
                    )
                    with spmd.no_typecheck():
                        loss.backward()
                losses.append(loss.detach())
            return torch.stack(losses)

        if cuda_graphs:
            # As the trainer with CUDA graphs: eager warmup steps, then one
            # capture of the forward-backward, replayed in later steps.
            forward_backward = wrap_fwd_bwd_with_cuda_graph(
                forward_backward,
                parameters=model.parameters(),
                num_warmup_iterations=NUM_CUDA_GRAPH_WARMUP_STEPS,
            )
        # One packed sequence per microbatch, as the data loader produces. The
        # data-parallel ranks get different data; their CP and TP peers share it.
        seq_len = _seq_len(parallelism_context)
        positions = torch.arange(seq_len, device=self.device_type)
        dp_mesh = parallelism_context.get_mesh("dp_shard")
        dp_rank, dp_size = dp_mesh.get_local_rank(), dp_mesh.size()
        history = []
        for step in steps:
            prepared_microbatches = []
            for microbatch in range(microbatches):
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
                # The trainer's global token counts: no padding, so every token
                # of every microbatch on every data-parallel rank.
                global_token_count = seq_len * dp_size * microbatches
                if "aux_loss_denominators" in model_kwargs:
                    model_kwargs["aux_loss_denominators"] = torch.tensor(
                        [global_token_count] * num_objectives, device=self.device_type
                    )
                # Created outside the type checker, which would type it R on tp.
                global_loss_token_counts = torch.tensor(
                    global_token_count
                    if num_objectives == 1
                    else [global_token_count] * num_objectives,
                    device=self.device_type,
                )
                prepared_microbatches.append(
                    (inputs, labels, model_kwargs, global_loss_token_counts)
                )
            # A replay overwrites the captured step's loss.
            losses = list(forward_backward(prepared_microbatches).clone())
            grad_norm = dist_utils.clip_grad_norm_(
                list(model.parameters()),
                max_norm=max_norm,
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
        if cuda_graphs:
            cuda_graph_teardown()
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
        spmd_typechecking=False,
        defer_gradient_reduction=True,
        cuda_graphs=False,
    ) -> None:
        # With CUDA graphs: the warmup steps, the captured step and two replays.
        steps = range(NUM_CUDA_GRAPH_WARMUP_STEPS + 3 if cuda_graphs else 3)
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
                fsdp2,
                *fsdp2_setup,
                microbatches,
                chunked_loss,
                steps=steps,
                spmd_typechecking=spmd_typechecking,
                defer_gradient_reduction=defer_gradient_reduction,
                cuda_graphs=cuda_graphs,
            )
            actual_history = self._train(
                flex,
                *flex_setup,
                microbatches,
                chunked_loss,
                steps=steps,
                spmd_typechecking=spmd_typechecking,
                defer_gradient_reduction=defer_gradient_reduction,
                cuda_graphs=cuda_graphs,
            )
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
        # FlexShard's hooks run outside spmd_types' checker, as FSDP2's do.
        self._check_matches_fsdp2(
            parallelism_context, [("default", 1)], spmd_typechecking=True
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
        self._check_matches_fsdp2(
            parallelism_context,
            [("default", 1)],
            build_deepseek_v3_model_config,
            spmd_typechecking=True,
        )

    @with_comms
    def test_dist_muon_without_optimizer_communication(self):
        # DistMuon as the DeepSeek V3 recipe sets it up, with whole experts on
        # each rank (ep 4). FlexShard stores every block parameter outside the
        # routed experts whole on owners (MLA_DIST_MUON_OWNED_PARAMS), one bucket
        # per block, and runs DistMuon on each rank's local storage: its losses
        # equal FSDP2's bit for bit over 3 steps, with clipping off (grad norms
        # may round apart, as owners split the sum of squares differently), and
        # its DistMuon step runs no collectives, while FSDP2's redistributes.
        # FSDP2's DistMuon takes FlexShard's layout (flex_shard_layout): each of
        # those matrices computes on the rank FlexShard stores it on.
        # pyrefly: ignore [missing-import]
        from flex_shard.dist_muon._optimizer_reshard_schedule import (
            _RedistributionBucketPlan,
        )
        from torchtitan.components.checkpointer.utils import canonical_fqn
        from torchtitan_recipes.tests.models.dist_muon import (
            mla_dist_muon_optimizer,
            MLA_DIST_MUON_OWNED_PARAMS,
        )

        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=self.world_size,
            cp=1,
            tp=1,
            pp=1,
            ep=self.world_size,
            world_size=self.world_size,
            enable_sequence_parallel=False,
        )
        losses, collectives = {}, {}
        owners: dict[str, dict[str, int]] = {"fsdp2": {}, "flex_shard": {}}
        for backend, owned in (
            ("fsdp2", ()),
            ("flex_shard", MLA_DIST_MUON_OWNED_PARAMS),
        ):
            model, *setup = self._build(
                backend,
                "default",
                parallelism_context,
                build_deepseek_v3_model_config,
                owned_params=owned,
            )
            model_config, parallelism, _ = setup
            if backend == "flex_shard":
                # pyrefly: ignore [missing-import]
                from flex_shard.custom_placements.owned import BucketedOwned

                buckets_per_block: dict[str, int] = {}
                for storage in model.sharded_bucket_storages:
                    fqns = [
                        fqn
                        for fqn in storage.param_infos
                        if fqn.startswith("layers.") and ".routed_experts." not in fqn
                    ]
                    if not fqns:
                        continue
                    block = ".".join(fqns[0].split(".")[:2])
                    buckets_per_block[block] = buckets_per_block.get(block, 0) + 1
                    for fqn in fqns:
                        placement = storage.param_infos[fqn].placement
                        self.assertIsInstance(placement, BucketedOwned, fqn)
                        (segment,) = placement.segments_by_fqn[fqn]
                        owners[backend][canonical_fqn(fqn)] = segment.owner_rank
                self.assertEqual(
                    set(buckets_per_block.values()), {1}, buckets_per_block
                )
            optim = mla_dist_muon_optimizer(
                model_config,
                muon_lr=8e-4,
                adamw_lr=8e-4,
                parallelism=parallelism,
                flex_shard_layout=True,
            ).build(model_parts=[model])
            (dist_muon,) = [
                o for o in optim.optimizers if "DistMuon" in type(o).__name__
            ]
            if backend == "fsdp2":
                for bucket in dist_muon._bucket_plans:
                    if not isinstance(bucket, _RedistributionBucketPlan):
                        continue
                    ranks = bucket.group.mesh_axis_participants
                    for item, plan in zip(
                        bucket.redistributed_items,
                        bucket.redistribution_plans,
                        strict=True,
                    ):
                        (owner,) = [
                            partition.participant
                            for partition in plan.compute_partitions
                            if partition.logical_regions
                        ]
                        owners[backend][item.fqn] = ranks.index(owner)
                muon_fqns = {
                    fqn
                    for fqn in dist_muon.param_groups[0]["param_names"]
                    if ".routed_experts." not in fqn
                }
                self.assertEqual(set(owners[backend]), muon_fqns)
            collectives[backend] = []
            step = dist_muon.step

            def profiled_step(*args, _step=step, _calls=collectives[backend]):
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as prof:
                    out = _step(*args)
                _calls.append(
                    sum(event.name.startswith("c10d::") for event in prof.events())
                )
                return out

            dist_muon.step = profiled_step
            history = self._train(model, *setup, 1, False, optim=optim, max_norm=1e9)
            losses[backend] = [losses_of_step for losses_of_step, *_ in history]
        for step, (expected, actual) in enumerate(
            zip(losses["fsdp2"], losses["flex_shard"], strict=True)
        ):
            for expected_loss, actual_loss in zip(expected, actual, strict=True):
                self.assertTrue(
                    torch.equal(expected_loss, actual_loss), msg=f"step {step} loss"
                )
        self.assertEqual(collectives["flex_shard"], [0, 0, 0])
        self.assertTrue(all(collectives["fsdp2"]), collectives["fsdp2"])
        self.assertEqual(
            owners["fsdp2"],
            {fqn: owners["flex_shard"][fqn] for fqn in owners["fsdp2"]},
        )

    @with_comms
    def test_matches_fsdp2_with_cuda_graphs(self):
        # As DeepSeek V3 recipes with CUDA graphs: the chunked loss, with and
        # without gradient accumulation. EP 1, since the all-to-all token
        # dispatcher doesn't support CUDA graphs.
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
            [("default", 1), ("default", 2), ("never", 2)],
            build_deepseek_v3_model_config,
            chunked_loss=True,
            cuda_graphs=True,
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
        self, model, optimizers, folder: str, *, sd_adapter=None, **config
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
            sd_adapter=sd_adapter,
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

    def _export_fsdp2_hf(self, parallelism_context) -> tuple[Any, str]:
        """Train a DeepSeek V3 model with FSDP2 for a step and export its
        weights in HF format; return the model and the export's path."""
        model, model_config, *setup = self._build(
            "fsdp2", "default", parallelism_context, build_deepseek_v3_model_config
        )
        optimizers = self._optimizers(model)
        # A step makes the weights differ from the initialization the loading
        # models start from, so a load that does nothing fails.
        self._train(
            model, model_config, *setup, 1, False, optim=optimizers, steps=range(1)
        )
        export_folder = os.path.join(self.temp_dir, "export")
        checkpointer = self._checkpointer(
            model,
            optimizers,
            export_folder,
            sd_adapter=DeepSeekV3StateDictAdapter(model_config, None),
            last_save_model_only=True,
            last_save_in_hf=True,
            export_dtype="float32",
        )
        checkpointer.save(1, last_step=True)
        checkpointer.close()
        return model, os.path.join(export_folder, "step-1")

    def _load_hf(
        self, backend: str, parallelism_context, path: str, **parallelism_options
    ):
        """Build a DeepSeek V3 model and load its weights from the HF export at
        ``path``; return the model and its config."""
        model, model_config, *_ = self._build(
            backend,
            "default",
            parallelism_context,
            build_deepseek_v3_model_config,
            **parallelism_options,
        )
        checkpointer = self._checkpointer(
            model,
            self._optimizers(model),
            os.path.join(self.temp_dir, "_".join([backend, *parallelism_options])),
            sd_adapter=DeepSeekV3StateDictAdapter(model_config, None),
            initial_load_path=path,
            initial_load_in_hf=True,
        )
        self.assertTrue(checkpointer.load())
        checkpointer.close()
        return model, model_config

    def _check_hf_load_matches_fsdp2(self, parallelism_context) -> None:
        model, path = self._export_fsdp2_hf(parallelism_context)
        expected = self._local_params(model)
        models = {}
        for backend in ("fsdp2", "flex_shard"):
            models[backend], model_config = self._load_hf(
                backend, parallelism_context, path
            )
        # FlexShard's parameters reach the adapter as views of the chunks
        # FlexShard declares to DCP, and from_hf returns them with the views'
        # placements, so the load redistributes nothing.
        flex = models["flex_shard"]
        state_dict = flex.state_dict()
        views = as_fsdp2_state_dict([flex], state_dict)
        adapter = DeepSeekV3StateDictAdapter(model_config, None)
        round_trip = adapter.from_hf(adapter.to_hf(views))
        local_round_trip = as_flex_shard_state_dict([flex], round_trip)

        for backend, model in models.items():
            self._assert_equal_tensors(
                expected, self._local_params(model), f"{backend} loaded"
            )
        layouts = flex_shard.get_flex_shard_global_layouts(flex)
        self.assertEqual(sorted(layouts), sorted(expected))
        for fqn, layout in layouts.items():
            view = views[fqn]
            chunks = [
                (tuple(chunk.offsets), tuple(chunk.sizes))
                for chunk in view.__create_chunk_list__()
            ]
            regions = list(zip(layout.global_offsets, layout.local_sizes, strict=True))
            self.assertEqual(
                (tuple(view.shape), chunks), (layout.global_shape, regions), msg=fqn
            )
            self.assertEqual(round_trip[fqn].placements, view.placements, msg=fqn)
            self.assertTrue(
                torch.equal(local_round_trip[fqn], state_dict[fqn]), msg=fqn
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
        self._check_matches_fsdp2(
            parallelism_context, [("default", 1)], spmd_typechecking=True
        )

    @with_comms
    def test_matches_fsdp2_with_tensor_and_context_parallelism(self):
        # FSDP2 lays parameters out on the flattened dp_shard x cp mesh and tp.
        self._check_matches_fsdp2(
            self._context(cp=2, tp=2, sp=True), [("default", 1), ("default", 2)]
        )

    @with_comms
    def test_matches_fsdp2_with_multi_token_prediction(self):
        # DeepSeek V3's MTP layer shards like a block after the decoder's, with
        # its routed experts on edp_shard. lm_head runs twice per forward, so
        # with deferred reduction its kept grads are reduce-scattered on their
        # own, as FSDP2's first post-backward of the two does.
        parallelism_context = self._context(ep=2)
        build_config = functools.partial(
            build_deepseek_v3_model_config, num_mtp_layers=1
        )
        # Every backward reduces, as the trainer's do by default.
        self._check_matches_fsdp2(
            parallelism_context,
            [("default", 1), ("default", 2)],
            build_config,
            defer_gradient_reduction=False,
        )
        self._check_matches_fsdp2(parallelism_context, [("default", 2)], build_config)

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
        parallelism_context = self._context(cp=2)
        self._check_matches_fsdp2(
            parallelism_context, product(("default", "always"), (1, 2))
        )
        self._check_matches_fsdp2(
            parallelism_context, [("default", 1)], spmd_typechecking=True
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

    @with_comms
    @with_temp_dir
    def test_hf_load_matches_fsdp2_with_expert_parallelism(self):
        # The routed experts also shard over edp_shard, two ranks per ep shard.
        self._check_hf_load_matches_fsdp2(self._context(ep=2))

    @with_comms
    @with_temp_dir
    def test_hf_load_matches_fsdp2_with_one_rank_edp_shard(self):
        # ep spans all ranks, so the routed experts' edp_shard axis has one rank.
        self._check_hf_load_matches_fsdp2(self._context(ep=4))

    @with_comms
    @with_temp_dir
    def test_hf_load_matches_fsdp2_with_blockwise_fp8(self):
        # A blockwise FP8 weight's local shard is a range of its matrix's rows,
        # in whole 128-row blocks, not FSDP2's chunk. Each rank's rows must hold
        # the same values as those rows of FSDP2's full weight; the other
        # parameters keep FSDP2's chunks. ep spans all ranks, as in the 256-GPU
        # job.
        from torchtitan.distributed.flex_shard_blockwise_fp8 import (
            blockwise_fp8_weights,
        )

        parallelism_context = self._context(ep=4)
        fsdp2, path = self._export_fsdp2_hf(parallelism_context)
        expected = self._local_params(fsdp2)
        full = {
            fqn: cast(DTensor, param).full_tensor()
            for fqn, param in fsdp2.named_parameters()
        }
        for option in (
            "flex_shard_blockwise_fp8",
            "flex_shard_blockwise_fp8_dense_reference",
        ):
            flex, _ = self._load_hf(
                "flex_shard", parallelism_context, path, **{option: True}
            )
            fp8_weights = blockwise_fp8_weights(flex)
            self.assertTrue(fp8_weights, msg=option)
            layouts = flex_shard.get_flex_shard_global_layouts(flex)
            for fqn, param in flex.named_parameters():
                local = _local(param).detach()
                if param not in fp8_weights:
                    self.assertTrue(
                        torch.equal(local, expected[fqn]), msg=f"{option} {fqn}"
                    )
                    continue
                layout = layouts[fqn]
                for global_offset, local_offset, size in zip(
                    layout.global_offsets,
                    layout.local_offsets,
                    layout.local_sizes,
                    strict=True,
                ):
                    self.assertTrue(
                        torch.equal(
                            local[_box(local_offset, size)],
                            full[fqn][_box(global_offset, size)],
                        ),
                        msg=f"{option} {fqn} rows at {global_offset}",
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
