# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DistMuon optimizer configuration shared by the recipes of decoders built from
DeepSeek V3's MLA attention and MoE blocks: Kimi K2.x, Moonlight and DeepSeek V3."""

from dataclasses import dataclass, replace
from typing import cast

from torch.distributed.tensor import Shard

from torchtitan.components.optim import AdamW, DistMuon, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.deepseek_v3.model import Attention as DeepSeekV3Attention
from torchtitan.trainer import Trainer


# With parallelism.fsdp_backend="flex_shard", the parameters FlexShard stores
# whole on owners (parallelism.flex_shard_owned_params): every block parameter
# outside the routed experts, which stay whole per expert-parallel rank. Each
# block is then one owner bucket, gathered and reduced in one collective as FSDP2
# does, and DistMuon's step needs no communication: each owner runs the Muon
# units of its parameters (whole matrices, heads, wkv_a's [kv_latent; k_rope],
# w13's gate and up as one batch of two), with the same bits as DistMuon's
# redistribution (checked on H100: a head block, or gate and up, gives the same
# bits in a batch of 1, 2 or 128). Norms are AdamW's.
MLA_DIST_MUON_OWNED_PARAMS = (
    r"\.(?:attention|attention_norm|ffn_norm|feed_forward|moe\.router"
    r"|moe\.shared_experts)\.",
)


def per_expert_compute_layout(parallelism: ParallelismConfig) -> ComputeLayout:
    ep_size = parallelism.expert_parallel_degree
    if ep_size <= 0:
        raise ValueError("expert_parallel_degree must be positive")
    if ep_size == 1:
        return ComputeLayout(
            shardings_by_mesh_axis={
                MeshAxisName.DP_SHARD.value: Shard(0),
            },
        )

    # Preserve exact EP-first DTensor ownership. If an EP-local expert count is
    # smaller than the edp_shard size, add balanced rank assignment only after
    # benchmarks show that the fixed nonempty edp_shard coordinates are a hotspot.
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.EDP_SHARD.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        },
        # EP splits the expert dimension first, then edp_shard repartitions each
        # EP-local expert domain, which reverses the storage-mesh axis order.
        shard_order_by_tensor_dim={
            0: (MeshAxisName.EP.value, MeshAxisName.EDP_SHARD.value),
        },
    )


def mla_dist_muon_optimizer(
    model_config: Decoder.Config,
    *,
    muon_lr: float,
    adamw_lr: float,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    """DistMuon for decoders built from DeepSeek V3's MLA attention and MoE
    blocks (DeepSeek V3, Kimi K2.x, Moonlight), with Kimi's Muon units: one
    matrix per logical projection, per MLA head where a projection is per head.
    Dense-MLP layers are those without ``moe``. AdamW takes the rest."""
    attention = cast(DeepSeekV3Attention.Config, model_config.first_base_attention)
    owned = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: Owned(),
        },
    )
    # Kimi runs Newton-Schulz per logical projection within each MLA head.
    per_query_head = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: BlockShard(
                dim=0,
                # per head: [q_nope_h; q_rope_h]
                block_sizes=(attention.qk_nope_head_dim, attention.qk_rope_head_dim),
            )
        },
    )
    kv_latent_and_rope = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: BlockShard(
                dim=0,
                # Shared projections: [kv_latent; k_rope]
                block_sizes=(attention.kv_lora_rank, attention.qk_rope_head_dim),
            )
        },
    )
    per_key_value_head = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: BlockShard(
                dim=0,
                # per head: [k_nope_h; v_h]
                block_sizes=(attention.qk_nope_head_dim, attention.v_head_dim),
            )
        },
    )
    per_expert = per_expert_compute_layout(parallelism)
    query_shardings: dict[str, ComputeLayout] = (
        {
            "wq_a": owned,
            "wq_b": per_query_head,
        }
        if attention.q_lora_rank
        else {"wq": per_query_head}
    )
    attention_shardings = {
        **query_shardings,
        "wkv_a": kv_latent_and_rope,
        "wkv_b": per_key_value_head,
        "wo": owned,
    }
    feed_forward_shardings = {
        "w13": ComputeLayout(
            shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Shard(0)},
        ),
        "w2": owned,
    }
    num_layers = len(model_config.layers)
    adamw_kwargs = {
        "lr": adamw_lr,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }
    expert_projections = ("w13.weight", "w2.weight")

    def compute_shardings_for_layer(
        layer_id: int,
    ) -> dict[str, ComputeLayout]:
        prefix = f"layers.{layer_id}"
        shardings = {
            f"{prefix}.attention.{projection}.weight": compute_sharding
            for projection, compute_sharding in attention_shardings.items()
        }
        if model_config.layers[layer_id].moe is None:
            shardings.update(
                {
                    f"{prefix}.feed_forward.{projection}.weight": compute_sharding
                    for projection, compute_sharding in feed_forward_shardings.items()
                }
            )
        else:
            shardings.update(
                {
                    f"{prefix}.moe.routed_experts.{projection}": per_expert
                    for projection in expert_projections
                }
            )
            shardings[f"{prefix}.moe.router.gate.weight"] = owned
            shardings.update(
                {
                    f"{prefix}.moe.shared_experts.{projection}.weight": compute_sharding
                    for projection, compute_sharding in feed_forward_shardings.items()
                }
            )
        return shardings

    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    layer_fqns = []
    for layer_id in range(num_layers):
        layer_shardings = compute_shardings_for_layer(layer_id)
        compute_sharding_by_fqn.update(layer_shardings)
        layer_fqns.append(tuple(layer_shardings))
    # Dense layers have much larger MLPs, so each gets its own bucket, while pairs
    # of consecutive MoE layers amortize collective launch overhead.
    bucket_layer_ids: list[tuple[int, ...]] = []
    open_moe_pair = False
    for layer_id, layer in enumerate(model_config.layers):
        if layer.moe is not None and open_moe_pair:
            bucket_layer_ids[-1] += (layer_id,)
            open_moe_pair = False
        else:
            bucket_layer_ids.append((layer_id,))
            open_moe_pair = layer.moe is not None
    bucket_configs = []
    for layer_ids in bucket_layer_ids:
        name = "layers." + "-".join(map(str, layer_ids))
        fqns = [fqn for layer_id in layer_ids for fqn in layer_fqns[layer_id]]
        routed_fqns = tuple(
            fqn for fqn in fqns if compute_sharding_by_fqn[fqn] is per_expert
        )
        non_routed_fqns = tuple(
            fqn for fqn in fqns if compute_sharding_by_fqn[fqn] is not per_expert
        )
        bucket_configs.append(BucketConfig(name=name, patterns=non_routed_fqns))
        if routed_fqns:
            bucket_configs.append(
                BucketConfig(name=f"{name}.routed-experts", patterns=routed_fqns)
            )
    # Muon is designed for matrix parameters; Moonlight uses AdamW for
    # non-matrix parameters such as RMSNorm, LM head, and embeddings. Expert
    # tensors below are batch-first stacks of matrices. See Sec. 2.2:
    # https://arxiv.org/abs/2502.16982
    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_shardings)})\.weight|"
        rf"routed_experts\.(?:{'|'.join(expert_projections)})|"
        r"feed_forward\.(?:w13|w2)\.weight|"
        # Keep the 2D router gate on Muon: Moonlight Figure 4 reports its
        # SVD-entropy gain over AdamW is larger than for other matrix groups.
        r"moe\.router\.gate\.weight|"
        r"moe\.shared_experts\.(?:w13|w2)\.weight"
        r")$"
    )
    return OptimizersContainer.Config(
        optimizers=[
            DistMuon.Config(
                pattern=muon_pattern,
                bucket_configs=tuple(bucket_configs),
                compute_sharding_by_fqn=compute_sharding_by_fqn,
                lr=muon_lr,
                weight_decay=0.1,
                # Kimi K2 uses 0.2 * sqrt(max(rows, columns)) for
                # shape-consistent AdamW-scale updates instead of Muon's
                # original aspect-ratio scaling.
                adjust_lr_fn="match_rms_adamw",
            ),
            # The remaining parameters are embeddings, norms, biases, LM head,
            # and the vision tower.
            AdamW.Config(
                pattern=r".*",
                foreach=True,
                fused=False,
                **adamw_kwargs,
            ),
        ],
    )


def align_dist_muon_expert_compute_layouts(
    optimizer_config: OptimizersContainer.Config,
    *,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    """Align routed-expert layouts with the final parallelism config.

    Config variants can replace the base recipe's parallelism before invoking
    ``__post_init__``. That parallelism decides whether routed experts use the
    1D ``dp_shard`` layout or the 2D ep/edp_shard layout, so their layouts have
    to be rebuilt here.
    """
    dist_muon_index = next(
        (
            index
            for index, config in enumerate(optimizer_config.optimizers)
            if isinstance(config, DistMuon.Config)
        ),
        None,
    )
    if dist_muon_index is None:
        return optimizer_config
    dist_muon_config = cast(
        DistMuon.Config,
        optimizer_config.optimizers[dist_muon_index],
    )
    compute_sharding_by_fqn = cast(
        dict[str, ComputeLayout],
        dist_muon_config.compute_sharding_by_fqn,
    )
    per_expert = per_expert_compute_layout(parallelism)
    aligned_shardings = {}
    changed = False
    for fqn, compute_layout in compute_sharding_by_fqn.items():
        if ".moe.routed_experts." in fqn and compute_layout != per_expert:
            aligned_shardings[fqn] = per_expert
            changed = True
        else:
            aligned_shardings[fqn] = compute_layout
    if not changed:
        return optimizer_config

    optimizers = list(optimizer_config.optimizers)
    optimizers[dist_muon_index] = replace(
        dist_muon_config,
        compute_sharding_by_fqn=aligned_shardings,
    )
    return replace(
        optimizer_config,
        optimizers=optimizers,
    )


@dataclass(kw_only=True, slots=True)
class DistMuonTrainerConfig(Trainer.Config):
    """Trainer config for recipes with ``mla_dist_muon_optimizer``: aligns the
    routed-expert layouts with the final parallelism, and with FlexShard stores
    the matrices DistMuon computes whole on one rank on owners."""

    def __post_init__(self) -> None:
        Trainer.Config.__post_init__(self)
        self.optim.optimizer = align_dist_muon_expert_compute_layouts(
            self.optim.optimizer,
            parallelism=self.parallelism,
        )
        if (
            self.parallelism.fsdp_backend == "flex_shard"
            and not self.parallelism.flex_shard_owned_params
        ):
            self.parallelism.flex_shard_owned_params = list(MLA_DIST_MUON_OWNED_PARAMS)
        # TODO(#3353): Support TP-produced _StridedShard layouts in DistMuon.
        if self.parallelism.tensor_parallel_degree > 1:
            # Fail during config parsing, before TP/FSDP creates _StridedShard
            # storage.
            raise ValueError(
                "DistMuon currently requires "
                "tensor_parallel_degree=1: tensor parallelism can produce "
                "unsupported _StridedShard parameter layouts."
            )
        # No PP gate: DistMuon is PP-safe. The one precondition -- every stage
        # must own at least one transformer layer, or its Muon pattern claims
        # nothing and OptimizersContainer rejects the empty param group -- needs
        # the stage count to approach the layer count. See README, Supported
        # Parallelisms.
