# FlexShard

FlexShard provides PyTorch-native building blocks for running optimizer compute
with layouts that differ from persistent DTensor parameter storage layouts. It
plans packed storage-to-compute redistribution and overlaps communication with
optimizer work. DistMuon is its initial consumer.

## Public API

The public API is exported from `torchtitan.distributed.flex_shard`:

- `ComputeLayout` describes temporary optimizer-compute sharding on named
  `DeviceMesh` axes using PyTorch DTensor placements plus `BlockShard` or
  `Owned`, and optionally the order in which several axes shard one tensor
  dimension.
- `ComputeLayout.compute_dtype` is the dtype the optimizer computes in for
  that parameter; `None` keeps the storage dtype. DistMuon runs Newton-Schulz
  in BF16 and accepts only `torch.bfloat16`, so every FP32 parameter sets it.
  Inputs are sent to the compute ranks and directions are sent back in this
  dtype.
- `BlockShard` shards complete contiguous blocks along one tensor dimension. It
  preserves the tensor's rank and global shape and never creates a tensor view.
- `Owned` assigns a complete subgroup-local logical tensor to one dynamically
  selected rank for the compute phase.
- `BucketConfig` groups and orders parameters by fully qualified name for
  packed redistribution and communication-compute overlap.
- `DistMuon(..., redistribution_meshes=...)` takes caller-created 1D process
  groups and gives each bucket the one whose ranks match the storage axes its
  transport spans.
- `BlockShard.block_sizes` is a nonempty tuple describing a repeating sequence
  of independently shardable block sizes. Use `(R,)` for uniform blocks.
  For example, `(128, 64)` partitions
  384 rows into four blocks of 128, 64, 128, and 64 rows. Blocks are distributed
  by count, so ranks can own different numbers of rows. DistMuon runs
  Newton-Schulz and the aspect-ratio learning-rate adjustment independently
  for each block. Kimi's shared `wkv_a` projection uses `(512, 64)` to distribute
  its KV latent and RoPE key matrices separately.
- `DistMuon` consumes optimizer-agnostic per-parameter `ComputeLayout`
  values in `compute_sharding_by_fqn`. DistMuon's `BlockShard` path accepts
  only a 2D parameter with contiguous local DTensor storage. The placement
  must target tensor dimension 0; the leading dimension must be nonzero and
  divisible by the sum of `block_sizes`.
  With `block_sizes=(R,)`, each consecutive `R` rows forms one independent
  `[R, C]` matrix. Multiple block sizes allow these matrices to have different
  row counts. FlexShard routes the flat 2D compute tensor using the block
  boundaries. DistMuon's planning code constructs zero-copy strided
  `[M, R, C]` views directly from each rank's compute shape and logical starting
  row. For `(R,)`, the view is equivalent to unflattening the row dimension.
  A native matrix batch `[..., R, C]` uses `Shard(0)` to distribute its
  outermost batch dimension, or `Owned` to assign the complete batch to one
  rank. A single 2D matrix without
  `BlockShard` uses whole-matrix compute such as `Owned`. The builder validates
  named DTensor parameters and plans their storage-to-compute transitions.

Storage placements describe persistent ownership only; they do not define
Muon matrix boundaries. Flat matrix-batch compute may use `BlockShard` on one
or more non-unit mesh axes. Callers pass a 1D process group over those axes
in `redistribution_meshes`. FlexShard routes complete matrices across it and restores the
original DTensor storage layout after the update. This supports combined DP
and TP storage, including the strided shards produced when TP and FSDP shard
the same tensor dimension. Non-transport mesh axes retain their storage
placement.

Native `[..., R, C]` parameters can redistribute `Replicate()`, a shard of the
matrix-row dimension, or a shard of the matrix-column dimension to `Shard(0)`
compute on one mesh axis, with every other non-unit storage mesh axis
preserving its existing batch ownership. Changed matrix axes may also combine
into one transport group; complete matrices are assigned across that group.
For example, expert storage `Shard(0), Shard(1)` on DP/TP preserves each DP
rank's expert range while TP redistributes matrix rows to whole-expert compute.

### Redistribution process groups

FlexShard never creates process groups. For each bucket, `DistMuon` resolves
the storage axes its transport spans (e.g. `Owned` on a replicated axis still
transports over it, and `BlockShard` rows already aligned with storage do
not) and uses the `redistribution_meshes` entry whose ranks equal those axes'
ranks in storage order; mesh names are not read. Without a match, a
single-axis transport reuses its storage group and a combined transport
raises `ValueError`. Two matches also raise.

```python
# (dp_shard=2, tp=2) storage, compute {"dp_shard": Owned(), "tp": Owned()}:
# the transport spans both axes, so it needs their ranks as one group.
ranks = parameter.device_mesh["dp_shard", "tp"].mesh.flatten().tolist()
group = dist.new_group(ranks)  # collective over WORLD
mesh = DeviceMesh.from_group(
    group, parameter.device_mesh.device_type, mesh=torch.tensor(ranks)
)
optimizer = DistMuon(
    param_groups,
    compute_sharding_by_fqn=compute_sharding_by_fqn,
    bucket_configs=bucket_configs,
    redistribution_meshes=[mesh],
)
```

`dist.new_group` must be called by every rank in the same order, so create
every group on every rank, including groups a pipeline stage does not use.
A group with the same ranks as a storage axis gives that axis's transports
their own communicator.

In TorchTitan, `ParallelismContext.get_redistribution_meshes` creates these
groups once on every rank (with the NCCL CTA cap, as dedicated copies of the
storage axes too), and the `DistMuon` wrapper passes them in.

Several mesh axes may shard the same tensor dimension. By default they apply
in storage-mesh order; `shard_order_by_tensor_dim` states a different order,
outermost axis first. For example, preserving an EP-axis `Shard(0)` while
repartitioning its local expert domain over a preceding `edp_shard` axis uses
`Shard(0)` on both axes with `shard_order_by_tensor_dim={0: ("ep", "edp_shard")}`.
FlexShard derives each axis's split factor from the bound mesh, then lowers the
`edp_shard` placement to subgroup-local `Shard(0)` for optimizer execution.

Compute sharding is construction-time configuration. It is validated and
frozen when the optimizer is built, but is not stored in its state dict;
checkpoint restore must rebuild the optimizer with matching values.

## TorchTitan Kimi integration

The [Kimi test recipes](../../../torchtitan_recipes/tests/models/kimi_k2_7.py)
provide the first TorchTitan integration. Their shared optimizer configuration:

- Selects matrix parameters from attention, dense MLPs, routed and shared
  experts, and routers for DistMuon. Other parameters continue to use
  AdamW.
- Defines each selected parameter's compute layout, including per-head Muon
  for compatible attention projections.
- Groups layers into buckets so compute-ready work can overlap packed
  redistribution.

The same construction can be used by Kimi-family, Kimi-VL, and Moonlight
recipes.

## Package boundary

FlexShard currently lives in TorchTitan while its API matures. We intend to
annex this directory into a standalone Python package and repository.

Keep this directory self-contained and PyTorch-only. Dependencies should flow
in one direction:

```text
TorchTitan components and models -> FlexShard -> PyTorch
```

FlexShard must not depend on TorchTitan model or training infrastructure.
