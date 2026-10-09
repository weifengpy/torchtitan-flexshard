# torchtitan with FlexShard

This fork of [pytorch/torchtitan](https://github.com/pytorch/torchtitan) adds
`parallelism.fsdp_backend`, which shards a model either with FSDP2 (`"fsdp2"`,
the default) or with [FlexShard](https://github.com/meta-pytorch/flex_shard)
(`"flex_shard"`). Training with FlexShard is bitwise identical to training with
FSDP2. This page compares the two.

## DeepSeek V3 16B on 8 H100s

| | FSDP2 | FlexShard |
| --- | --- | --- |
| Throughput, tokens/s per GPU | 9,281 | 9,456 (+1.9%) |
| TFLOPs per GPU (MFU) | 168.0 (17.0%) | 171.2 (17.3%) |
| Peak active memory | 48.09 GiB | 48.09 GiB |
| Peak reserved memory | 57.49 GiB | 57.43 GiB |
| GPU time of a profiled step | 1,664 ms | 1,622 ms |
| Exposed communication in that step | 340.5 ms | 265.0 ms |
| Loss and grad_norm, 40 steps | | bitwise equal to FSDP2 |

- `scripts/loss_compare.py --assert-equal --metrics loss,grad_norm` finds
  loss and grad_norm identical at every one of the 40 steps.
- Throughput, TFLOPs and MFU are medians over steps 10–40 of one run per
  backend. The per-step ranges overlap: 9,015–9,568 tokens/s for FSDP2 and
  8,997–9,693 for FlexShard.
- Memory is rank 0's peak.
- The profiled step is step 5 on rank 0. FlexShard hides more of its
  communication:
  - all-gathers: 30.3 vs 110.1 ms exposed;
  - reduce-scatters: 8.6 vs 14.3 ms;
  - expert-parallel all-to-all: 220.0 vs 235.4 ms.

### Configuration

Both backends run the `deepseek_v3_16b` recipe with the same settings; only
`parallelism.fsdp_backend` differs.

| Setting | Value |
| --- | --- |
| Hardware | 8 NVIDIA H100 GPUs, one node |
| Model | DeepSeek V3 16B: 15.7B parameters; 27 layers, the first dense and 26 MoE; dim 2,048; multi-head latent attention; per MoE layer, 64 routed experts (top-6, sigmoid routing) and 2 shared experts; vocabulary 102,400 |
| Data-parallel sharding | 8-way (`data_parallel_shard_degree=8`), no replication. Dense parameters are sharded over all 8 GPUs. |
| Expert parallelism | 4-way (`expert_parallel_degree=4`; the recipe's default is 8), with the all-to-all token dispatcher. Each expert-parallel rank owns 16 of the 64 routed experts, and their parameters are sharded over a 2-GPU mesh (`edp_shard`), so each GPU stores half of them. |
| Tensor, context, pipeline parallelism | None |
| Sharding groups | One per FSDP2 `fully_shard` group: the embedding; the final norm with lm_head; the dense layer; and, for each MoE layer, its dense parameters on the 8-GPU mesh and its routed experts on the 2-GPU mesh. FlexShard uses `Shard` placements on the same dims as FSDP2, with FSDP2's parameter order (`fsdp2_compatible=True`). |
| Reshard after forward | `default` policy: the embedding and transformer layers reshard after forward; the norm and lm_head group doesn't. |
| Prefetch | FSDP2's explicit expert-parallel schedule: each layer prefetches the next in forward and the previous in backward. |
| Mixed precision | bf16 parameters, fp32 gradient reduction; gradients are summed without division. |
| Loss | `ChunkedLossWrapper` with 8 chunks: each GPU's 16,384 tokens go through lm_head and cross-entropy (over the full 102,400-token vocabulary) in chunks of 2,048 tokens, each with its own backward. The norm and lm_head group stays gathered across the chunks. lm_head's gradient is reduce-scattered once, at the last chunk, and the norm's in the backward through the decoder. |
| Batch | 4 sequences of 4,096 tokens per GPU per step: 16,384 tokens per GPU and 131,072 per step, without gradient accumulation. The recipe's 16,384-token sequences ran out of memory on FSDP2 with plain cross-entropy. |
| Activation checkpointing | Per-op selective (`SelectiveAC`) |
| Compile | Local regions (`loss`, `fused_binary_activation`, `fp32_to_bf16_split`) and FlexAttention |
| CUDA graphs | Off: with expert parallelism, torchtitan supports them only with the HybridEP token dispatcher. |
| Data | C4 (`allenai/c4`, streamed), with the `deepseek-ai/deepseek-moe-16b-base` tokenizer |
| Optimizer | AdamW, learning rate 2.2e-4, warming up over all 40 steps |
| Determinism | On, as `scripts/loss_compare.py` sets it |

## Reproducing the DeepSeek V3 16B comparison

Requirements:

- A PyTorch build with FSDP's native collective copies, from
  [pytorch/pytorch#197204](https://github.com/pytorch/pytorch/pull/197204) and
  [pytorch/pytorch#200179](https://github.com/pytorch/pytorch/pull/200179). Any
  nightly from 2.16.0.dev20261009 on has both.
- FlexShard with meta-pytorch/flex_shard#51 through
  [#65](https://github.com/meta-pytorch/flex_shard/pull/65), until they land:

  ```bash
  pip install torchao
  pip install --no-deps "git+https://github.com/meta-pytorch/flex_shard.git@gh/weifengpy/55/head"
  ```

Download the tokenizer:

```bash
python scripts/download_hf_assets.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer
```

Save the two configs as `flex_shard_16b.py`:

```python
import dataclasses

from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_16b


def _config(backend: str):
    config = deepseek_v3_16b(seq_len=4096)
    config.training.disable_cuda_graphs = True
    config.parallelism = dataclasses.replace(
        config.parallelism, expert_parallel_degree=4, fsdp_backend=backend
    )
    return config


def fsdp2():
    return _config("fsdp2")


def flex_shard():
    return _config("flex_shard")
```

Then run both backends and compare them step by step:

```bash
PYTHONPATH=. python scripts/loss_compare.py . . \
    --baseline-module flex_shard_16b --baseline-config fsdp2 \
    --test-config flex_shard \
    --assert-equal --metrics loss,grad_norm --no-seed-checkpoint --steps 40
```

## Scope

The flex_shard backend shards over `data_parallel_shard_degree` only, with
expert, context, tensor and pipeline parallelism. It supports CUDA
graphs, activation checkpointing, local compile regions, FlexAttention, the
chunked loss, DeepSeek V3's multi-token prediction, SPMD type checking
(`debug.spmd_typechecking`), and checkpointing with `CheckpointManager`, whose
checkpoints load across the two backends. CUDA graphs are tested without
expert parallelism; with it, they need the HybridEP token dispatcher, which
hasn't been tested with FlexShard yet. It doesn't yet support:

- CPU offload, EMA and DistMuon;
- Hugging Face checkpoint conversion, `async_with_pinned_mem` checkpointing and
  the `torch_checkpointing` checkpointer;
- replicated data parallelism.

## License

torchtitan is BSD 3-Clause licensed, as found in the [LICENSE](./LICENSE) file.
