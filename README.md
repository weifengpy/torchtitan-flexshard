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

- Throughput, TFLOPs and MFU are medians over steps 10–40 of one run per
  backend. The per-step ranges overlap: 9,015–9,568 tokens/s for FSDP2 and
  8,997–9,693 for FlexShard.
- Memory is rank 0's peak.
- The profiled step is step 5 on rank 0. FlexShard hides more of its
  communication:
  - all-gathers: 30.3 vs 110.1 ms exposed;
  - reduce-scatters: 8.6 vs 14.3 ms;
  - expert-parallel all-to-all: 220.0 vs 235.4 ms.

Setup:

- 8 NVIDIA H100 GPUs with 8-way data-parallel sharding and 4-way expert
  parallelism; routed experts are sharded over 2-rank meshes.
- The `deepseek_v3_16b` recipe at sequence length 4,096, with 4 sequences per
  GPU per step. It keeps the recipe's chunked loss (`ChunkedLossWrapper`, 8
  chunks), selective activation checkpointing and FlexAttention. CUDA graphs are
  off, since FlexShard doesn't support them yet.
- Real C4 (`allenai/c4`, streamed) and the `deepseek-ai/deepseek-moe-16b-base`
  tokenizer.
- Deterministic mode, which `scripts/loss_compare.py` sets.

## Numerics

`scripts/loss_compare.py --assert-equal --metrics loss,grad_norm` compares
FlexShard against FSDP2. Every run below uses the recipe's chunked loss with
CUDA graphs off, and loss and grad_norm are identical at every step:

| Model | GPUs | Parallelism | Steps |
| --- | --- | --- | --- |
| Llama 3 debug model | 8 | 8-way data-parallel sharding | 100 |
| DeepSeek V3 debug model | 8 | 8-way sharding, 4-way expert parallelism | 100 |
| DeepSeek V3 16B | 8 | 8-way sharding, 4-way expert parallelism | 40 |

`tests/unit_tests/gpu/test_flex_shard.py` checks bitwise equality on 4 GPUs,
from initialization on. It covers each `fsdp_reshard_after_forward` policy,
gradient accumulation, expert parallelism and the chunked loss.

## Reproducing the DeepSeek V3 16B comparison

Requirements:

- A PyTorch build with FSDP's native collective copies, from
  [pytorch/pytorch#197204](https://github.com/pytorch/pytorch/pull/197204) and
  [pytorch/pytorch#200179](https://github.com/pytorch/pytorch/pull/200179). Any
  nightly from 2.16.0.dev20261009 on has both.
- FlexShard with meta-pytorch/flex_shard#51 through
  [#58](https://github.com/meta-pytorch/flex_shard/pull/58), until they land:

  ```bash
  pip install torchao
  pip install --no-deps "git+https://github.com/meta-pytorch/flex_shard.git@gh/weifengpy/48/head"
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
expert parallelism. It supports activation checkpointing, local compile
regions, FlexAttention and the chunked loss. It doesn't yet support:

- checkpointing, CUDA graphs, CPU offload, EMA, DistMuon and SPMD type checking;
- replicated data parallelism, and tensor, context or pipeline parallelism.

## License

torchtitan is BSD 3-Clause licensed, as found in the [LICENSE](./LICENSE) file.
