# FlexShard + DistMuon against FSDP2 + DistMuon: benchmark tools

Tools behind a benchmark of torchtitan's DistMuon on two data-parallel backends, on DeepSeek V3 671B with 256 H100s:

- **FSDP2 + DistMuon**: parameters are `Shard(0)` DTensors. Every optimizer step, DistMuon exchanges each Muon matrix with the rank that runs its Newton-Schulz.
- **FlexShard + DistMuon without optimizer communication**: FlexShard stores each Muon matrix whole on that rank, so the step runs locally.

The scripts read torch.profiler traces and torchtitan's structured logs, model both optimizer steps at other data-parallel sizes, and draw the figures. They're analysis only: nothing in torchtitan imports them. Some docstrings call the 256-GPU benchmark "gate 5".

## Setup measured

- DeepSeek V3 671B as defined (61 layers, 256 routed experts), no multi-token prediction module.
- 256 H100 80 GB, 32 hosts of 8 GPUs. Pipeline parallelism 4 with 4 virtual stages each (Interleaved1F1B), data parallelism 64, expert parallelism 64.
- 32 microbatches of one 4,096-token sequence per data-parallel rank. Full activation checkpointing. BF16 compute and gradient reduction, FP32 master weights.
- Muon with Kimi's units (`mla_dist_muon_optimizer`) for the matrices, fused AdamW for the rest. Balanced round-robin routing, no gradient clipping, deterministic, 10 steps, step 4 profiled on every rank.
- FSDP2 + DistMuon runs on FlexShard's layout: the same owners, units and batches as FlexShard (`flex_shard_layout=True`, owners from `flex_shard_owner_map`). FlexShard owns every block parameter and uses the grouped owner-row transport.
- Code: weifengpy/torchtitan-flexshard#23, #24 and #25, plus a follow-up commit for FlexShard's layout under FSDP2 (`Owned` blocks, fixed owners); meta-pytorch/flex_shard#76 and #77.
- Two jobs per backend. Each pair repeats bit for bit in losses and grad norms, and its optimizer time within 2-6 ms.

## Inputs

Each job's output directory holds:

- `profiling/iteration_4/rank<N>_trace.json.gz`: one rank's torch.profiler trace of step 4.
- `structured_logs/training.global_rank_<N>.*.jsonl`: every rank's step, forward-backward and optimizer events.
- `logs/`: the hosts' stdout, with every rank's device memory at step 10.

## Trace and log readers

| Script | Reads | Prints |
|---|---|---|
| `step_gpu_phases.py TRACE` | one rank's trace | the GPU step in phases (before the pipeline, the pipeline, the optimizer, after it) and the MoE all-to-all kernels |
| `optim_kernels_by_step.py TRACE` | one rank's trace | kernel time inside each DistMuon step: NCCL, matmuls, copies, other |
| `optim_gpu_bubbles.py TRACE` | one rank's trace | the optimizer window's busy and idle time, its gaps, and how far ahead the CPU launched |
| `optim_cpu_blocking.py TRACE` | one rank's trace | where the CPU blocks during DistMuon's steps |
| `fb_actions.py TRACE` | one rank's trace | pipeline actions and the compute stream's idle time, by what ran meanwhile |
| `fb_kernels.py TRACE OUT.json` | one rank's trace | the compute stream's kernel time by class inside the pipeline window |
| `extract_optim_kernels.py TRACE OUT.json` | one rank's trace | the optimizer window's kernels as JSON, for Figure 2 |
| `gate5_summary_events.py EVENTS.txt` | step events from every rank's structured log | step, forward-backward and optimizer spans: the slowest rank per step, then the median of steps 3 and 5-10 |
| `gate5_memory.py OUTPUT_DIR` | the hosts' stdout | device memory at step 10: max, median and the top ranks |

`gate5_summary_events.py` reads pre-extracted events, which is much faster than reading every log file in turn from network storage:

```bash
ls OUTPUT_DIR/structured_logs/*.jsonl \
  | xargs -P 32 grep -H -E '"log_type_name": "(step|forward_backward|optim_step)_(start|end)"' > EVENTS.txt
python gate5_summary_events.py EVENTS.txt
```

## Model

```bash
PYTHONPATH=<flex_shard checkout>/src python dist_muon_roofline.py
```

A discrete-event model of both optimizer steps on pipeline rank 0 as data parallelism grows, with FlexShard's real owner placement (so it needs flex_shard). It prints:

- the cost per element of Newton-Schulz and of the exchange, by Muon unit;
- both backends' optimizer time for the slowest rank at data parallelism 8 to 512, with FSDP2 + DistMuon's extra time split into exchange, per-model-part sync and bucket coupling;
- per model part and per rank at data parallelism 64, to check against traces.

Switches:

- `A2A_RULE=barrier`: the first version's all-to-all rule, where every rank leaves together after the last rank joins. The default lets each transfer start once both of its ranks have joined.
- `PARTS=...`: pipeline rank 0's model parts for another pipeline layout, for example `0-2,7-10,15-18,23-26,31-34,39-42,47-50,55-58` for pipeline parallelism 2.

## Figures

The figure scripts need matplotlib and write a PNG to the working directory. Numbers are inline unless noted.

| Figure | Script | Input |
|---|---|---|
| 1. Optimizer step speed as the cluster grows | `fig1_optimizer_speed.py` | the model's numbers and the measured slowest ranks |
| 2. The optimizer step on the GPU | `fig2_optimizer_timeline.py` | `extract_optim_kernels.py` output in the working directory: `fsdp2_strict_r0.json`, `fsdp2_strict_r1.json`, `fsdp2_strict_r2.json` and `flex_r0.json` |
| 3. Forward and backward by component | `fig3_forward_backward.py` | `fb_kernels.py` and `fb_actions.py` numbers for two jobs per backend |
| 4. Where one Muon matrix lives and moves | `fig4_dataflow.py` | none |
| 5. A made-up example of the synchronization cost | `fig5_toy_sync.py` | none |
| 6. FSDP2 + DistMuon's extra optimizer time, by cause | `fig6_extra_time.py` | the model's numbers |
