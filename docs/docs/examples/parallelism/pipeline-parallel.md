---
title: Pipeline parallel (PP)
sidebar_position: 2
---

# Pipeline parallel (PP)

> **What this demonstrates:** splitting a model's decoder layers
> across GPUs (one stage per GPU) so each iteration streams as
> micro-batches through the pipeline.

PP is the orthogonal axis to TP: TP shards weights *within* a
layer, PP shards layers *across* devices. Each GPU runs a
contiguous stretch of the decoder block stack and hands the
intermediate activations to the next stage.

LLMServingSim models PP at the **scheduler level**: the scheduler
keeps an `inflight` list of batches currently traversing the
pipeline, capped at `pp_size`. When the pipeline is full, the
scheduler returns `None` until ASTRA-Sim drains a stage. This
matches how production training frameworks (e.g., Megatron-LM)
stream micro-batches.

> ⚠️ **Caveat: PP currently affects scheduling only.** The trace
> generator still emits the full model per iteration, so
> inter-stage forwarding latency is not modeled in detail. PP
> scheduling depth prevents over-issuing batches but stage-to-stage
> activation shipment cost is not yet broken out. Treat PP results
> as a **lower bound** on real overhead. Tracked for a future
> release.

## Prerequisites

- Simulator container set up
- Bundled RTXPRO6000 profile for `meta-llama/Llama-3.1-8B`

## Cluster config

There's no PP-specific bundled config; pipeline-parallel runs by
flipping `pp_size` on a multi-GPU instance. Drop a
`single_node_pp_instance.json` next to the others:

```json title="configs/cluster/single_node_pp_instance.json"
{
  "num_nodes": 1,
  "link_bw": 16,
  "link_latency": 20000,
  "nodes": [
    {
      "num_instances": 1,
      "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0},
      "instances": [
        {
          "model_name": "meta-llama/Llama-3.1-8B",
          "hardware": "RTXPRO6000",
          "npu_mem": {"mem_size": 96, "mem_bw": 1597, "mem_latency": 0},
          "num_npus": 2,
          "tp_size": 1,
          "pp_size": 2,
          "pd_type": null
        }
      ]
    }
  ]
}
```

The two fields that matter:

- `num_npus: 2`, `tp_size: 1`, `pp_size: 2`: invariant is
  `num_npus = tp_size * pp_size`, so the simulator splits the model
  into two pipeline stages (each on its own GPU) with no TP within a
  stage.
- For combined TP × PP (e.g., 4 GPUs as `tp=2, pp=2`), set
  `num_npus: 4, tp_size: 2, pp_size: 2`.

## Run

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_pp_instance.json' \
  --dtype float16 --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/pp2_run.csv' \
  --log-interval 1.0
```

No new CLI flag, the parallelism degree is fully driven by the
cluster config.

## Expected output

The throughput log looks like a standard single-instance run:

```text
[INFO] step=20 batch=8 prompt_t=1.4k tok/s decode_t=540 tok/s npu_mem=44.0 GB
[INFO] step=21 batch=8 prompt_t=1.5k tok/s decode_t=560 tok/s npu_mem=44.1 GB
```

Two things to notice vs. the TP=1 baseline:

- **`npu_mem` is roughly halved** (each GPU holds half the
  decoder layers, so weights + KV cache per device shrink).
- **`batch` may saturate at lower values** during short bursts
  because the scheduler stops issuing once `inflight == pp_size`,
  this is the back-pressure that prevents over-injecting work into
  the pipeline.

## What's interesting

- **Memory split is real.** Even though forwarding cost isn't
  modeled in detail, the per-stage weight footprint *is*
  accurate: PP=2 lets you fit a model that doesn't fit on TP=1.
  Use this to study capacity, not latency.
- **Pipeline depth caps in-flight batches.** `inflight ≤ pp_size`
  is the only PP-driven scheduling constraint today. With
  `pp_size=2` and a token budget that allows 6 batches, you'll see
  the scheduler queue at most 2 batches in the pipeline at once.
- **No bubble modeling yet.** Real PP suffers from pipeline
  bubbles (idle stages while the pipeline fills/drains). The
  current PP path doesn't expose those, so steady-state numbers are
  optimistic.

## Related examples

- **[Tensor parallel](./tensor-parallel)**: the within-layer
  counterpart. TP × PP combinations are valid and common at
  large scale.
- **[Multi-instance LOAD routing](../disaggregated/multi-instance)**:
  the next-level-up scaling — replicate whole TP × PP groups
  across instances.

## Where to learn more

- **[Simulator → Parallelism mechanics](/docs/simulator/parallelism-mechanics)**:
  how `num_npus`, `tp_size`, and `pp_size` are validated and
  threaded through the scheduler / trace generator.
- The PP `inflight` list lives in `serving/core/scheduler.py`.
