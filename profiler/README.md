# llm_profile (v2 — LLMServingSim 2.0)

> **Note**: This directory (`llm_profile/`) is the legacy profiler kept for reference.
> LLMServingSim 2.0 uses the vLLM-based profiler in `profiler/`. See `profiler/README.md`
> for the current profiling workflow.

---

A PyTorch-based profiling tool for measuring LLM layer latencies, attention latencies, and
GPU/system-level power consumption. The outputs are used by LLMServingSim as performance and
power models.

## Overview

`llm_profile` loads models from Hugging Face and inserts PyTorch profiler hooks into key
layers to measure execution time on GPU. It supports dense and MoE architectures and
produces per-layer latency CSVs consumed by LLMServingSim's trace generator.

If direct profiling on a target hardware is not possible, use `extrapolate_hw.py` to
derive performance models from an existing hardware profile via hardware spec scaling.
See [Extrapolating to a New Hardware Target](#extrapolating-to-a-new-hardware-target) below.

## Profiler output (LLMServingSim 2.0 schema)

The current profiler (`profiler/`) writes results to:

```
profiler/perf/{hardware}/{model}/{variant}/
  meta.yaml                               # Profiler metadata (required by simulator)
  tp{N}/
    dense.csv          layer, tokens, time_us
    per_sequence.csv   layer, sequences, time_us
    attention.csv      prefill_chunk, kv_prefill, n_decode, kv_decode, time_us
    skew.csv           heterogeneous-decode sweep (optional)
    skew_fit.csv       per-bucket alpha table (optional)
```

These files are loaded automatically by `serving/core/trace_generator.py` at runtime.

### Running the profiler

```bash
# Edit MODEL / HARDWARE / TP_DEGREES in profiler/profile.sh, then:
./profiler/profile.sh
```

Key variables in `profiler/profile.sh`:

| Variable | Description | Current setting |
|---|---|---|
| `MODEL` | HF model id | `meta-llama/Llama-3.1-8B` |
| `HARDWARE` | Output folder name under `profiler/perf/` | `T-RTX` |
| `TP_DEGREES` | Comma-separated TP list | `1,4` |
| `MAX_NUM_BATCHED_TOKENS` | vLLM batch token cap | `2048` |
| `SKIP_SKEW` | Skip heterogeneous-decode sweep | unset (enabled) |

See `profiler/README.md` for full documentation.

## Pre-profiled hardware

| Hardware | Models | TP sizes | Notes |
|---|---|---|---|
| `A6000` | Llama-3.1-8B, Llama-3.1-70B | tp1, tp4, tp8 | Upstream |
| `H100` | Llama-3.1-8B, Llama-3.1-70B | tp1, tp4, tp8 | Upstream |
| `T-RTX` | Llama-3.1-8B | tp1, tp4 | TITAN RTX — source for WSC-LLM extrapolation |

> **Note**: `T-RTX` profiles are the source hardware for WSC-LLM extrapolation.
> Run `extrapolate_hw.py` after profiling to produce `profiler/perf/WSC-LLM/`.

## Extrapolating to a New Hardware Target

When a target hardware is not physically available for profiling, `extrapolate_hw.py`
scales an existing hardware profile to the target spec using hardware performance ratios.

### How it works

For each CSV in the source `tp<N>/` directories, the script applies:

- **Compute-bound** (`dense.csv`, `per_sequence.csv`): `time_us *= src_TFLOPS / dst_TFLOPS`
- **Memory-bound** (`attention.csv`): `time_us *= src_BW / dst_BW`
- **`meta.yaml`**: copied and `hardware` field patched to destination name
- **`skew.csv` / `skew_fit.csv`**: copied without scaling (alpha shape is hardware-invariant)

Hardware specs are defined in `KNOWN_HARDWARE` inside the script.

### Usage

```bash
# T-RTX → WSC-LLM (all TP degrees under the bf16 variant)
python profiler/extrapolate_hw.py \
    --src-hw T-RTX \
    --src-variant-dir profiler/perf/T-RTX/meta-llama/Llama-3.1-8B/bf16 \
    --dst-hw WSC-LLM \
    --dst-variant-dir profiler/perf/WSC-LLM/meta-llama/Llama-3.1-8B/bf16
```

| Argument | Description |
|---|---|
| `--src-hw` | Source hardware name (must exist in `KNOWN_HARDWARE`) |
| `--dst-hw` | Destination hardware name (must exist in `KNOWN_HARDWARE`) |
| `--src-variant-dir` | Path to `profiler/perf/<hw>/<model>/<variant>/` |
| `--dst-variant-dir` | Output path (created automatically) |
| `--tp N` | Process only TP=N (default: all `tp<N>/` subdirs) |
| `--skip-dense` | Skip `dense.csv` |
| `--skip-per-sequence` | Skip `per_sequence.csv` |
| `--skip-attention` | Skip `attention.csv` |
| `--dry-run` | Print scaling info without writing files |

### End-to-end workflow

```
1. Profile on source HW (TITAN RTX):
   ./profiler/profile.sh
   → profiler/perf/T-RTX/meta-llama/Llama-3.1-8B/bf16/tp{1,4}/
     {dense.csv, per_sequence.csv, attention.csv, skew.csv, skew_fit.csv, meta.yaml}

2. Extrapolate to target HW (WSC-LLM):
   python extrapolate_hw.py \
       --src-hw T-RTX \
       --src-variant-dir profiler/perf/T-RTX/meta-llama/Llama-3.1-8B/bf16 \
       --dst-hw WSC-LLM \
       --dst-variant-dir profiler/perf/WSC-LLM/meta-llama/Llama-3.1-8B/bf16
   → profiler/perf/WSC-LLM/meta-llama/Llama-3.1-8B/bf16/tp{1,4}/
     {dense.csv, per_sequence.csv, attention.csv, skew.csv, skew_fit.csv}
   → profiler/perf/WSC-LLM/meta-llama/Llama-3.1-8B/bf16/meta.yaml (hardware patched)

3. Set cluster config hardware field:
   "hardware": "WSC-LLM"

4. Run simulation:
   python -m serving \
       --cluster-config configs/cluster/<your_config>.json \
       --dataset workloads/azure_trace_conv_llama.jsonl
```

### Adding a new hardware spec

Add an entry to `KNOWN_HARDWARE` in `extrapolate_hw.py`:

```python
"MY-HW": HardwareSpec(
    name="MY-HW",
    TFLOPs=312.0,        # FP16 peak TFLOPS
    memory_bw_GBps=2000.0,
),
```

Or pass specs directly via CLI without editing the script:

```bash
python extrapolate_hw.py \
    --src-hw T-RTX \
    --src-variant-dir profiler/perf/T-RTX/meta-llama/Llama-3.1-8B/bf16 \
    --dst-hw CustomDie \
    --dst-variant-dir profiler/perf/CustomDie/meta-llama/Llama-3.1-8B/bf16 \
    --dst-tflops 50.0 \
    --dst-mem-bw 2000.0
```