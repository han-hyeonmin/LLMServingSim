# llm_profile

A PyTorch-based profiling tool for measuring LLM layer latencies, attention latencies, and
GPU/system-level power consumption. The outputs are used by LLMServingSim as performance and
power models.

To profile a new model or hardware target for use with LLMServingSim, follow the steps below.
See also the [Adding a New Model & Hardware](../README.md#adding-a-new-model--hardware) section
in the top-level README.

## Overview

`llm_profile` loads models from Hugging Face and inserts PyTorch profiler hooks into key
layers to measure execution time on GPU. It supports dense and MoE architectures and
produces per-layer latency CSVs and a scikit-learn-based attention latency predictor.
GPU and system-level power consumption are measured via `nvidia-smi` and `ipmitool`,
and the results feed into LLMServingSim's power model.

If direct profiling on a target hardware is not possible, use `extrapolate_hw.py` to
derive performance models from an existing hardware profile via hardware spec scaling.
See [Extrapolating to a New Hardware Target](#extrapolating-to-a-new-hardware-target) below.

## Usage

### 1. Environment

Run inside the provided Docker container or a native PyTorch + CUDA environment:

```bash
./docker.sh
```

For models that require access approval (e.g., LLaMA), provide your Hugging Face token
as described in `docker.sh`.

> **Non-Docker (server) setup**: If running directly on a server with a conda environment,
> activate the environment before running any scripts:
>
> ```bash
> source "$(conda info --base)/etc/profile.d/conda.sh"
> conda activate llm_profile
> cd /path/to/LLMServingSim/llm_profile
> ```
>
> Shell scripts in this directory already handle this internally. Do **not** chain
> `conda activate` using `\` — each command must be on a separate line or use `&&`.

### 2. Profile layers and attention

```bash
./profile_layers.sh    # Measures compute latency for non-attention layers
./profile_attn.sh      # Measures attention latency across batch sizes and sequence lengths
```

To reduce profiling time and memory usage, decrease the number of layers via `--num-layer`
in the respective profiling scripts.

### 3. Profile power (optional)

For power measurement, we provide example scripts under `profiler/power/` that use
`nvidia-smi` to measure GPU power consumption and `ipmitool` to measure system-level power:

```bash
./profiler/power/profile_gpu_power.sh      # GPU power via nvidia-smi
./profiler/power/profile_server_power.sh   # System-level power via ipmitool
```

Power profiling results are used by LLMServingSim's power model when a cluster config with
power settings is provided (e.g., `cluster_config/single_node_power_instance.json`).

### 4. Build the attention predictor

```bash
./build_predictor.sh
```

This trains a scikit-learn model on the profiled attention data to support real-time latency
prediction during simulation (`--enable-attn-prediction`). The inference space covered by
the predictor can be controlled via `--max-batch` and `--max-len`.

To build a predictor for an extrapolated hardware target (see below), pass the hardware name:

```bash
./build_predictor.sh --hardware WSC-LLM
```

## Output structure

Results are written to:

```
perf_models/{hardware}/{model}/tp{tp_size}/
  layers.csv                              # Per-layer compute latency
  attention.csv                           # Attention latency by (batch_size, seq_len)
  predictions/
    attn_decode_predictions.csv           # Predictor output for decode attention
    attn_prefill_predictions.csv          # Predictor output for prefill attention
```

These files are loaded automatically by LLMServingSim at runtime.

## Supported models

Model-specific profiling code is located in `models/`:

- `llama.py` — Llama architecture (Llama-3.1-8B, Llama-3.1-70B)
- `mixtral.py` — Mixtral-8x7B (MoE)
- `phimoe.py` — Phi-mini-MoE-instruct (MoE)

## Pre-profiled hardware

The following hardware profiles are included in `perf_models/`:

| Hardware | Models | TP sizes | Notes |
|---|---|---|---|
| `A6000` | Llama-3.1-8B, Llama-3.1-70B | tp1, tp4, tp8 | Upstream |
| `H100` | Llama-3.1-8B, Llama-3.1-70B | tp1, tp4, tp8 | Upstream |
| `TPU-v6e-1` | Llama-3.1-8B | tp1 | Upstream |
| `T-RTX` | Llama-3.1-8B | tp1, tp4 | Added (TITAN RTX) |

`T-RTX` profiles were measured on a TITAN RTX GPU and are used
as the source for WSC-LLM hardware extrapolation.

> **Note**: Raw profiler trace files (`profiler_traces/`) are excluded from the repository
> via `.gitignore` due to file size (~70 MB per trace). Only the summarized CSV outputs
> are committed.

## Extrapolating to a New Hardware Target

When a target hardware is not physically available for profiling, `extrapolate_hw.py`
scales an existing hardware profile to the target spec using hardware performance ratios
(peak FLOPS and memory bandwidth).

### How it works

For each row in `layers.csv` and `attention.csv`, the script applies:

- **Compute-bound layers**: latency scaled by `src_FLOPS / dst_FLOPS`
- **Memory-bound layers**: latency scaled by `src_BW / dst_BW`

The source and target hardware specs are defined in `HARDWARE_SPECS` inside the script.

### Usage

```bash
python extrapolate_hw.py \
  --src-hw T-RTX \
  --dst-hw WSC-LLM \
  --src-dir perf_models/T-RTX/meta-llama/Llama-3.1-8B/tp4/ \
  --dst-dir perf_models/WSC-LLM/meta-llama/Llama-3.1-8B/tp4/
```

| Argument | Description |
|---|---|
| `--src-hw` | Source hardware name (must exist in `HARDWARE_SPECS`) |
| `--dst-hw` | Target hardware name (must exist in `HARDWARE_SPECS`) |
| `--src-dir` | Path to the source `perf_models` directory |
| `--dst-dir` | Output path for the scaled profile |
| `--skip-layers` | Skip `layers.csv` processing |
| `--skip-attention` | Skip `attention.csv` processing |

### End-to-end workflow

```
1. Profile on source HW:
   ./profile_layers.sh && ./profile_attn.sh
   → perf_models/T-RTX/{model}/tp{N}/layers.csv, attention.csv

2. Extrapolate to target HW:
   python extrapolate_hw.py --src-hw T-RTX --dst-hw WSC-LLM ...
   → perf_models/WSC-LLM/{model}/tp{N}/layers.csv, attention.csv

3. (Optional) Build attention predictor for target HW:
   ./build_predictor.sh --hardware WSC-LLM

4. Create cluster_config referencing hardware="WSC-LLM"

5. Run simulation:
   python main.py --cluster-config cluster_config/wsc_llm_example.json ...
```

### Adding a new hardware spec

To add a hardware target to `HARDWARE_SPECS` in `extrapolate_hw.py`:

```python
"MY-HW": {
    "peak_flops_fp16": 312e12,   # peak FP16 FLOPS (e.g., 312 TFLOPS for H100 SXM)
    "mem_bandwidth":   2000e9,   # memory bandwidth in bytes/sec (e.g., 2 TB/s)
},
```

## Adding a new model or hardware

1. Add a model profiling script in `models/` following the existing examples.
2. Set the target hardware name and model identifier in the profiling shell scripts.
3. Run the profiling and predictor build steps above.
4. Create a `cluster_config` entry referencing the new hardware name.

Alternatively, if the hardware is not available, follow the
[extrapolation workflow](#extrapolating-to-a-new-hardware-target) above.