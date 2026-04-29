# find_linear_transform.py

## Overview

This script quantifies the linear relationship between two `layers.csv` latency tables:

| File | Description |
|------|-------------|
| `FILE_A` (`manual_cal_backup/layers.csv`) | **Manual calculation** — latency estimated analytically from hardware specs (FLOP count, memory access pattern) without actual measurement |
| `FILE_B` (`layers.csv`) | **Extrapolated profiling** — latency measured on NVIDIA TITAN RTX, then scaled to the WSC-LLM (Dojo-style die) target via Roofline-based ratio extrapolation (`extrapolate_hw.py`) |

The goal is to find a per-layer affine mapping `y = a·x + b` that converts manual-cal latency into extrapolated latency, and to assess how well manual calculation approximates the profiling-based ground truth.

---

## Background

### Pipeline context

```
TITAN RTX profiling
    └─ profile_layer.sh  →  layers.csv  (measured, GPU)
            │
            ▼
    extrapolate_hw.py
            │   scale factor = WSC-LLM TFLOPS / TITAN RTX TFLOPS  (compute-bound layers)
            │                  WSC-LLM BW     / TITAN RTX BW       (memory-bound layers)
            ▼
    layers.csv  (FILE_B — extrapolated, WSC-LLM target)

Manual calculation
    └─ Roofline model  →  manual_cal_backup/layers.csv  (FILE_A)
```

### Why compare the two?

Manual calculation and hardware extrapolation both target the same WSC-LLM hardware but use different estimation paths. A close linear fit (`R2 ≈ 1`, `a ≈ 1`, `b ≈ 0`) indicates the two methods agree; systematic deviation in `a` or `b` reveals a consistent bias in one of the estimation paths.

---

## Input Format

Both CSV files share the schema:

```
layer_name, input, kv_cache, tp_size, latency(ns)
embedding,  1,     0,        4,       2641
input_layernorm, 1, 0,       4,       17783
...
```

- `input` — number of input tokens (sequence length dimension)
- `kv_cache` — KV cache size
- `tp_size` — tensor parallelism degree (fixed at 4 for WSC-LLM)
- `latency(ns)` — layer latency in nanoseconds

---

## What the Script Does

### Step 1 — Schema Validation

Checks that both files cover the same set of `(layer_name, input, kv_cache, tp_size)` keys.

- If file lengths differ, **truncates to the shorter file** before comparison.
- Key-set comparison is **order-insensitive**: known scheduling anomalies (e.g., `o_proj` / `post_layernorm` order swap at `input` 2–16) are safely ignored.
- Aborts if the key sets do not match.

### Step 2 — Global Linear Regression

Fits a single `y = a·x + b` across all matched rows using ordinary least squares (OLS). Useful as a quick sanity check but is expected to have lower R2 due to heterogeneous layer types.

### Step 3 — Per-layer Linear Regression

Fits an independent `y = a·x + b` for each layer type, ordered as they appear in the transformer forward pass:

```
embedding → input_layernorm → q_proj → k_proj → v_proj → rope
→ o_proj → post_layernorm → gate_proj → up_proj → act_fn
→ down_proj → final_layernorm → lm_head
```

Reports `a`, `b`, `R2`, and sample count `n` per layer.

---

## Configuration

Edit the top of the script:

```python
root_dir = "/home/members/intern/hanhyunmin/LLMServingSim_2_0/llm_profile/perf_models/WSC-LLM/meta-llama/Llama-3.1-8B/tp4/"
FILE_A   = "manual_cal_backup/layers.csv"   # source: manual calculation
FILE_B   = "layers.csv"                     # target: extrapolated profiling
```

Paths are resolved as `Path(root_dir) / FILE_A` unconditionally.

---

## Usage

```bash
python find_linear_transform.py
```

No external dependencies — standard library only (`csv`, `pathlib`, `collections`).

---

## Interpreting Results

| Metric | Ideal | Interpretation |
|--------|-------|----------------|
| `R2 ≈ 1.0` | ✅ | Linear model explains variance well |
| `a ≈ 1, b ≈ 0` | ✅ | Manual cal and extrapolation agree |
| `a >> 1` | ⚠️ | Manual cal underestimates latency |
| `b >> 0` | ⚠️ | Constant offset — likely fixed overhead not captured in manual cal |
| `R2 < 0.9` | ❌ | Non-linear relationship; per-layer fit or additional features needed |

### Known patterns (Llama-3.1-8B, TP=4, WSC-LLM)

- **Norm layers** (`input_layernorm`, `post_layernorm`, `final_layernorm`): `a ≈ 49` — manual cal significantly underestimates, likely because normalization is memory-bound and manual cal uses compute-bound assumptions.
- **Projection layers** (`q/k/v/o_proj`, `gate/up/down_proj`): `a ≈ 2.3–2.6` — consistent scaling group.
- **`lm_head`**: large constant offset (`b ≈ 330k ns`) due to vocabulary size (128,256 tokens) dominating output projection cost.
- **`k_proj`, `v_proj`**: slightly lower R2 (~0.97) — KV cache size may introduce additional variance not captured by input token count alone.
