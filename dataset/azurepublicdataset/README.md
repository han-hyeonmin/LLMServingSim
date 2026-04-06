# Azure Trace Parser

Converts [Azure LLM Inference Public Trace](https://github.com/Azure/AzurePublicDataset) CSV
files into the `.jsonl` workload trace format consumed by LLMServingSim.

## Quick Start

```bash
# Convert the conversation trace (all valid requests)
python dataset/azurepublicdataset/azure_trace_parser.py conv

# Convert the code trace, limited to 100 requests
python dataset/azurepublicdataset/azure_trace_parser.py code --max-requests 100
```

You will be prompted for a Hugging Face token (required for the gated
Llama-3.1-8B tokenizer).

## Arguments

|Argument              |Default|Description                                       |
|----------------------|-------|--------------------------------------------------|
|`dataset` (positional)|—      |`conv` or `code`                                  |
|`--max-requests N`    |all    |Cap the number of output requests                 |
|`--max-input-tokens N`|2048   |Drop requests whose `input_toks > N`              |
|`--max-total-tokens N`|4096   |Drop requests whose `input_toks + output_toks > N`|

### How to choose `--max-input-tokens`

This value must satisfy **both** of the following constraints simultaneously:

```
max-input-tokens  ≤  main.py --max-num-batched-tokens   (scheduler upper bound)
max-input-tokens  ≤  layer.sh --max-len                  (layers.csv coverage)
```

If either constraint is violated:

- `> max-num-batched-tokens` → the scheduler removes the request from every
  batch, causing it to stall in the queue forever.
- `> layers.csv --max-len` → `_get_perf_row()` raises a `KeyError` because
  the exact-match lookup finds no entry for that `input_len`.

With the current project defaults (`--max-num-batched-tokens 2048`,
`--max-len 6000`), the effective ceiling is `min(2048, 6000) = 2048`.

## Output

Files are written to the `dataset/` directory (one level above the script).

|Example command            |Output file                                  |
|---------------------------|---------------------------------------------|
|`… conv`                   |`dataset/azure_trace_conv_llama.jsonl`       |
|`… conv --max-requests 100`|`dataset/azure_trace_conv_req100_llama.jsonl`|
|`… code --max-requests 500`|`dataset/azure_trace_code_req500_llama.jsonl`|

Each line is a JSON object:

```json
{
    "input_toks":      128,
    "output_toks":     64,
    "arrival_time_ns": 350000000,
    "input_tok_ids":   [91012, 4837, ...],
    "output_tok_ids":  [7231, 55902, ...]
}
```

|Field            |Type     |Description                                                   |
|-----------------|---------|--------------------------------------------------------------|
|`input_toks`     |int      |Number of prompt tokens                                       |
|`output_toks`    |int      |Number of generated tokens                                    |
|`arrival_time_ns`|int      |Arrival time relative to the first request (nanoseconds)      |
|`input_tok_ids`  |List[int]|Dummy token IDs in `[0, vocab_size)` for prefix cache matching|
|`output_tok_ids` |List[int]|Dummy token IDs in `[0, vocab_size)`                          |

## Request Filtering

Requests from the raw CSV pass through three filters in order:

```
raw CSV row
  │
  ├── input_toks ≤ 0  or  output_toks ≤ 0     →  SKIP  (data quality)
  ├── input_toks > --max-input-tokens          →  SKIP  (scheduler / profile coverage)
  ├── input_toks + output_toks > --max-total   →  SKIP  (simulator OOM prevention)
  │
  └── PASS  →  written to .jsonl
```

A breakdown of skip reasons is printed at the end of each run:

```
Done. 847 requests written, 153 skipped.
  Skipped breakdown:
    non-positive tokens   : 2
    input >  2048         : 89
    input+output >  4096  : 62
```

## Differences from `sharegpt_parser.py`

|Aspect             |`sharegpt_parser.py`                              |`azure_trace_parser.py`                            |
|-------------------|--------------------------------------------------|---------------------------------------------------|
|Arrival time       |Synthetic Poisson process                         |Real timestamps from Azure trace                   |
|Token lengths      |From actual ShareGPT conversations or fixed-length|From Azure CSV (`ContextTokens`, `GeneratedTokens`)|
|Token ID generation|`random.randint(0, vocab_size-1)`                 |Same                                               |
|Vocab size         |`tokenizer.vocab_size`                            |Same (128256 for Llama-3.1-8B)                     |
|Input length filter|N/A (controlled by `--fix-input`)                 |`--max-input-tokens`                               |

## Directory Layout

```
dataset/
├── azurepublicdataset/
│   ├── azure_trace_parser.py            ← this script
│   ├── README.md                        ← this file
│   ├── AzureLLMInferenceTrace_conv.csv
│   └── AzureLLMInferenceTrace_code.csv
├── azure_trace_conv_llama.jsonl         ← output
├── azure_trace_conv_req100_llama.jsonl  ← output (capped)
└── sharegpt_req100_rate10_llama.jsonl   ← sharegpt output (for comparison)
```

## End-to-End Example

```bash
# 1. Generate the trace
python dataset/azurepublicdataset/azure_trace_parser.py conv \
    --max-requests 100 \
    --max-input-tokens 2048

# 2. Run the simulator
python main.py \
    --cluster-config cluster_config/dojo_pd_3node.json \
    --dataset dataset/azure_trace_conv_req100_llama.jsonl \
    --max-num-batched-tokens 2048 \
    --max-batch 5 \
    --num-req 100 \
    --log-level INFO
```