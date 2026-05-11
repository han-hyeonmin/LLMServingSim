# outputs

Simulator result files and post-simulation analysis tools.

## Contents

| Path | Purpose |
| --- | --- |
| `*.csv` | Per-request metrics from `serving/__main__.py` (`--output`). One row per completed request. |
| `*_timeseries.csv` | Per-iteration timeseries written alongside the per-request CSV (P1 patch). One row per instance per heartbeat. |
| `convert_sim_output.py` | Post-processing: converts per-request CSV to benchmark-comparable format (decode token counts, ns → ms/s). See `[output/README.md](../output/README.md)`. |
| `analyze_steady_state.py` | Steady-state window detection on the P1 timeseries CSV. Primary analysis tool. |
| `v0/` | Historical artifacts from before the P1 patch. Includes `analyze_steady_state_v0.py` (reconstructs concurrent decode from per-request ITL lists). Frozen — do not use the v0 tool on new simulation outputs. |

## Per-request CSV

Written by every scheduler when `--output <path>` is set. Columns:

| Column | Unit | Meaning |
| --- | --- | --- |
| `instance id` | — | Scheduler instance that recorded this request (not the cluster-config index — see note below). |
| `request id` | — | Unique request id from the input dataset. |
| `input` | tokens | Prompt length. |
| `output` | tokens | **Decode tokens only** (not input + decode). |
| `arrival` | ns | Request arrival time in the simulator clock. |
| `end_time` | ns | Final decode token completion time. |
| `latency` | ns | `end_time - arrival`. |
| `queuing_delay` | ns | Time spent in the waiting queue before prefill started. |
| `TTFT` | ns | Time to first token (prefill completion). Measured at the simulator level — does not include client-side delivery latency. |
| `TPOT` | ns | Mean inter-token latency during decode. Integer division of `(latency - TTFT) // (output - 1)`. |
| `ITL` | list[ns] | Per-token inter-token latencies. `len(ITL) == output - 1`. `sum(ITL) == latency - TTFT` exactly. |

The `output` column changed semantics in the P1 series: it now records the number
of generated tokens (decode only), not `input + decode` as in earlier v0 outputs.
The two analysis scripts encode this — use `analyze_steady_state.py` for current
outputs and `v0/analyze_steady_state_v0.py` only for files in `v0/`.

## Per-iteration timeseries CSV (P1)

Written alongside the per-request CSV. Suffix: `_timeseries.csv`. One row per
instance per heartbeat (default 1 s, controlled by `--log-interval`).

| Column | Meaning |
| --- | --- |
| `time_ns`, `time_s` | Simulator clock at the heartbeat. |
| `instance_id` | Cluster-config instance index. |
| `pd_type` | `prefill` / `decode` / `unified` (from cluster config). |
| `node_id` | Cluster-config node index that owns this instance. |
| `running_total` | Inflight requests on this instance. |
| `running_prefill` | Subset of `running_total` still in prefill phase. |
| `running_decode` | Subset of `running_total` in decode phase. |
| `waiting` | Requests whose arrival has passed but are not yet inflight. |
| `npu_used_mb` | NPU memory used (MB). |
| `npu_util_pct` | NPU memory utilization (%). |

`pd_type` resolves the instance-id mapping at source. In a PD-disaggregated
setup, filter `pd_type == "decode"` to get the decode-pool timeseries. Note
that the per-request CSV's `instance id` column records the **prefill-pool**
instance that handled the request (where it was first scheduled), so the two
ids do not generally line up across files.

## Steady-state analysis

The goal is to measure performance only in the steady-state window — after
warmup, before drain — when the system is processing requests with overlapping
decode (the "full batch" window).

### Algorithm

1. Aggregate `running_decode` across decode-pool instances at each heartbeat.
2. Detect the longest contiguous window where `concurrent_decode >= peak * theta`
   (default `theta = 0.9`). Short dips below threshold (shorter than `hysteresis_s`)
   are absorbed so transient stalls don't fragment the window.
3. Inside the window, compute throughput and per-request statistics (TPOT, TTFT
   percentiles) using only requests that both started and ended within the window.

### Usage

```bash
python outputs/analyze_steady_state.py \
    outputs/<run>_timeseries.csv \
    --per-request-csv outputs/<run>.csv \
    --plot --verbose
```

Output files (next to the timeseries CSV unless `--out-dir` is set):

- `<run>_decode_pool_ts.csv` — aggregated decode-pool timeseries.
- `<run>_steady_summary.json` — window bounds, metrics, and pool composition.
- `<run>_decode_pool_ts.png` — diagnostic plot (concurrent decode + waiting + window).

Common flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--theta` | 0.9 | Steady-state threshold as a fraction of peak `concurrent_decode`. |
| `--hysteresis-s` | 2.0 | Absorb dips below threshold up to this duration. |
| `--instance-filter` | auto | Restrict to specific `instance_id`s. Default: auto-pick instances with `pd_type == "decode"`. |
| `--cross-validate` | off | Build the v0-style ITL-reconstructed timeseries from the per-request CSV and compare peak / mean against the recorded timeseries. Methods are expected to agree within ~20 %. |

### When the window is too short

If the steady window is shorter than the typical per-request decode duration,
no request fits entirely inside the window and TPOT statistics are reported
as `None`. This is the dominant failure mode at small request counts. The
tool warns explicitly when this happens.

To get a meaningful window, either increase `--num-reqs` (so the queue stays
non-empty long enough to form a plateau), or set `--max-num-seqs` to constrain
the pool capacity (which creates a hard ceiling on `concurrent_decode` and
prolongs the time spent at that ceiling).

## Notes

- `analyze_steady_state_v0.py` reconstructs the decode-pool timeseries from
  per-request ITL lists. It exists because v0 simulations predate the P1
  timeseries patch. For new simulations, prefer the timeseries-based tool —
  it sees inflight requests directly (no completion gating) and resolves
  `pd_type` ambiguity automatically.
- Cross-validation (the `--cross-validate` flag) was verified on a 100-request
  smoke test: timeseries-direct peak matched the ITL-reconstructed peak
  exactly (13 vs 13, 0 % difference), and the means agreed within 3.5 %.