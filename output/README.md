# output/

Simulation results and post-processing scripts for LLMServingSim.

## Directory structure

```
output/
  *.csv                     # Per-request results written by main.py --output
  convert_sim_output.py     # Converts simulation CSV to benchmark comparison format
```

---

## convert_sim_output.py

Converts a `main.py` output CSV into a format comparable with a reference benchmark table.

### Column mapping

| Output column | Unit | Derivation |
|---|---|---|
| `L_prefill` | tokens | `input` |
| `L_decode` | tokens | `output - input` |
| `Decode start (s)` | s | `(arrival + TTFT) × 1e-9` |
| `Decode end (s)` | s | `end_time × 1e-9` |
| `Decode time (ms)` | ms | `(end_time - arrival - TTFT) × 1e-6` |
| `stall_total (ms)` | ms | `queuing_delay × 1e-6` |

> All time values in the input CSV are in nanoseconds (ns), as written by `scheduler.py::save_output`.

### Usage

```bash
# Default: writes *_converted.csv in the same directory
python output/convert_sim_output.py output/example_run.csv

# Specify output path
python output/convert_sim_output.py output/example_run.csv -o output/result_bench.csv

# Change sort column (default: request id)
python output/convert_sim_output.py output/example_run.csv --sort-by "Decode start (s)"
```

### Notes

- The `output` column must be a cumulative token count (input + decode tokens).
  If not, rows with `L_decode < 0` will be flagged with a warning.
- The script exits with an error if any of the required columns
  (`input`, `output`, `arrival`, `end_time`, `queuing_delay`, `TTFT`) are missing.