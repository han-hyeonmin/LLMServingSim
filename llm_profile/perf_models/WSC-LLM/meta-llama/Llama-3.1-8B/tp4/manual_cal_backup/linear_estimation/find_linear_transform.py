"""
Find linear transformation y = ax + b between manual_cal and extrapolated latency.
See README.md for details.
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

# ── Configuration ─────────────────────────────────────
root_dir = "/home/members/intern/hanhyunmin/LLMServingSim_2_0/llm_profile/perf_models/WSC-LLM/meta-llama/Llama-3.1-8B/tp4/"
FILE_A = "manual_cal_backup/layers.csv"  # source: manual calculation
FILE_B = "layers.csv"  # target: extrapolated profiling
COL_KEY = "layer_name"
COL_INPUT = "input"
COL_KV = "kv_cache"
COL_TP = "tp_size"
COL_LATENCY = "latency(ns)"

LAYER_ORDER = [
    "embedding",
    "input_layernorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "rope",
    "o_proj",
    "post_layernorm",
    "gate_proj",
    "up_proj",
    "act_fn",
    "down_proj",
    "final_layernorm",
    "lm_head",
]
# ──────────────────────────────────────────────────────

FILE_A = Path(root_dir) / FILE_A
FILE_B = Path(root_dir) / FILE_B


def read_csv(path) -> list[dict]:
    """Return list of row dicts from CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


def validate_schema(rows_a: list, rows_b: list) -> tuple:
    """
    Check that both files share the same (layer_name, input, kv_cache, tp_size) key set.
    Truncates to the shorter file if lengths differ; order differences are ignored.
    Returns (is_valid, rows_a, rows_b).
    """
    if len(rows_a) != len(rows_b):
        n_short = min(len(rows_a), len(rows_b))
        shorter = "FILE_A" if len(rows_a) < len(rows_b) else "FILE_B"
        print(
            f"[Warning] Row count mismatch: FILE_A={len(rows_a)}, FILE_B={len(rows_b)}"
        )
        print(f"          Truncating to {n_short} rows (shorter: {shorter}).")
        rows_a = rows_a[:n_short]
        rows_b = rows_b[:n_short]

    def to_key_set(rows):
        return {(r[COL_KEY], r[COL_INPUT], r[COL_KV], r[COL_TP]) for r in rows}

    only_in_a = to_key_set(rows_a) - to_key_set(rows_b)
    only_in_b = to_key_set(rows_b) - to_key_set(rows_a)

    if only_in_a or only_in_b:
        print("[Validation FAIL] Key set mismatch:")
        for label, keys in [("FILE_A", only_in_a), ("FILE_B", only_in_b)]:
            if keys:
                print(f"  Only in {label} ({len(keys)} keys):")
                for k in sorted(keys)[:5]:
                    print(f"    {k}")
        return False, rows_a, rows_b

    print(f"[Validation OK] Key sets match ({len(to_key_set(rows_a))} unique keys).")
    return True, rows_a, rows_b


def build_latency_dict(rows: list) -> dict:
    """Return dict: (layer_name, input, kv_cache, tp_size) -> latency (float)."""
    data = {}
    for row in rows:
        key = (row[COL_KEY], row[COL_INPUT], row[COL_KV], row[COL_TP])
        try:
            data[key] = float(row[COL_LATENCY])
        except ValueError:
            pass
    return data


def least_squares(xs: list, ys: list) -> tuple:
    """Return (a, b) for y = ax + b via OLS."""
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_xx = sum(x * x for x in xs)
    denom = n * sum_xx - sum_x**2
    if abs(denom) < 1e-12:
        raise ValueError("Denominator is near zero — no variance in data.")
    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n
    return a, b


def r_squared(xs: list, ys: list, a: float, b: float) -> float:
    """Return R2 = 1 - SS_res / SS_tot."""
    y_mean = sum(ys) / len(ys)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0


def main():
    print(f"FILE_A: {FILE_A}")
    print(f"FILE_B: {FILE_B}\n")

    for f in [FILE_A, FILE_B]:
        if not Path(f).exists():
            print(f"[Error] File not found: {f}")
            sys.exit(1)

    rows_a = read_csv(FILE_A)
    rows_b = read_csv(FILE_B)

    print("=" * 52)
    print("  [Step 1] Schema Validation")
    print("=" * 52)
    is_valid, rows_a, rows_b = validate_schema(rows_a, rows_b)
    if not is_valid:
        print("[Aborted] Fix mismatches before fitting.")
        sys.exit(1)
    print()

    data_a = build_latency_dict(rows_a)
    data_b = build_latency_dict(rows_b)

    common_keys = sorted(set(data_a) & set(data_b))
    if not common_keys:
        print("[Error] No common keys found.")
        sys.exit(1)

    xs = [data_a[k] for k in common_keys]
    ys = [data_b[k] for k in common_keys]
    print(f"Matched rows: {len(common_keys)}\n")

    print("=" * 52)
    print("  [Step 2] Global Linear Transform: y = a*x + b")
    print("=" * 52)
    a, b = least_squares(xs, ys)
    r2 = r_squared(xs, ys, a, b)
    print(f"    a  = {a:.6f}")
    print(f"    b  = {b:.2f}")
    print(f"    R2 = {r2:.6f}\n")

    print("=" * 52)
    print("  [Step 3] Per-layer Linear Transform")
    print("=" * 52)
    layer_groups: dict = defaultdict(lambda: ([], []))
    for k in common_keys:
        layer_groups[k[0]][0].append(data_a[k])
        layer_groups[k[0]][1].append(data_b[k])

    ordered = [l for l in LAYER_ORDER if l in layer_groups]
    remaining = [l for l in sorted(layer_groups) if l not in LAYER_ORDER]

    print(f"  {'layer_name':<20} {'a':>10} {'b':>14} {'R2':>8}  {'n':>6}")
    print("  " + "-" * 63)
    for layer in ordered + remaining:
        lx, ly = layer_groups[layer]
        if len(lx) < 2:
            ratio = ly[0] / lx[0] if lx[0] != 0 else float("nan")
            print(f"  {layer:<20} ratio={ratio:>8.4f}  (n=1, b cannot be computed)")
            continue
        try:
            la, lb = least_squares(lx, ly)
            lr2 = r_squared(lx, ly, la, lb)
            print(f"  {layer:<20} {la:>10.4f} {lb:>14.2f} {lr2:>8.4f}  {len(lx):>6}")
        except ValueError as e:
            print(f"  {layer:<20} [Skipped: {e}]")


if __name__ == "__main__":
    main()
