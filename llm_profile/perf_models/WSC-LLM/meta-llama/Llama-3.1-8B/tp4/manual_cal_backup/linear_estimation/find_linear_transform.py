"""
find_linear_transform.py

Fit y = ax + b (OLS) between two layers.csv latency tables:
  FILE_A : manual calculation  (source)
  FILE_B : extrapolated profiling via extrapolate_hw.py  (target)

See README.md for full pipeline context and result interpretation.

Configuration
-------------
Set root_dir, FILE_A, FILE_B in the "Configuration" block below.
root_dir accepts:
  - An absolute path       : "/data/perf_models/WSC-LLM/.../tp4"
  - A relative path        : "../../perf_models/WSC-LLM/.../tp4"
  - An environment variable: os.environ.get("PERF_MODEL_DIR", ".")
The resolved paths are never printed; only FILE_A and FILE_B labels appear
in all output so that internal server paths are not exposed.
"""

import csv
import sys
from pathlib import Path

# ── Configuration ─────────────────────────────────────
# Paths are resolved relative to this script's location so that no absolute
# server path ever appears in source code.
#
# Directory layout assumed:
#   tp4/
#   ├── layers.csv                         ← FILE_B
#   └── manual_cal_backup/
#       ├── layers.csv                     ← FILE_A
#       └── linear_estimation/
#           └── find_linear_transform.py  ← this file (__file__)
_HERE = Path(__file__).resolve().parent  # linear_estimation/
FILE_A = _HERE / ".." / "layers.csv"  # manual_cal_backup/layers.csv
FILE_B = _HERE / ".." / ".." / "layers.csv"  # tp4/layers.csv
# ──────────────────────────────────────────────────────

# Human-readable labels used in all print() calls instead of raw paths.
LABEL_A = "FILE_A (manual_cal)"
LABEL_B = "FILE_B (extrapolated)"

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


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def build_latency_dict(path: Path) -> dict:
    """Return {(layer_name, input, kv_cache, tp_size): latency_ns} from CSV."""
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["layer_name"].strip(),
                row["input"].strip(),
                row["kv_cache"].strip(),
                row["tp_size"].strip(),
            )
            try:
                data[key] = float(row["latency(ns)"])
            except (ValueError, KeyError):
                pass
    return data


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_schema(path_a: Path, path_b: Path):
    """
    Check that both CSVs cover the same (layer_name, input, kv_cache, tp_size)
    key set. Truncate to the shorter file before comparison.
    Abort on mismatch; return (rows_a, rows_b) on success.
    """

    def read_rows(p):
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    rows_a = read_rows(path_a)
    rows_b = read_rows(path_b)

    n = min(len(rows_a), len(rows_b))
    if len(rows_a) != len(rows_b):
        print(
            f"[Warning] Row count differs: {LABEL_A}={len(rows_a)}, "
            f"{LABEL_B}={len(rows_b)} — truncating to {n} rows."
        )
    rows_a, rows_b = rows_a[:n], rows_b[:n]

    key_cols = ["layer_name", "input", "kv_cache", "tp_size"]

    def make_key(row):
        return tuple(row[c].strip() for c in key_cols)

    keys_a = {make_key(r) for r in rows_a}
    keys_b = {make_key(r) for r in rows_b}

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    if only_a or only_b:
        print("[Error] Key sets do not match after truncation.")
        if only_a:
            print(f"  Keys only in {LABEL_A} (up to 5): {sorted(only_a)[:5]}")
        if only_b:
            print(f"  Keys only in {LABEL_B} (up to 5): {sorted(only_b)[:5]}")
        sys.exit(1)

    return True, rows_a, rows_b


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def least_squares(xs: list, ys: list) -> tuple:
    """Return (a, b) for y = ax + b via OLS."""
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx**2
    if abs(denom) < 1e-12:
        raise ValueError("denominator near zero — no variance in x")
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return a, b


def r_squared(xs, ys, a, b) -> float:
    y_mean = sum(ys) / len(ys)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # File existence check — use labels, not raw paths
    for label, path in [(LABEL_A, FILE_A), (LABEL_B, FILE_B)]:
        if not path.exists():
            print(f"[Error] {label} not found.")
            sys.exit(1)

    _, rows_a, rows_b = validate_schema(FILE_A, FILE_B)

    dict_a = build_latency_dict(FILE_A)
    dict_b = build_latency_dict(FILE_B)

    common = sorted(set(dict_a) & set(dict_b))
    if not common:
        print("[Error] No common keys between the two files.")
        sys.exit(1)

    xs = [dict_a[k] for k in common]
    ys = [dict_b[k] for k in common]

    # ── Step 1: Global fit ────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Global linear fit  ({LABEL_A} → {LABEL_B})")
    print(f"  n = {len(xs)} matched rows")
    print(f"{'='*60}")
    try:
        a, b = least_squares(xs, ys)
        r2 = r_squared(xs, ys, a, b)
        print(f"  y = {a:.4f}·x + {b:.2f}    R² = {r2:.4f}")
    except ValueError as e:
        print(f"  [Skipped: {e}]")

    # ── Step 2: Per-layer fit ─────────────────────────
    from collections import defaultdict

    layer_groups: dict = defaultdict(lambda: ([], []))
    for k, x in zip(common, xs):
        layer_name = k[0]
        layer_groups[layer_name][0].append(x)
        layer_groups[layer_name][1].append(dict_b[k])

    print(f"\n{'='*60}")
    print(f"  Per-layer linear fit")
    print(f"{'='*60}")

    ordered = [l for l in LAYER_ORDER if l in layer_groups]
    remaining = sorted(l for l in layer_groups if l not in LAYER_ORDER)

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
