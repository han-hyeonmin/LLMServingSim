#!/usr/bin/env python3
"""
Extrapolate profiling data to a new hardware target using ratio-based scaling.

New profiler schema (LLMServingSim 2.0):
  profiler/perf/<hw>/<model>/<variant>/tp<N>/
    dense.csv          layer, tokens, time_us          (compute-bound)
    per_sequence.csv   layer, sequences, time_us       (compute-bound)
    attention.csv      prefill_chunk, kv_prefill, n_decode, kv_decode, time_us  (memory-bound)
    meta.yaml          profiler metadata (required by simulator)

Scaling rules:
  - dense.csv / per_sequence.csv : time_us ∝ 1 / TFLOPS  (compute-bound)
  - attention.csv                : time_us ∝ 1 / mem_BW  (memory-bound)
  - meta.yaml                    : copied and hardware field patched
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Hardware specs
# ---------------------------------------------------------------------------


@dataclass
class HardwareSpec:
    """Hardware specification for ratio-based extrapolation."""

    name: str
    TFLOPs: float  # FP16 peak TFLOPS
    memory_bw_GBps: float  # Memory bandwidth (GB/s)

    # Metadata — not used in scaling
    core_count: int = 0
    TFLOPs_per_core: float = 0.0


KNOWN_HARDWARE: dict[str, HardwareSpec] = {
    # NVIDIA TITAN RTX  (TU102 · FP16 Tensor Core ~32.6 TFLOPS · 672 GB/s GDDR6)
    "T-RTX": HardwareSpec(
        name="T-RTX",
        TFLOPs=32.6,
        memory_bw_GBps=672.0,
        core_count=72,
    ),
    # Dojo-style compute die  (36 cores × 1.02 TFLOPS = 36.72 TFLOPS · 3.35 TB/s)
    "WSC-LLM": HardwareSpec(
        name="WSC-LLM",
        TFLOPs=1.02 * (6**2),  # 36.72 TFLOPS
        memory_bw_GBps=3.35 * 1000,  # 3350 GB/s
        core_count=6**2,
        TFLOPs_per_core=1.02,
    ),
}


def load_hw_spec_from_yaml(path: str) -> dict[str, HardwareSpec]:
    """Load hardware specs from a YAML config file."""
    if not HAS_YAML:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    specs = {}
    for hw_name, hw_cfg in cfg.get("hardware", {}).items():
        specs[hw_name] = HardwareSpec(
            name=hw_name,
            TFLOPs=float(hw_cfg["TFLOPs"]),
            memory_bw_GBps=float(hw_cfg["memory_bw_GBps"]),
            core_count=int(hw_cfg.get("core_count", 0)),
            TFLOPs_per_core=float(hw_cfg.get("TFLOPs_per_core", 0.0)),
        )
    return specs


def get_hw_spec(name: str, extra_specs: Optional[dict] = None) -> HardwareSpec:
    catalog = dict(KNOWN_HARDWARE)
    if extra_specs:
        catalog.update(extra_specs)
    if name not in catalog:
        available = ", ".join(sorted(catalog.keys()))
        raise KeyError(f"Unknown hardware '{name}'. Available: {available}")
    return catalog[name]


# ---------------------------------------------------------------------------
# CSV scaling helpers — new schema (time_us column)
# ---------------------------------------------------------------------------


def scale_compute_csv(
    src_path: str,
    dst_path: str,
    src_spec: HardwareSpec,
    dst_spec: HardwareSpec,
    label: str,
) -> None:
    """Scale a compute-bound CSV (dense.csv or per_sequence.csv).

    Schema: layer, <key_col>, time_us
    Scaling: time_us *= src_TFLOPS / dst_TFLOPS
    """
    df = pd.read_csv(src_path)

    if "time_us" not in df.columns:
        raise KeyError(
            f"Expected 'time_us' column in {src_path}. " f"Found: {list(df.columns)}"
        )

    ratio = src_spec.TFLOPs / dst_spec.TFLOPs
    print(
        f"  [{label}] Compute ratio: {src_spec.TFLOPs:.2f} / {dst_spec.TFLOPs:.2f} = {ratio:.4f}"
    )
    print(f"  [{label}] Rows: {len(df)}")

    df["time_us"] = df["time_us"].astype(float) * ratio

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    df.to_csv(dst_path, index=False)
    print(f"  [{label}] Written → {dst_path}")


def scale_attention_csv(
    src_path: str,
    dst_path: str,
    src_spec: HardwareSpec,
    dst_spec: HardwareSpec,
) -> None:
    """Scale attention.csv (memory-bound).

    Schema: prefill_chunk, kv_prefill, n_decode, kv_decode, time_us
    Scaling: time_us *= src_BW / dst_BW
    """
    df = pd.read_csv(src_path)

    if "time_us" not in df.columns:
        raise KeyError(
            f"Expected 'time_us' column in {src_path}. " f"Found: {list(df.columns)}"
        )

    # Validate expected key columns are present
    expected_keys = {"prefill_chunk", "kv_prefill", "n_decode", "kv_decode"}
    missing = expected_keys - set(df.columns)
    if missing:
        raise KeyError(f"Missing attention key columns in {src_path}: {missing}")

    ratio = src_spec.memory_bw_GBps / dst_spec.memory_bw_GBps
    print(
        f"  [attention.csv] BW ratio: {src_spec.memory_bw_GBps:.1f} / {dst_spec.memory_bw_GBps:.1f} = {ratio:.4f}"
    )
    print(f"  [attention.csv] Rows: {len(df)}")

    df["time_us"] = df["time_us"].astype(float) * ratio

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    df.to_csv(dst_path, index=False)
    print(f"  [attention.csv] Written → {dst_path}")


def patch_meta_yaml(
    src_path: str,
    dst_path: str,
    dst_hw_name: str,
) -> None:
    """Copy meta.yaml from src and patch the hardware field.

    meta.yaml is required by trace_generator.py (_load_meta).
    The hardware field is updated to match the destination hardware name.
    All other fields (engine_effective, attention_grid, skew_fit, etc.)
    are preserved as-is — they describe the profiling conditions, which
    remain valid after ratio-based scaling.
    """
    if not HAS_YAML:
        raise ImportError("PyYAML is required: pip install pyyaml")

    with open(src_path) as f:
        meta = yaml.safe_load(f)

    meta["hardware"] = dst_hw_name

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        yaml.dump(meta, f, sort_keys=False)
    print(f"  [meta.yaml] Patched hardware='{dst_hw_name}' → {dst_path}")


# ---------------------------------------------------------------------------
# Per-TP directory processing
# ---------------------------------------------------------------------------


def process_tp_dir(
    src_tp_dir: str,
    dst_tp_dir: str,
    src_spec: HardwareSpec,
    dst_spec: HardwareSpec,
    skip_dense: bool,
    skip_per_sequence: bool,
    skip_attention: bool,
    dry_run: bool,
) -> None:
    """Process all CSVs under one tp<N>/ directory."""
    if dry_run:
        print(f"  [DRY RUN] Would process: {src_tp_dir} → {dst_tp_dir}")
        return

    # dense.csv — compute-bound
    if not skip_dense:
        src = os.path.join(src_tp_dir, "dense.csv")
        dst = os.path.join(dst_tp_dir, "dense.csv")
        if os.path.isfile(src):
            scale_compute_csv(src, dst, src_spec, dst_spec, "dense.csv")
        else:
            print(f"  WARNING: {src} not found — skipping.")

    # per_sequence.csv — compute-bound
    if not skip_per_sequence:
        src = os.path.join(src_tp_dir, "per_sequence.csv")
        dst = os.path.join(dst_tp_dir, "per_sequence.csv")
        if os.path.isfile(src):
            scale_compute_csv(src, dst, src_spec, dst_spec, "per_sequence.csv")
        else:
            print(f"  WARNING: {src} not found — skipping.")

    # attention.csv — memory-bound
    if not skip_attention:
        src = os.path.join(src_tp_dir, "attention.csv")
        dst = os.path.join(dst_tp_dir, "attention.csv")
        if os.path.isfile(src):
            scale_attention_csv(src, dst, src_spec, dst_spec)
        else:
            print(f"  WARNING: {src} not found — skipping.")

    # moe.csv — compute-bound (copy as-is if present; MoE not in Llama-3.1-8B)
    src_moe = os.path.join(src_tp_dir, "moe.csv")
    if os.path.isfile(src_moe):
        dst_moe = os.path.join(dst_tp_dir, "moe.csv")
        scale_compute_csv(src_moe, dst_moe, src_spec, dst_spec, "moe.csv")

    # skew.csv and skew_fit.csv — copy without scaling.
    # These encode the heterogeneous-decode alpha correction shape, which
    # depends on the attention kernel's skew pattern, not absolute latency.
    # The simulator uses them only for the alpha blend, not for raw latency
    # values, so ratio-based rescaling would corrupt the fit.
    for fname in ("skew.csv", "skew_fit.csv"):
        src_f = os.path.join(src_tp_dir, fname)
        if os.path.isfile(src_f):
            dst_f = os.path.join(dst_tp_dir, fname)
            os.makedirs(dst_tp_dir, exist_ok=True)
            shutil.copy2(src_f, dst_f)
            print(f"  [{fname}] Copied (alpha shape preserved) → {dst_f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrapolate LLMServingSim profiler data to a new hardware target.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # T-RTX → WSC-LLM, all TP degrees
  python extrapolate_hw.py \\
      --src-hw T-RTX \\
      --src-variant-dir profiler/perf/T-RTX/meta-llama/Llama-3.1-8B/bf16 \\
      --dst-hw WSC-LLM \\
      --dst-variant-dir profiler/perf/WSC-LLM/meta-llama/Llama-3.1-8B/bf16

  # Single TP directory (legacy-style invocation)
  python extrapolate_hw.py \\
      --src-hw T-RTX \\
      --src-variant-dir profiler/perf/T-RTX/meta-llama/Llama-3.1-8B/bf16 \\
      --dst-hw WSC-LLM \\
      --dst-variant-dir profiler/perf/WSC-LLM/meta-llama/Llama-3.1-8B/bf16 \\
      --tp 4
        """,
    )

    parser.add_argument(
        "--src-hw", required=True, help="Source hardware name (e.g., T-RTX)"
    )
    parser.add_argument(
        "--src-variant-dir",
        required=True,
        help="Path to source variant dir: profiler/perf/<hw>/<model>/<variant>/",
    )
    parser.add_argument(
        "--dst-hw", required=True, help="Destination hardware name (e.g., WSC-LLM)"
    )
    parser.add_argument(
        "--dst-variant-dir",
        required=True,
        help="Path to destination variant dir (will be created)",
    )

    # Optional: process only a specific TP
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Process only this TP degree. Default: all tp<N>/ subdirs.",
    )

    # Hardware spec overrides
    parser.add_argument(
        "--config", default=None, help="YAML file with custom hardware specs"
    )
    parser.add_argument("--src-tflops", type=float, default=None)
    parser.add_argument("--src-mem-bw", type=float, default=None)
    parser.add_argument("--dst-tflops", type=float, default=None)
    parser.add_argument("--dst-mem-bw", type=float, default=None)

    # Skip flags
    parser.add_argument(
        "--skip-dense", action="store_true", help="Skip dense.csv extrapolation"
    )
    parser.add_argument(
        "--skip-per-sequence",
        action="store_true",
        help="Skip per_sequence.csv extrapolation",
    )
    parser.add_argument(
        "--skip-attention", action="store_true", help="Skip attention.csv extrapolation"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scaling info without writing files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load specs
    extra_specs = {}
    if args.config:
        extra_specs = load_hw_spec_from_yaml(args.config)

    try:
        src_spec = get_hw_spec(args.src_hw, extra_specs)
    except KeyError:
        if args.src_tflops and args.src_mem_bw:
            src_spec = HardwareSpec(args.src_hw, args.src_tflops, args.src_mem_bw)
        else:
            print(
                f"ERROR: Unknown source hardware '{args.src_hw}'. "
                f"Use --src-tflops / --src-mem-bw or --config."
            )
            sys.exit(1)

    try:
        dst_spec = get_hw_spec(args.dst_hw, extra_specs)
    except KeyError:
        if args.dst_tflops and args.dst_mem_bw:
            dst_spec = HardwareSpec(args.dst_hw, args.dst_tflops, args.dst_mem_bw)
        else:
            print(
                f"ERROR: Unknown destination hardware '{args.dst_hw}'. "
                f"Use --dst-tflops / --dst-mem-bw or --config."
            )
            sys.exit(1)

    # CLI overrides
    if args.src_tflops is not None:
        src_spec.TFLOPs = args.src_tflops
    if args.src_mem_bw is not None:
        src_spec.memory_bw_GBps = args.src_mem_bw
    if args.dst_tflops is not None:
        dst_spec.TFLOPs = args.dst_tflops
    if args.dst_mem_bw is not None:
        dst_spec.memory_bw_GBps = args.dst_mem_bw

    src_variant = Path(args.src_variant_dir)
    dst_variant = Path(args.dst_variant_dir)

    # Print summary
    print("=" * 65)
    print("  Hardware Extrapolation Summary")
    print("=" * 65)
    print(f"  Source      : {src_spec.name}")
    print(f"    TFLOPS    : {src_spec.TFLOPs:.2f}")
    print(f"    Mem BW    : {src_spec.memory_bw_GBps:.1f} GB/s")
    print(f"    Src dir   : {src_variant}")
    print()
    print(f"  Destination : {dst_spec.name}")
    print(f"    TFLOPS    : {dst_spec.TFLOPs:.2f}")
    print(f"    Mem BW    : {dst_spec.memory_bw_GBps:.1f} GB/s")
    print(f"    Dst dir   : {dst_variant}")
    print()
    print(f"  Compute ratio (src/dst) : {src_spec.TFLOPs / dst_spec.TFLOPs:.4f}")
    print(
        f"  BW ratio      (src/dst) : {src_spec.memory_bw_GBps / dst_spec.memory_bw_GBps:.4f}"
    )
    print("=" * 65)

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Copy + patch meta.yaml (variant-level, not per-TP)
    src_meta = src_variant / "meta.yaml"
    dst_meta = dst_variant / "meta.yaml"
    if src_meta.is_file():
        print(f"\nProcessing meta.yaml ...")
        patch_meta_yaml(str(src_meta), str(dst_meta), dst_spec.name)
    else:
        print(f"\nWARNING: meta.yaml not found at {src_meta} — skipping.")

    # Collect TP directories to process
    if args.tp is not None:
        tp_dirs = [(f"tp{args.tp}", src_variant / f"tp{args.tp}")]
    else:
        tp_dirs = sorted(
            [
                (entry.name, entry)
                for entry in src_variant.iterdir()
                if entry.is_dir() and entry.name.startswith("tp")
            ],
            key=lambda x: int(x[0][2:]) if x[0][2:].isdigit() else 0,
        )

    if not tp_dirs:
        print(f"\nERROR: No tp<N>/ directories found under {src_variant}")
        sys.exit(1)

    for tp_name, src_tp_dir in tp_dirs:
        dst_tp_dir = dst_variant / tp_name
        print(f"\n--- {tp_name} ---")
        process_tp_dir(
            str(src_tp_dir),
            str(dst_tp_dir),
            src_spec,
            dst_spec,
            skip_dense=args.skip_dense,
            skip_per_sequence=args.skip_per_sequence,
            skip_attention=args.skip_attention,
            dry_run=False,
        )

    print(f"\nDone. Output written to: {dst_variant}/")
    print(f"Next steps:")
    print(f'  1. Ensure cluster config has "hardware": "{dst_spec.name}"')
    print(
        f"  2. Run: python -m serving --cluster-config configs/cluster/<your_config>.json ..."
    )


if __name__ == "__main__":
    main()
