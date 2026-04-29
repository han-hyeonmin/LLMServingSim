#!/usr/bin/env python3
"""
Extrapolate profiling data (layers.csv, attention.csv) to a new hardware target
using ratio-based scaling:
  - layers.csv   : latency ∝ 1 / TFLOPS  (compute-bound)
  - attention.csv: latency ∝ 1 / mem_BW  (memory-bound)

Output is a drop-in replacement under perf_models/{dst_hw}/{model}/tp{N}/.
Run with --help for usage examples.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class HardwareSpec:
    """Hardware specification for ratio-based extrapolation."""

    name: str
    TFLOPs: float  # FP16 peak TFLOPS
    memory_bw_GBps: float  # Memory bandwidth (GB/s)

    # Additive channel latency offset for attention (us). 0.0 = disabled.
    channel_latency_prefill_us: float = 0.0
    channel_latency_decode_us: float = 0.0

    # Metadata — not used in scaling
    core_count: int = 0
    TFLOPs_per_core: float = 0.0


KNOWN_HARDWARE: dict[str, HardwareSpec] = {
    # NVIDIA TITAN RTX  (TU102 · FP16 Tensor Core ~32.6 TFLOPS · 24 GB GDDR6 672 GB/s)
    "TITAN_RTX": HardwareSpec(
        name="TITAN_RTX",
        TFLOPs=32.6,
        memory_bw_GBps=672.0,
        core_count=72,
    ),
    # Dojo-style compute die  (36 cores × 1.02 TFLOPS = 36.72 TFLOPS · 3.35 TB/s)
    "Dojo_Die": HardwareSpec(
        name="Dojo_Die",
        TFLOPs=1.02 * (6**2),
        memory_bw_GBps=3.35 * 1000,
        channel_latency_prefill_us=0.4,
        channel_latency_decode_us=0.1,
        core_count=6**2,
        TFLOPs_per_core=1.02,
    ),
}


def load_hw_spec_from_yaml(path: str) -> dict[str, HardwareSpec]:
    """Load hardware specs from a YAML config file (see --help for format)."""
    if not HAS_YAML:
        raise ImportError(
            "PyYAML is required to load config files. "
            "Install with: pip install pyyaml"
        )
    with open(path) as f:
        cfg = yaml.safe_load(f)

    specs = {}
    for hw_name, hw_cfg in cfg.get("hardware", {}).items():
        specs[hw_name] = HardwareSpec(
            name=hw_name,
            TFLOPs=float(hw_cfg["TFLOPs"]),
            memory_bw_GBps=float(hw_cfg["memory_bw_GBps"]),
            channel_latency_prefill_us=float(
                hw_cfg.get("channel_latency_prefill_us", 0.0)
            ),
            channel_latency_decode_us=float(
                hw_cfg.get("channel_latency_decode_us", 0.0)
            ),
            core_count=int(hw_cfg.get("core_count", 0)),
            TFLOPs_per_core=float(hw_cfg.get("TFLOPs_per_core", 0.0)),
        )
    return specs


def get_hw_spec(
    name: str, extra_specs: Optional[dict[str, HardwareSpec]] = None
) -> HardwareSpec:
    """Look up a HardwareSpec by name from built-in catalog + any extras."""
    catalog = dict(KNOWN_HARDWARE)
    if extra_specs:
        catalog.update(extra_specs)
    if name not in catalog:
        available = ", ".join(sorted(catalog.keys()))
        raise KeyError(f"Unknown hardware '{name}'. Available: {available}")
    return catalog[name]


def scale_layers_csv(
    src_path: str,
    dst_path: str,
    src_spec: HardwareSpec,
    dst_spec: HardwareSpec,
) -> pd.DataFrame:
    """Scale layers.csv latency by compute TFLOPS ratio (src/dst).

    Schema: layer_name, input, kv_cache, tp_size, latency(ns)
    """
    df = pd.read_csv(src_path)

    required_cols = {"layer_name", "input", "kv_cache", "tp_size", "latency(ns)"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {src_path}: {missing}")

    compute_ratio = src_spec.TFLOPs / dst_spec.TFLOPs

    print(
        f"  [layers.csv] Compute scaling ratio: {src_spec.TFLOPs:.2f} / {dst_spec.TFLOPs:.2f} "
        f"= {compute_ratio:.4f}"
    )
    print(f"  [layers.csv] Rows: {len(df)}")

    df["latency(ns)"] = (df["latency(ns)"].astype(float) * compute_ratio).astype(int)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    df.to_csv(dst_path, index=False)
    print(f"  [layers.csv] Written to: {dst_path}")

    return df


def scale_attention_csv(
    src_path: str,
    dst_path: str,
    src_spec: HardwareSpec,
    dst_spec: HardwareSpec,
) -> pd.DataFrame:
    """Scale attention.csv latency by memory bandwidth ratio (src/dst).

    Latency columns: time_stats.attn_{prefill,decode}.median (ms), or p50_ns fallback.
    """
    df = pd.read_csv(src_path)

    bw_ratio = src_spec.memory_bw_GBps / dst_spec.memory_bw_GBps
    print(
        f"  [attention.csv] BW scaling ratio: {src_spec.memory_bw_GBps:.1f} / "
        f"{dst_spec.memory_bw_GBps:.1f} = {bw_ratio:.4f}"
    )
    print(f"  [attention.csv] Rows: {len(df)}")

    # Identify latency columns (time_stats.* or p50_ns fallback)
    time_cols = [c for c in df.columns if c.startswith("time_stats.")]
    if not time_cols:
        print(
            "  [attention.csv] WARNING: No time_stats.* columns found. "
            "Looking for alternative latency columns..."
        )
        # Fallback: check for 'p50_ns' (TPU-style) or other patterns
        if "p50_ns" in df.columns:
            time_cols = ["p50_ns"]
        else:
            raise KeyError(
                f"Cannot find latency columns in {src_path}. "
                f"Expected time_stats.* or p50_ns columns. "
                f"Available columns: {list(df.columns)}"
            )

    for col in time_cols:
        df[col] = df[col].astype(float) * bw_ratio

    print(
        f"  [attention.csv] Scaled {len(time_cols)} latency columns: "
        f"{time_cols[:5]}{'...' if len(time_cols) > 5 else ''}"
    )

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    df.to_csv(dst_path, index=False)
    print(f"  [attention.csv] Written to: {dst_path}")

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrapolate LLM profiling data to a new hardware target "
        "using ratio-based scaling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using built-in hardware specs
  python extrapolate_hw.py \\
      --src-hw TITAN_RTX \\
      --src-dir perf_models/TITAN_RTX/meta-llama/Llama-3.1-8B/tp1 \\
      --dst-hw Dojo_Die \\
      --dst-dir perf_models/Dojo_Die/meta-llama/Llama-3.1-8B/tp1

  # With custom YAML config
  python extrapolate_hw.py \\
      --src-hw TITAN_RTX \\
      --src-dir perf_models/TITAN_RTX/meta-llama/Llama-3.1-8B/tp1 \\
      --dst-hw MyAccel \\
      --dst-dir perf_models/MyAccel/meta-llama/Llama-3.1-8B/tp1 \\
      --config my_hardware.yaml

  # Override specs directly via CLI
  python extrapolate_hw.py \\
      --src-hw TITAN_RTX \\
      --src-dir perf_models/TITAN_RTX/meta-llama/Llama-3.1-8B/tp1 \\
      --dst-hw CustomDie \\
      --dst-dir perf_models/CustomDie/meta-llama/Llama-3.1-8B/tp1 \\
      --dst-tflops 50.0 \\
      --dst-mem-bw 2000.0
        """,
    )

    parser.add_argument(
        "--src-hw",
        type=str,
        required=True,
        help="Source hardware name (e.g., TITAN_RTX, A6000, H100)",
    )
    parser.add_argument(
        "--src-dir",
        type=str,
        required=True,
        help="Path to source perf_models/{hw}/{model}/tp{N}/ directory",
    )
    parser.add_argument(
        "--dst-hw",
        type=str,
        required=True,
        help="Destination hardware name for the output",
    )
    parser.add_argument(
        "--dst-dir", type=str, required=True, help="Path to output directory"
    )

    # YAML config for custom hardware
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML file with hardware specs (see --help for format)",
    )

    # CLI spec overrides (src)
    parser.add_argument(
        "--src-tflops", type=float, default=None, help="Override source FP16 TFLOPS"
    )
    parser.add_argument(
        "--src-mem-bw",
        type=float,
        default=None,
        help="Override source memory bandwidth (GB/s)",
    )

    # CLI spec overrides (dst)
    parser.add_argument(
        "--dst-tflops",
        type=float,
        default=None,
        help="Override destination FP16 TFLOPS",
    )
    parser.add_argument(
        "--dst-mem-bw",
        type=float,
        default=None,
        help="Override destination memory bandwidth (GB/s)",
    )

    # File selection
    parser.add_argument(
        "--skip-layers", action="store_true", help="Skip layers.csv extrapolation"
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

    extra_specs = {}
    if args.config:
        extra_specs = load_hw_spec_from_yaml(args.config)
        print(f"Loaded {len(extra_specs)} hardware spec(s) from {args.config}")

    try:
        src_spec = get_hw_spec(args.src_hw, extra_specs)
    except KeyError:
        if args.src_tflops and args.src_mem_bw:
            src_spec = HardwareSpec(
                name=args.src_hw,
                TFLOPs=args.src_tflops,
                memory_bw_GBps=args.src_mem_bw,
            )
        else:
            print(
                f"ERROR: Unknown source hardware '{args.src_hw}' and no CLI overrides provided."
            )
            print(
                f"       Use --src-tflops and --src-mem-bw, or define in --config YAML."
            )
            sys.exit(1)

    try:
        dst_spec = get_hw_spec(args.dst_hw, extra_specs)
    except KeyError:
        if args.dst_tflops and args.dst_mem_bw:
            dst_spec = HardwareSpec(
                name=args.dst_hw,
                TFLOPs=args.dst_tflops,
                memory_bw_GBps=args.dst_mem_bw,
            )
        else:
            print(
                f"ERROR: Unknown destination hardware '{args.dst_hw}' and no CLI overrides provided."
            )
            print(
                f"       Use --dst-tflops and --dst-mem-bw, or define in --config YAML."
            )
            sys.exit(1)

    # Apply CLI overrides on top of catalog/config values
    if args.src_tflops is not None:
        src_spec.TFLOPs = args.src_tflops
    if args.src_mem_bw is not None:
        src_spec.memory_bw_GBps = args.src_mem_bw
    if args.dst_tflops is not None:
        dst_spec.TFLOPs = args.dst_tflops
    if args.dst_mem_bw is not None:
        dst_spec.memory_bw_GBps = args.dst_mem_bw

    # ---- Print summary ----
    print("=" * 65)
    print("  Hardware Extrapolation Summary")
    print("=" * 65)
    print(f"  Source      : {src_spec.name}")
    print(f"    TFLOPS    : {src_spec.TFLOPs:.2f}")
    print(f"    Mem BW    : {src_spec.memory_bw_GBps:.1f} GB/s")
    print(f"    Src dir   : {args.src_dir}")
    print()
    print(f"  Destination : {dst_spec.name}")
    print(f"    TFLOPS    : {dst_spec.TFLOPs:.2f}")
    print(f"    Mem BW    : {dst_spec.memory_bw_GBps:.1f} GB/s")
    print(f"    Ch lat PF : {dst_spec.channel_latency_prefill_us:.2f} us")
    print(f"    Ch lat DC : {dst_spec.channel_latency_decode_us:.2f} us")
    print(f"    Dst dir   : {args.dst_dir}")
    print()
    print(f"  Compute ratio (src/dst) : {src_spec.TFLOPs / dst_spec.TFLOPs:.4f}")
    print(
        f"  BW ratio      (src/dst) : {src_spec.memory_bw_GBps / dst_spec.memory_bw_GBps:.4f}"
    )
    print("=" * 65)

    if args.dry_run:
        print("\n  [DRY RUN] No files written.")
        return

    processed_any = False

    # ---- Process layers.csv ----
    if not args.skip_layers:
        src_layers = os.path.join(args.src_dir, "layers.csv")
        dst_layers = os.path.join(args.dst_dir, "layers.csv")
        if os.path.isfile(src_layers):
            print(f"\nProcessing layers.csv ...")
            scale_layers_csv(src_layers, dst_layers, src_spec, dst_spec)
            processed_any = True
        else:
            print(f"\n  WARNING: {src_layers} not found — skipping.")

    # ---- Process attention.csv ----
    if not args.skip_attention:
        src_attn = os.path.join(args.src_dir, "attention.csv")
        dst_attn = os.path.join(args.dst_dir, "attention.csv")
        if os.path.isfile(src_attn):
            print(f"\nProcessing attention.csv ...")
            scale_attention_csv(src_attn, dst_attn, src_spec, dst_spec)
            processed_any = True
        else:
            print(f"\n  WARNING: {src_attn} not found — skipping.")

    # ---- Summary ----
    if processed_any:
        print(f"\nDone. Output written to: {args.dst_dir}/")
        print(f"Next steps:")
        print(f"  1. (Optional) Run build_predictor.sh with --hardware {args.dst_hw}")
        print(f"     to train attention predictors on the extrapolated data.")
        print(f'  2. Create a cluster_config JSON referencing hardware="{args.dst_hw}"')
        print(f"  3. Run LLMServingSim with the new config.")
    else:
        print(
            "\nNo files processed (both layers and attention were skipped or not found)."
        )


if __name__ == "__main__":
    main()
