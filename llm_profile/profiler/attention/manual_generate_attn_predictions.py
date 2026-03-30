import argparse
import json
import csv
import os
import numpy as np

# ============================================================
# 1. Load Configuration
# ============================================================
def load_config(config_path):
    """
    Load the hardware and model specifications from a JSON file.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================================
# 2. Core Performance Calculation (Roofline Model)
# ============================================================
def calculate_latency_ns(F, M, peak_flops, mem_bw_bytes):
    """
    Calculate the compute time and memory time.
    Returns the bottleneck (maximum of the two) in nanoseconds.
    """
    compute_time_s = F / peak_flops if peak_flops > 0 else 0
    memory_time_s = M / mem_bw_bytes if mem_bw_bytes > 0 else 0
    
    latency_s = max(compute_time_s, memory_time_s)
    return int(latency_s * 1e9)  # Convert seconds to nanoseconds

# ============================================================
# 3. Main Execution and Argument Parsing
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Analytically generate attention prediction CSVs")
    parser.add_argument("--hardware", type=str, required=True, help="Target hardware name")
    parser.add_argument("--model", type=str, required=True, help="Target model name")
    parser.add_argument("--tp-size", type=str, required=True, help="Tensor parallel sizes (e.g., '1' or '1, 2')")
    parser.add_argument("--max-len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--config-path", type=str, default="custom_hw_model_config.json", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config_path)

    if args.hardware not in cfg["hardware"]:
        raise ValueError(f"Hardware '{args.hardware}' not found in config.")
    if args.model not in cfg["model"]:
        raise ValueError(f"Model '{args.model}' not found in config.")


    # Extract hardware invariants
    hw_cfg = cfg["hardware"][args.hardware]
    peak_flops = hw_cfg["TFLOPs_per_die"] * 1e12
    mem_bw_bytes = hw_cfg["memory_bw_TBps"] * 1e12

    # Extract model parameters
    mdl_cfg = cfg["model"][args.model]
    H = mdl_cfg["hidden_size"]
    n_heads = mdl_cfg["num_attention_heads"]
    n_kv = mdl_cfg["num_key_value_heads"]
    bytes_el = mdl_cfg["bytes_per_element"]
    
    head_dim = H // n_heads
    H_kv = head_dim * n_kv

    # ----------------------------------------------------
    # Apply Custom Grid Conditions
    # ----------------------------------------------------
    max_len = args.max_len
    
    # prefill kv size = range(0, max-len + 1, 64)
    prefill_kv_range = np.arange(0, max_len + 1, 64)
    
    # prefill chunk size = range(32, max-len * 2 * 32 + 1, 32)
    prefill_chunk_range = np.arange(32, max_len * 2 * 32 + 1, 32)
    
    # decode kv size = range(0, max-len + 1, 64)
    decode_kv_range = np.arange(0, max_len + 1, 64)
    
    # decode batch size = range(1, max-len / 4 + 1)
    decode_batch_range = np.arange(1, (max_len // 4) + 1)
    
    tp_sizes = [int(tp.strip()) for tp in args.tp_size.split(",")]

    for P in tp_sizes:
        # Common scalars adapted from metrics_engine.py
        invP = 1.0 / P
        H_over_P = H * invP
        Hkv_over_P = H_kv * invP
        
        prefill_rows = []
        decode_rows = []

        # ----------------------------------------------------
        # A. Prefill Attention Prediction
        # ----------------------------------------------------
        for kv in prefill_kv_range:
            for chunk in prefill_chunk_range:
                N_past = kv       # Past KV cache length
                N_q = chunk       # Current query chunk length
                N_k = N_past + N_q # Total key length for this chunk

                # FLOPs = Cross-attention with past + Causal self-attention for chunk
                F_cross = 4 * N_q * N_past * H_over_P
                F_causal = 2 * (N_q ** 2) * H_over_P  # Divided by 2 for causal masking
                F_total = F_cross + F_causal
                
                # Memory = Read Q + Read K, V + Write O
                M_total = (
                    2 * N_q * H_over_P +            # Q Read, O Write
                    2 * N_k * Hkv_over_P            # K, V Read
                ) * bytes_el

                lat_ns = calculate_latency_ns(F_total, M_total, peak_flops, mem_bw_bytes)

                prefill_rows.append({
                    "kv_cache_size": kv,
                    "prefill_chunk_size": chunk,
                    "prediction": max(1, lat_ns)
                })

        # ----------------------------------------------------
        # B. Decode Attention Prediction
        # ----------------------------------------------------
        for b in decode_batch_range:
            for kv in decode_kv_range:
                N_k = kv # Total past KV cache length per request
                
                # FLOPs = 4 * B * N_k * (H / P)
                F_total = 4 * b * N_k * H_over_P
                
                # Memory = Read Q + Read K, V (from memory) + Write O
                M_total = (
                    2 * b * H_over_P +              # Q Read, O Write
                    2 * b * N_k * Hkv_over_P        # K, V Read for all batch requests
                ) * bytes_el

                lat_ns = calculate_latency_ns(F_total, M_total, peak_flops, mem_bw_bytes)

                decode_rows.append({
                    "batch_size": b,
                    "kv_cache_size": kv,
                    "prediction": max(1, lat_ns)
                })

        # ----------------------------------------------------
        # Save to CSV files
        # ----------------------------------------------------
        out_dir = f"../perf_models/{args.hardware}/{args.model}/tp{P}/predictions"
        os.makedirs(out_dir, exist_ok=True)
        
        prefill_file = os.path.join(out_dir, "attn_prefill_predictions.csv")
        with open(prefill_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["kv_cache_size", "prefill_chunk_size", "prediction"])
            writer.writeheader()
            writer.writerows(prefill_rows)
            
        decode_file = os.path.join(out_dir, "attn_decode_predictions.csv")
        with open(decode_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["batch_size", "kv_cache_size", "prediction"])
            writer.writeheader()
            writer.writerows(decode_rows)
            
        print(f"Generated Prefill/Decode predictions for TP={P} successfully.")

if __name__ == "__main__":
    main()