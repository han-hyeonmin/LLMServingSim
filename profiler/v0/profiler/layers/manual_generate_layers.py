import argparse
import json
import csv
import os

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
    Matches the 'assemble' logic in metrics_engine.py.
    """
    compute_time_s = F / peak_flops if peak_flops > 0 else 0
    memory_time_s = M / mem_bw_bytes if mem_bw_bytes > 0 else 0
    
    latency_s = max(compute_time_s, memory_time_s)
    return int(latency_s * 1e9)  # Convert seconds to nanoseconds

# ============================================================
# 3. Main Execution and Argument Parsing
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Analytically generate layers.csv")
    parser.add_argument("--hardware", type=str, required=True, help="Target hardware name")
    parser.add_argument("--model", type=str, required=True, help="Target model name")
    parser.add_argument("--tp-size", type=str, required=True, help="Tensor parallel sizes (e.g., '1' or '1, 2')")
    parser.add_argument("--max-len", type=int, default=8192, help="Maximum sequence length")
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
    I = mdl_cfg["intermediate_size"]
    vocab_size = mdl_cfg["vocab_size"]
    n_heads = mdl_cfg["num_attention_heads"]
    n_kv = mdl_cfg["num_key_value_heads"]
    bytes_el = mdl_cfg["bytes_per_element"]
    
    head_dim = H // n_heads
    H_kv = head_dim * n_kv

    # Generate sequence lengths dynamically up to max_len
    seq_lens = list(range(1, args.max_len + 1))
    
    tp_sizes = [int(tp.strip()) for tp in args.tp_size.split(",")]

    # Exact target order for LLMServingSim layer mapping
    target_layer_order = [
        "embedding", "input_layernorm", "q_proj", "k_proj", "v_proj", "rope", 
        "o_proj", "post_layernorm", "gate_proj", "up_proj", "act_fn", 
        "down_proj", "final_layernorm", "lm_head"
    ]

    for P in tp_sizes:
        # Common scalars strictly mapping metrics_engine.py logic
        invP = 1.0 / P
        H_over_P = H * invP
        I_over_P = I * invP
        Hkv_over_P = H_kv * invP
        
        output_rows = []
        for N_tokens in seq_lens:
            latencies = {}

            # ----------------------------------------------------
            # A. Calculate Linear Layers based on metrics_engine.py
            # ----------------------------------------------------
            def linear_cost(d_in, d_out_over_P):
                """
                F calculates FLOPs directly matching metrics_engine.py.
                M evaluates Weight + Input Act + Output Act for independent layer bottleneck.
                """
                F = 2 * N_tokens * d_in * d_out_over_P
                M = (d_in * d_out_over_P + N_tokens * d_in + N_tokens * d_out_over_P) * bytes_el
                return calculate_latency_ns(F, M, peak_flops, mem_bw_bytes)

            latencies["q_proj"] = linear_cost(H, H_over_P)
            latencies["k_proj"] = linear_cost(H, Hkv_over_P)
            latencies["v_proj"] = linear_cost(H, Hkv_over_P)
            latencies["o_proj"] = linear_cost(H, H_over_P)      
            latencies["gate_proj"] = linear_cost(H, I_over_P)
            latencies["up_proj"] = linear_cost(H, I_over_P)
            latencies["down_proj"] = linear_cost(I, H_over_P)   
            
            vocab_over_P = vocab_size * invP
            latencies["lm_head"] = linear_cost(H, vocab_over_P)

            # ----------------------------------------------------
            # B. Calculate Memory-bound Layers
            # ----------------------------------------------------
            # LayerNorms: M = 2 * N * H * bytes_el (Read + Write)
            M_ln = 2 * N_tokens * H * bytes_el
            lat_ln = calculate_latency_ns(0, M_ln, peak_flops, mem_bw_bytes)
            for name in ["input_layernorm", "post_layernorm", "final_layernorm"]:
                latencies[name] = lat_ln
            
            # Embedding: M = N * H * bytes_el
            M_emb = N_tokens * H * bytes_el
            latencies["embedding"] = calculate_latency_ns(0, M_emb, peak_flops, mem_bw_bytes)

            # RoPE: M = 2 * N * (H + H_kv) * invP * bytes_el
            M_rope = 2 * N_tokens * (H + H_kv) * invP * bytes_el
            latencies["rope"] = calculate_latency_ns(0, M_rope, peak_flops, mem_bw_bytes)

            # Act_fn (SwiGLU): M = 2 * N * I * invP * bytes_el
            M_act = 2 * N_tokens * I * invP * bytes_el
            latencies["act_fn"] = calculate_latency_ns(0, M_act, peak_flops, mem_bw_bytes)

            # ----------------------------------------------------
            # C. Convert to nanoseconds
            # ----------------------------------------------------
            for layer_name in target_layer_order:
                latency = latencies[layer_name]
                output_rows.append({
                    "layer_name": layer_name,
                    "input": N_tokens,
                    "kv_cache": 0,
                    "tp_size": P,
                    "latency(ns)": max(1, latency)
                })

        # Save to CSV
        out_dir = f"../perf_models/{args.hardware}/{args.model}/tp{P}"
        os.makedirs(out_dir, exist_ok=True)
        
        output_filename = os.path.join(out_dir, "layers.csv")
        with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["layer_name", "input", "kv_cache", "tp_size", "latency(ns)"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
            
        print(f'Generated "perf_models/{args.hardware}/{args.model}/tp{P}/layers.csv" successfully.')

if __name__ == "__main__":
    main()