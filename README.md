# LLMServingSim

> Personal research fork of [casys-kaist/LLMServingSim](https://github.com/casys-kaist/LLMServingSim).
> For publications, citation, and official documentation see the upstream repo or [llmservingsim.ai](https://llmservingsim.ai).

## About

LLMServingSim is a cycle-level simulator for LLM serving infrastructure. It pairs a Python frontend that mirrors vLLM's continuous-batching scheduler with the ASTRA-Sim C++ analytical network backend, and drives both from per-hardware latency data captured by a vLLM-based layerwise profiler. The result is a unified environment for studying heterogeneous accelerators, disaggregated memory tiers (CPU / CXL / PIM), MoE routing, and multi-instance parallelism (TP / PP / EP / DP) end-to-end.

## Repository structure

```
LLMServingSim/
├── serving/                # simulator core    (`python -m serving`)
├── profiler/               # vLLM-based layerwise profiler  (`python -m profiler`)
├── bench/                  # vLLM end-to-end benchmark + sim validation  (`python -m bench`)
├── workloads/              # JSONL workloads + ShareGPT generators  (`python -m workloads.generators`)
├── scripts/                # shared environment / build entry points
├── configs/                # cluster / model / PIM configurations
└── astra-sim/              # ASTRA-Sim C++ backend (submodule)
```

Each Python module has its own README under the directory.

## Getting Started

```bash
git clone --recurse-submodules https://github.com/han-hyeonmin/LLMServingSim.git
cd LLMServingSim
./scripts/docker-sim.sh           # launch the simulator container
./scripts/compile.sh              # build ASTRA-Sim + Chakra
./serving/run.sh                  # run the example simulations
```

For installation details, container choices, configuration layout, CLI
flags, and the full set of example workloads, see the
[documentation](https://llmservingsim.ai/docs/getting-started/overview).

### conda Installation (no Docker)

```bash
conda env create -f scripts/servingsim.yml
conda activate servingsim
conda env create -f scripts/vllm-env.yml
conda activate vllm-env
```
