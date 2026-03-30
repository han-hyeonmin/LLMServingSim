# Data Sources

## AzureLLMInferenceDataset2023

| Field | Details |
|-------|---------|
| Files | `dataset/azurepublicdataset_conv/AzureLLMInferenceTrace_conv.csv` <br> `dataset/azurepublicdataset_code/AzureLLMInferenceTrace_code.csv` |
| Source | https://github.com/Azure/AzurePublicDataset |
| Paper | Patel et al., "Splitwise: Efficient generative LLM inference using phase splitting", ISCA 2024 |
| License | [CC-BY Attribution](https://creativecommons.org/licenses/by/4.0/) |
| Collection Date | November 11, 2023 (as per original dataset) |
| Retrieved | 2026-03-30 |

### Description

A sample of two LLM inference services in Azure containing per-request input and output token counts.
Collected on November 11th, 2023. Used as the workload trace in the Splitwise (ISCA 2024) paper.

### Citation

```bibtex
@inproceedings{patel2024splitwise,
  title     = {Splitwise: Efficient Generative LLM Inference Using Phase Splitting},
  booktitle = {Proceedings of the 51st International Symposium on Computer Architecture (ISCA)},
  year      = {2024}
}
```