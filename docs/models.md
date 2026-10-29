# Models used so far

Sizes are the on-disk file sizes from `bench/catalog.json` (GiB = 2^30). "Used" means a benchmark run on real hardware has completed with it. Entries below the line are in the catalog for planning only.

## Used in runs

| model | kind | why we picked it | size on disk | quantization |
|---|---|---|---|---|
| Qwen3.8-27B | dense text LLM, 27B, coding-agent capable, tool calling | the fleet baseline for inference and host_pressure: the model AA-AgentPerf-Local's recipe uses, so our numbers line up with theirs; fits every box from 32 GB up at 65K ctx | 15.9 GiB (catalog pin); a 17.7 GiB unpinned copy was used on a GB10 and the Ryzen box | Q4_K_M GGUF |
| Qwen3.6-35B-A3B | MoE text LLM, 35B total, 3B active | residency preset member: a mid-size MoE that decodes fast but takes 21 GiB resident, so it is the model that gets evicted first on 128 GB boxes | 21.1 GiB | Q4_K_M GGUF |
| Qwen3.5-9B | dense text LLM, 9B | residency preset member and the model the ladder lands on for 16 GB machines; also the rpc_split cluster smoke model because it fits an M4 Mac mini's share | 5.5 GiB | Q4_K_M GGUF |
| Ling-3.0-flash | MoE text LLM, 124B total, 5B active | the ladder's top rung on 96 to 192 GB boxes and the residency model that does not fit beside the other three on 128 GB; large weights, cheap decode, which is exactly the shape that separates memory from bandwidth | 71.7 GiB | Q4_K_M GGUF |
| Nemotron 3.5 Lightning 30B-A3B | hybrid Mamba-2 + MoE text LLM, 30B total, 3B active, 6 attention layers | the cross-engine model: the one checkpoint family llama.cpp (Q4_0 GGUF), vLLM (NVFP4/FP8), MLX and TensorFold (MLX-4bit) all serve at about 4 bits, so engine rows compare like for like | 17.6 GiB GGUF, 17.3 GiB MLX-4bit, 20.1 GiB NVFP4 | Q4_0 / MLX 4-bit / NVFP4 |
| Qwen3.8-27B FP8 | the baseline model for vLLM | the "best engine on CUDA" row for the same model as the baseline | 28.7 GiB | FP8 safetensors (official Qwen) |
| Qwen3.5-9B MLX-4bit | the 9B for MLX | the "best engine on Apple" row for a 16 GB Mac | 5.5 GiB | MLX 4-bit |
| DeepSeek-V4-Flash-0731 | MoE text LLM, 284B total, 13B active, MLA attention with sliding-window layers | Ryzen-box-only run by request: the biggest model the 160 GB carve-out can hold, and one the GB10 cannot load; also the case that exposed the llama.cpp grammar + ignore_eos bug and the planner's conservative estimate for this architecture | 144.4 GiB over 5 shards | UD-Q4_K_XL GGUF (unsloth dynamic) |

## In the catalog, not yet run

| model | kind | why it is there | size on disk | quantization |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | dense text LLM, 8B | small verified reference with three precisions for engine-track comparisons | 4.6 / 8.0 / 15.0 GiB | Q4_K_M, Q8_0 GGUF, BF16 safetensors |
| DeepSeek-V4-Flash FP8 | as above | vLLM/SGLang variant; DGX Spark pair and Station candidate | 148.7 GiB | FP8 safetensors |
| DeepSeek-V4-Pro | MoE text LLM, 671B total, 37B active | the ladder's top rung for DGX Station (coherent spill) and 8x B300 (tensor parallel); planning only | 376 GiB / 643 GiB (estimates) | Q4_K_M GGUF, FP8 |
| Qwen2.5-VL-7B | vision-language | residency-workstation preset member once vision runners exist | 4.7 GiB (estimate) | Q4_K_M |
| FLUX.2-klein-4B | image generation | residency-workstation preset member; shows a GPU shared with non-LLM work | 7.5 GiB (estimate) | BF16 |
| Whisper-large-v3 | speech recognition | residency-workstation preset member | 2.9 GiB | FP16 |

Planned for the best-engine track but not in the catalog yet: Nemotron 3.5 Lightning 30B-A3B, the one model vLLM, MLX and TensorFold all serve at equivalent bits.
