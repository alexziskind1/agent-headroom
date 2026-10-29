# agentic_bench test outline

What each test measures, how it runs, and how different hardware is expected to behave. Measured numbers come from `STATUS.md` and `out/REPORT.md` (2026-10-26). Rows marked *assumption* have not been run yet and describe what the planner predicts or what the design expects.

The suite has no composite score. Each test isolates one question about a machine's memory system. The flow for every test is the same: `probe` measures the machine, `plan` fits the biggest workload from the catalog that the budget allows, `run` executes it with llama-server (or another engine) plus the AA-AgentPerf-Local replay, and `report` tabulates the results.

Shared assumptions behind every test:
- Budget is total memory minus a reserve (10% of GPU memory, at least 2 GiB) and, on unified machines, a host floor of 16 GiB (or 20% on small boxes) so the OS and tool sandboxes keep room. Planning uses totals, not free memory; if the box is busy the run refuses at admission.
- Cost of a workload is weights + KV cache (context x sequences, computed from the model's layer count, KV heads and head dim) + activations + engine overhead. Nothing is guessed from the model name.
- Same llama-server flags everywhere, mirrored from AA-AgentPerf-Local's managed mode: `--ctx-size N --parallel n --gpu-layers all --jinja --batch-size 4096 --ubatch-size 512 --flash-attn on`.
- Memory topology comes from the driver, not from comparing sizes: `unified` (integrated GPU, Apple and GB10), `discrete` (PCIe GPU, and an AMD APU with a BIOS carve-out), `coherent` (non-integrated GPU with hardware-coherent host page tables, DGX Station class), `cluster` (several nodes merged into one probe).

---

## 1. Inference (adopted from AA-AgentPerf-Local)

**Question.** How fast does this machine run one real coding agent on a fixed model at the longest context the replay needs?

**How it runs.**
- Model fixed by the operator (Qwen3.8-27B Q4_K_M is the fleet baseline), context fixed at 65,536, one slot.
- Replay `agentperf-default-v1`: 8 recorded coding-agent tasks, 168 turns, 172 tool calls, contexts up to 65K. Tool results are replayed from the recording with their recorded delays (`tool_mode none`), so only the model and the server are measured.
- Output length policy `exact` by default: every turn generates exactly the recorded number of tokens with end-of-sequence ignored, so decode tok/s is comparable across machines. (This is what tripped the DeepSeek-V4 tool-call grammar in llama.cpp; `--replay-arg=--output-token-policy --replay-arg=recorded` is the escape hatch and is recorded in the result.)
- Outputs: decode tok/s, end-to-end tok/s (includes prefill and tool delays), time to first token p50/p95, wall time, peak host and GPU memory, model file hash versus the catalog pin.

**Hardware behaviour.**

| machine | result or expectation | why |
|---|---|---|
| RTX PRO 6000 96 GB + 92 GB RAM | 75.1 tok/s, 8.3 min (measured) | GDDR7 bandwidth; the 19 GB model sits entirely in VRAM with 65K KV |
| GB10 128 GB unified | 11.6 tok/s, 50 min (measured, idle box) | LPDDR5X at ~270 GB/s; decode is bandwidth-bound |
| Ryzen AI Max+ 495, 192 GB, 160 GB carve-out, Vulkan | 10.9 tok/s, 70 min (measured) | same bandwidth class as the GB10; Vulkan path, not ROCm |
| M4 16 GB | cannot run the 27B at 65K; ladder picks the 9B (measured 16.1 tok/s at 61K) | 16 GB minus the host floor leaves no room for 19 GB of weights |
| DGX Station GB300, 249.5 GiB HBM + 743 GB LPDDR | llama.cpp Q4_K_M 88.8 tok/s, 8.1 min (measured, stage-llm's 22 GB beside it); vLLM FP8 135.2 tok/s, 3.1 min (measured, recorded policy; llama.cpp recorded 89.3) | HBM3e at 6.1 TB/s measured; the 27B leaves 200 GiB idle |
| 8x B300 rack | *assumption:* single-GPU numbers equal the Station; the test does not use more than one GPU for a 27B | tensor parallel only helps models that do not fit one device |

Inference separates machines by bandwidth and almost nothing else. It cannot see 128 GB versus 192 GB.

## 2. Capacity ladder

**Question.** What is the largest catalog model and the longest context this machine can hold, and how does that model run?

**How it runs.**
- Planner walks the text catalog from the largest model down, and for each model binary-searches the longest context (4,096-token steps) that fits the budget with one sequence.
- The first model that fits wins; its context is the answer. Then the full replay runs on that selection.
- Outputs: the chosen model and context (categorical), then decode and end-to-end tok/s, wall, peak memory.

**Hardware behaviour.**

| machine | result or expectation |
|---|---|
| M4 16 GB | Qwen3.5-9B at 61K (measured) |
| RTX PRO 6000 96 GB | Ling-3.0-flash (124B MoE) fits; run skipped overnight for disk |
| GB10 128 GB | Ling-3.0-flash Q4 at 131K, 45.2 tok/s, 16 min (measured) |
| Ryzen 192 GB (160 GB carve-out) | Ling-3.0-flash Q4 at 131K, 40.5 tok/s (measured); DeepSeek-V4-Flash 144 GiB loads at 65K with `--force` (measured, 14.2 tok/s) |
| DGX Station GB300 | DeepSeek-V4-Pro IQ2_XXS (433 GiB) at 131K via `coherent_spill`, 57% of weights in LPDDR over NVLink-C2C (388 GB/s measured vs 6134 GB/s HBM): 4.37 tok/s, 151 min, 168/168 (measured). The bandwidth-ratio estimate for this placement is 0.107 of all-HBM speed; naive unified-memory spill is demand-paged, so the slow tier sets the pace |
| 2x GB10 cluster over 200 GbE | *assumption:* Ling at 131K stays on one node (`single`); a 250 GB model needs `rpc_split`, and every token crosses the wire |
| 8x B300 rack | *assumption:* DeepSeek-V4-Pro FP8 (690 GB) under tensor parallel across 8 devices |

The ladder is where the catalog's lack of tiers matters: a bigger machine simply reaches further down the list, and the answer is a model name before it is a number. Note the GB10 and the Ryzen box land on the same rung; only the memory tests separate them.

## 3. Host pressure

**Question.** How many agents can this machine run at once before something breaks, and what breaks first: the GPU, the CPU, or system RAM?

**How it runs.**
- One llama-server with `--parallel n` and `--ctx-size n x 65,536`, n doubling 1, 2, 4, 8, 16. The planner caps the sweep where n x KV exceeds the budget.
- n independent agentperf clients run the replay concurrently against that server.
- Two tool modes. `fixed_delay` (`aa-mini-v1`, 6 turns per client) measures server concurrency alone. `live` (full replay, real Docker tool containers for pinchbench and SWE-bench tasks) measures the whole machine: each agent adds a client process and containers on the host.
- The sweep stops at the first level with any failed turn; the last clean level is "max sustained agents".
- Outputs per level: aggregate tok/s, per-agent p50 decode, wall time, peak host RAM and GPU memory.

**Hardware behaviour (live tools).**

| machine | sustained | shape | what capped it |
|---|---|---|---|
| RTX PRO 6000, 8 cores, 92 GB | 8 (measured) | aggregate peaks at 2 agents (53 tok/s) and falls to 35 at 8; wall grows 8.6x for 8x the work | CPU and host RAM (50 GiB at 8); the GPU is idle waiting for tools |
| GB10, 20 cores, 128 GB unified | 8 (measured, idle box) | aggregate keeps rising to 17.4 at 8; per-agent decode halves 10.6 to 5.1 | the unified pool: 102 GiB in use at 8, level 16 would not fit |
| Ryzen 192 GB, 32 GB host after carve-out | 2 (measured) | 5.3, 7.6, 7.9 tok/s at 1, 2, 4; one turn dropped at 4 with host RAM at 30.4 of 32 GiB | host RAM; the carve-out starves the tools |
| M4 16 GB | 2 at 16K ctx (measured) | | everything |
| DGX Station GB300, 72 cores, 743 GB | 16 (measured, sweep cap) | aggregate peaks at 4 agents (64.5 tok/s) and falls to 42 at 16; per-agent p50 89 to 13; wall 14.7x for 16x the work; host used 261 GiB at 16 (about 10.8 GiB per live agent) | nothing failed; tool containers compete with the server past 4 agents. Fixed delay on the same box kept rising to 93.3 at 16 |
| 8x B300 rack | *assumption:* host_pressure is per server; with vLLM the question becomes batching efficiency, not fit | |

This is the test where the same decode speed hides three different machines: an 8-core workstation chokes on tools, a GB10 runs out of pool, and the Ryzen box runs out of the 32 GB its BIOS left for the host.

## 4. Residency

**Question.** How many of a fixed set of models can stay loaded at once, and what does it cost when they cannot?

**How it runs.**
- A preset lists the models (text workstation preset: Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.5-9B, Ling-3.0-flash), each at 32K context.
- The planner packs them in order into the budget. What fits is `resident`; on coherent machines what overflows into the host tier is `resident_coherent`; the rest is `evicted`. Nothing is substituted with a smaller quant.
- The run starts a server per resident model, then cycles mini-replay tasks across all models in the preset. An evicted model is loaded on demand by evicting a resident one, timed as swap time, and the victim is restored afterwards.
- Outputs: resident count, evicted list, total swap seconds, cycle time, tasks per minute, peak memory.

**Hardware behaviour.**

| machine | result or expectation |
|---|---|
| GB10 128 GB | 3 resident, Ling evicted, 79 s swap, 445 s cycle, 0.54 tasks/min (measured) |
| Ryzen 192 GB | all 4 resident, 0 swap, 402 s cycle, 0.60 tasks/min (measured) |
| RTX PRO 6000 96 GB | *planner:* 3 resident (Ling does not fit beside the others) |
| M4 16 GB | not applicable; only the 9B fits |
| DGX Station GB300 | all 4 resident, 0 swap, 3.19 tasks/min all-llama.cpp and 4.39 with vLLM FP8 serving the 27B (measured); a larger preset would show the three-state answer: hot in HBM, warm in LPDDR via NVLink-C2C, cold on disk |
| 2x GB10 cluster | *assumption:* `single` placement spreads whole models across nodes, so 4 resident with nothing crossing the wire while serving |

Residency is the only test where 192 GB versus 128 GB is a different machine rather than a slightly different number: the Ryzen box wins it outright while losing every speed test.

---

Mixed engines (validated 2026-10-27 on a GB10): resident servers can be any engine the machine has, so a realistic workstation preset is vLLM for the dense model, llama.cpp for the MoE GGUF and TensorFold for the small helper. Measured with Qwen3.8-27B FP8 (vLLM), Qwen3.6-35B-A3B Q4_K_M (llama.cpp) and Qwen3.5-9B MLX-4bit (TensorFold) resident, Ling-3.0-flash evicted: all clean, 79.9 GiB resident, but the swap cost 400 s because the largest resident (vLLM) is evicted and vLLM pays 320 s of compile and warmup to come back, against 106 s for the all-llama.cpp preset. Expectation: on any machine, mixing vLLM into a residency preset trades decode speed for a far worse eviction penalty, and the right victim is the cheapest to reload, not the biggest. Cluster-worker residency stays llama.cpp only.

## 5. Placements the planner can choose

| placement | when | what the run does |
|---|---|---|
| single | model fits one device | one server, one device |
| tensor_parallel / pipeline | one node, several devices, model does not fit one | llama.cpp `--tensor-split` over local GPUs |
| rpc_split | cluster plan, or `--placement rpc_split` forced | `ggml-rpc-server` on each worker over ssh, `--rpc a:p,...` and a split proportional to each node's room; validated with a Qwen3.5-9B smoke run split 92/8 between a GB10 and an M4 Mac mini (16.8 tok/s, clean teardown) |
| coherent_spill | coherent topology, weights overflow HBM but fit HBM + host tier | `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`; decode estimate = 1 / ((1 - f) + f x bw_device / bw_link) where f is the spilled fraction |
| offload | `--allow-offload`, discrete GPU with weights partly in system RAM | CPU offload of layers |
| forced | `--force` with a fixed context when the estimate says no | runs anyway, records the shortfall; the server load is the real test (DeepSeek-V4-Flash on the Ryzen box loaded 13.7 GiB under the estimate) |

## 6. Engine tracks

Every machine gets a llama.cpp row (the baseline that keeps variables constant). Each platform also gets a "best available" row chosen by the catalog's engine preference: vLLM or SGLang on CUDA, MLX and TensorFold on Apple, TensorFold on CUDA where it qualifies. Runners exist for llama.cpp, vLLM (native or docker), MLX and TensorFold (`bench/engines.py`); `--engine best` or `--engine <name>` selects them and `bench fetch` pulls the matching checkpoint.

Measurement rule: AgentPerf's `exact` output policy (generate exactly the recorded length, ignore end-of-sequence) only works on llama.cpp. The other engines stop at tool calls or ignore `ignore_eos`, so engine rows run the `recorded` policy automatically, and the fair comparison is a llama.cpp row run with `--output-policy recorded` at the same context. Validated 2026-10-27 with Nemotron 3.5 Lightning 30B-A3B on a GB10 (llama.cpp 71 tok/s recorded, vLLM NVFP4 109, TensorFold 169 with its drafter) and Qwen3.5-9B on an M4 (llama.cpp 17.7, MLX 20.0).

Measured so far (aa-mini-v1 smokes, recorded policy, 2026-10-27):
- Inference, GB10, Nemotron 30B-A3B at 32K: llama.cpp 71 tok/s, vLLM NVFP4 109, TensorFold MLX-4bit with drafter 169. Load time is the other axis: 3 s, 275 s (compile and warmup), 14 s. Qwen3.8-27B FP8 on vLLM decodes at 12.4 tok/s against 12.1 for llama.cpp Q4_K_M under the same recorded policy: almost twice as fast per byte, a wash per token because FP8 is twice the bytes on a bandwidth-bound box, and slower end to end (7.6 vs 11.0) with 312 s of compile and warmup per launch.
- Inference, M4 16 GB, Qwen3.5-9B 4-bit at 16K: llama.cpp 17.7, MLX 20.0.
- Host pressure (fixed delays): MLX on the M4 batches decode and sustains 4 agents at 8K with aggregate rising 11.5 to 23.4 tok/s; TensorFold on the GB10 sustains 4 at 16K, aggregate 66 to 135.

Expected effect per test (not yet measured):
- Inference: vLLM on a discrete CUDA GPU should raise single-stream decode modestly and prefill a lot (CUDA graphs, paged attention).
- Host pressure: this is where vLLM changes the picture on CUDA, since continuous batching turns the "aggregate rises then falls" curve into a much flatter one; the host-side tool cost stays the same.
- Ladder: NVFP4 or MLX 4-bit checkpoints shift which model fits, so the categorical answer can change with the engine.
- Residency: unchanged in principle; engine overhead per server differs.

## 6b. warm_contexts test (how many agent contexts does host RAM keep warm?)

**Question.** With the GPU, model and slot count held constant, how many concurrent agents can a box serve before their contexts no longer fit in host RAM? This isolates system RAM: the agent execution layer (clients, contexts between turns) lives in host memory, the model never moves.

**How it runs.**
- One llama-server with `s` slots (default 8) at the per-agent context, unified KV, and a host-memory prompt cache (`--cache-ram`) sized from the probe: host RAM after the plan minus a server floor, times the KV share of each agent's footprint. Slot selection by prompt similarity is disabled (`--slot-prompt-similarity 0`): with it on, llama-server routes a request to any idle slot sharing more than 10% prefix (every agent shares the tools block) and never consults the host cache, so agents silently re-prefill. With LRU selection each task saves the outgoing slot state to the host cache and restores the incoming agent's own context instead of re-prefilling.
- n agents (n doubling from s) replay `agentperf-default-v1` concurrently with `tool_mode none` and a fixed 64 output tokens per turn, so time per level is dominated by prefill behaviour, not decode. Contexts and prompts are the recorded ones; the prompt cache namespace is unique per agent.
- Per agent the host carries one context of KV (64 KiB/token for Qwen3.8-27B, 4.0 GiB at 65K fp16) plus the replay client (1.5 GiB). The modeled knee is `(host RAM after plan - 4 GiB) / per-agent bytes`; the default levels cross 2x the knee.
- Per level: prompt-cache hit ratio (cached vs uncached prompt tokens from the server's own counters), TTFT p50 and worst p95, aggregate tok/s, tasks/min, wall, peak host RAM, page cache, swap, mean and peak CPU utilisation, and the server's /metrics counters. The measured knee is the first level whose hit ratio falls more than 0.15 below the first level's.

**Why it is RAM-bound and not CPU-bound.** A client waiting on an HTTP stream and a server copying KV between host and device are cheap; SWE-bench builds are not involved. The CPU column is recorded so the claim is checked, not assumed. Validated on a GB10: with the host cache on, a context evicted from its slot returns in 0.4 s; with it off the same turn re-prefills 4K tokens in 1.5 s.

**Expected behaviour (modeled, 2x RTX PRO 6000 box, Qwen3.8-27B Q4 at 65K, 8 slots).**

| host RAM | host cache | knee (fp16 KV) | knee (q8 KV) |
|---|---|---|---|
| 128 GiB | 67 GiB | 16 agents | 26 agents |
| 256 GiB | 137 GiB | 34 agents | 53 agents |
| 2 TiB | 1.1 TiB | 278 agents | 437 agents |

Below the knee every level has the same hit ratio and TTFT; above it the hit ratio collapses, TTFT grows by the full-context prefill time (seconds per turn) and tasks/min falls. Levels past the knee also show swap or page-cache pressure if `--cache-ram` is set above the real budget; the planner's budget keeps the server inside RAM so the degradation is re-prefill, not thrash.

**Honest limits.** llama.cpp only (vLLM's CPU KV offload `--kv-offloading-size` is the engine-track follow-up). The knee depends on the per-agent context: at 32K the same RAM holds twice the agents. The fixed 64-token output makes decode throughput unrepresentative; use the inference and host_pressure rows for tok/s.

## 7. What is not covered yet

- Vision, image and speech models are catalog entries only; residency is text-only.
- Coherent mode is validated on the DGX Station with naive unified-memory spill only (every tensor demand-paged). Tensor-override placement that keeps attention, shared experts and KV in HBM and spills only routed experts is not implemented; the measured 4.37 tok/s is the floor for "model bigger than HBM", not the ceiling.
- Cluster mode is llama.cpp RPC only; no NCCL or Ray paths.
- Speculative decoding, multimodal replays and power measurement on non-NVIDIA GPUs are out of scope for this version.
