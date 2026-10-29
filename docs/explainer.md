# Agent Headroom, explained

Agent Headroom is a memory-aware agentic benchmark suite. Tests: Rung (ladder), Pace (inference), Crowd (host_pressure), Tenants (residency), Recall (warm_contexts); badges Bridge (coherent memory), Fleet (clusters), Tracks (engine rows). It measures how much agentic work a machine can hold and sustain, not just how fast one model decodes tokens. Written for a video script; every number below comes from `STATUS.md` and `out/REPORT.md`.

## The one-line pitch

Token speed tells you how fast a GPU is. It does not tell you whether a 128 GB box and a 192 GB box behave differently, or whether 32 GB of system RAM next to the same GPU costs you anything. Agents are where that shows: long contexts, many concurrent sessions, tools running on the CPU, several models kept warm at once. Agent Headroom probes the machine, plans the biggest agentic workload that fits, runs it the same way everywhere, and reports where each machine breaks.

## The parts

### 1. Probe
`bench probe` measures the machine and writes `probe.json`.
- CPU, RAM, every accelerator with total and free memory, storage read speed, installed engines and versions.
- Memory topology is `unified` or `discrete`. It is decided only by the device's integrated flag (CUDA attribute 18, Apple GPUs), never by comparing sizes. An AMD APU with a 160 GB BIOS carve-out is treated as discrete, because the carve-out really is a separate pool.
- On unified GB10 boxes nvidia-smi reports N/A, so memory is read through libcuda directly. On macOS available memory comes from `kern.memorystatus_level`, which is what the kernel actually uses to decide pressure.

### 2. Catalog
`bench/catalog.json` is open-ended data, not tiers. Each model entry has the file size in bytes, SHA-256, layer count, KV heads and head dimension (so KV bytes per token is computed, not guessed), measured engine overhead, and the upstream pin (repo and revision). Four text models are verified from AA-AgentPerf-Local's recipes: Qwen3.8-27B, Qwen3.6-35B-A3B, Qwen3.5-9B, Ling-3.0-flash. Engines carry their fixed overhead and which backends and placements they support. Adding a model is adding a JSON block; nothing else changes.

### 3. Planner
`bench plan <test>` is a dry run that never downloads or launches anything. It:
- Budgets against total memory, not available memory. A plan describes the hardware. If the box is busy that is a warning, and the run refuses at admission time if memory is really gone.
- Keeps a GPU reserve (10%, at least 2 GiB) and on unified machines a host floor of min(16 GiB, 20% of RAM) so tools and the OS keep room.
- Costs each candidate as weights + KV(context x sequences) + activation + engine overhead, with placements single, tensor-parallel, or CPU offload.
- Picks per test: fixed model with maximum context (inference), largest model then largest context (ladder), maximum concurrent sequences (host_pressure), or packs a fixed model set in order and marks what has to be evicted (residency).
- Writes `plan.json` with every candidate's estimate and why it was or was not chosen (`--explain`).

### 4. Runner
`bench run plan.json` re-probes, refuses if device count or topology changed or the budget drifted more than 2%, hashes the model file against the catalog, launches `llama-server` with exactly AA-AgentPerf-Local's managed-mode flags, waits for health, samples host and GPU memory once a second, runs the replay, and writes `results.json` next to the server log, the memory trace and the raw AgentPerf output. Exit codes say why it stopped: refused, server failed, replay failed.

### 5. The five tests
No composite score. Each test isolates one bottleneck.

**inference** adopts AA-AgentPerf-Local unchanged: a frozen 8-task, 168-turn, 172-tool-call coding-agent trajectory with contexts up to 65K. Same model, same flags, same replay on every machine. Output: decode tok/s, end-to-end tok/s, time to first token, wall time, peak memory.

**capacity ladder** asks: what is the largest catalog model and the longest context this machine holds, and how does it run? The answer is categorical (which model) before it is numerical (tok/s). A 16 GB Mac picks the 9B at 61K. A 128 GB GB10 picks Ling-3.0-flash (124B MoE) at 131K.

**host_pressure** runs N concurrent agents against one server, N doubling until something fails or the budget is exceeded. Each agent adds a KV slot on the GPU and a client plus live Docker tool containers on the host. With recorded tool delays it measures server concurrency. With live tools it measures the whole machine: CPU cores and system RAM show up independently of the GPU.

**residency** loads a fixed set of models (four text models, 32K each) and cycles agent tasks across them. Models that do not fit are evicted and reloaded, never substituted. Output: how many stay resident, swap time, tasks per minute, peak memory. This is the test where 192 GB versus 128 GB is a different machine, not a slightly different number.

**warm_contexts** holds the GPU, model and slot count fixed (8 slots at 65K) and grows the agent fleet past the slots. Between turns each agent's context (4 GiB of KV at 65K) is parked in llama-server's host-RAM prompt cache and restored on its next turn; past the point where n contexts outgrow the cache, every extra agent re-prefills its whole context every turn. Output per level: prompt-cache hit ratio, time to first token, tasks per minute, peak host RAM, page cache, swap and CPU utilisation, plus the planner's modeled knee. For a 2x RTX PRO 6000 box with Qwen3.8-27B the knee is 16 agents at 128 GB, 34 at 256 GB, 278 at 2 TB. CPU per agent is near zero, so this is the system-RAM answer that does not depend on cores. One llama-server default had to be turned off: slot selection by prompt similarity routes agents into each other's slots and silently bypasses the host cache.

### 6. Report and remote
`bench report` folds every `results.json` into markdown tables per test. `bench remote host cmd` ships the package over SSH (jump hosts and per-hop keys supported), runs it there, and brings `out/` back. `bench ssh` runs a quoted script on a jump target. The whole fleet is driven from one session with no agents installed on the targets beyond Python, llama-server and agentperf-local.

### 7. Coherent memory (DGX Station class)
A GB300 station has two pools: HBM on the GPU and LPDDR5X on the Grace CPU, joined by NVLink-C2C so the GPU can read both. The probe detects this from the CUDA driver (hardware-coherent host page tables with a non-integrated GPU) and measures two bandwidths: device memory and the host-to-device link. The planner keeps KV cache, activations and engine overhead in HBM and lets weights overflow into the host tier, placement `coherent_spill`, with a decode-speed estimate from the bandwidth ratio. Residency gets a third state: hot in HBM, warm in LPDDR, cold on disk. llama-server runs with CUDA managed memory enabled so the overflow really happens.

Measured 2026-10-27 on the DGX Station GB300 (249.5 GiB HBM at 6.1 TB/s, 743 GB LPDDR over NVLink-C2C at 388 GB/s): DeepSeek-V4-Pro IQ2_XXS, 433 GiB, 57% of weights in LPDDR, 131K context, 4.37 tok/s with every turn clean in 151 minutes. The planner's bandwidth-ratio estimate for that placement is 0.107 of all-HBM speed. The same box ran the 27B at 88.8 tok/s (llama.cpp) and 135.2 (vLLM FP8), held all four residency models (4.39 tasks/min with vLLM serving the 27B), and sustained 16 live-tool agents.

### 8. Clusters
The head is whichever machine runs `bench`. A `clusters/<name>.json` lists the workers. `bench cluster probe` probes every node, measures each link (throughput and round trip), and merges everything into one probe with every device tagged by node. Two placements: `single` keeps a model whole on one node (residency spreads models this way, nothing crosses the wire while serving), `rpc_split` slices one model across all nodes in proportion to their room via llama.cpp RPC, where every token crosses the wire. `--placement rpc_split` forces the split for a model one node could hold, which is how you measure the interconnect cost. The runner starts `ggml-rpc-server` on each worker over ssh, samples every node's memory, and tears it all down.

### 9. Engine tracks
llama.cpp runs on every machine with identical flags; that row keeps the variables constant. A second row uses the engine people really run on that platform, chosen by the catalog's preference list: vLLM on CUDA, MLX and TensorFold on Apple silicon, TensorFold on CUDA where it is qualified. `bench/engines.py` launches all four the same way (native binary or, for vLLM, the official docker image), and `bench fetch` pulls the pinned checkpoint. Nemotron 3.5 Lightning 30B-A3B is the model every engine serves at about 4 bits, so the rows compare like for like. One rule: AgentPerf's exact output policy works only on llama.cpp, so engine rows use the recorded policy and are compared with a llama.cpp row run the same way; the result file records the policy. First numbers: on a GB10, Nemotron decodes at 71 tok/s on llama.cpp, 109 on vLLM (NVFP4) and 169 on TensorFold with its drafter; on an M4, the 9B goes from 17.7 on llama.cpp to 20.0 on MLX.

## What the overnight run showed

| machine | inference 27B 65K | live tools, agents sustained | residency | ladder |
|---|---|---|---|---|
| RTX PRO 6000 96 GB + 92 GB RAM, 8 cores | 75.1 tok/s, 8.3 min | 8, but throughput peaks at 2 and wall time grows 8.6x at 8 (CPU and RAM bound, 50 GiB host) | skipped (disk) | skipped (disk) |
| GB10 128 GB unified, 20 cores | 11.6 tok/s, 50.4 min | 8 of 8 clean, throughput still rising at 8 (17.4 agg), per-agent decode 10.6 to 5.1, 102 GiB of 128 used | 3 resident, Ling evicted, 79 s swap, 0.54 tasks/min | Ling-3.0-flash 131K, 45.2 tok/s |
| Ryzen AI Max+ 495 192 GB (160 GB to GPU), Vulkan | 11.0 tok/s, 69.5 min | 2 sustained; at 4 one turn of 672 dropped with host RAM at 30.4 of 32 GiB, wall 1.5x the GB10's at every level | all 4 resident, 0 swap, 0.60 tasks/min | Ling-3.0-flash 131K, 40.5 tok/s |
| M4 16 GB | 9B at 61K: 16.1 decode, 8.4 end-to-end | 2 at 16K ctx | not applicable | same 9B |

The GB10 and the Ryzen box decode at the same speed with one agent. With live tools the Ryzen box's 32 GB host pool (what is left after the 160 GB GPU carve-out) is the limit: it fills at 4 agents while the GB10 still has 58 GB of its shared pool free. The residency test still separates them by 11% cycle time and a 79 s swap, and the ladder shows the same model at the same context on both. Only the memory tests see the extra 64 GB.

## What it does differently from other agentic benchmarks

- **llama-bench, MLPerf Inference, vLLM bench**: measure tokens per second for one model at fixed settings. No tools, no concurrency growth, no multi-model, no planning against memory.
- **AA-AgentPerf-Local**: a real agent trajectory with live tools, and the best single-run harness available, so we adopted it as the inference test instead of rewriting it. It runs one model on one server and reports speed. It does not choose what fits, sweep concurrency, or keep several models resident.
- **SWE-bench, Terminal-Bench, GAIA, tau-bench**: score model intelligence, not hardware. Same model scores the same on any box that can run it.
- **Ollama and LM Studio style "max model size" charts**: categorical fit without a workload behind it.

Agent Headroom is the only one that starts from the machine's memory, plans the workload to fill it, and reports a fit answer, a concurrency answer, a residency answer, a warm-context answer and a speed answer separately. The same four numbers scale from a 16 GB laptop to a rack because the catalog has no ceiling and the planner has no tiers.

## Things to say honestly

- Four engines run: llama.cpp everywhere as the baseline, plus vLLM (CUDA), MLX (Apple) and TensorFold (Apple and CUDA) as the "fastest thing people actually run" row. The engine rows use AgentPerf's recorded output policy because only llama.cpp honors the exact policy, so each is compared against a llama.cpp row run the same way. SGLang is catalog-only.
- Residency is text-only; vision, image and speech models are catalog entries.
- Cluster mode is llama.cpp RPC only. Coherent mode is measured on one DGX Station with naive demand-paged spill: DeepSeek-V4-Pro IQ2_XXS (433 GiB, 57% in LPDDR) at 4.37 tok/s, 168/168 turns; tensor-override placement is not implemented. warm_contexts has run only as a smoke on a GB10; the 128 vs 256 GB sweep is pending.
- Results are reproducible per catalog version and replay version; the replay is frozen per major release.
