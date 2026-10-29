# Probe + Planner Specification

Version 0.1 — 2026-10-25

The probe measures the machine. The planner turns the probe, the model catalog, and a policy into an exact, reproducible plan. The runner executes a plan and nothing else. Nothing in this layer is allowed to download or launch a server.

```
bench probe               -> probe.json
bench plan  <test>        -> plan.json      (always a dry run)
bench run   <plan.json>   -> results        (separate command; re-probes and diffs first)
```

---

## 1. Probe

### 1.1 Rules

- Measure, never infer. Every memory number records the method that produced it.
- Report raw capacity. Reserves and safety margins belong to the planner.
- Never allocate GPU memory beyond what a query requires.
- Short measured tests are allowed (storage read speed, link bandwidth) and must each finish in under 10 s.
- Output is a single JSON document. Unknown values are `null`, never omitted.

### 1.2 Memory measurement by platform

| Platform | GPU addressable bytes | Method recorded as |
|---|---|---|
| NVIDIA discrete | `cudaMemGetInfo` total, per device | `cuda_mem_get_info` |
| NVIDIA unified (GB10, Grace) | `cudaMemGetInfo` total on device 0 | `cuda_mem_get_info` |
| Apple Silicon | `recommendedMaxWorkingSetSize` from Metal, plus `iogpu.wired_limit_mb` sysctl if set | `metal_working_set` |
| AMD ROCm | HIP `hipMemGetInfo` | `hip_mem_get_info` |
| CPU only | none | — |

`topology` is `discrete` when GPU memory is physically separate from host RAM, `unified` when they share one pool, `cpu_only` when there is no accelerator. It is decided only by the device's integrated flag (CUDA `CU_DEVICE_ATTRIBUTE_INTEGRATED`, Apple Silicon). It is never inferred from sizes: a 96 GB discrete card beside 92 GB of RAM looks unified by size and is not. On unified systems `system_total_bytes` and the device `memory_total_bytes` overlap and the planner must not add them.

### 1.3 probe.json schema

```json
{
  "schema_version": "1.0",
  "probe_version": "0.1.0",
  "timestamp": "2026-10-25T16:02:11Z",
  "probe_id": "sha256 of this document minus timestamp and probe_id",

  "host": {
    "hostname": "host1",
    "os": "linux", "os_version": "Ubuntu 24.04", "kernel": "6.17.0-1008-nvidia", "arch": "aarch64",
    "cpu": { "model": "NVIDIA Grace", "cores": 20, "threads": 20 }
  },

  "memory": {
    "topology": "unified",
    "system_total_bytes": 130000000000,
    "system_available_bytes": 120000000000,
    "swap_total_bytes": 0,
    "huge_pages": false
  },

  "devices": [
    {
      "index": 0,
      "vendor": "nvidia",
      "name": "NVIDIA GB10",
      "backend": "cuda",
      "memory_total_bytes": 128000000000,
      "memory_free_bytes": 119000000000,
      "memory_method": "cuda_mem_get_info",
      "compute_capability": "12.1",
      "driver_version": "580.65",
      "memory_clock_khz": 4266000,
      "memory_bus_width_bits": 256,
      "bandwidth_gbps_measured": null
    }
  ],

  "interconnect": {
    "intra_node": [
      { "from": 0, "to": 1, "type": "nvlink|pcie|none", "bandwidth_gbps_measured": null }
    ]
  },

  "storage": {
    "model_cache_path": "/home/user/.cache/bench/models",
    "free_bytes": 1800000000000,
    "read_gbps_measured": 3.1,
    "device_type": "nvme|sata|network|unknown"
  },

  "software": {
    "python": "3.12.4",
    "cuda": "13.0", "rocm": null, "metal": null,
    "docker": true,
    "engines": { "llama.cpp": "b6821", "vllm": null, "sglang": null, "mlx": null, "ollama": null }
  },

  "cluster": {
    "node_id": "host1",
    "nodes": ["host1"],
    "links": []
  },

  "warnings": [
    "nvidia-smi reports memory.total as N/A on unified systems; cudaMemGetInfo used instead"
  ]
}
```

A single node is a cluster of one. The cluster probe (later) merges per-node documents and fills `cluster.links` with measured pairwise bandwidth and latency.

---

## 2. Catalog

The planner needs these fields per model. Everything else in the catalog is informational.

```yaml
catalog_version: "2026.10"
models:
  - id: qwen3.8-27b
    modality: text            # text | vision | image_gen | asr | tts | embed
    params_total: 27000000000
    params_active: 27000000000
    arch: { n_layers: 64, n_kv_heads: 8, head_dim: 128 }
    kv_bytes_per_token: 262144          # 2 * n_layers * n_kv_heads * head_dim * 2 bytes (fp16)
    activation_overhead_fraction: 0.05  # of weight bytes
    context: { min: 8192, max: 131072 }
    files:
      - quant: q4_k_m
        bytes: 16500000000
        sha256: "..."
        url: "..."
        engines: [llama.cpp]
      - quant: fp8
        bytes: 28000000000
        sha256: "..."
        url: "..."
        engines: [vllm, sglang]
    rank: 27000000000                   # ladder ordering; defaults to params_total
```

Image, speech, and embedding entries carry a fixed `working_set_bytes` instead of `kv_bytes_per_token`.

Engine overhead constants live in the catalog too, under `engines:` (for example a CUDA context of 1 GiB for llama.cpp, 2 GiB for vLLM).

---

## 3. Planner

### 3.1 Inputs

- `probe.json`
- catalog file (pinned by `catalog_version`)
- test name: `inference | ladder | host_pressure | residency`
- test preset (for `residency`: the named model set; for `inference`/`host_pressure`: the fixed model id)
- policy, all with defaults:

| Policy key | Default | Meaning |
|---|---|---|
| `reserve_fraction` | 0.10 | fraction of addressable GPU memory left free |
| `reserve_min_bytes` | 2 GiB | floor for the reserve |
| `host_reserve_fraction` | 0.25 | host RAM left for tools and OS (discrete) |
| `host_min_bytes` | 16 GiB | unified only: RAM the GPU budget must leave for OS, tools, sandboxes |
| `allow_offload` | false | may weights spill to host RAM |
| `allow_multi_device` | true | may use tensor/pipeline parallel |
| `kv_quant` | fp16 | KV cache precision |
| `n_seq` | 1 | concurrent sequences the KV pool must hold |
| `target_context` | null | fixed context; if null the planner maximizes it |

Any override passed on the command line is written into `plan.policy`, so results carry it.

### 3.2 Budget

```
gpu_budget[d]  = device[d].memory_total_bytes * (1 - reserve_fraction), minus at least reserve_min_bytes
gpu_budget_total = sum over usable devices
host_budget    = system_total_bytes * (1 - host_reserve_fraction)
host_min       = min(host_min_bytes, 20% of system_total_bytes)
```

A plan describes the hardware, not what happens to be running on it, so budgets use totals. Two probes of the same hardware under different load produce the same `plan_id`. `system_available_bytes` only drives a "machine is busy" warning in the plan and the admission check in `bench run`.

On `unified` topology both draw from one pool, so `gpu_budget[d]` is additionally capped at `system_total_bytes - host_min`. The GPU budget is a ceiling, not an allocation: after selection the planner reports `host_bytes_after_plan = system_total_bytes - sum(selected totals)` and warns if it is below `host_min_bytes`.

Engine preference: for a given file, engines detected on the host (probe `software.engines`) are tried first. A selection on an engine that is not installed is allowed but produces a warning, so a plan can be built on one machine for another.

### 3.3 Cost of a candidate

```
weights   = file.bytes
kv        = kv_bytes_per_token * context * n_seq * kv_quant_factor
act       = weights * activation_overhead_fraction
engine    = engines[engine].overhead_bytes
total     = weights + kv + act + engine
```

Placement:

- `single`: total must fit `gpu_budget[d]` for one device.
- `tensor_parallel` over N devices: `(weights + kv + act) / N + engine` must fit each device. Requires `allow_multi_device` and a link type the recipe permits.
- `offload`: weights beyond the GPU budget go to host RAM. Requires `allow_offload`. Plan records `gpu_layers` and `host_bytes`. Status becomes `fits_with_offload`.

### 3.4 Selection per test

- **ladder**: walk candidates by descending `rank`, then descending quant precision. Pick the first that fits at `context.min`. Then raise context until the budget is exhausted or `context.max` is reached.
- **inference** / **host_pressure**: fixed model. Compute the largest context it fits; fail if below `context.min`. For `host_pressure` the planner also reports the max `n_seq` that fits the KV pool at `target_context`.
- **residency**: given the preset's model list, mark each as `resident` or `evicted` by packing in preset order. Report the total resident bytes and the eviction list. Never substitute a model.

Every rejected candidate is listed with the exact shortfall in bytes.

### 3.5 plan.json schema

```json
{
  "schema_version": "1.0",
  "plan_id": "sha256 of this document minus plan_id and created",
  "created": "2026-10-25T16:03:00Z",
  "probe_id": "sha256 from probe.json",
  "catalog_version": "2026.10",
  "test": "ladder",
  "preset": null,
  "policy": { "reserve_fraction": 0.10, "allow_offload": false, "n_seq": 1, "kv_quant": "fp16", "target_context": null },

  "budget": {
    "gpu_bytes_per_device": [115200000000],
    "gpu_bytes_total": 115200000000,
    "host_bytes": 3000000000,
    "host_bytes_after_plan": 67000000000,
    "system_available_bytes": 120000000000,
    "topology": "unified",
    "device_backends": ["cuda"],
    "installed_engines": ["llama.cpp"]
  },

  "selections": [
    {
      "role": "text",
      "model_id": "qwen3.8-27b",
      "quant": "q4_k_m",
      "engine": "llama.cpp",
      "recipe_id": "qwen38-27b-q4_k_m/cuda-unified",
      "placement": { "strategy": "single", "devices": [0], "gpu_layers": null, "host_bytes": 0 },
      "context_tokens": 131072,
      "n_seq": 1,
      "estimate": {
        "weights_bytes": 16500000000,
        "kv_bytes": 34359738368,
        "activation_bytes": 825000000,
        "engine_bytes": 1073741824,
        "total_bytes": 52758480192,
        "fit_margin_bytes": 62441519808
      },
      "status": "fits"
    }
  ],

  "rejected": [
    { "model_id": "ling-3.0-flash", "quant": "q4_k_m", "reason": "weights exceed gpu budget", "shortfall_bytes": 4100000000 }
  ],

  "downloads": [
    { "model_id": "qwen3.8-27b", "quant": "q4_k_m", "bytes": 16500000000, "present": false }
  ],
  "download_total_bytes": 16500000000,

  "estimated_runtime_s": 2400,
  "warnings": []
}
```

Status values: `fits`, `fits_with_offload`, `does_not_fit`, `resident`, `evicted`.

---

## 4. Dry-run behavior

`bench plan` is always a dry run. It:

1. Runs the probe (or accepts `--probe probe.json`).
2. Builds the plan and writes `plan.json`.
3. Prints a human summary: budget per device, each selection with context and margin, rejected candidates with shortfall, downloads needed with total size, estimated runtime, warnings.
4. Exits `0` if every required role fits, `2` if nothing fits, `3` on probe failure.

It never downloads, never launches a server, never writes outside the output directory.

Guarantees:

- **Deterministic.** Same `probe_id`, catalog version, and policy produce a byte-identical plan. `plan_id` is the proof.
- **Explainable.** `--explain` prints the full cost arithmetic for every candidate, accepted or rejected.
- **Binding.** `bench run` takes a `plan.json`, re-probes, and refuses to start if device memory or device count differ from the plan's probe by more than 2 percent. It then executes exactly the selections in the plan.
- **Overridable, visibly.** `--model`, `--context`, `--reserve`, `--allow-offload`, `--n-seq` are accepted and recorded in `plan.policy`. A plan with overrides is tagged `custom: true` in results.

---

## 5. Run (inference and host_pressure)

`bench run plan.json` is the only command that executes anything.

1. Re-probe. Refuse (exit 4) if device count or topology changed, any per-device budget drifted more than 2 percent, or available memory is below what the plan needs plus `host_min`.
2. Resolve the model file from `~/.cache/bench/models/<id>/<quant>/` or `--model-file`. Hash it. `artifact.verified` is true only if the SHA-256 matches the catalog. A size mismatch is a warning, not a refusal, so a local copy can be used for smoke tests.
3. Launch `llama-server` with the flags AA-AgentPerf-Local's managed mode uses: `--ctx-size`, `--parallel n_seq`, `--gpu-layers all`, `--jinja`, `--batch-size 4096 --ubatch-size 512`, `--flash-attn on`. Wait for `/health`. Record load time.
4. Sample host used bytes (and GPU used bytes on discrete) at 1 Hz to `memory.jsonl`.
5. Run `agentperf-local run --replay <name> --tool-mode <mode>` against the server. Pass an attached-server description when the catalog entry carries AA identity fields, so the run could be submitted.
6. Stop the server, write `results.json` with plan and probe ids, selection, artifact hash, server argv and version, AA summary subset, memory peaks, and duration.

Catalog files may carry `fixed_overhead_bytes` (constant state plus measured runtime overhead from an AA recipe). When present it replaces the activation fraction and engine overhead in the cost model. Multi-part GGUFs list `parts`; `bytes` is their sum and the runner opens the first part.

## 5b. host_pressure run

Sweeps agent concurrency against one server. Levels default to 1, 2, 4, ... up to the plan's `max_n_seq` (override with `--levels 1,2,4,8`). For each level n the server restarts with `--parallel n` and `n * context_tokens` total context (llama.cpp splits context across slots), then n `agentperf-local` clients start together, each with its own cache namespace and output directory. Default replay is `aa-mini-v1` with `fixed_delay` tools; `--tool-mode live` runs the recorded tools in Docker and is the mode that loads host RAM.

Per level: clients ok/failed, failed turns, aggregate output tokens/s (sum of client output tokens over the slowest client's wall time), per-client p50 and min tokens/s, server load time, peak host and GPU memory. The sweep stops at the first level with any failure or when the cost model says the next level exceeds the GPU budget. `max_sustained_agents` is the last clean level.

## 5c. residency run (v0, text only)

Every `resident` text model in the plan gets its own `llama-server` on its own port, all started together; startup time per model is recorded. The cycle then visits the preset in order and replays `aa-mini-v1` against each model. An `evicted` model costs a swap: the largest resident server is stopped, the evicted model is loaded and replayed, then the resident one is reloaded. Swap seconds (everything except the replay itself) are summed. Score is tasks per minute over the cycle plus total swap seconds. Non-text models in the preset are skipped and listed in the result. Use the `residency-text-workstation` preset for v0.

## 5d. report and remote

`bench report [roots...]` renders one markdown table per test from every `results.json` found under the given roots (default `out/`, which includes runs fetched from other machines). `bench remote <host> <bench args...>` ships the package over SSH, runs the command there with `~/.local/bin` on PATH, and copies the remote `out/` tree back to `out/remote/<host>/`.

## 5e. Coherent memory (DGX Station GB300, GH200) — v0.2

A third topology, `coherent`: two physical pools (HBM on the GPU, LPDDR on the CPU) that are both GPU-addressable over a hardware-coherent link (NVLink-C2C).

**Detection.** Only from CUDA device attributes, never from sizes: `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES` (100) = 1 with `INTEGRATED` (18) = 0. Measured: GB10 reports 100 = 1 but is integrated (so `unified`); an RTX PRO 6000 reports 100 = 0 while still reporting `PAGEABLE_MEMORY_ACCESS` (88) = 1 through HMM, which is why 88 alone is not the signal. The probe records `host_coherent`, `pageable_host_access`, `numa_config` and `host_link` (`nvlink-c2c` or `pcie`) per device.

**Bandwidth.** The full probe measures two numbers per CUDA device with the driver API: `bandwidth_gbps_measured` (device-to-device copy, 512 MiB x4, read+write counted) and `host_to_device_gbps_measured` (pinned host-to-device copy, the link the GPU would read spilled weights over). GB10 measured 222 and 59 GB/s. The old objection to a derived figure stands; these are measured.

**Budget.** Device budget as for discrete, plus `host_coherent_bytes` = system RAM x (1 - host_reserve_fraction): the host tier weights may spill into. Tools and sandboxes share that tier, so `host_bytes_after_plan` subtracts the spill.

**Placement `coherent_spill`.** Tried after `single` and the single-node split and before `offload`. KV, activations and engine overhead must fit device memory; weights fill the rest of the device and the remainder goes to the host tier. The selection records `host_bytes`, `gpu_weight_bytes`, `spill_fraction` and, when both bandwidths were measured, `decode_speed_relative_est` = 1 / ((1 - f) + f x bw_device / bw_link). Status `fits_coherent`. Discrete machines never take this path.

**Residency** gains a third state, `resident_coherent`: whole on one device plus the host tier. The run reports it separately from `resident` and `evicted`, so a station shows hot (HBM) / warm (LPDDR) / cold (reload) instead of a binary.

**Run.** `llama-server` is launched with `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` for `coherent_spill` selections; the sampler reads device memory through nvidia-smi as on discrete.

## 5f. Clusters — v0.2

The head is whichever machine runs `bench`. `clusters/<name>.json` lists workers from the head's point of view: `{"name": ..., "workers": [{"ssh": "worker1", "addr": "192.0.2.11", "rpc_port": 18052}]}`.

**`bench cluster probe <spec>`** probes the head, deploys the package to each worker and probes it there, measures each head-to-worker link with a built-in TCP test (256 MiB transfer plus 50 one-byte round trips: `gbps_measured`, `rtt_ms`), and writes one merged probe: topology `cluster`, every device in one list with `node`, `node_topology`, `node_system_total_bytes`, and `cluster.node_probes` carrying each node's RAM, topology, ssh target, address and rpc port.

**Budget.** Each device is budgeted by its own node's topology (a Spark is unified, a workstation is discrete). `budget.nodes` lists nodes with their device indices.

**Placements.**
- `single`: whole model on one node's device. Residency uses only this, so models spread across nodes and never cross the network.
- `rpc_split` (llama.cpp RPC): layers on every node's device in proportion to its room (`tensor_split`), every node paying its own engine overhead. Status `fits_cluster`. Chosen when no node holds the model alone, or forced with `--placement rpc_split` to measure the interconnect cost of a model that one node could run.

**Run.** For `rpc_split`, the runner starts `ggml-rpc-server -H 0.0.0.0 -p <port>` on every worker over ssh, waits for the port, launches the head's `llama-server --rpc a:p,b:p --tensor-split ...`, streams `bench sample` from every worker into `memory-<host>.jsonl`, and records per-node peaks under `results.cluster`. For residency, a model placed on a worker is launched there over ssh from that worker's own model cache and replayed at `addr:port`. Requires `ggml-rpc-server` and `llama-server` on every node (builds with `GGML_RPC=ON`).

**Admission.** The head re-probes itself; worker membership is verified when their rpc-servers come up.

## 6. Out of scope for 0.2

- vLLM/SGLang (`ray`) and MLX distributed cluster placements: catalog capabilities exist, no runner.
- Heterogeneous multi-GPU within one node (uneven splits), MoE expert offload, NUMA-aware placement, NVMe-streamed weights.
- Catalog in YAML. 0.1 ships `bench/catalog.json`; the loader can grow a YAML path once PyYAML is a dependency.
- Any execution.
