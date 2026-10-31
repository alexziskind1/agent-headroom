<p align="center"><img src="docs/brand/agent-headroom-wordmark.svg" alt="Agent Headroom" width="520"></p>
<p align="center"><a href="https://agentheadroom.com">agentheadroom.com</a></p>

# Agent Headroom

**Version 0.1.0, early release.** The five tests run end to end on CUDA, Metal, Vulkan and ROCm and have been cross-checked on five machines, but only inside one fleet so far. Expect install friction, read [Known limits](#known-limits) before quoting numbers, and treat the result schema as version 1.0 of a moving target.

Memory-aware benchmarks for agent fleets. Probes a machine, plans the largest workload that fits, runs it, reports.

Five tests, each measuring a different kind of headroom, all driven by the [AgentPerf](https://pypi.org/project/agentperf-local/) agentic replay:

| test | CLI id | headroom | question |
|---|---|---|---|
| **RUNG** | `ladder` | size | the largest model and context that fit |
| **PACE** | `inference` | speed | decode and end-to-end tok/s on the frozen replay |
| **CROWD** | `host_pressure` | concurrency | concurrent agents before the first failure |
| **TENANTS** | `residency` | model | models kept loaded, swap cost |
| **RECALL** | `warm_contexts` | context | agent contexts kept warm in host RAM |

> **Agents:** if you are an AI agent operating a machine, go to [Agents start here](#agents-start-here) at the end of this file and follow it in order.

Badges: **BRIDGE** (coherent memory: HBM plus host RAM over NVLink-C2C), **FLEET** (multi-node clusters), **TRACKS** (llama.cpp baseline plus the fastest engine per platform). Brand assets and rules: `docs/brand/`. Animated explainer: `docs/explainer.html`.

## Usage

Status: probe, planner, `run` for inference, host_pressure, and text-only residency (llama.cpp + AA-AgentPerf-Local replay), `report`, `remote`, coherent-memory placement (GB300 station class) and llama.cpp RPC clusters.

```
python3 -m bench probe                                   # measure this machine -> out/probe-<host>.json
python3 -m bench plan ladder                             # largest text model + context that fits
python3 -m bench plan inference --model qwen3.8-27b      # fixed model, max context
python3 -m bench plan host_pressure --model qwen3.8-27b --context 65536
python3 -m bench plan residency --preset residency-workstation
python3 -m bench plan warm_contexts --model qwen3.8-27b --context 65536 --slots 8   # how many agent contexts host RAM keeps warm
python3 -m bench plan ladder --probe out/probe-x.json --explain   # plan for another machine's probe
python3 -m bench run out/plan-<host>-inference.json --replay aa-mini-v1    # smoke run
python3 -m bench run out/plan-<host>-inference.json                         # full 8-task replay
python3 -m bench run out/plan-<host>-host_pressure.json --levels 1,2,4        # concurrency sweep
python3 -m bench run out/plan-<host>-residency-text.json                      # multi-model residency
python3 -m bench run out/plan-<host>-warm_contexts.json --levels 8,12,16,24     # fleet sweep past the modeled knee (warm-long-v1)
python3 -m bench run out/plan-<host>-warm_contexts.json --replay agentperf-default-v1 --levels 8   # full replay, one level
python3 -m bench remote <host> probe                                           # same commands on another machine
python3 -m bench cluster probe clusters/sparks.json                             # multi-node: merge node probes, measure links
python3 -m bench plan inference --probe out/probe-sparks.json --model qwen3.8-27b --placement rpc_split   # force the split
python3 -m bench plan residency --probe out/probe-sparks.json --preset residency-text-workstation         # one model per node
python3 -m bench report                                                       # tables across all results
python3 -m unittest discover -s tests
```

`run` re-probes, refuses if the hardware drifted or the machine is busy (exit 4), hashes the model file against the catalog, launches `llama-server` with the same flags AA-AgentPerf-Local's managed mode uses, samples memory at 1 Hz, replays the workload with `agentperf-local`, and writes `results.json` plus the raw AA output under `out/runs/<host>-<test>-<stamp>/`.

Requirements for `run`: `llama-server` and `agentperf-local` on PATH (`pip install agentperf-local` into a venv and symlink it), and the model file under `~/.cache/bench/models/<id>/<quant>/` or passed with `--model-file`.

`plan` is always a dry run. It never downloads or launches anything. Exit 0 if the plan fits, 2 if nothing fits, 3 on probe failure.

Spec: `docs/probe_planner_spec.md`. Test outline with per-hardware expectations: `docs/test_outline.md`. Written explainer: `docs/explainer.md`, animated: `docs/explainer.html`. Models used: `docs/models.md`. Catalog: `bench/catalog.json` (entries marked `verified: false` are estimates and produce warnings).

No dependencies beyond Python 3.10+. CUDA is read through `libcuda` via ctypes, so unified-memory boxes that report N/A in nvidia-smi still get real numbers.

## Topologies

| topology | detected by | budget | placements |
|---|---|---|---|
| discrete | CUDA integrated=0, not coherent | VRAM minus 10% reserve; host RAM for tools | single, pipeline split, offload (opt-in) |
| unified | CUDA integrated=1, Apple Silicon, AMD APU without carve-out | one pool minus a host floor of min(16 GiB, 20%) | single |
| coherent | CUDA attribute 100 (host page tables) with integrated=0: GH200, GB300 station | VRAM as discrete plus a GPU-addressable host tier | single, coherent_spill |
| cluster | `bench cluster probe` | every node's devices, each by its own topology | single per node, rpc_split |

Cluster runs need `ggml-rpc-server` and `llama-server` on every node (llama.cpp built with `-DGGML_RPC=ON`), the model file on the head (rpc_split) or in each worker's cache (residency), and ssh from the head to every worker.

## Engines

llama.cpp is the baseline row on every machine. The "best available" row uses the fastest engine the platform has: vLLM on CUDA (native `vllm serve`, or the `vllm/vllm-openai` docker image named in the catalog), MLX (`mlx_lm.server` from `~/.venvs/mlx`) and TensorFold (`~/.tensorfold/bin/tensorfold`) on Apple silicon, TensorFold also on CUDA. `bench/engines.py` holds the launch specs; `bench plan --engine best` picks by the catalog's `engine_preference`, `--engine <name>` forces one.

```
bench fetch nemotron-3.5-lightning-30b-a3b --quant mlx-4bit     # pinned HF revision into ~/.cache/bench/models/<id>/<quant>/
bench plan inference --model nemotron-3.5-lightning-30b-a3b --engine tensorfold --quant mlx-4bit --context 32768
bench run out/plan-<host>-inference.json --replay aa-mini-v1
bench run <plan> --output-policy recorded                          # the llama.cpp row to compare an engine row against
```

Measurement rule: AgentPerf's default `exact` output policy only works on llama.cpp (it needs `ignore_eos` and a reported context). Every other engine runs the `recorded` policy automatically and the result says so under `replay.extra_args`; compare it only with a llama.cpp row run with `--output-policy recorded` at the same context. Nemotron 3.5 Lightning 30B-A3B is the model every engine serves at about 4 bits.

## Known limits

What 0.1.0 does not do, or does only partly. Each is stated in the result file where it applies.

- **Residency is text-only and single-device.** On a multi-GPU box run it as a one-GPU row (`CUDA_VISIBLE_DEVICES=0` for probe, plan and run, plus `--replay-arg=--device --replay-arg=0`). llama-server is not pinned per device, so a two-GPU residency plan would split every resident across both cards and the per-device accounting would be wrong.
- **Clusters are llama.cpp RPC only.** Residency on cluster workers launches llama-server over ssh; other engines run only on the head.
- **Coherent memory is measured on one machine class** (DGX Station GB300) with naive demand-paged spill into LPDDR. Tensor-override placement is not implemented.
- **Engine rows use AgentPerf's recorded output policy.** Compare a vLLM, MLX or TensorFold row only with a llama.cpp row run with `--output-policy recorded` at the same context; the exact policy works on llama.cpp alone.
- **warm_contexts sweeps use the warm-long-v1 subset by default** (the two long meeting tasks, 41 turns). Level wall time grows with agents times turns times time to first token, so the full replay is opt-in with `--replay agentperf-default-v1` and is meant for one level, not a sweep.
- **Catalog entries marked `verified: false` are estimates** and produce planner warnings. The DeepSeek-V4-Pro IQ2_XXS quant has no download URL; supply it with `--model-files`.
- **No CI yet.** Run `python3 -m unittest discover tests` before sending changes; 80 tests, no network needed.

## Agents start here

This section is written for an AI agent operating a machine over a shell. Follow it in order. Humans: see [Usage](#usage) above.

**Contract.** `plan` never downloads or launches anything. `run` launches one inference server, replays a recorded agent workload against it, samples memory at 1 Hz, and writes one directory. Every command is idempotent except `run`, which must not be interrupted mid-level. Exit codes: 0 ok, 2 nothing fits, 3 probe failed, 4 refused (hardware drifted since the plan, machine busy, wrong model file, unknown replay), 5 server failed to start, 6 replay failed. A non-zero exit is a result, not a bug: read the last 40 lines of stderr and the result file, report them, do not retry blindly.

**Rules while a run is active.** One run per machine at a time. No model downloads, builds, package installs or other GPU or RAM consumers on that machine. Do not stop a server you did not start. If `run` exits 4 with "machine is busy", something else holds memory: find it, do not kill it, report it.

**Step 0: preconditions.** Run every check; fix or report before planning.

```
python3 --version                      # >= 3.10
llama-server --version                 # llama.cpp build; put ~/.local/bin or the build dir on PATH
agentperf-local --help | head -1       # pip install agentperf-local  (use a venv, symlink the script onto PATH)
nvidia-smi -L 2>/dev/null || true      # CUDA boxes; AMD boxes need rocm-smi or amdgpu sysfs; Macs need nothing
df -h ~ | tail -1                      # model files land in ~/.cache/bench/models; the 27B Q4_K_M is 16 GiB
```

**Step 1: probe.** Writes `out/probe-<host>.json`. Read `memory.topology` (unified, discrete, coherent, cluster), `memory.system_total_bytes`, `memory.system_available_bytes`, `devices[*].memory_total_bytes`, `devices[*].backend`, and `warnings[]`. If `devices` is empty the box is CPU-only and nothing below applies.

```
python3 -m bench probe
```

**Step 2: plan.** One plan per test. Each writes `out/plan-<host>-<test>.json`. Exit 2 means nothing in the catalog fits; the file still lists `rejected[]` with reasons. Read `selections[0].model_id`, `.quant`, `.context_tokens`, `.estimate.total_bytes`, `.status`, and `warnings[]`. A plan with `warnings` about `verified: false` entries is an estimate; say so when reporting.

```
python3 -m bench plan inference     --model qwen3.8-27b --context 65536
python3 -m bench plan host_pressure --model qwen3.8-27b --context 65536
python3 -m bench plan ladder
python3 -m bench plan residency     --preset residency-text-workstation
python3 -m bench plan warm_contexts --model qwen3.8-27b --context 65536 --slots 8
```

**Step 3: fetch the model.** The catalog pins every file by byte count and SHA-256. `run` refuses a file that does not match. Fetch only when no run is active on the machine.

```
python3 -m bench fetch qwen3.8-27b --quant q4_k_m     # -> ~/.cache/bench/models/qwen3.8-27b/q4_k_m/
```

**Step 4: smoke, then run.** Always smoke first with `aa-mini-v1` (one task, six turns, under a minute). Then run the real replay. Pass `--out-dir` so the directory name is yours. Expected durations on a 128 GB class box: inference 50 to 70 min, host_pressure 1 to 2 h for levels 1,2,4,8,16, residency 5 to 20 min, warm_contexts 1 to 3 h for levels 8,12,16,24, ladder 10 to 40 min.

```
python3 -m bench run out/plan-<host>-inference.json --replay aa-mini-v1 --out-dir out/runs/<host>-inference-smoke
python3 -m bench run out/plan-<host>-inference.json --out-dir out/runs/<host>-inference-01
python3 -m bench run out/plan-<host>-host_pressure.json --levels 1,2,4,8,16 --out-dir out/runs/<host>-host_pressure-01
python3 -m bench run out/plan-<host>-residency.json --out-dir out/runs/<host>-residency-01
python3 -m bench run out/plan-<host>-warm_contexts.json --levels 8,12,16,24 --out-dir out/runs/<host>-warm_contexts-01
python3 -m bench run out/plan-<host>-ladder.json --out-dir out/runs/<host>-ladder-01
```

Run them in the background with output to a log file and poll the log; a run can take hours and must survive your session. On a multi-GPU discrete box run residency as a one-GPU row: `CUDA_VISIBLE_DEVICES=0` on probe, plan and run, plus `--replay-arg=--device --replay-arg=0`.

**Step 5: read the result.** Every run directory holds `results.json`, `probe.json`, `server.log`, `memory.jsonl`, `attached-server.yaml` and the raw replay output under `agentperf/`. Top-level keys common to all tests: `status` (ok or failed), `test`, `host`, `selection`, `server.version`, `server.argv`, `artifact.sha256`, `artifact.verified`, `memory.peak_host_used_bytes`, `memory.peak_gpu_used_bytes`, `memory.peak_swap_used_bytes`, `duration_s`. Per test:

| test | read | means |
|---|---|---|
| inference | `agentperf.output_tokens_per_second`, `agentperf.end_to_end_output_tokens_per_second`, `agentperf.totals.successful_turns` of 168, `agentperf.wall_duration_ms` | decode speed, speed including prefill, completeness, wall |
| host_pressure | `max_sustained_agents`, `stopped_because`, `levels[*].n`, `levels[*].aggregate.aggregate_output_tokens_per_second`, `levels[*].aggregate.failed_turns`, `levels[*].peak_host_used_bytes` | concurrency headroom and why it ended |
| residency | `resident[]`, `evicted[]`, `cycle[*].swap_s`, `cycle[*].victims` (evicted entries only), `cycle[*].restore_page_cache_bytes`, `swap_s_total`, `tasks_per_minute` | models kept loaded, what a swap costs |
| warm_contexts | `max_sustained_agents`, `knee_measured_agents`, `levels[*].n`, `levels[*].aggregate.prompt_cache_hit_ratio`, `levels[*].aggregate.ttft_p50_ms`, `levels[*].aggregate.tasks_per_minute`, `levels[*].peak_host_used_bytes` | contexts kept warm before restores become re-prefills |
| ladder | `selection.model_id`, `selection.context_tokens`, `agentperf.output_tokens_per_second` | the top rung and its speed |

Then fold everything you have:

```
python3 -m bench report            # markdown tables per test over out/ and out/remote/
```

**Step 6: report.** Send the run directory path, `result_id`, `status`, the per-test numbers above, `memory.peak_*`, `server.version`, and every `warnings[]` from the probe and the plan. Quote numbers with their context (model, quant, context_tokens, replay name, output policy from `replay.extra_args`). Never compare an engine row to a llama.cpp row that did not use `--output-policy recorded`. Never present a planner estimate as a measurement.

**Remote machines.** The same commands run on another host from here; the package is shipped over ssh and `out/` comes back to `out/remote/<host>/`. The target needs Python, `llama-server` and `agentperf-local` on PATH for a non-interactive shell (`~/.local/bin` is added automatically). Jump hosts and per-hop keys: `jump/target?i=~/.ssh/key`.

```
python3 -m bench remote <host> probe
python3 -m bench remote <host> plan inference --model qwen3.8-27b --context 65536
python3 -m bench remote <host> run out/plan-<host>-inference.json --out-dir out/runs/<host>-inference-01
```

**When to stop and ask a human.** The probe reports a topology or backend you did not expect. The plan's `rejected[]` names the model you were told to run. `run` exits 4 twice for the same reason. A GPU error, ECC or row-remap counter changes during a run. Free disk under twice the model size. Anything that would require killing a process you did not start.

## License

Agent Headroom is released under the [Apache License 2.0](LICENSE). The Agent Headroom name and wordmark are not covered by that license; a modified copy of the suite must not present its results as Agent Headroom results.

Third-party components are used as installed tools or referenced by name, not redistributed here: [AgentPerf](https://github.com/ArtificialAnalysis/aa-agentperf-local) by Artificial Analysis (Apache-2.0, including its replay traces), [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), vLLM, SGLang, Diffusers and TensorFold (Apache-2.0), MLX (MIT). Models in the catalog carry their publishers' own licenses.
