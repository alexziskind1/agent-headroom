<p align="center"><img src="docs/brand/agent-headroom-wordmark.svg" alt="Agent Headroom" width="520"></p>

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

Badges: **BRIDGE** (coherent memory: HBM plus host RAM over NVLink-C2C), **FLEET** (multi-node clusters), **TRACKS** (llama.cpp baseline plus the fastest engine per platform). Brand assets and rules: `docs/brand/`. Animated explainer: `docs/explainer.html`.

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
python3 -m bench run out/plan-<host>-warm_contexts.json --levels 8,16,32,64     # fleet sweep past the modeled knee
python3 -m bench run out/plan-<host>-warm_contexts.json --replay warm-long-v1 --levels 8,12,16,24   # same knee, 4x shorter levels
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
- **The full-replay warm_contexts sweep is impractical.** Level wall time grows with agents times turns times time to first token. Use `--replay warm-long-v1` for the sweep; the full replay is for the smoke.
- **Catalog entries marked `verified: false` are estimates** and produce planner warnings. The DeepSeek-V4-Pro IQ2_XXS quant has no download URL; supply it with `--model-files`.
- **No one outside the original fleet has installed it yet.** agentperf-local, llama-server on PATH and the gated Llama 3.1 download are the likely first obstacles. Open an issue with the probe and the command.
- **No CI yet.** Run `python3 -m unittest discover tests` before sending changes; 80 tests, no network needed.

## License

Agent Headroom is released under the [Apache License 2.0](LICENSE). The Agent Headroom name and wordmark are not covered by that license; a modified copy of the suite must not present its results as Agent Headroom results.

Third-party components are used as installed tools or referenced by name, not redistributed here: [AgentPerf](https://github.com/ArtificialAnalysis/aa-agentperf-local) by Artificial Analysis (Apache-2.0, including its replay traces), [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT), vLLM, SGLang, Diffusers and TensorFold (Apache-2.0), MLX (MIT). Models in the catalog carry their publishers' own licenses.
