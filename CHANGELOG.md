# Changelog

## 0.1.0 (2026-10-31)

First tagged release.

- Five tests: ladder (RUNG), inference (PACE), host_pressure (CROWD), residency (TENANTS), warm_contexts (RECALL).
- Probe, plan, run, report, remote, cluster, fetch and ssh commands; results fetched back from remote hosts into `out/remote/`.
- Engines: llama.cpp everywhere, plus vLLM, MLX and TensorFold rows on the platforms where they qualify.
- Topologies: unified, discrete, coherent (DGX Station class) and clusters over llama.cpp RPC.
- Replays: agentperf-default-v1, aa-mini-v1, and the warm-long-v1 subset for warm_contexts sweeps.
- Known limits listed in the README.
