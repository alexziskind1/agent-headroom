"""host_pressure test: N concurrent agent replays against one server, swept upward.

Per level n: launch llama-server with --parallel n and n * context tokens, start n
agentperf-local clients at once, wait, collect. The server restarts between levels so
every level begins with an empty KV cache. Memory is sampled for the whole sweep and
peaks are recorded per level.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shlex
import subprocess
import time

from . import __version__
from .catalog import get_model
from .planner import RUNNABLE, cost, kv_bytes
from .probe import fmt_bytes, run_probe
from . import engines
from .runner import (free_port, EXIT_REPLAY_FAILED, EXIT_SERVER_FAILED, MemorySampler, Refused, admit,
                     ingest_summary, output_policy_args, resolve_model_file, stop, wait_health, write_attached_server)

SCHEMA_VERSION = "1.0"


def default_levels(max_n: int) -> list[int]:
    out, n = [], 1
    while n <= max_n:
        out.append(n)
        n *= 2
    if out and out[-1] != max_n:
        out.append(max_n)
    return out


def level_fits(cat: dict, sel: dict, plan: dict, n: int) -> tuple[bool, int]:
    """Re-run the planner's cost model with n sequences; return (fits, total_bytes)."""
    model = get_model(cat, sel["model_id"])
    file = next(f for f in model["files"] if f["quant"] == sel["quant"])
    eng = cat["engines"][sel["engine"]]
    est = cost(model, file, eng, sel["context_tokens"], n, plan["policy"]["kv_quant"], tuple(plan["budget"]["device_backends"]))
    room = max(plan["budget"]["gpu_bytes_per_device"])
    return est["total_bytes"] <= room, est["total_bytes"]


def run_clients(n: int, base_url: str, alias: str, out_root: str, replay: str | list[str], tool_mode: str, attached: str | None,
                extra: list[str] | None = None) -> list[dict]:
    """`replay` is a bundled replay id or the agentperf-local arguments that select one (e.g. a manifest path)."""
    replay_args = list(replay) if isinstance(replay, (list, tuple)) else ["--replay", replay]
    procs = []
    for i in range(n):
        out_dir = os.path.join(out_root, f"client-{i:02d}")
        cmd = ["agentperf-local", "run", *replay_args, "--base-url", base_url, "--model", alias,
               "--output-dir", out_dir, "--tool-mode", tool_mode, "--no-progress", "--no-power",
               "--cache-namespace", f"pressure-{n}-{i}-{int(time.time())}"]
        if attached:
            cmd += ["--attached-server", attached]
        cmd += list(extra or [])
        log = open(os.path.join(out_root, f"client-{i:02d}.log"), "w")
        log.write(shlex.join(cmd) + "\n")
        log.flush()
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        procs.append((i, p, log, t0, out_dir))
    results = []
    for i, p, log, t0, out_dir in procs:
        rc = p.wait()
        log.close()
        s = ingest_summary(out_dir)
        results.append({"client": i, "exit_code": rc, "wall_s": round(time.time() - t0, 1), "summary": s})
    return results


def aggregate(level: list[dict]) -> dict:
    ok = [c for c in level if c["exit_code"] == 0 and c["summary"].get("present") and c["summary"].get("success")]
    toks = sum((c["summary"]["totals"] or {}).get("total_local_output_tokens", 0) for c in ok)
    walls = [c["wall_s"] for c in level]
    tps = sorted(c["summary"]["output_tokens_per_second"] for c in ok if c["summary"].get("output_tokens_per_second"))
    failed_turns = sum((c["summary"].get("totals") or {}).get("failed_turns", 0) for c in level if c["summary"].get("present"))
    return {
        "clients_ok": len(ok),
        "clients_failed": len(level) - len(ok),
        "failed_turns": failed_turns,
        "aggregate_output_tokens_per_second": round(toks / max(walls), 2) if walls and max(walls) else None,
        "per_client_output_tps_p50": tps[len(tps) // 2] if tps else None,
        "per_client_output_tps_min": tps[0] if tps else None,
        "wall_s_max": max(walls) if walls else None,
    }


def run_pressure(plan: dict, cat: dict, opts: dict, log=print) -> tuple[int, dict]:
    sel = next((s for s in plan["selections"] if s["role"] == "text" and s["status"] in RUNNABLE), None)
    if sel is None:
        raise Refused("plan has no runnable text selection")
    if sel["engine"] not in engines.ENGINE_IDS:
        raise Refused(f"host_pressure has no launcher for engine {sel['engine']}")
    ok, why = engines.available(sel["engine"], cat)
    if not ok:
        raise Refused(why)

    started = _dt.datetime.now(_dt.timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out_dir = opts.get("out_dir") or os.path.join("out", "runs", f"{plan['host']}-host_pressure-{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    log(f"run dir {out_dir}")

    probe = run_probe(storage_test=False)
    for n in admit(plan, probe):
        log(f"note: {n}")
    with open(os.path.join(out_dir, "probe.json"), "w") as f:
        json.dump(probe, f, indent=2)

    model = get_model(cat, sel["model_id"])
    art = resolve_model_file(cat, sel, opts.get("model_file"), opts.get("verify", True), log)
    levels = opts.get("levels") or default_levels(sel.get("max_n_seq", 1))
    levels = [n for n in levels if n >= 1]
    port = free_port(opts.get("port", 8089))
    alias = f"{sel['model_id']}-{sel['quant']}"
    backend = probe["devices"][0]["backend"] if probe["devices"] else "cpu"

    cluster, endpoints = None, None
    if sel["placement"].get("strategy") == "rpc_split":
        from .cluster import ClusterRun
        cluster = ClusterRun(plan, out_dir, log)
        endpoints = cluster.start()

    sampler = MemorySampler(os.path.join(out_dir, "memory.jsonl"), probe["memory"]["topology"])
    sampler.start()
    result = {
        "schema_version": SCHEMA_VERSION, "bench_version": __version__, "result_id": None,
        "plan_id": plan["plan_id"], "probe_id": probe["probe_id"], "host": probe["host"]["hostname"],
        "test": "host_pressure", "catalog_version": plan["catalog_version"],
        "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection": sel, "artifact": art,
        "server": {"engine": sel["engine"], "version": engines.version(sel["engine"], cat)},
        "replay": {"name": opts.get("replay", "aa-mini-v1"), "tool_mode": opts.get("tool_mode", "fixed_delay"),
                   "extra_args": output_policy_args(sel["engine"], opts.get("replay_extra"))},
        "levels": [], "max_sustained_agents": 0, "stopped_because": None, "cluster": None,
        "memory": {"baseline_host_used_bytes": sampler._host_used()},
        "duration_s": None, "status": "failed",
    }
    code = 0
    try:
        for n in levels:
            fits, total = level_fits(cat, sel, plan, n)
            if not fits:
                result["stopped_because"] = f"n={n} needs {fmt_bytes(total)} which exceeds the GPU budget"
                log(result["stopped_because"])
                break
            lvl_dir = os.path.join(out_dir, f"n{n:02d}")
            os.makedirs(lvl_dir, exist_ok=True)
            per_seq = kv_bytes(model, sel["context_tokens"], 1, plan["policy"]["kv_quant"])
            if sel["engine"] == "llama.cpp":
                level_sel = dict(sel, n_seq=n, context_tokens=sel["context_tokens"] * n)  # llama.cpp splits ctx over slots
            else:
                level_sel = dict(sel, n_seq=n)  # vllm/mlx/tensorfold take per-sequence context and a sequence cap
            dev_total = probe["devices"][0]["memory_total_bytes"] if probe["devices"] else None
            spec = engines.launch_spec(sel["engine"], level_sel, art["path"], alias, port, opts.get("server_extra", []), cat=cat,
                                       rpc=endpoints, device_total_bytes=dev_total, tool_parser=(model.get("tool_parser") or {}).get(sel["engine"]))
            argv = spec["argv"]
            attached = os.path.join(lvl_dir, "attached-server.yaml")
            if not write_attached_server(attached, model, argv, backend, engine=sel["engine"]):
                attached = None
            log(f"level n={n}: launching server (ctx {level_sel['context_tokens']:,}, kv {fmt_bytes(per_seq * n)})")
            peak_before = (sampler.peak_host, sampler.peak_gpu, sampler.peak_cache, sampler.peak_swap)
            sampler.peak_host, sampler.peak_gpu, sampler.peak_cache, sampler.peak_swap = 0, 0, 0, 0
            with open(os.path.join(lvl_dir, "server.log"), "w") as slog:
                proc = subprocess.Popen(argv, stdout=slog, stderr=subprocess.STDOUT, env=spec["env"])
                level = {"n": n, "estimated_total_bytes": total, "server_load_s": None, "clients": [], "aggregate": None,
                         "peak_host_used_bytes": None, "peak_gpu_used_bytes": None}
                try:
                    level["server_load_s"] = round(wait_health(port, opts.get("load_timeout", 900), proc), 1)
                    t0 = time.time()
                    level["clients"] = run_clients(n, spec["base_url"], spec["model_name"], lvl_dir,
                                                   result["replay"]["name"], result["replay"]["tool_mode"], attached,
                                                   extra=result["replay"]["extra_args"])
                    level["wall_s"] = round(time.time() - t0, 1)
                    level["aggregate"] = aggregate(level["clients"])
                except RuntimeError as e:
                    level["error"] = str(e)
                    log(f"level n={n}: server failed: {e}")
                finally:
                    stop(proc)
                    engines.run_cleanup(spec, os.path.join(lvl_dir, "server.log"))
                    level["server_exit_code"] = proc.returncode
            level["peak_host_used_bytes"] = sampler.peak_host or None
            level["peak_gpu_used_bytes"] = sampler.peak_gpu or None
            level["peak_page_cache_bytes"] = sampler.peak_cache or None
            level["peak_swap_used_bytes"] = sampler.peak_swap or None
            sampler.peak_host = max(sampler.peak_host, peak_before[0])
            sampler.peak_gpu = max(sampler.peak_gpu, peak_before[1])
            sampler.peak_cache = max(sampler.peak_cache, peak_before[2])
            sampler.peak_swap = max(sampler.peak_swap, peak_before[3])
            result["levels"].append(level)
            agg = level.get("aggregate") or {}
            log(f"level n={n}: ok={agg.get('clients_ok')} failed={agg.get('clients_failed')} "
                f"agg_tps={agg.get('aggregate_output_tokens_per_second')} per_client_p50={agg.get('per_client_output_tps_p50')} "
                f"peak_host={fmt_bytes(level['peak_host_used_bytes'])} cache={fmt_bytes(level['peak_page_cache_bytes'])} swap={fmt_bytes(level['peak_swap_used_bytes'])}")
            if agg.get("clients_ok") == n and agg.get("failed_turns", 0) == 0 and not level.get("error"):
                result["max_sustained_agents"] = n
            else:
                result["stopped_because"] = f"n={n} had failures"
                break
        result["status"] = "ok" if result["max_sustained_agents"] >= 1 else "failed"
        if result["status"] != "ok":
            code = EXIT_SERVER_FAILED
    finally:
        sampler.stop()
        if cluster is not None:
            result["cluster"] = cluster.stop()
        result["memory"].update(peak_host_used_bytes=sampler.peak_host or None, peak_gpu_used_bytes=sampler.peak_gpu or None,
                                samples=sampler.n, peak_page_cache_bytes=sampler.peak_cache or None,
                                peak_swap_used_bytes=sampler.peak_swap or None)
        result["duration_s"] = round((_dt.datetime.now(_dt.timezone.utc) - started).total_seconds(), 1)
        clean = {k: v for k, v in result.items() if k != "result_id"}
        result["result_id"] = hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)
        log(f"wrote {os.path.join(out_dir, 'results.json')}")
    return code, result


def summarize(result: dict) -> str:
    L = [f"result {result['result_id'][:12]}  {result['status']}  host={result['host']}  test=host_pressure  {result['duration_s']}s"]
    s = result["selection"]
    L.append(f"model: {s['model_id']}/{s['quant']} ctx/agent={s['context_tokens']:,}  replay={result['replay']['name']} tool_mode={result['replay']['tool_mode']}")
    L.append(f"{'n':>3} {'ok':>3} {'fail':>4} {'agg tok/s':>10} {'p50 tok/s':>10} {'wall s':>7} {'peak host':>10} {'load s':>6}")
    for l in result["levels"]:
        a = l.get("aggregate") or {}
        L.append(f"{l['n']:>3} {a.get('clients_ok', 0):>3} {a.get('clients_failed', 0):>4} "
                 f"{(a.get('aggregate_output_tokens_per_second') or 0):>10.2f} {(a.get('per_client_output_tps_p50') or 0):>10.2f} "
                 f"{(l.get('wall_s') or 0):>7.0f} {fmt_bytes(l.get('peak_host_used_bytes')):>10} {(l.get('server_load_s') or 0):>6.1f}")
    L.append(f"max sustained agents: {result['max_sustained_agents']}" + (f"  (stopped: {result['stopped_because']})" if result.get("stopped_because") else ""))
    return "\n".join(L)
