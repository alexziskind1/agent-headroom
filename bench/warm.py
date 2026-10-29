"""warm_contexts test: how many agent contexts can host RAM keep warm?

A fleet of n agents shares one GPU server with s slots (s fixed, s < n). Between an agent's turns its
context lives in the server's host-memory prompt cache (llama.cpp --cache-ram with unified KV); when the
agent's next turn arrives the server restores the KV from host RAM instead of re-prefilling. Past the point
where n contexts exceed the cache budget, restores miss and every turn re-prefills its whole context on the
GPU: uncached prompt tokens and time-to-first-token jump and aggregate throughput falls. The knee is a
function of host RAM alone: the GPU, model, slots and per-agent work are identical across RAM sizes.

Per-agent CPU demand is small (a replay client waiting on HTTP, memcpy in the server), so the knee lands
well before any CPU wall; output per turn is capped (fixed policy) so decode time does not hide the
prefill signal and a level finishes in minutes rather than hours.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
import urllib.request

from . import __version__
from .catalog import get_model
from .planner import RUNNABLE
from .pressure import run_clients, aggregate
from .probe import fmt_bytes, run_probe
from . import engines
from .runner import (free_port, EXIT_SERVER_FAILED, MemorySampler, Refused, admit, resolve_model_file,
                     stop, wait_health, write_attached_server)

SCHEMA_VERSION = "1.0"
TASKS_PER_REPLAY = {"agentperf-default-v1": 8, "aa-mini-v1": 1}

# Subsets of agentperf-default-v1, built on demand from the installed agentperf-local package. The cache knee
# depends on how large each agent's saved context is, not on how many turns it runs: the two meeting tasks
# reach 44K and 56K prompt tokens in 41 turns (a quarter of the full replay's 168), so a level finishes about
# 4x sooner with the knee in the same place. aa-mini-v1 (one task, under 1K tokens) does not preserve it.
SUBSETS = {"warm-long-v1": ["task_meeting_council_votes", "task_meeting_council_budget"]}


def agentperf_data_root() -> str | None:
    """Where the agentperf-local on PATH keeps its bundled replays (it usually lives in another venv)."""
    exe = shutil.which("agentperf-local")
    if exe:
        try:
            with open(exe, "rb") as f:
                first = f.readline().decode(errors="replace").strip()
            py = first[2:].split()[0] if first.startswith("#!") else None
            if py and os.path.exists(py):
                r = subprocess.run([py, "-c", "from agentperf_local.common.package_paths import PACKAGE_DATA_ROOT; print(PACKAGE_DATA_ROOT)"],
                                   capture_output=True, text=True, timeout=30)
                if r.returncode == 0 and r.stdout.strip():
                    return r.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        from agentperf_local.common.package_paths import PACKAGE_DATA_ROOT  # noqa: PLC0415
        return str(PACKAGE_DATA_ROOT)
    except ImportError:
        return None


def bundled_manifest(replay_id: str = "agentperf-default-v1") -> str | None:
    root = agentperf_data_root()
    if not root:
        return None
    arch = "arm64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"
    for cand in (f"manifest-{arch}.json", "manifest.json"):
        path = os.path.join(root, "replays", replay_id, cand)
        if os.path.exists(path):
            return path
    return None


def resolve_replay(replay: str, out_root: str, log=print) -> dict:
    """A bundled replay id, a subset name from SUBSETS (materialized under out_root/replays/<name>/ with its
    traces copied beside the manifest, as agentperf-local requires), or a path to a custom manifest."""
    if replay in TASKS_PER_REPLAY:
        return {"args": ["--replay", replay], "name": replay, "tasks": TASKS_PER_REPLAY[replay], "task_ids": None, "manifest": None}
    if replay in SUBSETS:
        src = bundled_manifest()
        if not src:
            raise Refused(f"replay subset {replay} needs agentperf-local's bundled agentperf-default-v1; agentperf-local was not found")
        want = SUBSETS[replay]
        with open(src) as f:
            m = json.load(f)
        tasks = [t for t in m["tasks"] if t["task_id"] in want]
        missing = set(want) - {t["task_id"] for t in tasks}
        if missing:
            raise Refused(f"replay subset {replay}: tasks {sorted(missing)} are not in {src}")
        dest = os.path.join(out_root, "replays", replay)
        for t in tasks:
            os.makedirs(os.path.dirname(os.path.join(dest, t["trace"])), exist_ok=True)
            shutil.copyfile(os.path.join(os.path.dirname(src), t["trace"]), os.path.join(dest, t["trace"]))
        path = os.path.join(dest, "manifest.json")
        with open(path, "w") as f:
            json.dump(dict(m, name=replay, tasks=tasks), f, indent=1)
        log(f"replay {replay}: {len(tasks)} of {len(m['tasks'])} tasks from {src} -> {path}")
        return {"args": [path], "name": replay, "tasks": len(tasks), "task_ids": [t["task_id"] for t in tasks], "manifest": path}
    if replay.endswith(".json") and os.path.exists(replay):
        path = os.path.abspath(replay)
        with open(path) as f:
            m = json.load(f)
        return {"args": [path], "name": m.get("name") or os.path.basename(os.path.dirname(path)), "tasks": len(m.get("tasks") or []),
                "task_ids": [t.get("task_id") for t in m.get("tasks") or []], "manifest": path}
    raise Refused(f"unknown replay {replay}: use {', '.join(TASKS_PER_REPLAY)}, {', '.join(SUBSETS)} or a manifest path")


def aggregate_warm(level: list[dict], n: int, replay: str | int, wall_s: float | None) -> dict:
    """Pressure aggregate plus the cache signal: prompt-token hit ratio, TTFT and tasks/min."""
    agg = aggregate(level)
    ok = [c for c in level if c["summary"].get("present")]
    cached = sum((c["summary"].get("totals") or {}).get("total_server_cached_prompt_tokens") or 0 for c in ok)
    uncached = sum((c["summary"].get("totals") or {}).get("total_server_uncached_prompt_tokens") or 0 for c in ok)
    ttft50 = sorted(((c["summary"].get("latency_ms") or {}).get("time_to_first_token") or {}).get("p50") or 0 for c in ok)
    ttft95 = [((c["summary"].get("latency_ms") or {}).get("time_to_first_token") or {}).get("p95") or 0 for c in ok]
    agg.update({
        "cached_prompt_tokens": cached,
        "uncached_prompt_tokens": uncached,
        "prompt_cache_hit_ratio": round(cached / (cached + uncached), 4) if cached + uncached else None,
        "ttft_p50_ms": round(ttft50[len(ttft50) // 2]) if ttft50 else None,
        "ttft_p95_max_ms": round(max(ttft95)) if ttft95 else None,
        "tasks_per_minute": round((replay if isinstance(replay, int) else TASKS_PER_REPLAY.get(replay, 8)) * n / (wall_s / 60), 3) if wall_s else None,
    })
    return agg


def scrape_metrics(port: int) -> dict:
    """llama-server /metrics counters at the end of a level (needs --metrics)."""
    out = {}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("llamacpp:") and " " in line:
                    k, v = line.split(" ", 1)
                    try:
                        out[k.split(":", 1)[1]] = float(v)
                    except ValueError:
                        pass
    except Exception:  # noqa: BLE001
        pass
    return out


def run_warm(plan: dict, cat: dict, opts: dict, log=print) -> tuple[int, dict]:
    sel = next((s for s in plan["selections"] if s["role"] == "text" and s["status"] in RUNNABLE), None)
    if sel is None:
        raise Refused("plan has no runnable text selection")
    if sel["engine"] != "llama.cpp":
        raise Refused("warm_contexts runs on llama.cpp only (host prompt cache via --cache-ram)")
    warm = sel.get("warm")
    if not warm:
        raise Refused("plan has no warm_contexts parameters; re-plan with `bench plan warm_contexts`")
    ok, why = engines.available(sel["engine"], cat)
    if not ok:
        raise Refused(why)

    started = _dt.datetime.now(_dt.timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out_dir = opts.get("out_dir") or os.path.join("out", "runs", f"{plan['host']}-warm_contexts-{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    log(f"run dir {out_dir}")

    probe = run_probe(storage_test=False)
    for n in admit(plan, probe):
        log(f"note: {n}")
    with open(os.path.join(out_dir, "probe.json"), "w") as f:
        json.dump(probe, f, indent=2)

    model = get_model(cat, sel["model_id"])
    art = resolve_model_file(cat, sel, opts.get("model_file"), opts.get("verify", True), log)
    levels = [n for n in (opts.get("levels") or warm["levels"]) if n >= 1]
    slots = int(opts.get("slots") or warm["slots"])
    cache_ram = int(opts.get("cache_ram_bytes") or warm["cache_ram_bytes"])
    fixed_out = int(opts.get("fixed_output_tokens") or 64)
    rp = resolve_replay(opts.get("replay") or "agentperf-default-v1", out_dir, log)
    replay = rp["name"]
    port = free_port(opts.get("port", 8089))
    alias = f"{sel['model_id']}-{sel['quant']}"
    backend = probe["devices"][0]["backend"] if probe["devices"] else "cpu"
    dev_total = probe["devices"][0]["memory_total_bytes"] if probe["devices"] else None

    sampler = MemorySampler(os.path.join(out_dir, "memory.jsonl"), probe["memory"]["topology"])
    sampler.start()
    result = {
        "schema_version": SCHEMA_VERSION, "bench_version": __version__, "result_id": None,
        "plan_id": plan["plan_id"], "probe_id": probe["probe_id"], "host": probe["host"]["hostname"],
        "test": "warm_contexts", "catalog_version": plan["catalog_version"],
        "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection": sel, "artifact": art,
        "server": {"engine": sel["engine"], "version": engines.version(sel["engine"], cat)},
        "replay": {"name": replay, "tasks": rp["tasks"], "task_ids": rp["task_ids"], "manifest": rp["manifest"], "tool_mode": "none",
                   "extra_args": ["--output-token-policy", "fixed", "--max-output-tokens", str(fixed_out)] + list(opts.get("replay_extra") or [])},
        "warm": dict(warm, slots=slots, cache_ram_bytes=cache_ram, fixed_output_tokens=fixed_out, levels=levels),
        "levels": [], "max_sustained_agents": 0, "knee_measured_agents": None, "stopped_because": None,
        "memory": {"baseline_host_used_bytes": sampler._host_used()},
        "duration_s": None, "status": "failed",
    }
    code = 0
    try:
        for n in levels:
            lvl_dir = os.path.join(out_dir, f"n{n:03d}")
            os.makedirs(lvl_dir, exist_ok=True)
            level_sel = dict(sel, n_seq=slots, context_tokens=sel["context_tokens"] * slots)
            # --slot-prompt-similarity 0: llama-server otherwise routes a request to any idle slot whose content
            # shares >10% prefix with it (the shared tools/system block of every agent qualifies) and skips the
            # host cache unless that slot would lose half its context; with LRU selection every task saves the
            # outgoing state and restores the incoming agent's own context. Verified on a GB10 2026-10-28.
            extra = list(opts.get("server_extra") or []) + ["--kv-unified", "--cache-ram", str(cache_ram // 2**20),
                                                             "--slot-prompt-similarity", "0", "--metrics"]
            spec = engines.launch_spec(sel["engine"], level_sel, art["path"], alias, port, extra, cat=cat,
                                       device_total_bytes=dev_total, tool_parser=(model.get("tool_parser") or {}).get(sel["engine"]))
            argv = spec["argv"]
            attached = os.path.join(lvl_dir, "attached-server.yaml")
            if not write_attached_server(attached, model, argv, backend, engine=sel["engine"]):
                attached = None
            log(f"level n={n}: {slots} slots x {sel['context_tokens']:,} ctx, host cache {fmt_bytes(cache_ram)}, fixed {fixed_out} output tokens/turn")
            sampler.reset_level()
            with open(os.path.join(lvl_dir, "server.log"), "w") as slog:
                import subprocess
                proc = subprocess.Popen(argv, stdout=slog, stderr=subprocess.STDOUT, env=spec["env"])
                level = {"n": n, "slots": slots, "server_load_s": None, "clients": [], "aggregate": None, "server_metrics": None}
                try:
                    level["server_load_s"] = round(wait_health(port, opts.get("load_timeout", 900), proc), 1)
                    t0 = time.time()
                    level["clients"] = run_clients(n, spec["base_url"], spec["model_name"], lvl_dir, rp["args"], "none", attached,
                                                   extra=result["replay"]["extra_args"])
                    level["wall_s"] = round(time.time() - t0, 1)
                    level["aggregate"] = aggregate_warm(level["clients"], n, rp["tasks"], level["wall_s"])
                    level["server_metrics"] = scrape_metrics(port)
                except RuntimeError as e:
                    level["error"] = str(e)
                    log(f"level n={n}: server failed: {e}")
                finally:
                    stop(proc)
                    engines.run_cleanup(spec, os.path.join(lvl_dir, "server.log"))
                    level["server_exit_code"] = proc.returncode
            level.update(sampler.level_peaks())
            result["levels"].append(level)
            agg = level.get("aggregate") or {}
            log(f"level n={n}: ok={agg.get('clients_ok')} failed={agg.get('clients_failed')} hit={agg.get('prompt_cache_hit_ratio')} "
                f"ttft_p50={agg.get('ttft_p50_ms')}ms agg_tps={agg.get('aggregate_output_tokens_per_second')} tasks/min={agg.get('tasks_per_minute')} "
                f"wall={level.get('wall_s')}s peak_host={fmt_bytes(level['peak_host_used_bytes'])} cache={fmt_bytes(level['peak_page_cache_bytes'])} "
                f"swap={fmt_bytes(level['peak_swap_used_bytes'])} cpu={level.get('mean_cpu_util')}/{level.get('peak_cpu_util')}")
            if agg.get("clients_ok") == n and agg.get("failed_turns", 0) == 0 and not level.get("error"):
                result["max_sustained_agents"] = n
            else:
                result["stopped_because"] = f"n={n} had failures"
                break
        # measured knee: first level whose hit ratio drops more than 0.15 below the first level's
        hits = [(l["n"], (l.get("aggregate") or {}).get("prompt_cache_hit_ratio")) for l in result["levels"]]
        hits = [(n, h) for n, h in hits if h is not None]
        if hits:
            base = hits[0][1]
            result["knee_measured_agents"] = next((n for n, h in hits if h < base - 0.15), None)
        result["status"] = "ok" if result["max_sustained_agents"] >= 1 else "failed"
        if result["status"] != "ok":
            code = EXIT_SERVER_FAILED
    finally:
        sampler.stop()
        result["memory"].update(peak_host_used_bytes=sampler.peak_host or None, peak_gpu_used_bytes=sampler.peak_gpu or None,
                                samples=sampler.n, peak_page_cache_bytes=sampler.peak_cache or None,
                                peak_swap_used_bytes=sampler.peak_swap or None, peak_cpu_util=sampler.peak_cpu or None)
        result["duration_s"] = round((_dt.datetime.now(_dt.timezone.utc) - started).total_seconds(), 1)
        clean = {k: v for k, v in result.items() if k != "result_id"}
        result["result_id"] = hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)
        log(f"wrote {os.path.join(out_dir, 'results.json')}")
    return code, result


def summarize(result: dict) -> str:
    w = result["warm"]
    L = [f"result {result['result_id'][:12]}  {result['status']}  host={result['host']}  test=warm_contexts  {result['duration_s']}s"]
    s = result["selection"]
    L.append(f"model: {s['model_id']}/{s['quant']} ctx/agent={s['context_tokens']:,} slots={w['slots']} host cache={fmt_bytes(w['cache_ram_bytes'])} "
             f"modeled knee={w.get('knee_agents')} agents ({fmt_bytes(w.get('per_agent_host_bytes'))}/agent)  measured knee={result.get('knee_measured_agents')}")
    L.append(f"{'n':>4} {'ok':>3} {'hit':>6} {'ttft p50':>9} {'ttft p95':>9} {'agg tok/s':>10} {'tasks/min':>9} {'wall s':>7} {'peak host':>10} {'swap':>8} {'cpu':>5}")
    for l in result["levels"]:
        a = l.get("aggregate") or {}
        L.append(f"{l['n']:>4} {a.get('clients_ok', 0):>3} {(a.get('prompt_cache_hit_ratio') or 0):>6.2f} {(a.get('ttft_p50_ms') or 0) / 1000:>8.1f}s "
                 f"{(a.get('ttft_p95_max_ms') or 0) / 1000:>8.1f}s {(a.get('aggregate_output_tokens_per_second') or 0):>10.2f} "
                 f"{(a.get('tasks_per_minute') or 0):>9.2f} {(l.get('wall_s') or 0):>7.0f} {fmt_bytes(l.get('peak_host_used_bytes')):>10} "
                 f"{fmt_bytes(l.get('peak_swap_used_bytes')):>8} {(l.get('peak_cpu_util') or 0):>5.2f}")
    L.append(f"max sustained agents: {result['max_sustained_agents']}" + (f"  ({result['stopped_because']})" if result.get("stopped_because") else ""))
    return "\n".join(L)
