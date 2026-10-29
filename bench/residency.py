"""residency test (v0, text-only): keep a fixed set of models resident and cycle work across them.

The plan marks each preset model `resident` or `evicted`. Every resident model gets its
own llama-server, all started together. The cycle then visits each model in preset
order and replays the mini workload against it. An evicted model costs a swap: one
resident server is stopped, the evicted model is loaded, its replay runs, then the
resident one is reloaded. Load and reload seconds are part of the score.

Score: tasks per minute over the cycle, plus total swap seconds. A machine that holds
the whole set pays zero swap time.
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
from .probe import fmt_bytes, run_probe
from .planner import RESIDENT
from .remote import ssh_argv
from . import engines
from .runner import (output_policy_args, EXIT_SERVER_FAILED, MemorySampler, Refused, admit, engine_version, ingest_summary,
                     llama_server_argv, resolve_model_file, server_env, stop, wait_health, write_attached_server, page_cache_resident_bytes)

SCHEMA_VERSION = "1.0"


class Server:
    """One llama-server. `node` (cluster plans) is a worker dict {hostname, ssh, addr}: the server is launched
    over ssh on that machine with its own copy of the model, and replayed against addr:port."""

    def __init__(self, sel: dict, path: str, port: int, out_dir: str, extra: list[str], node: dict | None = None, cat: dict | None = None,
                 device_total_bytes: int | None = None):
        self.sel, self.path, self.port, self.out_dir, self.extra, self.node, self.cat = sel, path, port, out_dir, extra, node, cat
        self.device_total_bytes = device_total_bytes
        self.alias = f"{sel['model_id']}-{sel['quant']}"
        self.proc = None
        self.load_s = None
        self.spec = None

    @property
    def engine(self) -> str:
        return self.sel.get("engine") or "llama.cpp"

    @property
    def model_name(self) -> str:
        """What clients send as `model` (mlx_lm.server keys on the path)."""
        return engines.model_name_for(self.engine, self.alias, self.path)

    @property
    def host(self) -> str:
        return self.node["addr"] if self.node else "127.0.0.1"

    def start(self, timeout: float) -> float:
        bind = "0.0.0.0" if self.node else "127.0.0.1"
        argv = llama_server_argv(self.sel, self.path, self.alias, self.port, self.extra, host=bind, bin_="llama-server" if self.node else None)
        log = open(os.path.join(self.out_dir, f"server-{self.sel['model_id']}-{self.port}.log"), "a")
        if self.node:
            env = "GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 " if self.sel["placement"].get("strategy") == "coherent_spill" else ""
            # the model path may be a shell glob on the worker (its own ~/.cache/bench/models/...)
            script = f"export PATH=$HOME/.local/bin:$PATH; exec {env}" + " ".join(shlex.quote(a) if a != self.path else f"$(ls {self.path} | head -1)" for a in argv)
            log.write(f"[{self.node['hostname']}] {script}\n")
            log.flush()
            self.proc = subprocess.Popen(ssh_argv(self.node["ssh"], script), stdout=log, stderr=subprocess.STDOUT)
        else:
            tool_parser = None
            if self.cat:
                tool_parser = (get_model(self.cat, self.sel["model_id"]).get("tool_parser") or {}).get(self.engine)
            # device_total sizes vLLM's --gpu-memory-utilization to this model's share, so several servers can share one GPU
            self.spec = engines.launch_spec(self.engine, self.sel, self.path, self.alias, self.port, self.extra, cat=self.cat,
                                            device_total_bytes=self.device_total_bytes, tool_parser=tool_parser)
            argv = self.spec["argv"]
            log.write(shlex.join(argv) + "\n")
            log.flush()
            self.proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=self.spec["env"])
        self.load_s = round(wait_health(self.port, timeout, self.proc, host=self.host), 1)
        return self.load_s

    def stop(self):
        if self.proc is not None:
            stop(self.proc)
            engines.run_cleanup(self.spec)
            self.proc = None
            if self.node:
                subprocess.run(ssh_argv(self.node["ssh"], f"pkill -f 'llama-server.*--port {self.port}' || true"), capture_output=True, timeout=30)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


def replay_once(server: Server, out_dir: str, replay: str, tool_mode: str, attached: str | None) -> dict:
    cmd = ["agentperf-local", "run", "--replay", replay, "--base-url", server.base_url, "--model", server.model_name,
           "--output-dir", out_dir, "--tool-mode", tool_mode, "--no-progress", "--no-power"]
    if attached:
        cmd += ["--attached-server", attached]
    cmd += output_policy_args(server.engine, None)
    t0 = time.time()
    with open(out_dir + ".log", "w") as log:
        log.write(shlex.join(cmd) + "\n")
        log.flush()
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return {"exit_code": rc, "wall_s": round(time.time() - t0, 1), "summary": ingest_summary(out_dir)}


def run_residency(plan: dict, cat: dict, opts: dict, log=print) -> tuple[int, dict]:
    sels = [s for s in plan["selections"] if s["modality"] == "text"]
    skipped = [s for s in plan["selections"] if s["modality"] != "text"]
    if not sels:
        raise Refused("residency v0 needs at least one text model in the preset")
    for eng in sorted({s["engine"] for s in sels if s["status"] in RESIDENT or s["status"] == "evicted"}):
        if eng not in engines.ENGINE_IDS:
            raise Refused(f"residency has no launcher for engine {eng}")
        ok, why = engines.available(eng, cat)
        if not ok:
            raise Refused(why)

    started = _dt.datetime.now(_dt.timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out_dir = opts.get("out_dir") or os.path.join("out", "runs", f"{plan['host']}-residency-{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    log(f"run dir {out_dir}")
    probe = run_probe(storage_test=False)
    for n in admit(plan, probe):
        log(f"note: {n}")
    with open(os.path.join(out_dir, "probe.json"), "w") as f:
        json.dump(probe, f, indent=2)

    # Cluster plans: a model placed on a worker runs there, from that worker's own model cache.
    nodes = plan["budget"].get("nodes") or []
    def node_of(sel):
        devs = (sel.get("placement") or {}).get("devices") or []
        for nd in nodes:
            if devs and devs[0] in nd["devices"] and nd["node"] != 0:
                return nd
        return None
    if any(s["engine"] != "llama.cpp" and node_of(s) for s in sels):
        raise Refused("residency on cluster workers is llama.cpp only (remote launch runs llama-server over ssh)")

    # Resolve every model file up front so a missing file fails before any server starts.
    arts = {}
    for s in sels:
        override = (opts.get("model_files") or {}).get(s["model_id"])
        nd = node_of(s)
        if nd:
            arts[s["model_id"]] = {"path": override or f"~/.cache/bench/models/{s['model_id']}/{s['quant']}/*.gguf", "node": nd["hostname"],
                                   "bytes": None, "catalog_bytes": None, "catalog_sha256": None, "sha256": None, "verified": False}
        else:
            arts[s["model_id"]] = resolve_model_file(cat, s, override, opts.get("verify", True), log)

    extra = opts.get("server_extra", [])
    base_port = opts.get("port", 8089)
    load_timeout = opts.get("load_timeout", 900)
    replay = opts.get("replay", "aa-mini-v1")
    tool_mode = opts.get("tool_mode", "none")
    backend = probe["devices"][0]["backend"] if probe["devices"] else "cpu"

    sampler = MemorySampler(os.path.join(out_dir, "memory.jsonl"), probe["memory"]["topology"])
    sampler.start()
    result = {
        "schema_version": SCHEMA_VERSION, "bench_version": __version__, "result_id": None,
        "plan_id": plan["plan_id"], "probe_id": probe["probe_id"], "host": probe["host"]["hostname"],
        "test": "residency", "preset": plan["preset"], "catalog_version": plan["catalog_version"],
        "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "server": {"engine": "llama.cpp", "version": engine_version(),
                   "engines": {e: engines.version(e, cat) for e in sorted({x.get("engine") or "llama.cpp" for x in plan["selections"]})}},
        "replay": {"name": replay, "tool_mode": tool_mode},
        "resident": [s["model_id"] for s in sels if s["status"] in RESIDENT],
        "resident_coherent": [s["model_id"] for s in sels if s["status"] == "resident_coherent"],
        "placement": {s["model_id"]: dict(s["placement"], node=(node_of(s) or {}).get("hostname")) for s in sels},
        "evicted": [s["model_id"] for s in sels if s["status"] == "evicted"],
        "skipped_non_text": [s["model_id"] for s in skipped],
        "artifacts": arts,
        "startup": [], "cycle": [], "swap_s_total": 0.0, "tasks": 0, "tasks_per_minute": None,
        "memory": {"baseline_host_used_bytes": sampler._host_used()},
        "duration_s": None, "status": "failed",
    }
    servers: dict[str, Server] = {}
    dev_total = probe["devices"][0]["memory_total_bytes"] if probe.get("devices") else None
    code = EXIT_SERVER_FAILED
    try:
        # 1. Bring every resident model up, all at once.
        port = base_port
        for s in sels:
            if s["status"] not in RESIDENT:
                continue
            srv = Server(s, arts[s["model_id"]]["path"], port, out_dir, extra, node=node_of(s), cat=cat, device_total_bytes=dev_total)
            port += 1
            log(f"starting resident {s['model_id']} on :{srv.port}")
            servers[s["model_id"]] = srv  # registered before start so a failed launch is still cleaned up
            load = srv.start(load_timeout)
            result["startup"].append({"model_id": s["model_id"], "port": srv.port, "load_s": load})
            log(f"  up in {load}s")
        result["memory"]["resident_host_used_bytes"] = sampler._host_used()

        # 2. Cycle through the preset in order. Evicted models swap in and back out.
        t_cycle = time.time()
        for s in sels:
            mid = s["model_id"]
            entry = {"model_id": mid, "status": s["status"], "swap_s": 0.0}
            attached = os.path.join(out_dir, f"attached-{mid}.yaml")
            model = get_model(cat, mid)
            if s["status"] in RESIDENT:
                srv = servers[mid]
                if not write_attached_server(attached, model, (srv.spec or {}).get("argv") or llama_server_argv(s, srv.path, srv.alias, srv.port, extra), backend, engine=srv.engine):
                    attached = None
                entry["replay"] = replay_once(srv, os.path.join(out_dir, f"replay-{mid}"), replay, tool_mode, attached)
            else:
                # Evict the largest resident server, load this one, run, restore.
                victim_id = max(servers, key=lambda k: servers[k].sel["estimate"]["total_bytes"]) if servers else None
                t0 = time.time()
                if victim_id:
                    log(f"evicting {victim_id} to make room for {mid}")
                    servers[victim_id].stop()
                victim_node = servers[victim_id].node if victim_id else None
                tmp = Server(s, arts[mid]["path"], port, out_dir, extra, node=victim_node, cat=cat, device_total_bytes=dev_total)
                try:
                    load = tmp.start(load_timeout)
                except RuntimeError as e:
                    entry["error"] = f"could not load evicted model: {e}"
                    tmp.stop()
                    load = None
                entry["load_s"] = load
                if load is not None:
                    if not write_attached_server(attached, model, (tmp.spec or {}).get("argv") or llama_server_argv(s, tmp.path, tmp.alias, tmp.port, extra), backend, engine=tmp.engine):
                        attached = None
                    entry["replay"] = replay_once(tmp, os.path.join(out_dir, f"replay-{mid}"), replay, tool_mode, attached)
                    tmp.stop()
                if victim_id:
                    v = servers[victim_id]
                    if not v.node:
                        entry["restore_page_cache_bytes"] = page_cache_resident_bytes(v.path)
                        entry["restore_file_bytes"] = os.path.getsize(v.path) if os.path.isfile(v.path) else None
                    log(f"restoring {victim_id}" + (f" ({fmt_bytes(entry['restore_page_cache_bytes'])} of {fmt_bytes(entry.get('restore_file_bytes'))} in page cache)" if entry.get("restore_page_cache_bytes") is not None else ""))
                    try:
                        entry["restore_s"] = servers[victim_id].start(load_timeout)
                    except RuntimeError as e:
                        # a failed restore is a residency result (the swap did not round-trip), not a crash
                        entry["error"] = f"could not restore {victim_id}: {e}"
                        entry["restore_s"] = None
                        servers[victim_id].stop()
                entry["swap_s"] = round((time.time() - t0) - (entry.get("replay", {}).get("wall_s") or 0), 1)
                result["swap_s_total"] = round(result["swap_s_total"] + entry["swap_s"], 1)
            rp = entry.get("replay") or {}
            ok = rp.get("exit_code") == 0 and (rp.get("summary") or {}).get("success")
            entry["ok"] = bool(ok)
            if ok:
                result["tasks"] += (rp["summary"].get("totals") or {}).get("successful_turns", 0) and 1
            result["cycle"].append(entry)
            log(f"{mid}: {'ok' if ok else 'FAILED'}  wall={rp.get('wall_s')}s  swap={entry['swap_s']}s  tps={(rp.get('summary') or {}).get('output_tokens_per_second')}")
        cycle_s = time.time() - t_cycle
        result["cycle_s"] = round(cycle_s, 1)
        result["tasks_per_minute"] = round(result["tasks"] / (cycle_s / 60), 3) if cycle_s else None
        result["status"] = "ok" if all(e["ok"] for e in result["cycle"]) else "partial"
        code = 0 if result["status"] == "ok" else EXIT_SERVER_FAILED
    finally:
        for srv in servers.values():
            srv.stop()
        sampler.stop()
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
    L = [f"result {result['result_id'][:12]}  {result['status']}  host={result['host']}  test=residency preset={result['preset']}  {result['duration_s']}s"]
    L.append(f"resident: {', '.join(result['resident']) or 'none'}   evicted: {', '.join(result['evicted']) or 'none'}")
    if result.get("skipped_non_text"):
        L.append(f"skipped (non-text, v0): {', '.join(result['skipped_non_text'])}")
    for s in result["startup"]:
        L.append(f"  up {s['model_id']} :{s['port']} in {s['load_s']}s")
    for e in result["cycle"]:
        rp = e.get("replay") or {}
        tps = (rp.get("summary") or {}).get("output_tokens_per_second")
        L.append(f"  {e['status']:>8} {e['model_id']:<18} {'ok' if e['ok'] else 'FAIL':<4} wall={rp.get('wall_s')}s swap={e['swap_s']}s tps={tps and round(tps, 2)}")
    L.append(f"tasks/min: {result['tasks_per_minute']}   swap total: {result['swap_s_total']}s   cycle: {result.get('cycle_s')}s")
    m = result["memory"]
    L.append(f"memory: baseline={fmt_bytes(m.get('baseline_host_used_bytes'))} resident={fmt_bytes(m.get('resident_host_used_bytes'))} peak={fmt_bytes(m.get('peak_host_used_bytes'))}")
    return "\n".join(L)
