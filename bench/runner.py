"""bench run: execute a plan. Inference test = llama-server + agentperf-local replay.

Flow: load plan -> re-probe and admit -> resolve + verify model file -> launch server ->
sample memory -> run agentperf-local -> stop server -> write results.json.
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from . import __version__
from .catalog import get_model
from .planner import RUNNABLE
from .probe import GiB, default_cache_path, fmt_bytes, run_probe

SCHEMA_VERSION = "1.0"
EXIT_REFUSED = 4
EXIT_SERVER_FAILED = 5
EXIT_REPLAY_FAILED = 6
DRIFT_TOLERANCE = 0.02


class Refused(Exception):
    pass


# --------------------------------------------------------------------------- admission

def admit(plan: dict, probe: dict) -> list[str]:
    """Raise Refused if the machine no longer matches the plan or cannot hold it now."""
    notes = []
    pb = plan["budget"]
    if probe["host"]["hostname"] != plan["host"]:
        notes.append(f"hostname differs: plan={plan['host']} now={probe['host']['hostname']}")
    if pb["topology"] == "cluster":
        # The local probe only sees the head. Node membership is checked when the cluster is brought up.
        head = [d for d in probe["devices"]]
        planned_head = pb["nodes"][0]["devices"] if pb.get("nodes") else []
        if len(head) != len(planned_head):
            raise Refused(f"head device count changed: plan had {len(planned_head)}, now {len(head)}")
        notes.append(f"cluster plan: {len(pb['nodes'])} nodes; workers are checked at rpc-server start")
        return notes
    if len(probe["devices"]) != len(pb["gpu_bytes_per_device"]):
        raise Refused(f"device count changed: plan had {len(pb['gpu_bytes_per_device'])}, now {len(probe['devices'])}")
    if probe["memory"]["topology"] != pb["topology"]:
        raise Refused(f"topology changed: plan={pb['topology']} now={probe['memory']['topology']}")
    # Recompute the per-device budget and compare: catches memory changes beyond tolerance.
    pol = plan["policy"]
    for d, planned in zip(probe["devices"], pb["gpu_bytes_per_device"]):
        total = d["memory_total_bytes"]
        reserve = max(int(total * pol["reserve_fraction"]), pol["reserve_min_bytes"])
        now = max(total - reserve, 0)
        if pb["topology"] == "unified":
            now = min(now, max((probe["memory"]["system_total_bytes"] or 0) - pb["host_min_bytes"], 0))
        if planned and abs(now - planned) / planned > DRIFT_TOLERANCE:
            raise Refused(f"device {d['index']} budget drifted: plan={fmt_bytes(planned)} now={fmt_bytes(now)}")
    used = sum(s["estimate"]["total_bytes"] for s in plan["selections"] if s["status"] != "evicted")
    avail = probe["memory"]["system_available_bytes"] or 0
    if any(s["status"] == "forced" for s in plan["selections"]):
        # The plan was forced past the estimate on purpose; the server launch is the real test. Only the host floor is checked.
        notes.append(f"forced plan: estimate {fmt_bytes(used)} exceeds the budget by {fmt_bytes(max(-min(s['estimate']['fit_margin_bytes'] for s in plan['selections']), 0))}; skipping the fit check")
        if avail < pb["host_min_bytes"]:
            raise Refused(f"host has {fmt_bytes(avail)} available, below host_min {fmt_bytes(pb['host_min_bytes'])}")
        return notes
    if pb["topology"] == "unified":
        need = used + pb["host_min_bytes"]
        if avail < need:
            raise Refused(f"machine is busy: {fmt_bytes(avail)} available, plan needs {fmt_bytes(need)}")
    else:
        # Only the device-resident part counts against HBM: coherent_spill and offload placements keep
        # placement.host_bytes of the weights in host RAM (the DGX Station ladder was refused by the full estimate).
        dev_used = sum(s["estimate"]["total_bytes"] - int((s.get("placement") or {}).get("host_bytes") or 0)
                       for s in plan["selections"] if s["status"] != "evicted")
        for d in probe["devices"]:
            free = d.get("memory_free_bytes")
            if free is not None and free < dev_used * 1.0 / max(len(probe["devices"]), 1):
                raise Refused(f"device {d['index']} has {fmt_bytes(free)} free, plan needs {fmt_bytes(dev_used)} on device")
        if avail < pb["host_min_bytes"]:
            raise Refused(f"host has {fmt_bytes(avail)} available, below host_min {fmt_bytes(pb['host_min_bytes'])}")
    return notes


# --------------------------------------------------------------------------- model file

def sha256_file(path: str, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def resolve_model_file(cat: dict, sel: dict, override: str | None, verify: bool, log) -> dict:
    model = get_model(cat, sel["model_id"])
    file = next(f for f in model["files"] if f["quant"] == sel["quant"])
    if engines.artifact_kind(file) == "dir":
        path = override or engines.artifact_path(sel["model_id"], file)
        if not engines.dir_artifact_ok(path):
            raise Refused(f"no checkpoint directory at {path}; run `bench fetch {sel['model_id']} --quant {sel['quant']}` or pass --model-file <dir>")
        size = engines.dir_artifact_bytes(path)
        info = {"path": path, "kind": "dir", "bytes": size, "catalog_bytes": file["bytes"], "catalog_sha256": None,
                "sha256": None, "verified": size == file["bytes"], "revision": file.get("revision")}
        if size != file["bytes"]:
            log(f"warning: checkpoint is {size:,} bytes of safetensors, catalog says {file['bytes']:,}; this is not the pinned artifact")
        return info
    if override:
        path = override
    else:
        d = os.path.join(default_cache_path(), sel["model_id"], sel["quant"])
        cands = sorted(glob.glob(os.path.join(d, "*.gguf")))
        if not cands:
            raise Refused(f"no model file in {d}; download {file.get('url')} there or pass --model-file")
        path = cands[0]
    size = os.path.getsize(path)
    info = {"path": path, "kind": "file", "bytes": size, "catalog_bytes": file["bytes"], "catalog_sha256": file.get("sha256"),
            "sha256": None, "verified": False}
    parts = file.get("parts") or []
    if parts:
        # split GGUF: the catalog's bytes is the sum of all shards; the sha256 pins the first one. Every shard must be
        # present with its pinned size next to the first, or llama-server loads a truncated model.
        d = os.path.dirname(path)
        info["parts"] = []
        for pt in parts:
            pp = os.path.join(d, os.path.basename(pt["filename"]))
            have = os.path.getsize(pp) if os.path.exists(pp) else None
            info["parts"].append({"path": pp, "bytes": have, "catalog_bytes": pt["bytes"]})
            if have is None:
                raise Refused(f"split model is missing shard {pp}")
            if have != pt["bytes"]:
                log(f"warning: shard {os.path.basename(pp)} is {have:,} bytes, catalog says {pt['bytes']:,}")
        size = sum(x["bytes"] for x in info["parts"])
        info["bytes"] = size
    if verify:
        log(f"hashing {path} ({fmt_bytes(size)})")
        t0 = time.time()
        info["sha256"] = sha256_file(path)
        log(f"sha256 {info['sha256'][:16]}... in {time.time() - t0:.0f}s")
        info["verified"] = bool(file.get("sha256")) and info["sha256"] == file["sha256"]
    if size != file["bytes"]:
        log(f"warning: file is {size:,} bytes, catalog says {file['bytes']:,}; this is not the pinned artifact")
    return info


# --------------------------------------------------------------------------- server

SERVER_BIN = os.environ.get("BENCH_LLAMA_SERVER", "llama-server")
from . import engines  # noqa: E402  (engine launch specs: llama.cpp, vllm, mlx, tensorfold)


def llama_server_argv(sel: dict, model_path: str, alias: str, port: int, extra: list[str],
                      rpc: list[str] | None = None, host: str = "127.0.0.1", bin_: str | None = None) -> list[str]:
    """Mirror agentperf-local's managed llama.cpp launch so numbers are comparable.
    `rpc` (cluster plans): worker endpoints `addr:port`; layers are split by the plan's tensor_split."""
    argv = [
        bin_ or SERVER_BIN,
        "--model", model_path,
        "--alias", alias,
        "--host", host, "--port", str(port),
        "--ctx-size", str(sel["context_tokens"]),
        "--parallel", str(sel["n_seq"]),
        "--gpu-layers", "all",
        "--jinja",
        "--batch-size", "4096", "--ubatch-size", "512",
        "--flash-attn", "on",
        "--no-webui",
        "--verbosity", "1",
    ]
    pl = sel.get("placement") or {}
    if rpc and pl.get("strategy") == "rpc_split":
        argv += ["--rpc", ",".join(rpc), "--tensor-split", ",".join(str(x) for x in pl["tensor_split"])]
    return argv + list(extra)


def server_env(sel: dict) -> dict:
    """Environment for llama-server. coherent_spill: let CUDA allocations oversubscribe into host RAM
    (managed memory over NVLink-C2C), which is how a GB300 station serves weights from LPDDR."""
    env = dict(os.environ)
    if (sel.get("placement") or {}).get("strategy") == "coherent_spill":
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    return env


def free_port(preferred: int, host: str = "127.0.0.1", span: int = 50) -> int:
    """`preferred` if nothing listens on it, else the next free port. Two runs on one box (a chain plus a
    manual smoke) must not fight over 8089; the port used is recorded in the server argv."""
    import socket
    for p in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return preferred


def wait_health(port: int, timeout: float, proc: subprocess.Popen, host: str = "127.0.0.1") -> float:
    t0 = time.time()
    url = f"http://{host}:{port}/health"
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before becoming healthy")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    raise RuntimeError(f"server not healthy after {timeout:.0f}s")


def stop(proc: subprocess.Popen, grace: float = 20) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(10)


def engine_version() -> str | None:
    try:
        out = subprocess.run([SERVER_BIN, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for l in (out.stdout + out.stderr).splitlines():
        if l.strip().startswith("version"):
            return l.strip()
    return None


# --------------------------------------------------------------------------- memory sampler

def page_cache_resident_bytes(path: str, limit_bytes: int = 1 << 40) -> int | None:
    """Bytes of `path` currently in the page cache (mincore over a private read-only mmap). This is the
    warm tier of model residency: a restore from page cache is a memcpy, a restore from disk is a read.
    None on failure, on non-POSIX platforms, or if the path is not a plain local file."""
    import ctypes
    import mmap as _mmap
    try:
        if not os.path.isfile(path):
            return None
        size = os.path.getsize(path)
        if size == 0 or size > limit_bytes:
            return None
        page = _mmap.PAGESIZE
        n = (size + page - 1) // page
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        fd = os.open(path, os.O_RDONLY)
        try:
            addr = libc.mmap(None, size, _mmap.PROT_READ, _mmap.MAP_SHARED, fd, 0)
        finally:
            os.close(fd)
        if addr in (None, -1, 2**64 - 1):
            return None
        vec = (ctypes.c_ubyte * n)()
        try:
            if libc.mincore(addr, size, vec) != 0:
                return None
        finally:
            libc.munmap(addr, size)
        table = bytes(1 if b & 1 else 0 for b in range(256))
        resident = bytes(vec).translate(table).count(b"\x01")
        return min(resident * page, size)
    except Exception:  # noqa: BLE001
        return None


class MemorySampler:
    """1 Hz samples of host used bytes and GPU used bytes to a jsonl file."""

    def __init__(self, path: str, topology: str, interval: float = 1.0):
        self.path, self.topology, self.interval = path, topology, interval
        self._stop = threading.Event()
        self.peak_host = 0
        self.peak_gpu = 0
        self.peak_cache = 0   # page cache (Buffers + Cached); not in host_used, which is MemTotal - MemAvailable
        self.peak_swap = 0    # SwapTotal - SwapFree
        self.peak_cpu = 0.0   # fraction of all cores busy, from /proc/stat deltas (Linux)
        self._cpu_sum, self._cpu_n = 0.0, 0
        self._cpu_prev = None
        self.n = 0
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _host_used(self) -> int | None:
        if sys.platform == "darwin":
            try:
                total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout.strip())
                lvl = int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, timeout=5).stdout.strip())
                return int(total * (100 - lvl) / 100)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                return None
        try:
            vals = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    vals[k] = int(v.split()[0]) * 1024
            return vals["MemTotal"] - vals["MemAvailable"]
        except (OSError, KeyError, ValueError):
            return None

    def _host_cache_swap(self) -> tuple[int | None, int | None]:
        """Linux only: (page cache bytes, swap used bytes). Cache pressure does not show in host_used,
        so a RAM-bound tool workload is visible here (and in wall time) before anything fails."""
        if sys.platform == "darwin":
            return None, None
        try:
            vals = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    vals[k] = int(v.split()[0]) * 1024
            return vals["Buffers"] + vals["Cached"], vals["SwapTotal"] - vals["SwapFree"]
        except (OSError, KeyError, ValueError):
            return None, None

    def _cpu_util(self) -> float | None:
        if sys.platform == "darwin":
            return None
        try:
            with open("/proc/stat") as f:
                parts = f.readline().split()
            vals = [int(x) for x in parts[1:]]
            idle, total = vals[3] + vals[4], sum(vals)
        except (OSError, ValueError, IndexError):
            return None
        prev, self._cpu_prev = self._cpu_prev, (idle, total)
        if prev is None or total == prev[1]:
            return None
        return round(1 - (idle - prev[0]) / (total - prev[1]), 4)

    def reset_level(self):
        """Start per-level peaks (the run-wide peaks are kept in level_peaks())."""
        self._run_peaks = (self.peak_host, self.peak_gpu, self.peak_cache, self.peak_swap, self.peak_cpu)
        self.peak_host = self.peak_gpu = self.peak_cache = self.peak_swap = 0
        self.peak_cpu, self._cpu_sum, self._cpu_n = 0.0, 0.0, 0

    def level_peaks(self) -> dict:
        out = {"peak_host_used_bytes": self.peak_host or None, "peak_gpu_used_bytes": self.peak_gpu or None,
               "peak_page_cache_bytes": self.peak_cache or None, "peak_swap_used_bytes": self.peak_swap or None,
               "peak_cpu_util": self.peak_cpu or None, "mean_cpu_util": round(self._cpu_sum / self._cpu_n, 4) if self._cpu_n else None}
        rp = getattr(self, "_run_peaks", (0, 0, 0, 0, 0.0))
        self.peak_host, self.peak_gpu = max(self.peak_host, rp[0]), max(self.peak_gpu, rp[1])
        self.peak_cache, self.peak_swap, self.peak_cpu = max(self.peak_cache, rp[2]), max(self.peak_swap, rp[3]), max(self.peak_cpu, rp[4])
        return out

    _amd_used_files = None

    def _gpu_used(self) -> int | None:
        if self.topology not in ("discrete", "coherent"):
            return None
        if self._amd_used_files is None:
            import glob as _g
            self._amd_used_files = [f for f in _g.glob("/sys/class/drm/card[0-9]*/device/mem_info_vram_used")
                                    if os.path.exists(os.path.join(os.path.dirname(f), "vendor"))
                                    and open(os.path.join(os.path.dirname(f), "vendor")).read().strip() == "0x1002"]
        if self._amd_used_files:
            try:
                return sum(int(open(f).read().strip()) for f in self._amd_used_files)
            except (OSError, ValueError):
                return None
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5)
            return sum(int(x) for x in out.stdout.split()) * 1024 * 1024
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None

    def _loop(self):
        with open(self.path, "w") as f:
            while not self._stop.is_set():
                h, g = self._host_used(), self._gpu_used()
                c, s = self._host_cache_swap()
                u = self._cpu_util()
                self.n += 1
                if u is not None:
                    self.peak_cpu = max(self.peak_cpu, u)
                    self._cpu_sum += u
                    self._cpu_n += 1
                if h:
                    self.peak_host = max(self.peak_host, h)
                if g:
                    self.peak_gpu = max(self.peak_gpu, g)
                if c:
                    self.peak_cache = max(self.peak_cache, c)
                if s:
                    self.peak_swap = max(self.peak_swap, s)
                f.write(json.dumps({"t": round(time.time(), 1), "host_used_bytes": h, "gpu_used_bytes": g,
                                    "page_cache_bytes": c, "swap_used_bytes": s, "cpu_util": u}) + "\n")
                f.flush()
                self._stop.wait(self.interval)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(5)


# --------------------------------------------------------------------------- agentperf

def write_attached_server(path: str, model: dict, argv: list[str], backend: str, engine: str = "llama.cpp") -> bool:
    aa = model.get("aa")
    if not aa:
        return False
    ver = (engines.version(engine) if engine != "llama.cpp" else engine_version()) or "unknown"
    # agentperf-local's attached-server schema accepts only these frameworks; for others the run proceeds
    # without the attachment (it is provenance metadata, not part of the measurement).
    framework = {"llama.cpp": "llama-cpp", "vllm": "vllm", "sglang": "sglang"}.get(engine)
    if framework is None:
        return False
    doc = {
        "model_release_slug": aa["model_release_slug"],
        "hf_repository": aa["hf_repository"],
        "hf_revision": aa["hf_revision"],
        "framework": framework,
        "framework_version": ver,
        "accelerator_backend": backend,
        "server_launch_command": shlex.join(argv),
    }
    with open(path, "w") as f:
        for k, v in doc.items():
            f.write(f"{k}: {json.dumps(v)}\n")
    return True


def output_policy_args(engine: str, extra: list[str] | None) -> list[str]:
    """AgentPerf's default `exact` policy needs a server that honors ignore_eos and reports its context
    (llama.cpp does). vLLM stops at tool calls and mlx_lm/tensorfold ignore ignore_eos, so every other
    engine runs the `recorded` policy (model stops at its own EOS, capped at the recorded length) unless
    the caller already chose one. The policy is recorded in results.replay.extra_args."""
    extra = list(extra or [])
    if any(a.startswith("--output-token-policy") for a in extra):
        return extra
    if engine != "llama.cpp":
        extra = ["--output-token-policy", "recorded"] + extra
    return extra


def run_agentperf(base_url: str, alias: str, out_dir: str, replay: str, tool_mode: str, attached: str | None, log_path: str,
                  extra: list[str] | None = None) -> int:
    """`extra` goes to agentperf-local verbatim (e.g. --output-token-policy recorded); it is recorded in results."""
    cmd = ["agentperf-local", "run", "--replay", replay, "--base-url", base_url, "--model", alias,
           "--output-dir", out_dir, "--tool-mode", tool_mode, "--no-progress", "--power"]
    if attached:
        cmd += ["--attached-server", attached]
    cmd += list(extra or [])
    with open(log_path, "w") as log:
        log.write(shlex.join(cmd) + "\n")
        log.flush()
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return p.returncode


def ingest_summary(out_dir: str) -> dict:
    path = os.path.join(out_dir, "summary.json")
    if not os.path.exists(path):
        return {"present": False}
    with open(path) as f:
        s = json.load(f)

    def pick(d, *keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return None
        return d

    lat = s.get("latency_distributions_ms") or {}
    return {
        "present": True,
        "path": path,
        "success": s.get("success"),
        "run_id": s.get("run_id"),
        "output_tokens_per_second": s.get("output_tokens_per_second"),
        "end_to_end_output_tokens_per_second": s.get("end_to_end_output_tokens_per_second"),
        "measured_duration_ms": s.get("measured_duration_ms"),
        "wall_duration_ms": s.get("wall_duration_ms"),
        "totals": s.get("totals"),
        "latency_ms": {k: {m: v.get(m) for m in ("count", "mean", "p50", "p95")} for k, v in lat.items() if isinstance(v, dict)},
        "failed_turn_ids": s.get("failed_turn_ids"),
        "short_output_warning_turn_ids": s.get("short_output_warning_turn_ids"),
    }


# --------------------------------------------------------------------------- main

def run_plan(plan: dict, cat: dict, opts: dict, log=print) -> tuple[int, dict]:
    if plan["test"] not in ("inference", "host_pressure", "ladder"):
        raise Refused(f"run does not implement test {plan['test']} yet")
    text = [s for s in plan["selections"] if s["role"] == "text" and s["status"] in RUNNABLE]
    if not text:
        raise Refused("plan has no runnable text selection")
    sel = text[0]
    if sel["engine"] not in engines.ENGINE_IDS:
        raise Refused(f"run has no launcher for engine {sel['engine']} (have {', '.join(engines.ENGINE_IDS)})")
    ok, why = engines.available(sel["engine"], cat)
    if not ok:
        raise Refused(why)
    if shutil.which("agentperf-local") is None:
        raise Refused("agentperf-local not on PATH")

    started = _dt.datetime.now(_dt.timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out_dir = opts.get("out_dir") or os.path.join("out", "runs", f"{plan['host']}-{plan['test']}-{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    log(f"run dir {out_dir}")

    log("re-probing")
    probe = run_probe(storage_test=False)
    notes = admit(plan, probe)
    for n in notes:
        log(f"note: {n}")
    with open(os.path.join(out_dir, "probe.json"), "w") as f:
        json.dump(probe, f, indent=2)

    model = get_model(cat, sel["model_id"])
    art = resolve_model_file(cat, sel, opts.get("model_file"), opts.get("verify", True), log)

    port = free_port(opts.get("port", 8089))
    if port != opts.get("port", 8089):
        log(f"port {opts.get('port', 8089)} busy, using {port}")
    alias = f"{sel['model_id']}-{sel['quant']}"
    cluster = None
    endpoints = None
    if sel["placement"].get("strategy") == "rpc_split":
        from .cluster import ClusterRun
        cluster = ClusterRun(plan, out_dir, log)
        endpoints = cluster.start()
    dev_total = probe["devices"][0]["memory_total_bytes"] if probe["devices"] else None
    spec = engines.launch_spec(sel["engine"], sel, art["path"], alias, port, opts.get("server_extra", []), cat=cat,
                               rpc=endpoints, device_total_bytes=dev_total, tool_parser=(model.get("tool_parser") or {}).get(sel["engine"]))
    argv = spec["argv"]
    backend = probe["devices"][0]["backend"] if probe["devices"] else "cpu"
    attached = os.path.join(out_dir, "attached-server.yaml")
    if not write_attached_server(attached, model, argv, backend, engine=sel["engine"]):
        attached = None

    sampler = MemorySampler(os.path.join(out_dir, "memory.jsonl"), probe["memory"]["topology"])
    sampler.start()
    baseline_host = sampler._host_used()

    log("launching: " + shlex.join(argv))
    server_log = open(os.path.join(out_dir, "server.log"), "w")
    proc = subprocess.Popen(argv, stdout=server_log, stderr=subprocess.STDOUT, env=spec["env"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "bench_version": __version__,
        "result_id": None,
        "plan_id": plan["plan_id"],
        "probe_id": probe["probe_id"],
        "host": probe["host"]["hostname"],
        "test": plan["test"],
        "catalog_version": plan["catalog_version"],
        "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection": sel,
        "artifact": art,
        "server": {"engine": sel["engine"], "version": engines.version(sel["engine"], cat), "argv": argv, "load_s": None, "exit_code": None},
        "replay": {"name": opts.get("replay", "agentperf-default-v1"), "tool_mode": opts.get("tool_mode", "none"), "exit_code": None,
                   "extra_args": output_policy_args(sel["engine"], opts.get("replay_extra"))},
        "agentperf": None,
        "memory": {"baseline_host_used_bytes": baseline_host, "peak_host_used_bytes": None, "peak_gpu_used_bytes": None, "samples": None},
        "cluster": None,
        "duration_s": None,
        "status": "failed",
    }
    code = EXIT_SERVER_FAILED
    try:
        result["server"]["load_s"] = round(wait_health(port, opts.get("load_timeout", 900), proc), 1)
        log(f"server healthy after {result['server']['load_s']}s")
        ap_dir = os.path.join(out_dir, "agentperf")
        rc = run_agentperf(spec["base_url"], spec["model_name"], ap_dir, result["replay"]["name"],
                           result["replay"]["tool_mode"], attached, os.path.join(out_dir, "agentperf.log"),
                           extra=result["replay"]["extra_args"])
        result["replay"]["exit_code"] = rc
        result["agentperf"] = ingest_summary(ap_dir)
        if rc == 0 and result["agentperf"]["present"]:
            result["status"] = "ok"
            code = 0
        else:
            code = EXIT_REPLAY_FAILED
            log(f"agentperf-local exited {rc}; see {os.path.join(out_dir, 'agentperf.log')}")
    except RuntimeError as e:
        log(f"server failed: {e}; see {os.path.join(out_dir, 'server.log')}")
    finally:
        stop(proc)
        server_log.close()
        engines.run_cleanup(spec, os.path.join(out_dir, "server.log"))
        result["server"]["exit_code"] = proc.returncode
        sampler.stop()
        if cluster is not None:
            result["cluster"] = cluster.stop()
        result["memory"].update(peak_host_used_bytes=sampler.peak_host or None,
                                peak_gpu_used_bytes=sampler.peak_gpu or None, samples=sampler.n,
                                peak_page_cache_bytes=sampler.peak_cache or None, peak_swap_used_bytes=sampler.peak_swap or None)
        result["duration_s"] = round((_dt.datetime.now(_dt.timezone.utc) - started).total_seconds(), 1)
        clean = {k: v for k, v in result.items() if k != "result_id"}
        result["result_id"] = hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)
        log(f"wrote {os.path.join(out_dir, 'results.json')}")
    return code, result


def summarize(result: dict) -> str:
    L = [f"result {result['result_id'][:12]}  {result['status']}  host={result['host']}  test={result['test']}  {result['duration_s']}s"]
    s = result["selection"]
    L.append(f"model: {s['model_id']}/{s['quant']} ctx={s['context_tokens']:,} n_seq={s['n_seq']}  artifact verified={result['artifact']['verified']}")
    L.append(f"server: {result['server']['version']}  load={result['server']['load_s']}s")
    ap = result.get("agentperf") or {}
    if ap.get("present"):
        tot = ap.get("totals") or {}
        L.append(f"replay {result['replay']['name']} tool_mode={result['replay']['tool_mode']}: success={ap.get('success')}  "
                 f"turns={tot.get('successful_turns')} failed={tot.get('failed_turns')}  measured={(ap.get('measured_duration_ms') or 0) / 60000:.1f} min")
        L.append(f"output tokens/s: {ap.get('output_tokens_per_second') or 0:.2f} (decode)  {ap.get('end_to_end_output_tokens_per_second') or 0:.2f} (end to end)")
        for k, v in (ap.get("latency_ms") or {}).items():
            if k in ("ttft", "e2e", "decode", "first_token", "turn") or len(ap["latency_ms"]) <= 6:
                L.append(f"  {k}: p50={v.get('p50') and round(v['p50'])} ms  p95={v.get('p95') and round(v['p95'])} ms  n={v.get('count')}")
    m = result["memory"]
    L.append(f"memory: baseline host used={fmt_bytes(m['baseline_host_used_bytes'])} peak host used={fmt_bytes(m['peak_host_used_bytes'])} peak gpu used={fmt_bytes(m['peak_gpu_used_bytes'])} ({m['samples']} samples)")
    return "\n".join(L)
