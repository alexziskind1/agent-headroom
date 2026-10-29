"""Clusters: several machines driven from the head, which is wherever `bench` runs.

cluster.json (written from the head's point of view):
    {"name": "example", "workers": [{"ssh": "worker1", "addr": "192.0.2.11", "rpc_port": 50052}]}

    bench cluster probe cluster.json                 -> out/probe-<name>.json: every node's probe merged, links measured
    bench plan inference --probe out/probe-<name>.json --model qwen3.8-27b
    bench plan residency --probe out/probe-<name>.json --preset residency-text-workstation
    bench run out/plan-<name>-inference.json         -> rpc-server on each worker over ssh, llama-server --rpc on the head

Placement on a cluster:
    single      one model whole on one node's device (residency spreads models across nodes this way)
    rpc_split   llama.cpp RPC: layers on every node in proportion to its room; every token crosses the network

The merged probe keeps every device in one list; device["node"] says which machine it lives on and
device["node_topology"] how that machine budgets it (a Spark is unified, a workstation is discrete).
"""
from __future__ import annotations

import json
import os
import shlex
import socket
import struct
import subprocess
import sys
import threading
import time

from .probe import _stable_id, run_probe
from .remote import PKG_DIR, _q, deploy, ssh_argv

NETTEST_BYTES = 256 * 1024 * 1024
RPC_BIN = os.environ.get("BENCH_RPC_SERVER", "ggml-rpc-server")
REMOTE_PREFIX = (f"export PATH=$HOME/.local/bin:$PATH; cd {PKG_DIR} && "
                 "PY=$(for c in python3.12 python3.13 python3.14 python3; do command -v $c >/dev/null && echo $c && break; done); ")


def load_spec(path: str) -> dict:
    with open(path) as f:
        spec = json.load(f)
    spec.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    for w in spec["workers"]:
        w.setdefault("rpc_port", 18052)  # macOS ephemeral ports start at 49152; stay below
        w.setdefault("addr", w["ssh"].split("/")[-1].split("?")[0])
    spec["_path"] = os.path.abspath(path)
    return spec


def remote_bench(host: str, args: list[str], **popen) -> subprocess.Popen:
    """`python3 -m bench <args>` on a node (package already deployed), as a Popen."""
    script = REMOTE_PREFIX + "exec $PY -m bench " + " ".join(_q(a) for a in args)
    return subprocess.Popen(ssh_argv(host, script), **popen)


def remote_shell(host: str, script: str, **popen) -> subprocess.Popen:
    return subprocess.Popen(ssh_argv(host, "export PATH=$HOME/.local/bin:$PATH; " + script), **popen)


# --------------------------------------------------------------------------- link test

def nettest_serve(port: int, once: bool = True) -> None:
    """Accept a client, swallow its payload, then echo 50 one-byte pings. Prints `ready` when listening."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    print(f"ready {srv.getsockname()[1]}", flush=True)  # port 0 = let the OS pick one
    while True:
        conn, _ = srv.accept()
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            n = struct.unpack("!Q", _recv_exact(conn, 8))[0]
            left = n
            buf = bytearray(4 * 1024 * 1024)
            while left > 0:
                k = conn.recv_into(buf, min(len(buf), left))
                if not k:
                    break
                left -= k
            conn.sendall(b"OK")
            for _ in range(50):
                b = conn.recv(1)
                if not b:
                    break
                conn.sendall(b)
        if once:
            break
    srv.close()


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    out = b""
    while len(out) < n:
        b = conn.recv(n - len(out))
        if not b:
            raise ConnectionError("peer closed")
        out += b
    return out


def nettest_client(addr: str, port: int, nbytes: int = NETTEST_BYTES) -> dict:
    s = socket.create_connection((addr, port), timeout=60)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    with s:
        chunk = b"\0" * (4 * 1024 * 1024)
        s.sendall(struct.pack("!Q", nbytes))
        t0 = time.perf_counter()
        left = nbytes
        while left > 0:
            k = min(len(chunk), left)
            s.sendall(chunk[:k])
            left -= k
        _recv_exact(s, 2)
        dt = time.perf_counter() - t0
        rtts = []
        for _ in range(50):
            t1 = time.perf_counter()
            s.sendall(b"x")
            _recv_exact(s, 1)
            rtts.append(time.perf_counter() - t1)
        rtts.sort()
    return {"gbps_measured": round(nbytes / dt / 1e9, 2), "rtt_ms": round(rtts[len(rtts) // 2] * 1000, 3), "bytes": nbytes}


def measure_link(worker: dict, port: int = 0) -> dict:
    p = remote_bench(worker["ssh"], ["nettest", "serve", "--port", str(port), "--once"],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout
    t0 = time.time()
    ready = False
    while time.time() - t0 < 60:
        line = p.stdout.readline()
        if not line:
            break
        if line.startswith("ready"):
            ready = True
            port = int(line.split()[1])
            break
    if not ready:
        p.kill()
        _remote_pkill(worker, f"bench nettest serve --port {port}")
        return {"gbps_measured": None, "rtt_ms": None, "error": "nettest server did not start"}
    try:
        res = nettest_client(worker["addr"], port)
    except OSError as e:
        res = {"gbps_measured": None, "rtt_ms": None, "error": f"nettest to {worker['addr']}:{port} failed: {e}"}
    finally:
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
            _remote_pkill(worker, f"bench nettest serve --port {port}")
    return res


def _remote_pkill(worker: dict, pattern: str) -> None:
    subprocess.run(ssh_argv(worker["ssh"], f"pkill -f {_q(pattern)} || true"), capture_output=True, timeout=30)


# --------------------------------------------------------------------------- cluster probe

def probe_node(worker: dict, log=print) -> dict:
    deploy(worker["ssh"])
    p = remote_bench(worker["ssh"], ["probe", "--no-storage-test", "--json", "--out", "out/probe-node.json"],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(timeout=300)
    i = out.find("{")
    if p.returncode != 0 or i < 0:
        raise RuntimeError(f"probe on {worker['ssh']} failed: {err.strip()[:400] or out[:400]}")
    return json.loads(out[i:])


def _node_summary(doc: dict, worker: dict | None) -> dict:
    return {"hostname": doc["host"]["hostname"], "topology": doc["memory"]["topology"],
            "system_total_bytes": doc["memory"]["system_total_bytes"],
            "system_available_bytes": doc["memory"]["system_available_bytes"],
            "cpu_cores": doc["host"]["cpu"]["cores"], "os": doc["host"]["os"], "arch": doc["host"]["arch"],
            "engines": doc["software"]["engines"],
            "ssh": worker["ssh"] if worker else None, "addr": worker["addr"] if worker else None,
            "rpc_port": worker["rpc_port"] if worker else None}


def probe_cluster(spec: dict, storage_test: bool = True, log=print) -> dict:
    head = run_probe(storage_test=storage_test)
    node_probes = [_node_summary(head, None)]
    devices = []
    warnings = [f"{head['host']['hostname']}: {w}" for w in head["warnings"]]
    for d in head["devices"]:
        devices.append(dict(d, node=0, node_index=d["index"], node_hostname=head["host"]["hostname"],
                            node_topology=head["memory"]["topology"], node_system_total_bytes=head["memory"]["system_total_bytes"]))
    links = []
    for i, w in enumerate(spec["workers"], 1):
        log(f"probing node {i}: {w['ssh']}")
        doc = probe_node(w, log)
        node_probes.append(_node_summary(doc, w))
        warnings += [f"{doc['host']['hostname']}: {x}" for x in doc["warnings"]]
        for d in doc["devices"]:
            devices.append(dict(d, index=len(devices), node=i, node_index=d["index"], node_hostname=doc["host"]["hostname"],
                                node_topology=doc["memory"]["topology"], node_system_total_bytes=doc["memory"]["system_total_bytes"]))
        log(f"measuring link to {w['addr']}")
        link = measure_link(w)
        links.append(dict(link, **{"from": head["host"]["hostname"], "to": doc["host"]["hostname"], "type": "tcp"}))
        if link.get("error"):
            warnings.append(link["error"])
    merged = dict(head)
    merged["memory"] = dict(head["memory"], topology="cluster", head_topology=head["memory"]["topology"])
    merged["devices"] = devices
    merged["interconnect"] = dict(head.get("interconnect", {}), inter_node=links)
    merged["cluster"] = {"name": spec["name"], "node_id": head["host"]["hostname"],
                         "nodes": [n["hostname"] for n in node_probes], "node_probes": node_probes,
                         "links": links, "spec_path": spec.get("_path"),
                         "workers": [{k: w[k] for k in ("ssh", "addr", "rpc_port")} for w in spec["workers"]]}
    merged["warnings"] = warnings
    merged["probe_id"] = _stable_id(merged, drop=("probe_id", "timestamp"))
    return merged


# --------------------------------------------------------------------------- run-time: rpc servers + remote sampling

def wait_port(addr: str, port: int, timeout: float, proc: subprocess.Popen | None = None) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"process exited with {proc.returncode} before {addr}:{port} opened (see the rpc-<host>.log in the run dir)")
        try:
            with socket.create_connection((addr, port), timeout=2):
                return time.time() - t0
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"{addr}:{port} not reachable after {timeout:.0f}s")


class RemoteSampler:
    """Streams `bench sample` lines from a worker into memory-<host>.jsonl and tracks peaks."""

    def __init__(self, worker: dict, hostname: str, out_dir: str):
        self.worker, self.hostname = worker, hostname
        self.path = os.path.join(out_dir, f"memory-{hostname}.jsonl")
        self.peak_host = self.peak_gpu = 0
        self.n = 0
        self.proc = None
        self._t = None

    def start(self):
        self.proc = remote_bench(self.worker["ssh"], ["sample", "--interval", "1"], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True)
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()

    def _pump(self):
        assert self.proc and self.proc.stdout
        with open(self.path, "w") as f:
            for line in self.proc.stdout:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                self.n += 1
                self.peak_host = max(self.peak_host, d.get("host_used_bytes") or 0)
                self.peak_gpu = max(self.peak_gpu, d.get("gpu_used_bytes") or 0)
                f.write(line)
                f.flush()

    def stop(self) -> dict:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._t:
            self._t.join(5)
        return {"hostname": self.hostname, "peak_host_used_bytes": self.peak_host or None,
                "peak_gpu_used_bytes": self.peak_gpu or None, "samples": self.n}


class ClusterRun:
    """Bring up rpc-server on every worker named in the plan, sample their memory, tear down afterwards."""

    def __init__(self, plan: dict, out_dir: str, log=print):
        self.nodes = plan["budget"]["nodes"]
        self.workers = [n for n in self.nodes if n["node"] != 0]
        self.out_dir, self.log = out_dir, log
        self.procs: list[tuple[dict, subprocess.Popen]] = []
        self.samplers: list[RemoteSampler] = []
        self.endpoints: list[str] = []
        self.up_s: dict = {}

    def start(self) -> list[str]:
        for w in self.workers:
            deploy(w["ssh"])
            logf = open(os.path.join(self.out_dir, f"rpc-{w['hostname']}.log"), "w")
            cmd = f"exec {RPC_BIN} -H 0.0.0.0 -p {w['rpc_port']}"
            logf.write(cmd + "\n")
            logf.flush()
            p = remote_shell(w["ssh"], cmd, stdout=logf, stderr=subprocess.STDOUT)
            self.procs.append((w, p))
            self.log(f"node {w['node']} {w['hostname']}: starting {RPC_BIN} on {w['addr']}:{w['rpc_port']}")
            self.up_s[w["hostname"]] = round(wait_port(w["addr"], w["rpc_port"], 90, p), 1)
            self.endpoints.append(f"{w['addr']}:{w['rpc_port']}")
            s = RemoteSampler(w, w["hostname"], self.out_dir)
            s.start()
            self.samplers.append(s)
        return self.endpoints

    def stop(self) -> dict:
        nodes = [s.stop() for s in self.samplers]
        for w, p in self.procs:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
            # ssh termination does not always reach the remote child; make sure.
            subprocess.run(ssh_argv(w["ssh"], f"pkill -f '^{RPC_BIN} ' || true"), capture_output=True, timeout=30)
        return {"endpoints": self.endpoints, "rpc_up_s": self.up_s, "nodes": nodes}


# --------------------------------------------------------------------------- `bench sample` (worker side)

def sample_loop(interval: float) -> None:
    from .runner import MemorySampler
    ms = MemorySampler(os.devnull, "discrete")
    try:
        while True:
            print(json.dumps({"t": round(time.time(), 1), "host_used_bytes": ms._host_used(), "gpu_used_bytes": ms._gpu_used()}), flush=True)
            time.sleep(interval)
    except (KeyboardInterrupt, BrokenPipeError):
        return


def summarize(doc: dict) -> str:
    c = doc["cluster"]
    L = [f"cluster {c['name']}: {len(c['nodes'])} nodes, head={c['node_id']}"]
    for i, n in enumerate(c["node_probes"]):
        devs = [d for d in doc["devices"] if d.get("node") == i]
        dv = ", ".join(f"{d['name']} {d['memory_total_bytes'] // (1024 ** 3)} GiB" for d in devs) or "no accelerator"
        L.append(f"  node {i} {n['hostname']}: {n['topology']}  ram={n['system_total_bytes'] // (1024 ** 3)} GiB  {dv}")
    for l in c["links"]:
        L.append(f"  link {l['from']} -> {l['to']}: {l.get('gbps_measured')} GB/s  rtt {l.get('rtt_ms')} ms" + (f"  ({l['error']})" if l.get("error") else ""))
    for w in doc["warnings"]:
        L.append(f"warning: {w}")
    return "\n".join(L)
