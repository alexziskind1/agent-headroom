"""Run bench commands on another machine over SSH and bring the outputs back.

    bench remote <host> probe
    bench remote <host> plan ladder
    bench remote <host> run out/plan-<host>-inference.json --replay aa-mini-v1

The package is shipped as a tarball on every call (it is small), extracted to
~/.cache/bench/pkg on the remote, and `python3 -m bench ...` runs there with
~/.local/bin on PATH. Everything the remote writes under its out/ comes back
into out/remote/<host>/.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile

PKG_DIR = "~/.cache/bench/pkg"
SSH_BASE = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


def _package_bytes() -> bytes:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in ("bench", "tests"):
            tar.add(os.path.join(root, name), arcname=name,
                    filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
    return buf.getvalue()


def target_name(host: str) -> str:
    """`jump/target?i=key` -> `target`; plain hosts unchanged."""
    return host.split("?", 1)[0].split("/")[-1]


def ssh_argv(host: str, script: str) -> list[str]:
    """`host` may be `target`, or `jump/target[?i=keyfile]` to hop through `jump` using the jump's own key."""
    if "/" not in host:
        return SSH_BASE + [host, script]
    jump, target = host.split("/", 1)
    key = None
    if "?i=" in target:
        target, key = target.split("?i=", 1)
    inner = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    if key:
        inner += ["-i", key, "-o", "IdentitiesOnly=yes"]
    inner += [target, _q(script)]
    return SSH_BASE + [jump, " ".join(inner)]


def ssh(host: str, script: str, stdin: bytes | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(ssh_argv(host, script), input=stdin, capture_output=True, timeout=timeout)


def deploy(host: str) -> None:
    script = f"mkdir -p {PKG_DIR} && cd {PKG_DIR} && rm -rf bench tests && tar xzf -"
    r = ssh(host, script, stdin=_package_bytes(), timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"deploy to {host} failed: {r.stderr.decode(errors='replace').strip()[:500]}")


def run_remote(host: str, args: list[str], stream: bool = True) -> int:
    """Run `python3 -m bench <args>` on the host, streaming its output; return its exit code."""
    quoted = " ".join(_q(a) for a in args)
    env = " ".join(f"export {k}={_q(v)};" for k, v in (os.environ.get("BENCH_REMOTE_ENV") and
                   dict(kv.split("=", 1) for kv in os.environ["BENCH_REMOTE_ENV"].split(",")) or {}).items())
    script = (f"export PATH=$HOME/.local/bin:$PATH; {env} cd {PKG_DIR} && "
              f"(for c in python3.12 python3.13 python3.14 python3; do command -v $c >/dev/null && echo $c && break; done) | xargs -I{{}} sh -c '{{}} -m bench {quoted}'")
    p = subprocess.Popen(ssh_argv(host, script), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout
    for line in p.stdout:
        if stream:
            print(f"[{target_name(host)}] {line}", end="")
    return p.wait()


def fetch(host: str, local_root: str = "out/remote") -> list[str]:
    """Copy the remote out/ tree into out/remote/<host>/ and return the file list."""
    dest = os.path.join(local_root, target_name(host))
    os.makedirs(dest, exist_ok=True)
    r = ssh(host, f"cd {PKG_DIR} && [ -d out ] && tar czf - out || true", timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"fetch from {host} failed: {r.stderr.decode(errors='replace')[:500]}")
    if not r.stdout:
        return []
    names = []
    with tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile() or ".." in m.name or m.name.startswith("/"):
                continue
            tar.extract(m, dest)
            names.append(os.path.join(dest, m.name))
    return names


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def run_script(host: str, script: str, stdin_path: str | None = None) -> int:
    """`bench ssh <host> <script>`: run a shell script string on a (jump) host with correct quoting, streaming output."""
    stdin = open(stdin_path, "rb") if stdin_path else None
    p = subprocess.Popen(ssh_argv(host, script), stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout
    for line in p.stdout:
        print(f"[{target_name(host)}] {line}", end="", flush=True)
    code = p.wait()
    if stdin:
        stdin.close()
    return code


def main(host: str, args: list[str]) -> int:
    deploy(host)
    code = run_remote(host, args)
    files = fetch(host)
    for f in files:
        if f.endswith((".json",)) and "/runs/" not in f:
            print(f"fetched {f}")
    runs = sorted({os.path.dirname(f) for f in files if f.endswith("results.json")})
    for r in runs:
        print(f"fetched run {r}")
    return code
