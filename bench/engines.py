"""Engine launch layer: one place that knows how to start, health-check and version each inference server.

Every engine exposes the same shape to the runners:
    spec = launch_spec(engine, sel, artifact_path, alias, port, extra, rpc=None, host="127.0.0.1")
    spec["argv"], spec["env"], spec["health_url"], spec["base_url"], spec["cleanup"] (argv to run after stop, or None)

Engines:
    llama.cpp   llama-server, GGUF file. The baseline on every machine (flags mirror AA-AgentPerf-Local managed mode).
    vllm        `vllm serve` natively when on PATH, else the OpenAI docker image with --gpus all. Directory artifact.
    mlx         mlx_lm.server (Apple Metal). Directory artifact (MLX safetensors).
    tensorfold  tensorfold-native serve (Metal or CUDA). Directory artifact (MLX 4/6-bit or NVFP4).
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess

GiB = 1024 ** 3

ENGINE_IDS = ("llama.cpp", "vllm", "mlx", "tensorfold")

LLAMA_BIN = os.environ.get("BENCH_LLAMA_SERVER", "llama-server")
VLLM_IMAGE_ENV = "BENCH_VLLM_IMAGE"
MLX_PYTHON = os.environ.get("BENCH_MLX_PYTHON") or os.path.expanduser("~/.venvs/mlx/bin/python")
TENSORFOLD_BIN = os.environ.get("BENCH_TENSORFOLD") or os.path.expanduser("~/.tensorfold/bin/tensorfold")


def models_root() -> str:
    return os.path.expanduser(os.environ.get("BENCH_MODEL_CACHE", "~/.cache/bench/models"))


# --------------------------------------------------------------------------- availability

def _tensorfold_bin() -> str | None:
    for cand in (TENSORFOLD_BIN, shutil.which("tensorfold-native"), shutil.which("tensorfold")):
        if cand and os.path.exists(cand):
            return cand
    return None


def _vllm_mode(cat: dict | None = None) -> tuple[str | None, str | None]:
    """('native', None) | ('docker', image) | (None, None)."""
    if shutil.which("vllm"):
        return "native", None
    image = os.environ.get(VLLM_IMAGE_ENV) or ((cat or {}).get("engines", {}).get("vllm", {}) or {}).get("docker_image")
    if image and shutil.which("docker"):
        return "docker", image
    return None, None


def available(engine: str, cat: dict | None = None) -> tuple[bool, str]:
    """(ok, reason). Cheap checks only; the launch is the real test."""
    if engine == "llama.cpp":
        return (shutil.which(LLAMA_BIN) is not None or os.path.exists(LLAMA_BIN),
                f"{LLAMA_BIN} not found (set BENCH_LLAMA_SERVER or put llama-server on PATH)")
    if engine == "vllm":
        mode, image = _vllm_mode(cat)
        return mode is not None, "vllm not on PATH and no docker image (engines.vllm.docker_image or BENCH_VLLM_IMAGE)"
    if engine == "mlx":
        return os.path.exists(MLX_PYTHON), f"{MLX_PYTHON} missing (uv venv ~/.venvs/mlx && uv pip install mlx-lm; or BENCH_MLX_PYTHON)"
    if engine == "tensorfold":
        return _tensorfold_bin() is not None, "tensorfold-native not found (curl -fsSL https://tensorfold.dev/install.sh | sh; or BENCH_TENSORFOLD)"
    return False, f"unknown engine {engine}"


def version(engine: str, cat: dict | None = None) -> str | None:
    def first_line(argv):
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return None
        for l in (out.stdout + out.stderr).splitlines():
            if l.strip():
                return l.strip()
        return None
    if engine == "llama.cpp":
        try:
            out = subprocess.run([LLAMA_BIN, "--version"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        for l in (out.stdout + out.stderr).splitlines():
            if l.strip().startswith("version"):
                return l.strip()
        return None
    if engine == "vllm":
        mode, image = _vllm_mode(cat)
        if mode == "native":
            return first_line(["vllm", "--version"])
        return f"docker {image}" if image else None
    if engine == "mlx":
        return first_line([MLX_PYTHON, "-c", "import mlx_lm, mlx.core as mx; print(f'mlx_lm {mlx_lm.__version__} mlx {mx.__version__}')"])
    if engine == "tensorfold":
        b = _tensorfold_bin()
        return first_line([b, "--version"]) if b else None
    return None


# --------------------------------------------------------------------------- artifacts

def artifact_kind(file: dict) -> str:
    """'file' for GGUF (single or split), 'dir' for safetensors/MLX checkpoints."""
    if file.get("kind"):
        return file["kind"]
    q = (file.get("quant") or "").lower()
    return "dir" if (q in ("fp8", "bf16", "fp16", "nvfp4", "awq") or q.startswith("mlx")) else "file"


def artifact_path(model_id: str, file: dict) -> str:
    return os.path.join(models_root(), model_id, file["quant"])


def dir_artifact_bytes(path: str) -> int:
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(path, "**", "*.safetensors"), recursive=True))


def dir_artifact_ok(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")) and dir_artifact_bytes(path) > 0


# --------------------------------------------------------------------------- launch specs

def _docker_mount(path: str) -> tuple[str, str]:
    """(host dir, container path) for an artifact: the resolved real directory is bound read-only at /models/<id>/<quant>.
    Binding the artifact itself (not the cache root) keeps symlinked checkpoints working: on the DGX Station the cache
    entry is a symlink to ~/models/..., which dangles inside a container that only sees the cache root."""
    real = os.path.realpath(os.path.expanduser(path))  # existence is checked by the runner before launch
    rel = os.path.relpath(os.path.abspath(os.path.expanduser(path)), models_root())
    inside = os.path.join("/models", rel if not rel.startswith("..") else os.path.basename(real))
    return real, inside


def llama_argv(sel: dict, model_path: str, alias: str, port: int, extra: list[str], rpc: list[str] | None = None,
               host: str = "127.0.0.1", bin_: str | None = None) -> list[str]:
    """Mirror agentperf-local's managed llama.cpp launch so numbers are comparable."""
    argv = [
        bin_ or LLAMA_BIN,
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


def vllm_argv(sel: dict, model_path: str, alias: str, port: int, extra: list[str], cat: dict | None = None,
              host: str = "127.0.0.1", device_total_bytes: int | None = None, tool_parser: str | None = None) -> tuple[list[str], dict, str | None]:
    """vllm serve with the plan's budget: max-model-len = ctx, max-num-seqs = n_seq, gpu-memory-utilization sized
    from the plan's own estimate so vLLM's preallocation matches what llama.cpp would take. Returns (argv, env, cleanup_name)."""
    mode, image = _vllm_mode(cat)
    est = sel.get("estimate") or {}
    util = None
    if device_total_bytes and est.get("total_bytes"):
        # 10% plus 2 GiB over the plan's estimate: vLLM needs activation/CUDA-graph headroom beyond weights + KV,
        # and on shared unified memory the neighbours' caches grow after launch (a +2% margin failed on restore).
        want = est["total_bytes"] * 1.10 + 2 * 2**30
        util = min(0.95, max(0.30, round(want / device_total_bytes, 2)))
    pl = sel.get("placement") or {}
    tp = len(pl.get("devices") or [0]) if pl.get("strategy") == "tensor_parallel" else 1
    serve = [
        "--served-model-name", alias,
        "--host", host, "--port", str(port),
        "--max-model-len", str(sel["context_tokens"]),
        "--max-num-seqs", str(sel["n_seq"]),
        "--enable-auto-tool-choice", "--tool-call-parser", tool_parser or "hermes",
    ]
    if util is not None:
        serve += ["--gpu-memory-utilization", f"{util:.2f}"]
    if tp > 1:
        serve += ["--tensor-parallel-size", str(tp)]
    env = {}
    if mode == "docker":
        name = f"bench-vllm-{port}"
        # no --rm: the container's full log is pulled with `docker logs` at cleanup (the stdout pipe loses the
        # engine-core lines when the container is stopped), then the container is removed.
        host_dir, inside = _docker_mount(model_path)
        argv = ["docker", "run", "--name", name, "--gpus", "all", "--ipc=host", "--network", "host", "-e", "PYTHONUNBUFFERED=1",
                "-v", f"{host_dir}:{inside}:ro", "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_NO_USAGE_STATS=1",
                image, inside] + serve + list(extra)
        return argv, env, name
    argv = ["vllm", "serve", model_path] + serve + list(extra)
    env = {"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1"}
    return argv, env, None


def mlx_argv(sel: dict, model_path: str, alias: str, port: int, extra: list[str], host: str = "127.0.0.1") -> list[str]:
    """mlx_lm.server. The served model id is the path; agentperf is given that path as --model (see model_name_for)."""
    return [MLX_PYTHON, "-m", "mlx_lm.server", "--model", model_path, "--host", host, "--port", str(port),
            "--max-tokens", "16384", "--decode-concurrency", str(max(sel["n_seq"], 1)), "--prompt-concurrency", "8",
            "--prompt-cache-size", "16", "--log-level", "WARNING"] + list(extra)


def tensorfold_argv(sel: dict, model_path: str, alias: str, port: int, extra: list[str], host: str = "127.0.0.1") -> list[str]:
    b = _tensorfold_bin() or TENSORFOLD_BIN
    return [b, "serve", model_path, "--name", alias, "--host", host, "--port", str(port),
            "--parallel", str(max(sel["n_seq"], 1)), "--context", str(sel["context_tokens"]),
            "--max-tokens", "16384", "--no-update-check"] + list(extra)


def model_name_for(engine: str, alias: str, model_path: str) -> str:
    """The `model` field clients must send. mlx_lm.server keys on the path it was given."""
    return model_path if engine == "mlx" else alias


def launch_spec(engine: str, sel: dict, model_path: str, alias: str, port: int, extra: list[str], cat: dict | None = None,
                rpc: list[str] | None = None, host: str = "127.0.0.1", device_total_bytes: int | None = None,
                tool_parser: str | None = None) -> dict:
    env = dict(os.environ)
    cleanup = None
    if engine == "llama.cpp":
        argv = llama_argv(sel, model_path, alias, port, extra, rpc=rpc, host=host)
        if (sel.get("placement") or {}).get("strategy") == "coherent_spill":
            env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
    elif engine == "vllm":
        argv, extra_env, name = vllm_argv(sel, model_path, alias, port, extra, cat=cat, host=host,
                                          device_total_bytes=device_total_bytes, tool_parser=tool_parser)
        env.update(extra_env)
        cleanup = {"container": name} if name else None
    elif engine == "mlx":
        argv = mlx_argv(sel, model_path, alias, port, extra, host=host)
    elif engine == "tensorfold":
        argv = tensorfold_argv(sel, model_path, alias, port, extra, host=host)
    else:
        raise ValueError(f"no launcher for engine {engine}")
    return {
        "engine": engine,
        "argv": argv,
        "env": env,
        "health_url": f"http://{host}:{port}/health",
        "base_url": f"http://{host}:{port}/v1",
        "model_name": model_name_for(engine, alias, model_path),
        "cleanup": cleanup,
    }


def run_cleanup(spec: dict | None, log_path: str | None = None) -> None:
    """Stop and remove an engine's container, appending its complete log to log_path first."""
    c = (spec or {}).get("cleanup")
    if not c:
        return
    name = c["container"]
    try:
        subprocess.run(["docker", "stop", "-t", "20", name], capture_output=True, timeout=90)
        if log_path:
            out = subprocess.run(["docker", "logs", name], capture_output=True, text=True, timeout=60)
            with open(log_path, "a") as f:
                f.write("\n===== docker logs (complete) =====\n")
                f.write(out.stdout)
                f.write(out.stderr)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pass


# --------------------------------------------------------------------------- fetch

def fetch_argv(model_id: str, file: dict, dest: str) -> list[list[str]]:
    """Commands that download one catalog artifact into dest (hf for directories, curl for GGUF parts)."""
    kind = artifact_kind(file)
    if kind == "dir":
        hf = shutil.which("hf") or os.path.expanduser("~/.venvs/hf/bin/hf")
        cmd = [hf, "download", file["repo"], "--local-dir", dest]
        if file.get("revision"):
            cmd += ["--revision", file["revision"]]
        if file.get("include"):
            cmd += ["--include", file["include"]]
        return [cmd]
    parts = file.get("parts") or [{"filename": file.get("filename") or os.path.basename(file["url"]), "url": file["url"], "bytes": file["bytes"]}]
    cmds = []
    for part in parts:
        out = os.path.join(dest, os.path.basename(part["filename"]))
        cmds.append(["curl", "-L", "-sS", "-C", "-", "-o", out, part["url"]])
    return cmds
