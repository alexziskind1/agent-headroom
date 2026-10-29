"""Hardware probe. Measures, never infers. Output follows docs/probe_planner_spec.md section 1."""
from __future__ import annotations

import ctypes
import glob as glob_mod
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

from . import __version__

SCHEMA_VERSION = "1.0"
GiB = 1024 ** 3


# --------------------------------------------------------------------------- helpers

def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def _docker_image_version(engine: str) -> str | None:
    """Engines that run from a docker image count as installed when the image is present locally."""
    image = os.environ.get("BENCH_VLLM_IMAGE") if engine == "vllm" else None
    if not image:
        try:
            with open(os.path.join(os.path.dirname(__file__), "catalog.json")) as f:
                image = (json.load(f).get("engines", {}).get(engine) or {}).get("docker_image")
        except (OSError, ValueError):
            image = None
    if not image or shutil.which("docker") is None:
        return None
    try:
        r = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return f"docker {image}" if r.returncode == 0 else None


def _version_of(cmd: list[str], timeout: float = 5.0) -> str | None:
    """First line mentioning a version, from stdout or stderr (llama.cpp prints its version to stderr)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    for stream in (out.stdout, out.stderr):
        lines = [l.strip() for l in stream.splitlines() if l.strip()]
        for l in lines:
            if "version" in l.lower() and not l.lower().startswith("warning"):
                return l[:80]
    lines = [l.strip() for l in (out.stdout + out.stderr).splitlines() if l.strip()]
    return lines[0][:80] if lines else None


# --------------------------------------------------------------------------- host

def probe_host() -> dict:
    uname = platform.uname()
    cpu_model = None
    cores = os.cpu_count()
    if uname.system == "Linux":
        info = _read("/proc/cpuinfo") or ""
        for line in info.splitlines():
            if line.lower().startswith(("model name", "hardware")):
                cpu_model = line.split(":", 1)[1].strip()
                break
        if cpu_model is None:
            out = _run(["lscpu"])
            if out:
                for line in out.splitlines():
                    if line.startswith("Model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        os_version = None
        rel = _read("/etc/os-release") or ""
        for line in rel.splitlines():
            if line.startswith("PRETTY_NAME="):
                os_version = line.split("=", 1)[1].strip().strip('"')
    elif uname.system == "Darwin":
        cpu_model = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        os_version = _run(["sw_vers", "-productVersion"])
    else:
        os_version = uname.release
    return {
        "hostname": platform.node(),
        "os": uname.system.lower(),
        "os_version": os_version,
        "kernel": uname.release,
        "arch": uname.machine,
        "cpu": {"model": cpu_model, "cores": cores, "threads": cores},
    }


# --------------------------------------------------------------------------- memory

def probe_system_memory() -> dict:
    total = available = swap = None
    huge = False
    if sys.platform == "linux":
        info = _read("/proc/meminfo") or ""
        vals = {}
        for line in info.splitlines():
            k, _, v = line.partition(":")
            vals[k.strip()] = int(v.strip().split()[0]) * 1024 if v.strip() else 0
        total = vals.get("MemTotal")
        available = vals.get("MemAvailable")
        swap = vals.get("SwapTotal")
        huge = vals.get("HugePages_Total", 0) > 0
        # Inside a container or VM slice /proc/meminfo shows the host; the cgroup limit is the real ceiling.
        for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            raw = (_read(path) or "").strip()
            if raw.isdigit() and total and int(raw) < total:
                cgroup = int(raw)
                available = min(available or cgroup, cgroup)
                return {"topology": None, "system_total_bytes": cgroup, "system_available_bytes": available,
                        "swap_total_bytes": swap, "huge_pages": huge, "cgroup_limit_bytes": cgroup, "physical_total_bytes": total}
    elif sys.platform == "darwin":
        t = _run(["sysctl", "-n", "hw.memsize"])
        total = int(t) if t else None
        # MemAvailable has no direct macOS analogue; use vm_stat free+inactive+speculative pages.
        vm = _run(["vm_stat"]) or ""
        page = 16384
        pages = {}
        for line in vm.splitlines():
            if "page size of" in line:
                page = int(line.split("page size of")[1].split()[0])
            elif ":" in line:
                k, _, v = line.partition(":")
                try:
                    pages[k.strip()] = int(v.strip().rstrip("."))
                except ValueError:
                    pass
        # kern.memorystatus_level is the percentage macOS itself treats as available (file cache counts).
        # vm_stat's free+inactive undercounts by roughly half on a loaded machine.
        lvl = _run(["sysctl", "-n", "kern.memorystatus_level"])
        if lvl and lvl.strip().isdigit() and total:
            available = int(total * int(lvl.strip()) / 100)
        else:
            free = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative"))
            available = free * page if pages else None
        sw = _run(["sysctl", "-n", "vm.swapusage"]) or ""
        swap = 0
        for tok in sw.replace("=", " ").split():
            pass
    return {
        "topology": None,  # filled after devices are known
        "system_total_bytes": total,
        "system_available_bytes": available,
        "swap_total_bytes": swap,
        "huge_pages": huge,
        "cgroup_limit_bytes": None,
        "physical_total_bytes": total,
    }


# --------------------------------------------------------------------------- cuda (driver API via ctypes)

_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
_CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
_CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36          # kHz
_CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37    # bits
_CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88     # GPU can touch pageable host memory (HMM or hardware)
_CU_DEVICE_ATTRIBUTE_PAGEABLE_USES_HOST_PAGE_TABLES = 100  # hardware coherent (NVLink-C2C / ATS): GH200, GB300, GB10
_CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130               # 1 when the GPU's memory is exposed as a host NUMA node
_CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2
_CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3  # true over NVLink, false over PCIe


def probe_cuda_bandwidth(cu, dev_index: int, warnings: list[str], mib: int = 512, reps: int = 4) -> dict:
    """Measured GB/s: device-to-device copy (memory bandwidth) and pinned host-to-device copy (the link the GPU
    would read spilled weights over: PCIe, NVLink-C2C, or the same LPDDR on unified parts)."""
    out = {"device_gbps": None, "host_to_device_gbps": None}
    dev = ctypes.c_int(0)
    cu.cuDeviceGet(ctypes.byref(dev), dev_index)
    ctx = ctypes.c_void_p()
    fn_ctx_create = getattr(cu, "cuCtxCreate_v2", None) or cu.cuCtxCreate
    fn_ctx_destroy = getattr(cu, "cuCtxDestroy_v2", None) or cu.cuCtxDestroy
    if fn_ctx_create(ctypes.byref(ctx), 0, dev) != 0:
        return out
    n = mib * 1024 * 1024
    try:
        alloc = getattr(cu, "cuMemAlloc_v2", None) or cu.cuMemAlloc
        free = getattr(cu, "cuMemFree_v2", None) or cu.cuMemFree
        dtod = getattr(cu, "cuMemcpyDtoD_v2", None) or cu.cuMemcpyDtoD
        htod = getattr(cu, "cuMemcpyHtoD_v2", None) or cu.cuMemcpyHtoD
        a, b = ctypes.c_void_p(), ctypes.c_void_p()
        if alloc(ctypes.byref(a), ctypes.c_size_t(n)) != 0 or alloc(ctypes.byref(b), ctypes.c_size_t(n)) != 0:
            warnings.append(f"device {dev_index}: could not allocate {mib} MiB x2 for the bandwidth test (busy?)")
            return out
        try:
            dtod(b, a, ctypes.c_size_t(n)); cu.cuCtxSynchronize()  # warm
            t0 = time.perf_counter()
            for _ in range(reps):
                dtod(b, a, ctypes.c_size_t(n))
            cu.cuCtxSynchronize()
            dt = time.perf_counter() - t0
            out["device_gbps"] = round(2 * n * reps / dt / 1e9, 1)  # read + write
            host = ctypes.c_void_p()
            if cu.cuMemHostAlloc(ctypes.byref(host), ctypes.c_size_t(n), 0) == 0:
                try:
                    htod(a, host, ctypes.c_size_t(n)); cu.cuCtxSynchronize()
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        htod(a, host, ctypes.c_size_t(n))
                    cu.cuCtxSynchronize()
                    out["host_to_device_gbps"] = round(n * reps / (time.perf_counter() - t0) / 1e9, 1)
                finally:
                    cu.cuMemFreeHost(host)
        finally:
            free(a); free(b)
    finally:
        fn_ctx_destroy(ctx)
    return out


def _load_cuda():
    names = ["libcuda.so.1", "libcuda.so", "nvcuda.dll", "/usr/local/cuda/compat/libcuda.so.1"]
    for n in names:
        try:
            return ctypes.CDLL(n)
        except OSError:
            continue
    return None


def probe_cuda_devices(warnings: list[str]) -> list[dict]:
    cu = _load_cuda()
    if cu is None:
        return []
    if cu.cuInit(0) != 0:
        warnings.append("cuInit failed; CUDA devices not probed")
        return []
    count = ctypes.c_int(0)
    if cu.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value == 0:
        return []

    drv = ctypes.c_int(0)
    cu.cuDriverGetVersion(ctypes.byref(drv))
    driver_str = None
    smi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if smi:
        driver_str = smi.splitlines()[0].strip()

    devices = []
    for i in range(count.value):
        dev = ctypes.c_int(0)
        cu.cuDeviceGet(ctypes.byref(dev), i)
        name = ctypes.create_string_buffer(256)
        cu.cuDeviceGetName(name, 256, dev)
        total = ctypes.c_size_t(0)
        fn_total = getattr(cu, "cuDeviceTotalMem_v2", None) or cu.cuDeviceTotalMem
        fn_total(ctypes.byref(total), dev)

        def attr(a: int) -> int:
            v = ctypes.c_int(0)
            cu.cuDeviceGetAttribute(ctypes.byref(v), a, dev)
            return v.value

        major, minor = attr(_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR), attr(_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        integrated = attr(_CU_DEVICE_ATTRIBUTE_INTEGRATED) == 1
        # Hardware-coherent host access (ATS over NVLink-C2C). Measured: GB10 = 1 (but integrated), RTX PRO 6000 = 0
        # even though it reports pageable access = 1 through HMM. GH200 / GB300 report 1 with integrated = 0.
        host_coherent = attr(_CU_DEVICE_ATTRIBUTE_PAGEABLE_USES_HOST_PAGE_TABLES) == 1 and not integrated
        pageable = attr(_CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS) == 1
        numa_cfg = attr(_CU_DEVICE_ATTRIBUTE_NUMA_CONFIG)
        clock_khz = attr(_CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
        bus_bits = attr(_CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
        # Raw attributes only. Deriving GB/s from them is off by 2x on LPDDR parts, so no derived figure is emitted.

        # cuMemGetInfo needs a context. Create, query, destroy. The context itself is the only allocation.
        free = ctypes.c_size_t(0)
        tot2 = ctypes.c_size_t(0)
        ctx = ctypes.c_void_p()
        fn_ctx_create = getattr(cu, "cuCtxCreate_v2", None) or cu.cuCtxCreate
        fn_ctx_destroy = getattr(cu, "cuCtxDestroy_v2", None) or cu.cuCtxDestroy
        fn_meminfo = getattr(cu, "cuMemGetInfo_v2", None) or cu.cuMemGetInfo
        method = "cuda_mem_get_info"
        if fn_ctx_create(ctypes.byref(ctx), 0, dev) == 0:
            if fn_meminfo(ctypes.byref(free), ctypes.byref(tot2)) != 0:
                warnings.append(f"cuMemGetInfo failed on device {i}; free memory unknown")
                free = ctypes.c_size_t(0)
            fn_ctx_destroy(ctx)
        else:
            warnings.append(f"cuCtxCreate failed on device {i}; using cuDeviceTotalMem only")
            method = "cu_device_total_mem"

        devices.append({
            "index": i,
            "vendor": "nvidia",
            "name": name.value.decode(errors="replace"),
            "backend": "cuda",
            "integrated": integrated,
            "host_coherent": host_coherent,
            "pageable_host_access": pageable,
            "numa_config": numa_cfg,
            "memory_total_bytes": int(tot2.value or total.value),
            "memory_free_bytes": int(free.value) if free.value else None,
            "memory_method": method,
            "compute_capability": f"{major}.{minor}",
            "driver_version": driver_str,
            "driver_api_version": f"{drv.value // 1000}.{(drv.value % 1000) // 10}",
            "memory_clock_khz": clock_khz or None,
            "memory_bus_width_bits": bus_bits or None,
            "bandwidth_gbps_measured": None,
            "host_to_device_gbps_measured": None,
            "host_link": "nvlink-c2c" if (host_coherent or integrated) else "pcie",
        })

    # P2P / interconnect between devices: native atomics over the peer link only exist on NVLink.
    links = []
    for a in range(count.value):
        for b in range(a + 1, count.value):
            can, nat = ctypes.c_int(0), ctypes.c_int(0)
            da, db = ctypes.c_int(0), ctypes.c_int(0)
            cu.cuDeviceGet(ctypes.byref(da), a)
            cu.cuDeviceGet(ctypes.byref(db), b)
            cu.cuDeviceCanAccessPeer(ctypes.byref(can), da, db)
            if can.value and hasattr(cu, "cuDeviceGetP2PAttribute"):
                cu.cuDeviceGetP2PAttribute(ctypes.byref(nat), _CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED, da, db)
            kind = ("nvlink" if nat.value else "pcie_p2p") if can.value else "none"
            links.append({"from": a, "to": b, "type": kind, "bandwidth_gbps_measured": None})
    probe_cuda_devices.links = links  # type: ignore[attr-defined]
    probe_cuda_devices.cu = cu  # type: ignore[attr-defined]
    return devices


# --------------------------------------------------------------------------- apple / rocm (best effort)

def probe_metal_devices(warnings: list[str], system_total: int | None) -> list[dict]:
    if sys.platform != "darwin" or system_total is None:
        return []
    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple GPU"
    wired = _run(["sysctl", "-n", "iogpu.wired_limit_mb"])
    if wired and wired.isdigit() and int(wired) > 0:
        addressable = int(wired) * 1024 * 1024
        method = "iogpu_wired_limit"
    else:
        # macOS default: ~75% of RAM above 36 GB, ~67% below. Marked as inferred; replace with Metal query.
        frac = 0.75 if system_total > 36 * GiB else 0.67
        addressable = int(system_total * frac)
        method = "macos_default_fraction"
        warnings.append("Metal working set inferred from default fraction; set iogpu.wired_limit_mb or add Metal query")
    return [{
        "index": 0, "vendor": "apple", "name": chip, "backend": "metal", "integrated": True,
        "memory_total_bytes": addressable, "memory_free_bytes": None, "memory_method": method,
        "compute_capability": None, "driver_version": None, "driver_api_version": None,
        "memory_clock_khz": None, "memory_bus_width_bits": None, "bandwidth_gbps_measured": None,
    }]


def probe_amdgpu_sysfs(warnings: list[str]) -> list[dict]:
    """AMD GPUs via /sys/class/drm (no ROCm needed). VRAM total is the UMA carve-out on APUs."""
    devices = []
    for card in sorted(glob_mod.glob("/sys/class/drm/card[0-9]*/device")):
        vendor = _read(os.path.join(card, "vendor"))
        if not vendor or vendor.strip() != "0x1002":
            continue
        vram = _read(os.path.join(card, "mem_info_vram_total"))
        if not vram:
            continue
        used = _read(os.path.join(card, "mem_info_vram_used"))
        gtt = _read(os.path.join(card, "mem_info_gtt_total"))
        name = None
        prod = _read(os.path.join(card, "product_name"))
        if prod:
            name = prod.strip()
        if not name:
            out = _run(["lspci", "-d", "1002:", "-mm"]) or ""
            for line in out.splitlines():
                if "VGA" in line or "Display" in line:
                    parts = line.split('"')
                    name = parts[5] if len(parts) > 5 else line
                    break
        # APU: the device is integrated when its VRAM is a carve-out of system memory.
        integrated = os.path.exists(os.path.join(card, "mem_info_preempt_used")) and ("APU" in (prod or "") or True)
        hw = _read("/sys/class/dmi/id/product_name")
        devices.append({
            "index": len(devices), "vendor": "amd", "name": name or "AMD GPU", "backend": "vulkan", "integrated": None,
            "memory_total_bytes": int(vram.strip()), "memory_free_bytes": int(vram.strip()) - int(used.strip()) if used else None,
            "memory_method": "amdgpu_sysfs_vram_total", "gtt_total_bytes": int(gtt.strip()) if gtt else None,
            "compute_capability": None, "driver_version": (_read("/sys/module/amdgpu/version") or "").strip() or None,
            "driver_api_version": None, "memory_clock_khz": None, "memory_bus_width_bits": None, "bandwidth_gbps_measured": None,
        })
    if devices:
        warnings.append("AMD GPU probed via amdgpu sysfs; VRAM total is the UMA carve-out on APUs and GTT is reported separately")
    return devices


def probe_rocm_devices(warnings: list[str]) -> list[dict]:
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    devices = []
    for i, (k, v) in enumerate(sorted(data.items())):
        tot = v.get("VRAM Total Memory (B)")
        used = v.get("VRAM Total Used Memory (B)")
        if tot is None:
            continue
        tot, used = int(tot), int(used or 0)
        devices.append({
            "index": i, "vendor": "amd", "name": k, "backend": "rocm", "integrated": None,
            "memory_total_bytes": tot, "memory_free_bytes": tot - used, "memory_method": "rocm_smi",
            "compute_capability": None, "driver_version": None, "driver_api_version": None,
            "memory_clock_khz": None, "memory_bus_width_bits": None, "bandwidth_gbps_measured": None,
        })
    if devices:
        warnings.append("ROCm probed via rocm-smi, not hipMemGetInfo")
    return devices


# --------------------------------------------------------------------------- storage

def default_cache_path() -> str:
    return os.environ.get("BENCH_MODEL_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "bench", "models"))


def probe_storage(warnings: list[str], size_mb: int = 256) -> dict:
    path = default_cache_path()
    os.makedirs(path, exist_ok=True)
    usage = shutil.disk_usage(path)
    read_gbps = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=".bench_io_", dir=path)
        try:
            chunk = os.urandom(4 * 1024 * 1024)
            if sys.platform == "darwin":
                import fcntl
                fcntl.fcntl(fd, 48, 1)  # F_NOCACHE on the write side too, or the read hits page cache
            with os.fdopen(fd, "wb") as f:
                for _ in range(size_mb // 4):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())
            rfd = os.open(tmp, os.O_RDONLY)
            try:
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(rfd, 0, 0, os.POSIX_FADV_DONTNEED)
                if sys.platform == "darwin":
                    import fcntl
                    fcntl.fcntl(rfd, 48, 1)  # F_NOCACHE
                t0 = time.perf_counter()
                n = 0
                while True:
                    b = os.read(rfd, 8 * 1024 * 1024)
                    if not b:
                        break
                    n += len(b)
                dt = time.perf_counter() - t0
                read_gbps = round(n / dt / 1e9, 2) if dt > 0 else None
            finally:
                os.close(rfd)
        finally:
            os.unlink(tmp)
    except OSError as e:
        warnings.append(f"storage read test skipped: {e}")

    dev_type = "unknown"
    if sys.platform == "linux":
        out = _run(["findmnt", "-n", "-o", "SOURCE", "--target", path])
        if out:
            base = os.path.basename(out)
            if base.startswith("nvme"):
                dev_type = "nvme"
            elif base.startswith("sd"):
                dev_type = "sata"
            elif ":" in out or out.startswith("//"):
                dev_type = "network"
    return {
        "model_cache_path": path,
        "free_bytes": usage.free,
        "read_gbps_measured": read_gbps,
        "device_type": dev_type,
    }


# --------------------------------------------------------------------------- software

def probe_software() -> dict:
    cuda = None
    nvcc = _run(["nvcc", "--version"])
    if nvcc:
        for tok in nvcc.split():
            if tok.startswith("V") and tok[1:2].isdigit():
                cuda = tok[1:]
    if cuda is None:
        txt = _read("/usr/local/cuda/version.json")
        if txt:
            try:
                cuda = json.loads(txt).get("cuda", {}).get("version")
            except json.JSONDecodeError:
                pass
    engines = {
        "llama.cpp": _version_of([os.environ.get("BENCH_LLAMA_SERVER", "llama-server"), "--version"]) or _version_of(["llama-cli", "--version"]),
        "vllm": _version_of(["vllm", "--version"]) or _version_of([sys.executable, "-c", "import vllm;print(vllm.__version__)"]) or _docker_image_version("vllm"),
        "sglang": _version_of([sys.executable, "-c", "import sglang;print(sglang.__version__)"]),
        "mlx": _version_of([sys.executable, "-c", "import mlx.core as m;print(m.__version__)"]) or _version_of([os.environ.get("BENCH_MLX_PYTHON") or os.path.expanduser("~/.venvs/mlx/bin/python"), "-c", "import mlx_lm, mlx.core as m;print(f'mlx_lm {mlx_lm.__version__} mlx {m.__version__}')"]),
        "tensorfold": _version_of([os.environ.get("BENCH_TENSORFOLD") or os.path.expanduser("~/.tensorfold/bin/tensorfold"), "--version"]) or _version_of(["tensorfold-native", "--version"]) or _version_of(["tensorfold", "--version"]),
        "ollama": _version_of(["ollama", "--version"]),
    }
    return {
        "python": platform.python_version(),
        "cuda": cuda,
        "rocm": _version_of(["rocm-smi", "--version"]),
        "metal": None,
        "docker": shutil.which("docker") is not None and _run(["docker", "info", "--format", "{{.ServerVersion}}"]) is not None,
        "engines": engines,
    }


# --------------------------------------------------------------------------- assemble

def _stable_id(doc: dict, drop: tuple[str, ...]) -> str:
    clean = {k: v for k, v in doc.items() if k not in drop}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_probe(storage_test: bool = True, bandwidth: bool | None = None) -> dict:
    # bandwidth defaults to storage_test; `bench plan` skips the storage test but needs the
    # two bandwidths for the coherent_spill decode estimate.
    if bandwidth is None:
        bandwidth = storage_test
    warnings: list[str] = []
    host = probe_host()
    mem = probe_system_memory()
    devices = probe_cuda_devices(warnings)
    links = getattr(probe_cuda_devices, "links", [])
    if not devices:
        devices = probe_rocm_devices(warnings)
    if not devices:
        devices = probe_amdgpu_sysfs(warnings)
        # An AMD APU carve-out is unified memory: decide by CPU model, since sysfs has no integrated flag.
        cpu = (host["cpu"]["model"] or "").lower()
        sys_total = mem["system_total_bytes"] or 0
        for d in devices:
            apu = any(k in cpu for k in ("ryzen ai", "radeon", "apu", "ryzen 7 8", "ryzen 9 8", "ryzen ai max"))
            carveout = apu and d["memory_total_bytes"] > 0.5 * sys_total
            # A BIOS UMA carve-out is invisible to the OS: VRAM and host RAM are separate pools,
            # so for budgeting it is `discrete` even though the silicon is unified.
            d["integrated"] = apu and not carveout
            if carveout:
                warnings.append(f"AMD APU with a fixed UMA carve-out of {d['memory_total_bytes'] // GiB} GiB: budgeted as separate pools (topology=discrete)")
    if not devices:
        devices = probe_metal_devices(warnings, mem["system_total_bytes"])

    if not devices:
        mem["topology"] = "cpu_only"
    else:
        # Topology comes only from the device's own flags (CUDA attributes, Apple Silicon).
        # Never infer it from sizes: a 96 GB discrete card next to 92 GB of RAM looks unified by size.
        #   unified:  one physical pool (integrated GPU)
        #   coherent: two pools, both GPU-addressable over a hardware-coherent link (GH200, GB300 station)
        #   discrete: two pools, GPU reaches host memory only by explicit copies
        integrated = any(d.get("integrated") for d in devices)
        coherent = any(d.get("host_coherent") for d in devices)
        mem["topology"] = "unified" if integrated else ("coherent" if coherent else "discrete")
        if any(d["vendor"] == "nvidia" for d in devices) and mem["topology"] == "unified":
            warnings.append("nvidia-smi reports memory.total as N/A on unified systems; cuMemGetInfo used instead")
        if mem["topology"] == "coherent":
            warnings.append("coherent topology: GPU can address host memory directly; the planner may spill weights to host RAM (placement coherent_spill)")
        if bandwidth and getattr(probe_cuda_devices, "cu", None) is not None:
            for d in devices:
                if d["vendor"] != "nvidia":
                    continue
                bw = probe_cuda_bandwidth(probe_cuda_devices.cu, d["index"], warnings)
                d["bandwidth_gbps_measured"] = bw["device_gbps"]
                d["host_to_device_gbps_measured"] = bw["host_to_device_gbps"]
    if mem.get("cgroup_limit_bytes"):
        warnings.append(f"cgroup memory limit {mem['cgroup_limit_bytes'] // GiB} GiB is below physical RAM {mem['physical_total_bytes'] // GiB} GiB; budgeting against the limit")
    host["numa_nodes"] = len(glob_mod.glob("/sys/devices/system/node/node[0-9]*")) or 1

    doc = {
        "schema_version": SCHEMA_VERSION,
        "probe_version": __version__,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "probe_id": None,
        "host": host,
        "memory": mem,
        "devices": devices,
        "interconnect": {"intra_node": links},
        "storage": probe_storage(warnings) if storage_test else {"model_cache_path": default_cache_path(), "free_bytes": None, "read_gbps_measured": None, "device_type": "unknown"},
        "software": probe_software(),
        "cluster": {"node_id": host["hostname"], "nodes": [host["hostname"]], "links": []},
        "warnings": warnings,
    }
    doc["probe_id"] = _stable_id(doc, drop=("probe_id", "timestamp"))
    return doc


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    return f"{n / GiB:,.1f} GiB"


def summarize(doc: dict) -> str:
    lines = [f"{doc['host']['hostname']}  {doc['host']['os']} {doc['host']['arch']}  cpu={doc['host']['cpu']['model']} x{doc['host']['cpu']['cores']}"]
    m = doc["memory"]
    lines.append(f"memory: topology={m['topology']}  system={fmt_bytes(m['system_total_bytes'])} total, {fmt_bytes(m['system_available_bytes'])} available")
    for d in doc["devices"]:
        line = f"device {d['index']}: {d['name']} [{d['backend']}] total={fmt_bytes(d['memory_total_bytes'])} free={fmt_bytes(d['memory_free_bytes'])} via {d['memory_method']} cc={d['compute_capability']}"
        if d.get("node") is not None:
            line = f"node {d['node']} " + line
        if d.get("bandwidth_gbps_measured"):
            line += f"  mem={d['bandwidth_gbps_measured']} GB/s"
        if d.get("host_to_device_gbps_measured"):
            line += f"  host->gpu={d['host_to_device_gbps_measured']} GB/s ({d.get('host_link')})"
        lines.append(line)
    for l in doc.get("interconnect", {}).get("inter_node", []):
        lines.append(f"link {l['from']} -> {l['to']}: {l['type']} {l.get('gbps_measured')} GB/s rtt={l.get('rtt_ms')} ms")
    s = doc["storage"]
    lines.append(f"storage: {s['model_cache_path']} free={fmt_bytes(s['free_bytes'])} read={s['read_gbps_measured']} GB/s ({s['device_type']})")
    sw = doc["software"]
    eng = ", ".join(f"{k}={v}" for k, v in sw["engines"].items() if v) or "none"
    lines.append(f"software: python={sw['python']} cuda={sw['cuda']} docker={sw['docker']} engines: {eng}")
    for w in doc["warnings"]:
        lines.append(f"warning: {w}")
    return "\n".join(lines)
