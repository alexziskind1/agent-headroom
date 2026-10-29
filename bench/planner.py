"""Planner: probe + catalog + policy -> plan.json. See docs/probe_planner_spec.md section 3."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import glob
import os

from .catalog import KV_QUANT_FACTOR, get_model, quant_rank
from .probe import GiB, fmt_bytes

SCHEMA_VERSION = "1.0"
TESTS = ("inference", "ladder", "host_pressure", "residency", "warm_contexts")
RUNNABLE = ("fits", "fits_with_offload", "fits_coherent", "fits_cluster", "forced")
BASELINE_ENGINE = "llama.cpp"   # the controlled row: same engine, same flags, every machine
RESIDENT = ("resident", "resident_coherent")

DEFAULT_POLICY = {
    "reserve_fraction": 0.10,
    "reserve_min_bytes": 2 * GiB,
    "host_reserve_fraction": 0.25,
    "host_min_bytes": 16 * GiB,   # unified only: RAM the GPU budget must leave for OS, tools, sandboxes
    "allow_offload": False,
    "allow_multi_device": True,
    "kv_quant": "fp16",
    "n_seq": 1,
    "target_context": None,
    "engine": None,          # force an engine, or "best": fastest installed engine per catalog engine_preference
    "model": None,           # force a model id (inference/host_pressure require one)
    "quant": None,           # force a quant
    "placement": None,       # force a strategy: "rpc_split" splits across cluster nodes even when one node would do
    "force": False,          # with target_context: emit the selection even when the estimate does not fit (status "forced")
    "slots": 8,              # warm_contexts: GPU slots the fleet shares (fixed across RAM sizes)
    "client_bytes": int(1.5 * 2**30),   # warm_contexts: host RAM per replay client process (measured 1-2 GiB)
    "server_host_bytes": 4 * 2**30,     # warm_contexts: llama-server host-side overhead outside the prompt cache
}

# Rough per-test runtime estimates, seconds. Replaced by measured history later.
RUNTIME_EST = {"inference": 1800, "ladder": 2400, "host_pressure": 5400, "residency": 3600, "warm_contexts": 7200}


# --------------------------------------------------------------------------- budget

def _host_min(policy: dict, sys_total: int) -> int:
    return min(policy["host_min_bytes"], int(sys_total * 0.2)) if sys_total else policy["host_min_bytes"]


def compute_budget(probe: dict, policy: dict) -> dict:
    """Topologies:
      discrete  - VRAM budget per device; host RAM is for tools (and CPU offload if allowed)
      unified   - one pool; the device budget is capped so host_min stays free for OS and tools
      coherent  - VRAM budget per device plus a GPU-addressable host tier (host_coherent_bytes) that weights may spill into
      cluster   - every node's devices in one list (device["node"]); each device is budgeted by its own node's topology
      cpu_only  - no device budget
    A plan describes the hardware, not what happens to be running. Budget against totals;
    `available` only drives the "machine is busy" warning and the run-time admission check."""
    devices = probe["devices"]
    topo = probe["memory"]["topology"]
    sys_total = probe["memory"]["system_total_bytes"] or 0
    sys_avail = probe["memory"]["system_available_bytes"] or 0
    host_min = _host_min(policy, sys_total)
    per_dev = []
    for d in devices:
        total = d["memory_total_bytes"]
        reserve = max(int(total * policy["reserve_fraction"]), policy["reserve_min_bytes"])
        b = max(total - reserve, 0)
        node_topo = d.get("node_topology", topo)
        node_sys = d.get("node_system_total_bytes", sys_total)
        if node_topo == "unified":
            # One pool: the GPU ceiling must still leave host_min for OS, tools and sandboxes.
            b = min(b, max(node_sys - _host_min(policy, node_sys), 0))
        per_dev.append(b)
    gpu_total = sum(per_dev)
    if topo == "unified":
        # Ceiling only. The real host figure is computed after selection (host_bytes_after_plan).
        host = int(max(sys_total - gpu_total, 0) * (1 - policy["host_reserve_fraction"]))
    else:
        host = int(sys_total * (1 - policy["host_reserve_fraction"]))
    installed = [k for k, v in (probe.get("software", {}).get("engines") or {}).items() if v]
    nodes = None
    if topo == "cluster":
        nodes = []
        for i, np_ in enumerate(probe["cluster"].get("node_probes", [])):
            nodes.append({"node": i, "hostname": np_["hostname"], "topology": np_["topology"],
                          "system_total_bytes": np_["system_total_bytes"],
                          "devices": [d["index"] for d in devices if d.get("node") == i],
                          "ssh": np_.get("ssh"), "addr": np_.get("addr"), "rpc_port": np_.get("rpc_port")})
    dev0 = devices[0] if devices else {}
    return {
        "gpu_bytes_per_device": per_dev,
        "gpu_bytes_total": gpu_total,
        "host_bytes": host,
        "host_coherent_bytes": host if topo == "coherent" else 0,
        "host_bytes_after_plan": None,
        "host_min_bytes": host_min,
        "system_total_bytes": sys_total,
        "system_available_bytes": sys_avail,
        "topology": topo,
        "nodes": nodes,
        "device_gbps": dev0.get("bandwidth_gbps_measured"),
        "host_to_device_gbps": dev0.get("host_to_device_gbps_measured"),
        "device_backends": sorted({d["backend"] for d in devices}),
        "installed_engines": installed,
    }


# --------------------------------------------------------------------------- cost

def kv_bytes(model: dict, context: int, n_seq: int, kv_quant: str) -> int:
    if model["modality"] in ("text", "vision"):
        per_tok = int(model["kv_bytes_per_token"] * context * n_seq * KV_QUANT_FACTOR[kv_quant])
        return per_tok + int(model.get("state_bytes_per_seq", 0)) * n_seq  # hybrid SSM state is per sequence
    return 0


def cost(model: dict, file: dict, engine: dict, context: int, n_seq: int, kv_quant: str, backends: tuple = ("cuda",)) -> dict:
    weights = file["bytes"]
    kv = kv_bytes(model, context, n_seq, kv_quant)
    # Measured overheads in the catalog come from CUDA recipes; on other backends they overstate
    # (a Metal llama-server has no 4 GiB CUDA runtime), so fall back to the generic model there.
    if file.get("fixed_overhead_bytes") is not None and "cuda" in backends:
        # Measured on real hardware (e.g. imported from an AA recipe): replaces both the
        # activation fraction and the generic engine overhead.
        act, eng = int(file["fixed_overhead_bytes"]), 0
    else:
        if model["modality"] in ("text", "vision"):
            # Activations and compute buffers scale with the parameters a token touches, not the file size:
            # a 284B/13B-active MoE needs nowhere near 5% of its weights. Floor at 10% so routing and expert
            # scratch space is never estimated at zero. Measured on DeepSeek-V4-Flash (GMK, 2026-10-26):
            # weights 144.4 GiB, everything else 5.3 GiB; the flat 5% rule had said 7.2 GiB for activations alone.
            scale = 1.0
            tot, act_p = model.get("params_total") or 0, model.get("params_active") or 0
            if tot and act_p and act_p < tot:
                scale = max(act_p / tot, 0.1)
            act = int(weights * model["activation_overhead_fraction"] * scale)
        else:
            act = max(model["working_set_bytes"] - weights, 0)
        eng = engine["overhead_bytes"]
    return {
        "weights_bytes": weights,
        "kv_bytes": kv,
        "activation_bytes": act,
        "engine_bytes": eng,
        "total_bytes": weights + kv + act + eng,
    }


# --------------------------------------------------------------------------- placement

def place(est: dict, budget: dict, engine_caps: dict, policy: dict) -> dict | None:
    """Return a placement dict with fit_margin_bytes, or None if nothing fits."""
    per_dev = budget["gpu_bytes_per_device"]
    if not per_dev:
        return None
    total = est["total_bytes"]
    forced = policy.get("placement")
    # single device: pick the device with the most room
    best = max(range(len(per_dev)), key=lambda i: per_dev[i])
    if total <= per_dev[best] and forced in (None, "single"):
        return {"strategy": "single", "devices": [best], "gpu_layers": None, "host_bytes": 0,
                "fit_margin_bytes": per_dev[best] - total, "status": "fits"}
    n = len(per_dev)
    nodes = budget.get("nodes")
    # split across the devices of ONE node (tensor parallel, or llama.cpp's layer split): equal shares
    mode = next((m for m in ("tensor_parallel", "pipeline") if m in engine_caps.get("multi_device", [])), None)
    if not nodes and n > 1 and policy["allow_multi_device"] and mode and forced in (None, mode):
        per_share = (est["weights_bytes"] + est["kv_bytes"] + est["activation_bytes"]) / n + est["engine_bytes"]
        if all(per_share <= b for b in per_dev):
            return {"strategy": mode, "devices": list(range(n)), "gpu_layers": None, "host_bytes": 0,
                    "fit_margin_bytes": int(min(per_dev) - per_share), "status": "fits"}
    # cluster: llama.cpp RPC places layers on every node's device in proportion to its room.
    # Every node pays its own engine overhead; KV travels with the layers.
    if nodes and n > 1 and policy["allow_multi_device"] and "rpc" in engine_caps.get("cluster", []):
        room = sum(per_dev)
        need = total + est["engine_bytes"] * (n - 1)
        if need <= room:
            return {"strategy": "rpc_split", "devices": list(range(n)), "gpu_layers": None, "host_bytes": 0,
                    "tensor_split": [round(b / room, 4) for b in per_dev],
                    "fit_margin_bytes": int(room - need), "status": "fits_cluster"}
    # coherent: weights spill into GPU-addressable host RAM; KV, activations and engine stay in device memory.
    # The GPU reads spilled weights over the host link every token, so decode slows by the spilled fraction
    # times the bandwidth ratio (estimated when the probe measured both).
    if budget["topology"] == "coherent" and engine_caps.get("coherent") and budget.get("host_coherent_bytes"):
        gpu_room = per_dev[best]
        non_weight = est["kv_bytes"] + est["activation_bytes"] + est["engine_bytes"]
        if non_weight <= gpu_room:
            on_gpu = gpu_room - non_weight
            host_needed = est["weights_bytes"] - on_gpu
            if 0 < host_needed <= budget["host_coherent_bytes"]:
                frac = host_needed / est["weights_bytes"]
                rel = None
                if budget.get("device_gbps") and budget.get("host_to_device_gbps"):
                    rel = round(1 / ((1 - frac) + frac * budget["device_gbps"] / budget["host_to_device_gbps"]), 3)
                return {"strategy": "coherent_spill", "devices": [best], "gpu_layers": None, "host_bytes": int(host_needed),
                        "gpu_weight_bytes": int(on_gpu), "spill_fraction": round(frac, 3), "decode_speed_relative_est": rel,
                        "fit_margin_bytes": int(budget["host_coherent_bytes"] - host_needed), "status": "fits_coherent"}
    # offload remainder to host
    if policy["allow_offload"] and engine_caps.get("offload"):
        gpu_room = per_dev[best]
        non_weight = est["kv_bytes"] + est["activation_bytes"] + est["engine_bytes"]
        if non_weight <= gpu_room:
            on_gpu = gpu_room - non_weight
            host_needed = est["weights_bytes"] - on_gpu
            if host_needed <= budget["host_bytes"]:
                return {"strategy": "offload", "devices": [best], "gpu_layers": None, "host_bytes": int(host_needed),
                        "gpu_weight_bytes": int(on_gpu),
                        "fit_margin_bytes": int(budget["host_bytes"] - host_needed), "status": "fits_with_offload"}
    return None


def shortfall(est: dict, budget: dict) -> int:
    per_dev = budget["gpu_bytes_per_device"]
    room = max(per_dev) if per_dev else 0
    return max(est["total_bytes"] - room, 0)


# --------------------------------------------------------------------------- candidates

def best_engine(cat: dict, model: dict, budget: dict) -> str | None:
    """`--engine best`: the first engine in the catalog's per-backend preference that is installed here
    and has a file for this model. llama.cpp is always last in those lists, so it is the fallback."""
    installed = set(budget.get("installed_engines", []))
    prefs = []
    for b in budget["device_backends"]:
        prefs += (cat.get("engine_preference") or {}).get(b, [])
    have = {e for f in model["files"] for e in f["engines"]}
    for e in prefs:
        if e in installed and e in have:
            return e
    return None


def candidates(cat: dict, model: dict, budget: dict, policy: dict) -> list[tuple[dict, str, dict]]:
    """(file, engine_name, engine_caps) combos usable on this hardware, best precision first."""
    backends = set(budget["device_backends"])
    installed = set(budget.get("installed_engines", []))
    want = best_engine(cat, model, budget) if policy["engine"] == "best" else policy["engine"]
    out = []
    for f in model["files"]:
        if policy["quant"] and f["quant"] != policy["quant"]:
            continue
        for eng_name in f["engines"]:
            if want and eng_name != want:
                continue
            caps = cat["engines"].get(eng_name)
            if not caps:
                continue
            if backends and not (backends & set(caps["backends"])):
                continue
            out.append((f, eng_name, caps))
    # Runnable beats precise: installed engines first; the baseline engine (llama.cpp, the one every box runs)
    # before faster platform engines unless --engine picked one; then highest precision; then name for stability.
    out.sort(key=lambda t: (t[1] not in installed, t[1] != BASELINE_ENGINE if not want else False, quant_rank(t[0]["quant"]), t[1]))
    return out


def max_context(model: dict, file: dict, caps: dict, budget: dict, policy: dict) -> tuple[int | None, dict | None, dict | None]:
    """Largest context (power-of-two steps of 4096 granularity) that fits. Returns (ctx, est, placement)."""
    lo, hi = model["context"]["min"], model["context"]["max"]
    if policy["target_context"]:
        ctx = policy["target_context"]
        est = cost(model, file, caps, ctx, policy["n_seq"], policy["kv_quant"], tuple(budget["device_backends"]))
        pl = place(est, budget, caps, policy)
        return (ctx, est, pl) if pl else (None, est, None)
    best = None
    # binary search on context in 4096-token steps
    lo_s, hi_s = lo // 4096, hi // 4096
    while lo_s <= hi_s:
        mid = (lo_s + hi_s) // 2
        ctx = mid * 4096
        est = cost(model, file, caps, ctx, policy["n_seq"], policy["kv_quant"], tuple(budget["device_backends"]))
        pl = place(est, budget, caps, policy)
        if pl:
            best = (ctx, est, pl)
            lo_s = mid + 1
        else:
            hi_s = mid - 1
    if best:
        return best
    est = cost(model, file, caps, lo, policy["n_seq"], policy["kv_quant"], tuple(budget["device_backends"]))
    return None, est, None


def _selection(role: str, model: dict, file: dict, eng: str, ctx: int, est: dict, pl: dict, policy: dict, budget: dict) -> dict:
    pl = dict(pl)
    status = pl.pop("status")
    margin = pl.pop("fit_margin_bytes")
    est = dict(est, fit_margin_bytes=margin)
    return {
        "role": role,
        "model_id": model["id"],
        "modality": model["modality"],
        "quant": file["quant"],
        "engine": eng,
        "recipe_id": f"{model['id']}-{file['quant']}/{'-'.join(budget['device_backends'])}-{budget['topology']}",
        "placement": pl,
        "context_tokens": ctx,
        "n_seq": policy["n_seq"],
        "estimate": est,
        "status": status,
        "verified": model.get("verified", False),
    }


def _rejection(model: dict, file: dict, eng: str, est: dict, budget: dict, reason: str) -> dict:
    return {"model_id": model["id"], "quant": file["quant"], "engine": eng, "reason": reason,
            "shortfall_bytes": shortfall(est, budget), "total_bytes": est["total_bytes"]}


# --------------------------------------------------------------------------- per-test selection

def select_fixed(cat: dict, model_id: str, budget: dict, policy: dict, role: str = "text") -> tuple[list, list]:
    model = get_model(cat, model_id)
    rejected = []
    for f, eng, caps in candidates(cat, model, budget, policy):
        ctx, est, pl = max_context(model, f, caps, budget, policy)
        if pl:
            return [_selection(role, model, f, eng, ctx, est, pl, policy, budget)], rejected
        rejected.append(_rejection(model, f, eng, est, budget, "does not fit at min context"))
    if policy.get("force") and policy["target_context"]:
        # "Try it anyway": the estimate says no, the user wants the real answer. Single-device placement at the
        # requested context, status "forced", negative margin kept so the result records how far over it was.
        for f, eng, caps in candidates(cat, model, budget, policy):
            est = cost(model, f, caps, policy["target_context"], policy["n_seq"], policy["kv_quant"], tuple(budget["device_backends"]))
            pl = {"strategy": "single", "devices": [0], "status": "forced", "fit_margin_bytes": -(shortfall(est, budget) or 0)}
            return [_selection(role, model, f, eng, policy["target_context"], est, pl, policy, budget)], rejected
    if not rejected:
        rejected.append({"model_id": model_id, "quant": None, "engine": None, "reason": "no file/engine usable on this backend", "shortfall_bytes": None, "total_bytes": None})
    return [], rejected


def select_ladder(cat: dict, budget: dict, policy: dict) -> tuple[list, list]:
    rejected = []
    models = [m for m in cat["models"] if m["modality"] == "text"]
    if policy["model"]:
        models = [m for m in models if m["id"] == policy["model"]]
    for model in sorted(models, key=lambda m: -m["rank"]):
        for f, eng, caps in candidates(cat, model, budget, policy):
            ctx, est, pl = max_context(model, f, caps, budget, policy)
            if pl:
                return [_selection("text", model, f, eng, ctx, est, pl, policy, budget)], rejected
            rejected.append(_rejection(model, f, eng, est, budget, "does not fit at min context"))
    return [], rejected


def select_host_pressure(cat: dict, model_id: str, budget: dict, policy: dict) -> tuple[list, list]:
    """Fixed model; also report max n_seq that fits the KV pool at target context."""
    sels, rejected = select_fixed(cat, model_id, budget, policy)
    if sels:
        s = sels[0]
        model = get_model(cat, model_id)
        per_seq = kv_bytes(model, s["context_tokens"], 1, policy["kv_quant"])
        if per_seq > 0:
            headroom = s["estimate"]["fit_margin_bytes"] + s["estimate"]["kv_bytes"]
            s["max_n_seq"] = max(int(headroom // per_seq), 1)
    return sels, rejected


def select_warm_contexts(cat: dict, model_id: str, budget: dict, policy: dict) -> tuple[list, list]:
    """Fixed model with `slots` sequences on the GPU; the host prompt cache is the budget under test.
    Per agent the host carries one context of KV (restored into a slot on its next turn) plus its client
    process. The modeled knee is where n agents' KV exceeds the cache; the levels cross it."""
    sels, rejected = select_fixed(cat, model_id, budget, dict(policy, n_seq=max(int(policy["slots"]), 1)))
    if sels:
        s = sels[0]
        model = get_model(cat, model_id)
        kv1 = kv_bytes(model, s["context_tokens"], 1, policy["kv_quant"])
        per_agent = kv1 + int(policy["client_bytes"])
        s["warm"] = {"slots": s["n_seq"], "kv_per_agent_bytes": kv1, "client_bytes": int(policy["client_bytes"]),
                     "per_agent_host_bytes": per_agent, "cache_ram_bytes": None, "knee_agents": None, "levels": None}
    return sels, rejected


def finalize_warm_contexts(sels: list, budget: dict, policy: dict) -> None:
    """Needs host_bytes_after_plan (set once the GPU selection is known)."""
    for s in sels:
        w = s.get("warm")
        if not w:
            continue
        host = max(int(budget["host_bytes_after_plan"] or 0) - int(policy["server_host_bytes"]), 0)
        kv_share = w["kv_per_agent_bytes"] / w["per_agent_host_bytes"] if w["per_agent_host_bytes"] else 1.0
        w["cache_ram_bytes"] = int(host * kv_share)
        w["knee_agents"] = int(host // w["per_agent_host_bytes"]) if w["per_agent_host_bytes"] else None
        levels, n = [], w["slots"]
        while len(levels) < 8 and (not levels or levels[-1] < 2 * (w["knee_agents"] or n)):
            levels.append(n)
            n *= 2
        w["levels"] = levels


def select_residency(cat: dict, preset_name: str, budget: dict, policy: dict) -> tuple[list, list]:
    preset = cat["presets"][preset_name]
    ctx = policy["target_context"] or preset.get("context", 32768)
    remaining = list(budget["gpu_bytes_per_device"])
    remaining_host = budget.get("host_coherent_bytes", 0)
    sels, rejected = [], []
    resident_bytes = 0
    for mid in preset["models"]:
        model = get_model(cat, mid)
        placed = False
        for f, eng, caps in candidates(cat, model, budget, policy):
            c = min(ctx, model["context"]["max"]) if model["modality"] in ("text", "vision") else 0
            est = cost(model, f, caps, c, policy["n_seq"], policy["kv_quant"], tuple(budget["device_backends"]))
            sub_budget = dict(budget, gpu_bytes_per_device=remaining, host_coherent_bytes=remaining_host, nodes=None)
            pl = place(est, sub_budget, caps, dict(policy, allow_offload=False, allow_multi_device=False))
            # Residency keeps each model whole on one device (or one device plus the coherent host tier);
            # on a cluster that means one model per node, never split across the network.
            if pl and pl["strategy"] in ("single", "coherent_spill"):
                d = pl["devices"][0]
                remaining[d] -= est["total_bytes"] - pl.get("host_bytes", 0)
                remaining_host -= pl.get("host_bytes", 0)
                sel = _selection(model["modality"], model, f, eng, c, est, pl, policy, budget)
                sel["status"] = "resident" if pl["strategy"] == "single" else "resident_coherent"
                sels.append(sel)
                resident_bytes += est["total_bytes"]
                placed = True
                break
        if not placed:
            f0 = model["files"][0]
            est = cost(model, f0, cat["engines"][f0["engines"][0]], ctx if model["modality"] in ("text", "vision") else 0, policy["n_seq"], policy["kv_quant"], tuple(budget["device_backends"]))
            sel = _selection(model["modality"], model, f0, f0["engines"][0], ctx, est,
                             {"strategy": "evicted", "devices": [], "gpu_layers": None, "host_bytes": 0,
                              "fit_margin_bytes": max(budget["gpu_bytes_per_device"] or [0]) - est["total_bytes"], "status": "evicted"}, policy, budget)
            sels.append(sel)
    return sels, rejected


# --------------------------------------------------------------------------- plan

def _stable_id(doc: dict, drop: tuple[str, ...]) -> str:
    clean = {k: v for k, v in doc.items() if k not in drop}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _downloads(cat: dict, sels: list, cache_path: str) -> list:
    out, seen = [], set()
    for s in sels:
        if s["status"] == "evicted":
            continue
        key = (s["model_id"], s["quant"])
        if key in seen:
            continue
        seen.add(key)
        model = get_model(cat, s["model_id"])
        f = next(f for f in model["files"] if f["quant"] == s["quant"])
        local = os.path.join(cache_path, s["model_id"], s["quant"])
        if os.path.isdir(local):
            if os.path.exists(os.path.join(local, "config.json")):   # checkpoint directory (safetensors / MLX)
                have = sum(os.path.getsize(q) for q in glob.glob(os.path.join(local, "**", "*.safetensors"), recursive=True))
            else:                                                    # GGUF file(s)
                have = sum(os.path.getsize(q) for q in glob.glob(os.path.join(local, "*.gguf")))
        else:
            have = 0
        # present means complete: a half-downloaded artifact is reported as missing with what is there
        out.append({"model_id": s["model_id"], "quant": s["quant"], "bytes": f["bytes"], "url": f.get("url"),
                    "present": have >= f["bytes"], "have_bytes": have})
    return out


def build_plan(probe: dict, cat: dict, test: str, preset: str | None, policy_overrides: dict) -> dict:
    assert test in TESTS, f"unknown test {test}"
    policy = dict(DEFAULT_POLICY)
    overrides = {k: v for k, v in policy_overrides.items() if v is not None}
    policy.update(overrides)
    budget = compute_budget(probe, policy)
    warnings = list(probe.get("warnings", []))

    if test == "ladder":
        sels, rejected = select_ladder(cat, budget, policy)
    elif test == "inference":
        if not policy["model"]:
            raise SystemExit("inference test needs --model <id>")
        sels, rejected = select_fixed(cat, policy["model"], budget, policy)
    elif test == "host_pressure":
        if not policy["model"]:
            raise SystemExit("host_pressure test needs --model <id>")
        sels, rejected = select_host_pressure(cat, policy["model"], budget, policy)
    elif test == "warm_contexts":
        if not policy["model"]:
            raise SystemExit("warm_contexts test needs --model <id>")
        sels, rejected = select_warm_contexts(cat, policy["model"], budget, policy)
    else:
        if not preset:
            raise SystemExit("residency test needs --preset <name>")
        sels, rejected = select_residency(cat, preset, budget, policy)

    used = sum(s["estimate"]["total_bytes"] for s in sels if s["status"] != "evicted")
    host_used = sum(s["placement"].get("host_bytes", 0) for s in sels)
    if budget["topology"] == "unified":
        budget["host_bytes_after_plan"] = max(budget["system_total_bytes"] - used, 0)
        need_now = used + budget["host_min_bytes"]
    else:
        budget["host_bytes_after_plan"] = budget["host_bytes"] - host_used
        need_now = host_used + budget["host_min_bytes"]
    if budget["host_bytes_after_plan"] < budget["host_min_bytes"]:
        warnings.append(f"only {fmt_bytes(budget['host_bytes_after_plan'])} host RAM left after plan; tools and sandboxes may starve")
    if test == "warm_contexts":
        finalize_warm_contexts(sels, budget, policy)
    if sels and budget["system_available_bytes"] < need_now:
        warnings.append(f"machine is busy: {fmt_bytes(budget['system_available_bytes'])} available now but plan needs {fmt_bytes(need_now)}; run will refuse until memory is freed")
    busy_dev = [d for d in probe["devices"] if d.get("memory_free_bytes") is not None and d["memory_free_bytes"] < 0.5 * d["memory_total_bytes"]]
    if busy_dev and budget["topology"] == "discrete":
        warnings.append("GPU memory is more than half used by other processes; free it before running")
    for s_ in sels:
        if s_["status"] == "forced":
            warnings.append(f"FORCED: {s_['model_id']}/{s_['quant']} at {s_['context_tokens']:,} ctx is estimated {fmt_bytes(-s_['estimate']['fit_margin_bytes'])} over budget; the run will show whether the driver (GTT, managed memory) absorbs it")
    for s in sels:
        if s["status"] == "evicted":
            continue
        pl = s["placement"]
        if pl["strategy"] == "coherent_spill":
            rel = pl.get("decode_speed_relative_est")
            warnings.append(f"{s['model_id']}: {pl['spill_fraction'] * 100:.0f}% of weights ({fmt_bytes(pl['host_bytes'])}) in coherent host RAM"
                            + (f"; decode estimated at {rel * 100:.0f}% of in-device speed" if rel else "; no bandwidth figures, speed unknown"))
        if pl["strategy"] == "rpc_split":
            warnings.append(f"{s['model_id']}: split over {len(pl['devices'])} devices across nodes via RPC (tensor-split {pl['tensor_split']}); every token crosses the network")
        if not s["verified"]:
            warnings.append(f"{s['model_id']}/{s['quant']}: catalog entry unverified; sizes are estimates")
        if budget["installed_engines"] and s["engine"] not in budget["installed_engines"]:
            warnings.append(f"{s['model_id']}: engine {s['engine']} not detected on this host (installed: {', '.join(budget['installed_engines'])})")
        elif not budget["installed_engines"]:
            warnings.append(f"{s['model_id']}: engine {s['engine']} not detected on this host (no engines installed)")

    downloads = _downloads(cat, sels, probe["storage"]["model_cache_path"])
    dl_total = sum(d["bytes"] for d in downloads if not d["present"])
    free = probe["storage"].get("free_bytes")
    if free is not None and dl_total > free:
        warnings.append(f"downloads need {fmt_bytes(dl_total)} but cache has {fmt_bytes(free)} free")

    if policy["engine"] == "best" and sels:
        warnings.append(f"engine=best resolved to {sels[0]['engine']} on {'/'.join(budget['device_backends'])} (installed: {', '.join(budget['installed_engines']) or 'none'})")
    doc = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": None,
        "created": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "probe_id": probe["probe_id"],
        "host": probe["host"]["hostname"],
        "catalog_version": cat["catalog_version"],
        "test": test,
        "preset": preset,
        "policy": policy,
        "custom": bool(overrides),
        "budget": budget,
        "selections": sels,
        "rejected": rejected,
        "downloads": downloads,
        "download_total_bytes": dl_total,
        "estimated_runtime_s": RUNTIME_EST[test] if sels else 0,
        "warnings": warnings,
    }
    doc["plan_id"] = _stable_id(doc, drop=("plan_id", "created"))
    return doc


def summarize(plan: dict, explain: bool = False) -> str:
    b = plan["budget"]
    L = [f"plan {plan['plan_id'][:12]}  test={plan['test']}" + (f" preset={plan['preset']}" if plan["preset"] else "") + f"  host={plan['host']}  catalog={plan['catalog_version']}" + ("  [custom policy]" if plan["custom"] else "")]
    L.append(f"budget: topology={b['topology']}  gpu=" + " + ".join(fmt_bytes(x) for x in b["gpu_bytes_per_device"]) + f" = {fmt_bytes(b['gpu_bytes_total'])}"
             + (f"  coherent host tier={fmt_bytes(b['host_coherent_bytes'])}" if b.get("host_coherent_bytes") else "")
             + f"  host after plan={fmt_bytes(b['host_bytes_after_plan'])}")
    if b.get("nodes"):
        for nd in b["nodes"]:
            L.append(f"  node {nd['node']} {nd['hostname']}: {nd['topology']}  devices={nd['devices']}  ram={fmt_bytes(nd['system_total_bytes'])}")
    if not plan["selections"]:
        L.append("selections: NONE FIT")
    for s in plan["selections"]:
        e = s["estimate"]
        pl = s["placement"]
        line = (f"  [{s['status']:>17}] {s['role']:<9} {s['model_id']}/{s['quant']} on {s['engine']}  "
                f"ctx={s['context_tokens']:,}  n_seq={s['n_seq']}  total={fmt_bytes(e['total_bytes'])}  margin={fmt_bytes(e['fit_margin_bytes'])}  {pl['strategy']}")
        if pl.get("host_bytes"):
            line += f" (host {fmt_bytes(pl['host_bytes'])}" + (f", {pl['spill_fraction'] * 100:.0f}% spilled" if pl.get("spill_fraction") else "") + ")"
        if pl.get("tensor_split"):
            line += f" split={pl['tensor_split']}"
        if "max_n_seq" in s:
            line += f"  max_n_seq={s['max_n_seq']}"
        if s.get("warm"):
            w = s["warm"]
            line += (f"\n    warm_contexts: {w['slots']} GPU slots, host prompt cache {fmt_bytes(w['cache_ram_bytes'])}, "
                     f"{fmt_bytes(w['per_agent_host_bytes'])} host RAM per agent ({fmt_bytes(w['kv_per_agent_bytes'])} KV + {fmt_bytes(w['client_bytes'])} client), "
                     f"modeled knee {w['knee_agents']} agents, levels {w['levels']}")
        L.append(line)
        if explain:
            L.append(f"               weights={fmt_bytes(e['weights_bytes'])} kv={fmt_bytes(e['kv_bytes'])} act={fmt_bytes(e['activation_bytes'])} engine={fmt_bytes(e['engine_bytes'])}")
    if plan["rejected"]:
        L.append("rejected:")
        for r in plan["rejected"]:
            sf = fmt_bytes(r["shortfall_bytes"]) if r["shortfall_bytes"] is not None else "n/a"
            L.append(f"  {r['model_id']}/{r['quant']} on {r['engine']}: {r['reason']}  short by {sf}")
    if plan["downloads"]:
        need = [d for d in plan["downloads"] if not d["present"]]
        L.append(f"downloads: {len(need)} of {len(plan['downloads'])} missing, {fmt_bytes(plan['download_total_bytes'])} to fetch")
        for d in plan["downloads"]:
            L.append(f"  {'have' if d['present'] else 'need'} {d['model_id']}/{d['quant']} {fmt_bytes(d['bytes'])}")
    L.append(f"estimated runtime: {plan['estimated_runtime_s'] // 60} min")
    for w in plan["warnings"]:
        L.append(f"warning: {w}")
    return "\n".join(L)
