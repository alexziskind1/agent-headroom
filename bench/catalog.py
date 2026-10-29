"""Model catalog loader. See docs/probe_planner_spec.md section 2."""
from __future__ import annotations

import json
import os

MODALITIES = {"text", "vision", "image_gen", "asr", "tts", "embed"}
KV_QUANT_FACTOR = {"fp16": 1.0, "bf16": 1.0, "q8": 0.5, "q4": 0.25}
QUANT_PRECISION_ORDER = ["bf16", "fp16", "fp8", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_0", "iq4_xs", "q3_k_m", "q2_k"]


def default_catalog_path() -> str:
    return os.environ.get("BENCH_CATALOG", os.path.join(os.path.dirname(__file__), "catalog.json"))


def load_catalog(path: str | None = None) -> dict:
    path = path or default_catalog_path()
    with open(path) as f:
        cat = json.load(f)
    validate_catalog(cat)
    return cat


def validate_catalog(cat: dict) -> None:
    assert "catalog_version" in cat, "catalog_version missing"
    assert "engines" in cat, "engines missing"
    ids = set()
    for m in cat["models"]:
        assert m["id"] not in ids, f"duplicate model id {m['id']}"
        ids.add(m["id"])
        assert m["modality"] in MODALITIES, f"{m['id']}: bad modality"
        if m["modality"] == "text" or m["modality"] == "vision":
            if "kv_bytes_per_token" not in m:
                a = m["arch"]
                # K and V per layer, 2 bytes each at fp16. MLA models (kv_single_latent) cache one compressed
                # latent of head_dim per layer instead of separate K and V.
                kv_tensors = 1 if a.get("kv_single_latent") else 2
                m["kv_bytes_per_token"] = kv_tensors * a["n_layers"] * a["n_kv_heads"] * a["head_dim"] * 2
        else:
            assert "working_set_bytes" in m, f"{m['id']}: non-text model needs working_set_bytes"
        m.setdefault("activation_overhead_fraction", 0.05)
        m.setdefault("rank", m.get("params_total", 0))
        m.setdefault("verified", False)
        assert m["files"], f"{m['id']}: no files"
        for f in m["files"]:
            assert f["bytes"] > 0
            assert f["engines"], f"{m['id']}/{f['quant']}: no engines"
    for name, e in cat["engines"].items():
        assert "overhead_bytes" in e, f"engine {name}: overhead_bytes missing"
    for pname, p in cat.get("presets", {}).items():
        for mid in p["models"]:
            assert mid in ids, f"preset {pname}: unknown model {mid}"


def quant_rank(q: str) -> int:
    """Higher precision sorts first."""
    q = q.lower()
    return QUANT_PRECISION_ORDER.index(q) if q in QUANT_PRECISION_ORDER else len(QUANT_PRECISION_ORDER)


def get_model(cat: dict, model_id: str) -> dict:
    for m in cat["models"]:
        if m["id"] == model_id:
            return m
    raise KeyError(f"model {model_id} not in catalog")
