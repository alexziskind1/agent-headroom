"""bench report: one table per test across every results.json under out/ (local and fetched remote)."""
from __future__ import annotations

import glob
import json
import os

from .probe import fmt_bytes


def load_results(roots: list[str]) -> list[dict]:
    out = []
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "results.json"), recursive=True)):
            if "/aborted/" in path.replace(os.sep, "/"):
                continue  # out/aborted holds smoke tests and failed attempts kept for forensics
            try:
                with open(path) as f:
                    r = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            r["_path"] = path
            out.append(r)
    return out


def _placement(sel: dict, r: dict | None = None) -> str:
    pl = sel.get("placement") or {}
    st = pl.get("strategy") or "?"
    if st == "coherent_spill":
        return f"coherent_spill {pl.get('spill_fraction', 0) * 100:.0f}% host"
    if st == "rpc_split":
        n = len((r or {}).get("cluster", {}).get("nodes") or []) + 1 if r else len(pl.get("devices") or [])
        return f"rpc_split x{n} {pl.get('tensor_split')}"
    return st


def _policy(r: dict) -> str:
    extra = (r.get("replay") or {}).get("extra_args") or []
    for i, a in enumerate(extra):
        if a == "--output-token-policy" and i + 1 < len(extra):
            return extra[i + 1]
        if a.startswith("--output-token-policy="):
            return a.split("=", 1)[1]
    return "exact"


def _row_inference(r: dict) -> list[str]:
    s, ap = r["selection"], r.get("agentperf") or {}
    tot = ap.get("totals") or {}
    return [r["host"], f"{s['model_id']}/{s['quant']}", (r.get("server") or {}).get("engine") or s.get("engine") or "?", _policy(r), _placement(s, r), f"{s['context_tokens']:,}", r["replay"]["name"], r["replay"]["tool_mode"],
            f"{ap.get('output_tokens_per_second') or 0:.2f}", f"{(ap.get('measured_duration_ms') or 0) / 60000:.1f}",
            f"{tot.get('successful_turns', '?')}/{tot.get('failed_turns', '?')}",
            fmt_bytes(r["memory"].get("peak_host_used_bytes")), "yes" if r["artifact"].get("verified") else "no", r["status"]]


def _row_pressure(r: dict) -> list[str]:
    s = r["selection"]
    levels = ", ".join(f"n{l['n']}={(l.get('aggregate') or {}).get('aggregate_output_tokens_per_second') or 0:.1f}" for l in r["levels"])
    return [r["host"], f"{s['model_id']}/{s['quant']}", (r.get("server") or {}).get("engine") or s.get("engine") or "?", _placement(s, r), f"{s['context_tokens']:,}", r["replay"]["tool_mode"],
            str(r["max_sustained_agents"]), levels, fmt_bytes(r["memory"].get("peak_host_used_bytes")),
            fmt_bytes(r["memory"].get("peak_swap_used_bytes")) if r["memory"].get("peak_swap_used_bytes") is not None else "-", r["status"]]


def _row_warm(r: dict) -> list[str]:
    s, w = r["selection"], r.get("warm") or {}
    lv = "; ".join(f"n{l['n']}: hit {((l.get('aggregate') or {}).get('prompt_cache_hit_ratio') or 0):.2f}, ttft {((l.get('aggregate') or {}).get('ttft_p50_ms') or 0) / 1000:.1f}s, {((l.get('aggregate') or {}).get('tasks_per_minute') or 0):.2f} tasks/min" for l in r["levels"])
    return [r["host"], f"{s['model_id']}/{s['quant']}", f"{s['context_tokens']:,}", str(w.get("slots")), fmt_bytes(w.get("cache_ram_bytes")),
            str(w.get("knee_agents")), str(r.get("knee_measured_agents")), str(r["max_sustained_agents"]), lv,
            fmt_bytes(r["memory"].get("peak_host_used_bytes")), f"{(r['memory'].get('peak_cpu_util') or 0):.2f}", r["status"]]


def _engines(r: dict) -> str:
    srv = r.get("server") or {}
    return "+".join(sorted((srv.get("engines") or {}).keys()) or [srv.get("engine") or "llama.cpp"])


def _row_residency(r: dict) -> list[str]:
    return [r["host"], r["preset"] or "", _engines(r), str(len(r["resident"])), str(len(r.get("resident_coherent") or [])), str(len(r["evicted"])), str(r.get("tasks_per_minute")),
            f"{r['swap_s_total']:.0f}", fmt_bytes(r["memory"].get("peak_host_used_bytes")), r["status"]]


TABLES = {
    "inference": (["host", "model", "engine", "policy", "placement", "ctx", "replay", "tools", "tok/s", "min", "turns ok/fail", "peak host", "verified", "status"], _row_inference),
    "ladder": (["host", "model", "engine", "policy", "placement", "ctx", "replay", "tools", "tok/s", "min", "turns ok/fail", "peak host", "verified", "status"], _row_inference),
    "host_pressure": (["host", "model", "engine", "placement", "ctx/agent", "tools", "max agents", "agg tok/s by level", "peak host", "peak swap", "status"], _row_pressure),
    "warm_contexts": (["host", "model", "ctx/agent", "slots", "host cache", "knee (model)", "knee (measured)", "max agents", "by level", "peak host", "peak cpu", "status"], _row_warm),
    "residency": (["host", "preset", "engines", "resident", "in host tier", "evicted", "tasks/min", "swap s", "peak host", "status"], _row_residency),
}


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    L = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    return "\n".join(L)


def render(results: list[dict]) -> str:
    out = []
    for test, (header, fn) in TABLES.items():
        rows = [fn(r) for r in results if r.get("test") == test]
        if rows:
            out.append(f"### {test}\n\n{markdown_table(header, rows)}\n")
    return "\n".join(out) if out else "no results found"
