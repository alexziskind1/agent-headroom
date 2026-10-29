"""bench CLI: probe, plan. `plan` is always a dry run and never downloads or launches anything."""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

from . import __version__
from .catalog import load_catalog
from .planner import RESIDENT, RUNNABLE, TESTS, build_plan
from .planner import summarize as plan_summary
from .probe import run_probe
from .probe import summarize as probe_summary
from .runner import EXIT_REFUSED, Refused, run_plan
from .runner import summarize as run_summary
from .remote import main as remote_main
from .remote import run_script as remote_script
from .pressure import run_pressure
from .pressure import summarize as pressure_summary
from .residency import run_residency
from .residency import summarize as residency_summary
from .report import load_results, render

EXIT_OK, EXIT_NOTHING_FITS, EXIT_PROBE_FAILED = 0, 2, 3


def _write_json(path: str, doc: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=False)
        f.write("\n")


def cmd_probe(args: argparse.Namespace) -> int:
    try:
        doc = run_probe(storage_test=not args.no_storage_test)
    except Exception as e:  # noqa: BLE001
        print(f"probe failed: {e}", file=sys.stderr)
        return EXIT_PROBE_FAILED
    out = args.out or os.path.join("out", f"probe-{doc['host']['hostname']}.json")
    _write_json(out, doc)
    if args.json:
        print(json.dumps(doc, indent=2))
    else:
        print(probe_summary(doc))
        print(f"wrote {out}")
    return EXIT_OK


def cmd_plan(args: argparse.Namespace) -> int:
    if args.probe:
        with open(args.probe) as f:
            probe = json.load(f)
    else:
        try:
            probe = run_probe(storage_test=False, bandwidth=True)
        except Exception as e:  # noqa: BLE001
            print(f"probe failed: {e}", file=sys.stderr)
            return EXIT_PROBE_FAILED
    cat = load_catalog(args.catalog)
    overrides = {
        "model": args.model,
        "quant": args.quant,
        "engine": args.engine,
        "target_context": args.context,
        "reserve_fraction": args.reserve,
        "allow_offload": True if args.allow_offload else None,
        "allow_multi_device": False if args.single_device else None,
        "n_seq": args.n_seq,
        "kv_quant": args.kv_quant,
        "placement": args.placement,
        "force": True if args.force else None,
        "slots": args.slots,
    }
    plan = build_plan(probe, cat, args.test, args.preset, overrides)
    out = args.out or os.path.join("out", f"plan-{plan['host']}-{args.test}.json")
    _write_json(out, plan)
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(plan_summary(plan, explain=args.explain))
        print(f"wrote {out}")
    fits = any(s["status"] in RUNNABLE + RESIDENT for s in plan["selections"])
    return EXIT_OK if fits else EXIT_NOTHING_FITS


def cmd_run(args: argparse.Namespace) -> int:
    with open(args.plan) as f:
        plan = json.load(f)
    cat = load_catalog(args.catalog)
    if cat["catalog_version"] != plan["catalog_version"]:
        print(f"catalog {cat['catalog_version']} differs from plan's {plan['catalog_version']}; re-plan first", file=sys.stderr)
        return EXIT_REFUSED
    opts = {
        "replay": args.replay, "tool_mode": args.tool_mode, "model_file": args.model_file,
        "verify": not args.no_verify, "port": args.port, "out_dir": args.out_dir,
        "server_extra": args.server_arg or [],
        "replay_extra": (["--output-token-policy", args.output_policy] if args.output_policy else []) + (args.replay_arg or []),
        "load_timeout": args.load_timeout,
        "levels": [int(x) for x in args.levels.split(",")] if args.levels else None,
        "slots": args.slots, "fixed_output_tokens": args.fixed_output_tokens,
        "cache_ram_bytes": int(args.cache_ram_gib * 2**30) if args.cache_ram_gib else None,
    }
    try:
        if plan["test"] == "residency":
            opts["replay"] = args.replay or "aa-mini-v1"
            opts["tool_mode"] = args.tool_mode or "none"
            opts["model_files"] = dict(kv.split("=", 1) for kv in (args.model_files or []))
            code, result = run_residency(plan, cat, opts)
            print(residency_summary(result))
        elif plan["test"] == "warm_contexts":
            from .warm import run_warm, summarize as warm_summary
            code, result = run_warm(plan, cat, opts)
            print(warm_summary(result))
        elif plan["test"] == "host_pressure":
            if args.replay is None:
                opts["replay"] = "aa-mini-v1"
            if args.tool_mode is None:
                opts["tool_mode"] = "fixed_delay"
            code, result = run_pressure(plan, cat, opts)
            print(pressure_summary(result))
        else:
            opts["replay"] = args.replay or "agentperf-default-v1"
            opts["tool_mode"] = args.tool_mode or "none"
            code, result = run_plan(plan, cat, opts)
            print(run_summary(result))
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    return code


def cmd_cluster(args: argparse.Namespace) -> int:
    from .cluster import load_spec, probe_cluster
    from .cluster import summarize as cluster_summary
    spec = load_spec(args.spec)
    try:
        doc = probe_cluster(spec, storage_test=not args.no_storage_test)
    except Exception as e:  # noqa: BLE001
        print(f"cluster probe failed: {e}", file=sys.stderr)
        return EXIT_PROBE_FAILED
    out = args.out or os.path.join("out", f"probe-{spec['name']}.json")
    _write_json(out, doc)
    if args.json:
        print(json.dumps(doc, indent=2))
    else:
        print(cluster_summary(doc))
        print(f"wrote {out}")
    return EXIT_OK


def cmd_nettest(args: argparse.Namespace) -> int:
    from .cluster import nettest_client, nettest_serve
    if args.role == "serve":
        nettest_serve(args.port, once=args.once)
        return 0
    print(json.dumps(nettest_client(args.addr, args.port)))
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from .cluster import sample_loop
    sample_loop(args.interval)
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from . import engines
    from .planner import get_model
    cat = load_catalog(args.catalog)
    model = get_model(cat, args.model)
    file = next((f for f in model["files"] if not args.quant or f["quant"] == args.quant), None)
    if not file:
        print(f"no file with quant {args.quant} for {args.model}", file=sys.stderr)
        return EXIT_REFUSED
    dest = engines.artifact_path(args.model, file)
    os.makedirs(dest, exist_ok=True)
    for cmd in engines.fetch_argv(args.model, file, dest):
        print("$ " + " ".join(cmd), flush=True)
        if not args.dry_run:
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                print(f"fetch failed with {rc}", file=sys.stderr)
                return rc
    if not args.dry_run:
        if engines.artifact_kind(file) == "dir":
            have = engines.dir_artifact_bytes(dest)
        else:
            have = sum(os.path.getsize(p) for p in glob.glob(os.path.join(dest, "*.gguf")))
        print(f"{dest}: {have:,} bytes (catalog {file['bytes']:,}) {'OK' if have == file['bytes'] else 'SIZE MISMATCH'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results = load_results(args.roots or ["out"])
    print(render(results))
    if args.out:
        with open(args.out, "w") as f:
            f.write(render(results) + "\n")
        print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench", description="Memory-aware agentic benchmark")
    p.add_argument("--version", action="version", version=f"bench {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("probe", help="measure this machine and write probe.json")
    pp.add_argument("--out", help="output path (default out/probe-<host>.json)")
    pp.add_argument("--json", action="store_true", help="print full JSON instead of summary")
    pp.add_argument("--no-storage-test", action="store_true", help="skip the 256 MB disk read test")
    pp.set_defaults(fn=cmd_probe)

    pl = sub.add_parser("plan", help="dry run: choose models and contexts for a test, write plan.json")
    pl.add_argument("test", choices=TESTS)
    pl.add_argument("--probe", help="use an existing probe.json instead of probing now")
    pl.add_argument("--catalog", help="catalog path (default bundled)")
    pl.add_argument("--preset", help="residency preset name")
    pl.add_argument("--model", help="model id (required for inference/host_pressure; filters ladder)")
    pl.add_argument("--quant", help="force a quant")
    pl.add_argument("--engine", help="force an engine, or `best`: fastest installed engine for this backend (catalog engine_preference); default is the baseline llama.cpp-first order")
    pl.add_argument("--context", type=int, help="fixed context tokens instead of maximizing")
    pl.add_argument("--reserve", type=float, help="GPU reserve fraction (default 0.10)")
    pl.add_argument("--allow-offload", action="store_true", help="let weights spill to host RAM")
    pl.add_argument("--single-device", action="store_true", help="disallow tensor/pipeline parallel")
    pl.add_argument("--n-seq", type=int, help="concurrent sequences the KV pool must hold")
    pl.add_argument("--kv-quant", choices=["fp16", "bf16", "q8", "q4"], help="KV cache precision")
    pl.add_argument("--force", action="store_true", help="with --context: plan it even if the estimate says it does not fit (status forced); the run reports what really happens")
    pl.add_argument("--slots", type=int, help="warm_contexts: GPU slots the agent fleet shares (default 8)")
    pl.add_argument("--placement", choices=["single", "rpc_split"], help="force a placement (cluster probes: rpc_split splits even when one node would do)")
    pl.add_argument("--explain", action="store_true", help="print cost arithmetic per selection")
    pl.add_argument("--json", action="store_true", help="print full JSON instead of summary")
    pl.add_argument("--out", help="output path (default out/plan-<host>-<test>.json)")
    pl.set_defaults(fn=cmd_plan)

    pr = sub.add_parser("run", help="execute a plan.json (inference/host_pressure via llama-server + agentperf-local)")
    pr.add_argument("plan")
    pr.add_argument("--catalog", help="catalog path (default bundled)")
    pr.add_argument("--replay", help="agentperf-default-v1 (default for inference) or aa-mini-v1 (default for host_pressure); "
                    "warm_contexts also takes warm-long-v1 (the two 44K/56K-token meeting tasks, 41 turns) or a custom manifest path")
    pr.add_argument("--tool-mode", choices=["none", "fixed_delay", "live"], help="default: none (inference), fixed_delay (host_pressure)")
    pr.add_argument("--model-files", action="append", metavar="MODEL_ID=PATH", help="residency: per-model file override (repeatable)")
    pr.add_argument("--slots", type=int, help="warm_contexts: override the plan's GPU slot count")
    pr.add_argument("--fixed-output-tokens", type=int, help="warm_contexts: output tokens per turn (fixed policy, default 64)")
    pr.add_argument("--cache-ram-gib", type=float, help="warm_contexts: override the plan's host prompt cache budget")
    pr.add_argument("--levels", help="host_pressure: comma-separated agent counts, e.g. 1,2,4,8 (default: doubling up to the plan's max_n_seq)")
    pr.add_argument("--model-file", help="use this GGUF instead of the cache dir")
    pr.add_argument("--no-verify", action="store_true", help="skip SHA-256 of the model file")
    pr.add_argument("--port", type=int, default=8089)
    pr.add_argument("--out-dir", help="run directory (default out/runs/<host>-<test>-<stamp>)")
    pr.add_argument("--server-arg", action="append", help="extra llama-server argument (repeatable)")
    pr.add_argument("--replay-arg", action="append", help="extra agentperf-local run argument, verbatim (repeatable)")
    pr.add_argument("--output-policy", choices=["exact", "recorded", "fixed"], help="agentperf output-token policy (default: exact for llama.cpp, recorded for every other engine)")
    pr.add_argument("--load-timeout", type=float, default=900, help="seconds to wait for the server to become healthy")
    pr.set_defaults(fn=cmd_run)

    fe = sub.add_parser("fetch", help="download a catalog artifact into the model cache (hf for checkpoint dirs, curl for GGUF)")
    fe.add_argument("model")
    fe.add_argument("--quant", help="quant to fetch (default: first file)")
    fe.add_argument("--catalog", default=None)
    fe.add_argument("--dry-run", action="store_true", help="print the commands only")
    fe.set_defaults(fn=cmd_fetch)
    rp = sub.add_parser("report", help="tables of every results.json under out/ (and fetched remote runs)")
    rp.add_argument("roots", nargs="*", help="directories to scan (default: out)")
    rp.add_argument("--out", help="also write the markdown here")
    rp.set_defaults(fn=cmd_report)

    rm = sub.add_parser("remote", help="deploy to <host> over ssh, run a bench command there, fetch its out/")
    rm.add_argument("host")
    rm.add_argument("args", nargs=argparse.REMAINDER, help="bench command and arguments to run remotely")
    rm.set_defaults(fn=lambda a: remote_main(a.host, a.args))

    sh = sub.add_parser("ssh", help="run a shell script string on <host> (supports jump/target?i=key), streaming output")
    sh.add_argument("host")
    sh.add_argument("script")
    sh.add_argument("--stdin", help="file to feed to the remote script's stdin")
    sh.set_defaults(fn=lambda a: remote_script(a.host, a.script, a.stdin))

    cl = sub.add_parser("cluster", help="multi-node: `cluster probe cluster.json` merges every node's probe and measures links")
    cl.add_argument("action", choices=["probe"])
    cl.add_argument("spec", help="cluster.json with workers [{ssh, addr, rpc_port}]; the head is this machine")
    cl.add_argument("--out", help="output path (default out/probe-<name>.json)")
    cl.add_argument("--no-storage-test", action="store_true")
    cl.add_argument("--json", action="store_true")
    cl.set_defaults(fn=cmd_cluster)

    nt = sub.add_parser("nettest", help="TCP bandwidth/RTT probe between nodes (used by `cluster probe`)")
    nt.add_argument("role", choices=["serve", "client"])
    nt.add_argument("--port", type=int, default=50099)
    nt.add_argument("--addr", help="client: server address")
    nt.add_argument("--once", action="store_true", help="serve: exit after one client")
    nt.set_defaults(fn=cmd_nettest)

    sm = sub.add_parser("sample", help="print host/GPU memory used as JSON lines until killed (used on cluster workers)")
    sm.add_argument("--interval", type=float, default=1.0)
    sm.set_defaults(fn=cmd_sample)

    args = p.parse_args(argv)
    return args.fn(args)
