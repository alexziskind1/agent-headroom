"""warm_contexts: planner knee model, level aggregate, report row."""
import copy
import unittest

from bench.catalog import load_catalog
from bench.planner import build_plan, GiB
from bench.report import render
import json
import os
import shutil
import tempfile

from bench.warm import SUBSETS, aggregate_warm, bundled_manifest, resolve_replay
from tests.test_planner import probe


class WarmPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def plan(self, p, **kw):
        return build_plan(p, copy.deepcopy(self.cat), "warm_contexts", None, kw)

    def test_knee_scales_with_host_ram(self):
        small = self.plan(probe("discrete", [96, 96], 128), model="qwen3.8-27b", target_context=65536)
        big = self.plan(probe("discrete", [96, 96], 256), model="qwen3.8-27b", target_context=65536)
        ws, wb = small["selections"][0]["warm"], big["selections"][0]["warm"]
        self.assertEqual(small["selections"][0]["n_seq"], 8)
        self.assertEqual(ws["kv_per_agent_bytes"], 65536 * 64 * 1024)  # 16 layers x 4 kv heads x 256 x K+V x fp16
        self.assertGreater(wb["cache_ram_bytes"], 2 * ws["cache_ram_bytes"] * 0.9)
        self.assertGreater(wb["knee_agents"], ws["knee_agents"] * 1.8)
        self.assertGreaterEqual(ws["levels"][-1], 2 * ws["knee_agents"])  # the sweep crosses the knee
        self.assertEqual(ws["levels"][0], 8)

    def test_slots_override(self):
        pl = self.plan(probe("discrete", [96], 128), model="qwen3.8-27b", target_context=65536, slots=4)
        self.assertEqual(pl["selections"][0]["warm"]["slots"], 4)
        self.assertEqual(pl["selections"][0]["estimate"]["kv_bytes"], 4 * 65536 * 64 * 1024)


def client(i, cached, uncached, ttft50, ttft95, ok=True):
    return {"client": i, "exit_code": 0 if ok else 1, "wall_s": 100.0,
            "summary": {"present": True, "success": ok, "output_tokens_per_second": 10.0,
                        "totals": {"total_local_output_tokens": 1000, "failed_turns": 0,
                                   "total_server_cached_prompt_tokens": cached, "total_server_uncached_prompt_tokens": uncached},
                        "latency_ms": {"time_to_first_token": {"p50": ttft50, "p95": ttft95}}}}


class WarmAggregateTests(unittest.TestCase):
    def test_hit_ratio_ttft_tasks(self):
        a = aggregate_warm([client(0, 900, 100, 500, 900), client(1, 700, 300, 700, 2000)], 2, "agentperf-default-v1", 120.0)
        self.assertEqual(a["clients_ok"], 2)
        self.assertAlmostEqual(a["prompt_cache_hit_ratio"], 0.8)
        self.assertEqual(a["ttft_p50_ms"], 700)
        self.assertEqual(a["ttft_p95_max_ms"], 2000)
        self.assertAlmostEqual(a["tasks_per_minute"], 8.0)

    def test_report_row(self):
        r = {"test": "warm_contexts", "host": "h", "status": "ok", "selection": {"model_id": "m", "quant": "q", "context_tokens": 65536},
             "warm": {"slots": 8, "cache_ram_bytes": 80 * GiB, "knee_agents": 15}, "knee_measured_agents": 16, "max_sustained_agents": 32,
             "levels": [{"n": 8, "aggregate": {"prompt_cache_hit_ratio": 0.95, "ttft_p50_ms": 800, "tasks_per_minute": 4.0}}],
             "memory": {"peak_host_used_bytes": 70 * GiB, "peak_cpu_util": 0.12}}
        out = render([r])
        self.assertIn("### warm_contexts", out)
        self.assertIn("n8: hit 0.95, ttft 0.8s, 4.00 tasks/min", out)
        self.assertIn("0.12", out)


if __name__ == "__main__":
    unittest.main()


class ReplaySubsetTest(unittest.TestCase):
    def test_bundled_ids_pass_through(self):
        rp = resolve_replay("aa-mini-v1", "/nonexistent", log=lambda *a: None)
        self.assertEqual(rp["args"], ["--replay", "aa-mini-v1"])
        self.assertEqual(rp["tasks"], 1)

    def test_unknown_replay_is_refused(self):
        from bench.runner import Refused
        with self.assertRaises(Refused):
            resolve_replay("nope-v9", "/nonexistent", log=lambda *a: None)

    @unittest.skipUnless(shutil.which("agentperf-local") and bundled_manifest(), "agentperf-local not installed")
    def test_subset_is_built_beside_its_traces(self):
        with tempfile.TemporaryDirectory() as d:
            rp = resolve_replay("warm-long-v1", d, log=lambda *a: None)
            self.assertEqual(rp["tasks"], len(SUBSETS["warm-long-v1"]))
            self.assertEqual(rp["args"], [os.path.join(d, "replays", "warm-long-v1", "manifest.json")])
            m = json.load(open(rp["args"][0]))
            self.assertEqual([t["task_id"] for t in m["tasks"]], rp["task_ids"])
            for t in m["tasks"]:
                self.assertTrue(os.path.exists(os.path.join(d, "replays", "warm-long-v1", t["trace"])))
                self.assertEqual(t["required_context_tokens"], 65536)
