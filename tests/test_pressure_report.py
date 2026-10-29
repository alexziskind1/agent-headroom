"""Tests for host_pressure helpers and the report renderer."""
import unittest

from bench.pressure import aggregate, default_levels
from bench.report import render


class PressureTests(unittest.TestCase):
    def test_default_levels_double_then_cap(self):
        self.assertEqual(default_levels(1), [1])
        self.assertEqual(default_levels(8), [1, 2, 4, 8])
        self.assertEqual(default_levels(22), [1, 2, 4, 8, 16, 22])

    def test_aggregate_counts_failures(self):
        ok = {"exit_code": 0, "wall_s": 100.0, "summary": {"present": True, "success": True, "output_tokens_per_second": 10.0,
                                                           "totals": {"total_local_output_tokens": 1000, "failed_turns": 0}}}
        bad = {"exit_code": 1, "wall_s": 50.0, "summary": {"present": True, "success": False, "output_tokens_per_second": None,
                                                           "totals": {"total_local_output_tokens": 100, "failed_turns": 2}}}
        a = aggregate([ok, ok, bad])
        self.assertEqual(a["clients_ok"], 2)
        self.assertEqual(a["clients_failed"], 1)
        self.assertEqual(a["failed_turns"], 2)
        self.assertEqual(a["aggregate_output_tokens_per_second"], 20.0)
        self.assertEqual(a["per_client_output_tps_p50"], 10.0)


class ReportTests(unittest.TestCase):
    def test_render_each_test(self):
        inf = {"test": "inference", "host": "h", "status": "ok", "selection": {"model_id": "m", "quant": "q", "context_tokens": 4096},
               "replay": {"name": "r", "tool_mode": "none"}, "agentperf": {"output_tokens_per_second": 12.5, "measured_duration_ms": 60000,
               "totals": {"successful_turns": 6, "failed_turns": 0}}, "memory": {"peak_host_used_bytes": 1 << 30}, "artifact": {"verified": True}}
        pr = {"test": "host_pressure", "host": "h", "status": "ok", "selection": {"model_id": "m", "quant": "q", "context_tokens": 4096},
              "replay": {"name": "r", "tool_mode": "fixed_delay"}, "max_sustained_agents": 2,
              "levels": [{"n": 1, "aggregate": {"aggregate_output_tokens_per_second": 10}}, {"n": 2, "aggregate": {"aggregate_output_tokens_per_second": 18}}],
              "memory": {"peak_host_used_bytes": 2 << 30}}
        rs = {"test": "residency", "host": "h", "status": "ok", "preset": "p", "resident": ["a", "b"], "evicted": ["c"], "tasks_per_minute": 1.5,
              "swap_s_total": 42.0, "memory": {"peak_host_used_bytes": 3 << 30}}
        out = render([inf, pr, rs])
        for head in ("### inference", "### host_pressure", "### residency"):
            self.assertIn(head, out)
        self.assertIn("n1=10.0, n2=18.0", out)
        self.assertIn("| h | p | llama.cpp | 2 | 0 | 1 | 1.5 | 42 |", out)

    def test_render_empty(self):
        self.assertEqual(render([]), "no results found")


if __name__ == "__main__":
    unittest.main()
