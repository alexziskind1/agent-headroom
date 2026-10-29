"""Residency servers must size engine launches to their own share of the device, not the whole GPU."""
import unittest

from bench import engines
from bench.catalog import load_catalog
from bench.residency import Server

GiB = 2**30


class ResidencyServerTests(unittest.TestCase):
    def setUp(self):
        self.cat = load_catalog()
        self.sel = {"model_id": "qwen3.8-27b", "quant": "fp8", "engine": "vllm", "status": "resident", "modality": "text",
                    "context_tokens": 32768, "n_seq": 1, "estimate": {"total_bytes": 34 * GiB}, "placement": {"strategy": "single", "devices": [0]}}
        self.path = "/home/user/.cache/bench/models/qwen3.8-27b/fp8"

    def test_vllm_share_is_passed_through(self):
        srv = Server(self.sel, self.path, 8089, "/tmp", [], cat=self.cat, device_total_bytes=120 * GiB)
        spec = engines.launch_spec("vllm", self.sel, self.path, srv.alias, 8089, [], cat=self.cat,
                                   device_total_bytes=srv.device_total_bytes, tool_parser="hermes")
        argv = spec["argv"]
        util = float(argv[argv.index("--gpu-memory-utilization") + 1])
        self.assertAlmostEqual(util, (34 * 1.1 + 2) / 120, places=2)
        self.assertLess(util, 0.5)

    def test_without_device_total_no_util_flag(self):
        spec = engines.launch_spec("vllm", self.sel, self.path, "a", 8089, [], cat=self.cat, device_total_bytes=None)
        self.assertNotIn("--gpu-memory-utilization", spec["argv"])


if __name__ == "__main__":
    unittest.main()


class PickVictimsTests(unittest.TestCase):
    """Eviction frees enough room for the incoming model, largest resident first (azbeast2, one 96 GB GPU:
    evicting only the 35b-a3b left 30 GB + 75 GB Ling > 96 GB and Ling OOMed on load)."""
    RES = {"qwen3.8-27b": 20 * GiB, "qwen3.6-35b-a3b": 24 * GiB, "qwen3.5-9b": 7 * GiB}

    def test_one_victim_when_it_fits(self):
        from bench.residency import pick_victims
        # 103.6 GiB budget (GB10): free 52.6 + 24 = 76.6 >= 75 after one victim
        self.assertEqual(pick_victims(self.RES, 75 * GiB, int(103.6 * GiB)), ["qwen3.6-35b-a3b"])

    def test_two_victims_on_a_96gb_device(self):
        from bench.residency import pick_victims
        # free 45 + 24 = 69 < 75, + 20 = 89 >= 75
        self.assertEqual(pick_victims(self.RES, 75 * GiB, 96 * GiB), ["qwen3.6-35b-a3b", "qwen3.8-27b"])

    def test_no_capacity_falls_back_to_largest(self):
        from bench.residency import pick_victims
        self.assertEqual(pick_victims(self.RES, 75 * GiB, None), ["qwen3.6-35b-a3b"])

    def test_nothing_resident(self):
        from bench.residency import pick_victims
        self.assertEqual(pick_victims({}, 75 * GiB, 96 * GiB), [])

    def test_everything_evicted_when_even_that_is_short(self):
        from bench.residency import pick_victims
        self.assertEqual(pick_victims(self.RES, 200 * GiB, 96 * GiB), ["qwen3.6-35b-a3b", "qwen3.8-27b", "qwen3.5-9b"])


class ReplayExtraTests(unittest.TestCase):
    def test_replay_once_forwards_device_arg(self):
        from unittest import mock
        from bench import residency
        sel = {"model_id": "qwen3.5-9b", "quant": "q4_k_m", "engine": "llama.cpp", "status": "resident", "modality": "text",
               "context_tokens": 32768, "n_seq": 1, "estimate": {"total_bytes": 7 * GiB}, "placement": {"strategy": "single", "devices": [0]}}
        srv = Server(sel, "/home/user/x.gguf", 8089, "/tmp", [])
        seen = {}
        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return mock.Mock(returncode=0)
        import tempfile, os
        with tempfile.TemporaryDirectory() as d, mock.patch.object(residency.subprocess, "run", fake_run), \
                mock.patch.object(residency, "ingest_summary", return_value={}):
            residency.replay_once(srv, os.path.join(d, "r"), "aa-mini-v1", "none", None, ["--device", "0"])
        self.assertIn("--device", seen["cmd"])
        self.assertEqual(seen["cmd"][seen["cmd"].index("--device") + 1], "0")
