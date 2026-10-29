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
