"""Admission-check tests for bench run. Run: python3 -m unittest discover -s tests"""
import copy
import unittest

from bench.catalog import load_catalog
from bench.planner import build_plan
from bench.runner import Refused, admit, llama_server_argv
from test_planner import GiB, probe


class AdmitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()
        cls.p = probe("unified", [128], 128, engines={"llama.cpp": "b1"})
        cls.plan = build_plan(cls.p, copy.deepcopy(cls.cat), "inference", None, {"model": "qwen3.8-27b", "target_context": 65536})

    def test_same_machine_admits(self):
        self.assertEqual(admit(self.plan, self.p), [])

    def test_device_count_change_refuses(self):
        q = probe("unified", [128, 128], 128)
        with self.assertRaises(Refused):
            admit(self.plan, q)

    def test_memory_drift_refuses(self):
        q = probe("unified", [96], 96)
        with self.assertRaises(Refused):
            admit(self.plan, q)

    def test_small_drift_admits(self):
        q = copy.deepcopy(self.p)
        q["devices"][0]["memory_total_bytes"] = int(128 * GiB * 0.99)
        q["memory"]["system_total_bytes"] = int(128 * GiB * 0.99)
        admit(self.plan, q)

    def test_busy_refuses(self):
        q = copy.deepcopy(self.p)
        q["memory"]["system_available_bytes"] = 8 * GiB
        with self.assertRaises(Refused):
            admit(self.plan, q)

    def test_discrete_gpu_busy_refuses(self):
        p = probe("discrete", [96], 92, engines={"llama.cpp": "b1"})
        plan = build_plan(p, copy.deepcopy(self.cat), "inference", None, {"model": "qwen3.8-27b", "target_context": 65536})
        p["devices"][0]["memory_free_bytes"] = 4 * GiB
        with self.assertRaises(Refused):
            admit(plan, p)

    def test_argv_mirrors_aa(self):
        sel = self.plan["selections"][0]
        argv = llama_server_argv(sel, "/m.gguf", "alias", 8089, [])
        for flag in ("--ctx-size", "--parallel", "--gpu-layers", "--jinja", "--flash-attn", "--no-webui"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[argv.index("--ctx-size") + 1], "65536")


if __name__ == "__main__":
    unittest.main()


class PlacementArgvTests(unittest.TestCase):
    def test_rpc_split_adds_rpc_and_tensor_split(self):
        from bench.runner import llama_server_argv, server_env
        sel = {"context_tokens": 8192, "n_seq": 1, "placement": {"strategy": "rpc_split", "tensor_split": [0.6, 0.4]}}
        argv = llama_server_argv(sel, "/m.gguf", "a", 8089, [], rpc=["10.0.0.2:18052"])
        self.assertIn("--rpc", argv)
        self.assertEqual(argv[argv.index("--rpc") + 1], "10.0.0.2:18052")
        self.assertEqual(argv[argv.index("--tensor-split") + 1], "0.6,0.4")
        self.assertNotIn("GGML_CUDA_ENABLE_UNIFIED_MEMORY", server_env(sel))

    def test_single_has_no_rpc_even_with_endpoints(self):
        from bench.runner import llama_server_argv
        sel = {"context_tokens": 8192, "n_seq": 1, "placement": {"strategy": "single"}}
        self.assertNotIn("--rpc", llama_server_argv(sel, "/m.gguf", "a", 8089, [], rpc=["x:1"]))

    def test_coherent_spill_sets_unified_memory_env(self):
        from bench.runner import server_env
        sel = {"placement": {"strategy": "coherent_spill"}}
        self.assertEqual(server_env(sel).get("GGML_CUDA_ENABLE_UNIFIED_MEMORY"), "1")


class OutputPolicyTests(unittest.TestCase):
    def test_llama_keeps_exact_default(self):
        from bench.runner import output_policy_args
        self.assertEqual(output_policy_args("llama.cpp", None), [])

    def test_other_engines_default_to_recorded(self):
        from bench.runner import output_policy_args
        for eng in ("vllm", "mlx", "tensorfold"):
            self.assertEqual(output_policy_args(eng, ["--x"]), ["--output-token-policy", "recorded", "--x"])

    def test_explicit_policy_wins(self):
        from bench.runner import output_policy_args
        self.assertEqual(output_policy_args("mlx", ["--output-token-policy", "fixed"]), ["--output-token-policy", "fixed"])


class FreePortTests(unittest.TestCase):
    def test_skips_a_bound_port(self):
        import socket
        from bench.runner import free_port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0)); s.listen(1)
            taken = s.getsockname()[1]
            self.assertNotEqual(free_port(taken), taken)


class AdmitCoherentTests(unittest.TestCase):
    def test_spill_counts_only_device_bytes(self):
        """A coherent_spill plan whose total exceeds free HBM is admitted when the device part fits."""
        p = probe("discrete", [128], 256, engines={"llama.cpp": "b1"})  # device branch of the free-memory check
        cat = load_catalog()
        plan = build_plan(p, copy.deepcopy(cat), "inference", None, {"model": "qwen3.8-27b", "target_context": 65536})
        sel = plan["selections"][0]
        sel["estimate"]["total_bytes"] = 150 * GiB
        sel["placement"] = dict(sel["placement"], strategy="coherent_spill", host_bytes=100 * GiB)
        q = copy.deepcopy(p)
        q["devices"][0]["memory_free_bytes"] = 60 * GiB
        self.assertEqual([n for n in admit(plan, q) if "refused" in n.lower()], [])
        q["devices"][0]["memory_free_bytes"] = 40 * GiB
        with self.assertRaises(Refused):
            admit(plan, q)
