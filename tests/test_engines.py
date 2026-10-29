import os
import unittest
from unittest import mock

from bench import engines


SEL = {"model_id": "qwen3.8-27b", "quant": "fp8", "engine": "vllm", "context_tokens": 65536, "n_seq": 1,
       "placement": {"strategy": "single", "devices": [0]}, "estimate": {"total_bytes": 40 * engines.GiB}}
CAT = {"engines": {"vllm": {"docker_image": "vllm/vllm-openai:v0.30.0-ubuntu2404"}}}


class ArtifactKindTests(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(engines.artifact_kind({"quant": "q4_k_m"}), "file")
        self.assertEqual(engines.artifact_kind({"quant": "ud-q4_k_xl"}), "file")
        self.assertEqual(engines.artifact_kind({"quant": "fp8"}), "dir")
        self.assertEqual(engines.artifact_kind({"quant": "mlx-4bit"}), "dir")
        self.assertEqual(engines.artifact_kind({"quant": "nvfp4"}), "dir")
        self.assertEqual(engines.artifact_kind({"quant": "q4_0", "kind": "file"}), "file")


class LaunchSpecTests(unittest.TestCase):
    def test_llama_matches_aa_flags(self):
        sel = dict(SEL, engine="llama.cpp", quant="q4_k_m")
        spec = engines.launch_spec("llama.cpp", sel, "/m/x.gguf", "a", 8089, [])
        a = spec["argv"]
        for flag in ("--jinja", "--flash-attn", "--batch-size", "--ubatch-size", "--no-webui"):
            self.assertIn(flag, a)
        self.assertEqual(a[a.index("--ctx-size") + 1], "65536")
        self.assertEqual(spec["model_name"], "a")
        self.assertIsNone(spec["cleanup"])

    def test_llama_coherent_spill_sets_unified_memory(self):
        sel = dict(SEL, engine="llama.cpp", quant="q4_k_m", placement={"strategy": "coherent_spill", "devices": [0], "spill_fraction": 0.3})
        spec = engines.launch_spec("llama.cpp", sel, "/m/x.gguf", "a", 8089, [])
        self.assertEqual(spec["env"].get("GGML_CUDA_ENABLE_UNIFIED_MEMORY"), "1")

    def test_vllm_docker_mode(self):
        with mock.patch("shutil.which", side_effect=lambda n: "/usr/bin/docker" if n == "docker" else None), \
             mock.patch.dict(os.environ, {"BENCH_MODEL_CACHE": "/cache/models"}, clear=False):
            spec = engines.launch_spec("vllm", SEL, "/cache/models/qwen3.8-27b/fp8", "qwen3.8-27b-fp8", 8089, [],
                                       cat=CAT, device_total_bytes=128 * engines.GiB, tool_parser="hermes")
        a = spec["argv"]
        self.assertEqual(a[:2], ["docker", "run"])
        self.assertIn("--gpus", a)
        self.assertIn("vllm/vllm-openai:v0.30.0-ubuntu2404", a)
        self.assertIn("/models/qwen3.8-27b/fp8", a)
        self.assertEqual(a[a.index("--max-model-len") + 1], "65536")
        self.assertEqual(a[a.index("--max-num-seqs") + 1], "1")
        self.assertEqual(a[a.index("--gpu-memory-utilization") + 1], "0.36")  # (40*1.1 + 2)/128
        self.assertEqual(a[a.index("--tool-call-parser") + 1], "hermes")
        self.assertEqual(spec["cleanup"], {"container": "bench-vllm-8089"})
        self.assertNotIn("--rm", a)
        self.assertEqual(spec["model_name"], "qwen3.8-27b-fp8")

    def test_vllm_native_mode(self):
        with mock.patch("shutil.which", side_effect=lambda n: "/usr/bin/vllm" if n == "vllm" else None):
            spec = engines.launch_spec("vllm", SEL, "/x/fp8", "a", 8089, ["--kv-cache-dtype", "fp8"], cat=CAT)
        self.assertEqual(spec["argv"][:3], ["vllm", "serve", "/x/fp8"])
        self.assertEqual(spec["argv"][-2:], ["--kv-cache-dtype", "fp8"])
        self.assertIsNone(spec["cleanup"])
        self.assertEqual(spec["env"]["HF_HUB_OFFLINE"], "1")

    def test_vllm_tensor_parallel(self):
        sel = dict(SEL, placement={"strategy": "tensor_parallel", "devices": [0, 1, 2, 3]})
        with mock.patch("shutil.which", side_effect=lambda n: "/usr/bin/vllm" if n == "vllm" else None):
            a = engines.launch_spec("vllm", sel, "/x/fp8", "a", 8089, [], cat=CAT)["argv"]
        self.assertEqual(a[a.index("--tensor-parallel-size") + 1], "4")

    def test_mlx_uses_path_as_model_name(self):
        sel = dict(SEL, engine="mlx", quant="mlx-4bit", n_seq=4)
        spec = engines.launch_spec("mlx", sel, "/c/qwen/mlx-4bit", "alias", 8090, [])
        a = spec["argv"]
        self.assertEqual(a[1:3], ["-m", "mlx_lm.server"])
        self.assertEqual(a[a.index("--decode-concurrency") + 1], "4")
        self.assertEqual(spec["model_name"], "/c/qwen/mlx-4bit")
        self.assertEqual(spec["health_url"], "http://127.0.0.1:8090/health")

    def test_tensorfold(self):
        sel = dict(SEL, engine="tensorfold", quant="mlx-4bit", n_seq=2)
        a = engines.launch_spec("tensorfold", sel, "/c/nemo/mlx-4bit", "nemo", 8091, [])["argv"]
        self.assertEqual(a[1:3], ["serve", "/c/nemo/mlx-4bit"])
        self.assertEqual(a[a.index("--parallel") + 1], "2")
        self.assertEqual(a[a.index("--context") + 1], "65536")
        self.assertEqual(a[a.index("--name") + 1], "nemo")


class FetchTests(unittest.TestCase):
    def test_dir_fetch_uses_hf_with_revision(self):
        cmds = engines.fetch_argv("m", {"quant": "fp8", "repo": "Org/M-FP8", "revision": "abc", "bytes": 1}, "/d")
        self.assertEqual(len(cmds), 1)
        self.assertIn("download", cmds[0])
        self.assertEqual(cmds[0][cmds[0].index("--revision") + 1], "abc")

    def test_gguf_fetch_curl_per_part(self):
        f = {"quant": "q4", "bytes": 2, "parts": [{"filename": "a-1.gguf", "url": "http://x/a-1.gguf", "bytes": 1},
                                                  {"filename": "a-2.gguf", "url": "http://x/a-2.gguf", "bytes": 1}]}
        cmds = engines.fetch_argv("m", f, "/d")
        self.assertEqual(len(cmds), 2)
        self.assertEqual(cmds[1][-1], "http://x/a-2.gguf")


if __name__ == "__main__":
    unittest.main()


class DockerMountTests(unittest.TestCase):
    def test_symlinked_checkpoint_binds_real_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            real = os.path.join(tmp, "elsewhere", "Qwen-FP8"); os.makedirs(real)
            root = os.path.join(tmp, "cache"); os.makedirs(os.path.join(root, "qwen3.8-27b"))
            link = os.path.join(root, "qwen3.8-27b", "fp8"); os.symlink(real, link)
            with mock.patch.dict(os.environ, {"BENCH_MODEL_CACHE": root}, clear=False):
                host_dir, inside = engines._docker_mount(link)
            self.assertEqual(host_dir, os.path.realpath(real))
            self.assertEqual(inside, "/models/qwen3.8-27b/fp8")
