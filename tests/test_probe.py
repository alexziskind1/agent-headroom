import unittest

from bench import probe


class LlamaServerBackendTest(unittest.TestCase):
    def test_rocm_build_reports_rocm(self):
        from unittest import mock
        with mock.patch.object(probe, "_run", return_value="Available devices:\n  ROCm0: AMD Radeon Graphics (163840 MiB, 163680 MiB free)"):
            self.assertEqual(probe.llama_server_backend(), "rocm")

    def test_vulkan_build_reports_vulkan(self):
        from unittest import mock
        with mock.patch.object(probe, "_run", return_value="Available devices:\n  Vulkan0: AMD Radeon Graphics (RADV GFX1151) (163840 MiB, 163680 MiB free)"):
            self.assertEqual(probe.llama_server_backend(), "vulkan")

    def test_missing_binary_reports_none(self):
        from unittest import mock
        with mock.patch.object(probe, "_run", return_value=None):
            self.assertIsNone(probe.llama_server_backend())
