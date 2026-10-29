"""Planner tests on synthetic probes. Run: python3 -m unittest discover -s tests"""
import copy
import json
import os
import unittest

from bench.catalog import load_catalog
from bench.planner import build_plan

GiB = 1024 ** 3


def probe(topology, devices_gib, system_gib, backend="cuda", engines=None):
    devs = []
    for i, g in enumerate(devices_gib):
        devs.append({
            "index": i, "vendor": "x", "name": f"dev{i}", "backend": backend, "integrated": topology == "unified",
            "memory_total_bytes": g * GiB, "memory_free_bytes": None, "memory_method": "test",
            "compute_capability": None, "driver_version": None, "driver_api_version": None,
            "memory_clock_khz": None, "memory_bus_width_bits": None, "bandwidth_gbps_measured": None,
        })
    return {
        "schema_version": "1.0", "probe_version": "t", "timestamp": "t", "probe_id": "p",
        "host": {"hostname": "t", "os": "linux", "os_version": None, "kernel": None, "arch": "x", "cpu": {"model": None, "cores": 1, "threads": 1}},
        "memory": {"topology": topology, "system_total_bytes": system_gib * GiB, "system_available_bytes": int(system_gib * 0.93) * GiB, "swap_total_bytes": 0, "huge_pages": False},
        "devices": devs,
        "interconnect": {"intra_node": []},
        "storage": {"model_cache_path": "/nonexistent", "free_bytes": 10 ** 13, "read_gbps_measured": None, "device_type": "unknown"},
        "software": {"python": "3", "cuda": None, "rocm": None, "metal": None, "docker": False, "engines": engines or {}},
        "cluster": {"node_id": "t", "nodes": ["t"], "links": []},
        "warnings": [],
    }


class _PlannerBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = load_catalog()

    def plan(self, p, test, **kw):
        preset = kw.pop("preset", None)
        return build_plan(p, copy.deepcopy(self.cat), test, preset, kw)



class PlannerTests(_PlannerBase):
    def test_ladder_picks_bigger_on_bigger_unified(self):
        small = self.plan(probe("unified", [128], 128), "ladder")
        big = self.plan(probe("unified", [1024], 1024), "ladder")
        self.assertEqual(small["selections"][0]["status"], "fits")
        self.assertGreater(big["selections"][0]["estimate"]["weights_bytes"], small["selections"][0]["estimate"]["weights_bytes"])

    def test_unified_does_not_double_count(self):
        p = self.plan(probe("unified", [128], 128), "ladder")
        self.assertLessEqual(p["budget"]["gpu_bytes_total"], 128 * GiB)
        self.assertGreaterEqual(p["budget"]["host_bytes_after_plan"], 0)

    def test_unified_leaves_host_min(self):
        p = self.plan(probe("unified", [128], 128), "ladder")
        self.assertGreaterEqual(p["budget"]["host_bytes_after_plan"], p["policy"]["host_min_bytes"])

    def test_discrete_tensor_parallel(self):
        p = self.plan(probe("discrete", [32, 32], 128), "inference", model="qwen3.8-27b", quant="fp8")
        self.assertEqual(p["selections"][0]["placement"]["strategy"], "tensor_parallel")

    def test_discrete_no_offload_by_default(self):
        p = self.plan(probe("discrete", [32], 128), "inference", model="ling-3.0-flash")
        self.assertEqual(p["selections"], [])
        self.assertTrue(p["rejected"])
        self.assertGreater(p["rejected"][0]["shortfall_bytes"], 0)

    def test_discrete_offload_when_allowed(self):
        p = self.plan(probe("discrete", [32], 128), "inference", model="ling-3.0-flash", allow_offload=True)
        self.assertEqual(p["selections"][0]["status"], "fits_with_offload")
        self.assertGreater(p["selections"][0]["placement"]["host_bytes"], 0)

    def test_residency_evicts_without_substitution(self):
        p = self.plan(probe("unified", [128], 128), "residency", preset="residency-workstation")
        ids = [s["model_id"] for s in p["selections"]]
        self.assertEqual(ids, self.cat["presets"]["residency-workstation"]["models"])
        statuses = {s["status"] for s in p["selections"]}
        self.assertIn("evicted", statuses)
        big = self.plan(probe("unified", [192], 192), "residency", preset="residency-workstation")
        self.assertNotIn("evicted", {s["status"] for s in big["selections"]})

    def test_host_pressure_reports_max_n_seq(self):
        p = self.plan(probe("unified", [128], 128), "host_pressure", model="qwen3.8-27b", target_context=65536)
        self.assertGreaterEqual(p["selections"][0]["max_n_seq"], 1)

    def test_installed_engine_preferred(self):
        p = self.plan(probe("unified", [128], 128, engines={"llama.cpp": "b1"}), "inference", model="qwen3.8-27b")
        self.assertEqual(p["selections"][0]["engine"], "llama.cpp")

    def test_deterministic(self):
        a = self.plan(probe("unified", [128], 128), "ladder")
        b = self.plan(probe("unified", [128], 128), "ladder")
        self.assertEqual(a["plan_id"], b["plan_id"])

    def test_load_does_not_change_plan(self):
        quiet = probe("unified", [128], 128)
        busy = probe("unified", [128], 128)
        busy["memory"]["system_available_bytes"] = 40 * GiB
        a, b = self.plan(quiet, "ladder"), self.plan(busy, "ladder")
        self.assertEqual(a["selections"][0]["model_id"], b["selections"][0]["model_id"])
        self.assertTrue(any("busy" in w for w in b["warnings"]))
        self.assertFalse(any("busy" in w for w in a["warnings"]))

    def test_small_machine_scales_host_min(self):
        p = self.plan(probe("unified", [10], 16), "ladder")
        self.assertLessEqual(p["budget"]["host_min_bytes"], int(16 * GiB * 0.2))

    def test_overrides_mark_custom(self):
        self.assertFalse(self.plan(probe("unified", [128], 128), "ladder")["custom"])
        self.assertTrue(self.plan(probe("unified", [128], 128), "ladder", reserve_fraction=0.2)["custom"])


if __name__ == "__main__":
    unittest.main()


def coherent_probe(hbm_gib, ram_gib, dev_gbps=None, link_gbps=None, engines=None):
    p = probe("discrete", [hbm_gib], ram_gib, engines=engines or {"llama.cpp": "x"})
    p["memory"]["topology"] = "coherent"
    d = p["devices"][0]
    d.update(host_coherent=True, pageable_host_access=True, host_link="nvlink-c2c",
             bandwidth_gbps_measured=dev_gbps, host_to_device_gbps_measured=link_gbps)
    return p


def cluster_probe(nodes, engines=None):
    """nodes: list of (topology, [device GiB...], ram GiB)."""
    head = probe(nodes[0][0], nodes[0][1], nodes[0][2], engines=engines or {"llama.cpp": "x"})
    devices, node_probes = [], []
    for i, (topo, devs, ram) in enumerate(nodes):
        node_probes.append({"hostname": f"n{i}", "topology": topo, "system_total_bytes": ram * GiB,
                            "system_available_bytes": ram * GiB, "ssh": None if i == 0 else f"n{i}", "addr": None if i == 0 else f"10.0.0.{i}",
                            "rpc_port": None if i == 0 else 50052, "engines": {"llama.cpp": "x"}})
        for g in devs:
            devices.append(dict(probe(topo, [g], ram)["devices"][0], index=len(devices), node=i, node_index=0, node_hostname=f"n{i}",
                                node_topology=topo, node_system_total_bytes=ram * GiB))
    head["memory"]["topology"] = "cluster"
    head["devices"] = devices
    head["cluster"] = {"name": "t", "node_id": "n0", "nodes": [n["hostname"] for n in node_probes], "node_probes": node_probes,
                       "links": [], "workers": [{"ssh": n["ssh"], "addr": n["addr"], "rpc_port": 18052} for n in node_probes[1:]]}
    return head


class CoherentTests(_PlannerBase):
    def test_station_spills_weights_to_host_tier(self):
        # GB300-class station: 252 GiB HBM, 480 GiB LPDDR. DeepSeek-class Q4 (unverified, ~230+ GiB) will not fit HBM alone.
        p = self.plan(coherent_probe(252, 480, dev_gbps=8000, link_gbps=900), "ladder")
        s = p["selections"][0]
        self.assertIn(s["status"], ("fits", "fits_coherent"))
        if s["status"] == "fits_coherent":
            pl = s["placement"]
            self.assertEqual(pl["strategy"], "coherent_spill")
            self.assertGreater(pl["host_bytes"], 0)
            self.assertLess(pl["decode_speed_relative_est"], 1.0)
            self.assertTrue(any("coherent host RAM" in w for w in p["warnings"]))

    def test_coherent_budget_has_host_tier(self):
        p = self.plan(coherent_probe(252, 480), "ladder")
        self.assertEqual(p["budget"]["topology"], "coherent")
        self.assertGreater(p["budget"]["host_coherent_bytes"], 300 * GiB)

    def test_spill_only_when_hbm_is_short(self):
        p = self.plan(coherent_probe(252, 480), "inference", model="qwen3.8-27b", quant="q4_k_m", target_context=65536)
        self.assertEqual(p["selections"][0]["placement"]["strategy"], "single")

    def test_small_hbm_large_ram_spills(self):
        p = self.plan(coherent_probe(16, 480), "inference", model="qwen3.8-27b", quant="q4_k_m", target_context=65536)
        self.assertEqual(p["selections"][0]["status"], "fits_coherent")
        self.assertGreater(p["selections"][0]["placement"]["spill_fraction"], 0)

    def test_discrete_never_spills(self):
        p = self.plan(probe("discrete", [16], 480, engines={"llama.cpp": "x"}), "inference", model="qwen3.8-27b", quant="q4_k_m", target_context=65536)
        self.assertEqual(p["selections"], [])

    def test_residency_uses_host_tier(self):
        p = self.plan(coherent_probe(96, 480), "residency", preset="residency-text-workstation")
        statuses = [s["status"] for s in p["selections"]]
        self.assertIn("resident_coherent", statuses)
        self.assertNotIn("evicted", statuses)
        # 64 GiB of HBM leaves no room for Ling's KV and activations, which must stay in device memory: evicted, not spilled
        p = self.plan(coherent_probe(64, 480), "residency", preset="residency-text-workstation")
        self.assertIn("evicted", [s["status"] for s in p["selections"]])


class ClusterTests(_PlannerBase):
    def test_two_sparks_split_a_model_neither_holds(self):
        p = self.plan(cluster_probe([("unified", [128], 128), ("unified", [128], 128)]), "ladder")
        s = p["selections"][0]
        self.assertEqual(p["budget"]["topology"], "cluster")
        self.assertEqual(len(p["budget"]["nodes"]), 2)
        single = self.plan(probe("unified", [128], 128, engines={"llama.cpp": "x"}), "ladder")["selections"][0]
        self.assertGreaterEqual(s["estimate"]["weights_bytes"], single["estimate"]["weights_bytes"])

    def test_rpc_split_shares_follow_room(self):
        # Ling (75 GiB) fits neither a 64 GB nor a 48 GB Spark alone; together they hold it.
        p = self.plan(cluster_probe([("unified", [64], 64), ("unified", [48], 48)]), "inference",
                      model="ling-3.0-flash", quant="q4_k_m", target_context=8192)
        s = p["selections"][0]
        self.assertEqual(s["status"], "fits_cluster")
        self.assertEqual(s["placement"]["strategy"], "rpc_split")
        self.assertAlmostEqual(sum(s["placement"]["tensor_split"]), 1.0, places=2)
        self.assertGreater(s["placement"]["tensor_split"][0], s["placement"]["tensor_split"][1])

    def test_residency_spreads_models_across_nodes(self):
        p = self.plan(cluster_probe([("unified", [128], 128), ("unified", [128], 128)]), "residency", preset="residency-text-workstation")
        res = [s for s in p["selections"] if s["status"] == "resident"]
        self.assertEqual(len(res), 4)
        self.assertEqual(len({s["placement"]["devices"][0] for s in res}), 2)
        self.assertTrue(all(s["placement"]["strategy"] == "single" for s in res))

    def test_forced_rpc_split(self):
        p = self.plan(cluster_probe([("unified", [128], 128), ("unified", [128], 128)]), "inference", model="qwen3.8-27b", quant="q4_k_m", target_context=65536, placement="rpc_split")
        self.assertEqual(p["selections"][0]["placement"]["strategy"], "rpc_split")

    def test_single_node_fits_stay_single(self):
        p = self.plan(cluster_probe([("unified", [128], 128), ("unified", [128], 128)]), "inference", model="qwen3.8-27b", quant="q4_k_m", target_context=65536)
        self.assertEqual(p["selections"][0]["placement"]["strategy"], "single")


class EngineTrackTests(_PlannerBase):
    def test_best_prefers_vllm_on_cuda_when_installed_and_file_exists(self):
        p = self.plan(probe("discrete", [96], 92, engines={"llama.cpp": "x", "vllm": "y"}), "inference", model="qwen3.8-27b", engine="best")
        # qwen3.8-27b has fp8 files for vllm in the catalog
        self.assertEqual(p["selections"][0]["engine"], "vllm")
        self.assertTrue(any("engine=best resolved to vllm" in w for w in p["warnings"]))

    def test_best_falls_back_to_llama_cpp_when_nothing_faster_installed(self):
        p = self.plan(probe("discrete", [96], 92, engines={"llama.cpp": "x"}), "inference", model="qwen3.8-27b", engine="best")
        self.assertEqual(p["selections"][0]["engine"], "llama.cpp")

    def test_default_is_baseline_llama_cpp_even_with_vllm_installed(self):
        p = self.plan(probe("discrete", [96], 92, engines={"llama.cpp": "x", "vllm": "y"}), "inference", model="qwen3.8-27b")
        self.assertEqual(p["selections"][0]["engine"], "llama.cpp")

    def test_catalog_lists_tensorfold_for_metal(self):
        self.assertIn("tensorfold", self.cat["engines"])
        self.assertEqual(self.cat["engine_preference"]["metal"][0], "tensorfold")


class ForceTests(_PlannerBase):
    def test_force_emits_selection_over_budget(self):
        p = self.plan(probe("discrete", [32], 128), "inference", model="ling-3.0-flash", target_context=65536, force=True)
        self.assertEqual(len(p["selections"]), 1)
        s = p["selections"][0]
        self.assertEqual(s["status"], "forced")
        self.assertEqual(s["context_tokens"], 65536)
        self.assertLess(s["estimate"]["fit_margin_bytes"], 0)
        self.assertTrue(any(w.startswith("FORCED") for w in p["warnings"]))
        self.assertTrue(p["rejected"])  # the honest rejection is kept beside the forced selection

    def test_force_without_context_changes_nothing(self):
        p = self.plan(probe("discrete", [32], 128), "inference", model="ling-3.0-flash", force=True)
        self.assertEqual(p["selections"], [])


class CalibrationTests(_PlannerBase):
    def test_moe_activations_scale_with_active_params(self):
        from bench.planner import cost
        eng = self.cat["engines"]["llama.cpp"]
        dense = {"modality": "text", "params_total": 100, "params_active": 100, "activation_overhead_fraction": 0.05, "kv_bytes_per_token": 0}
        moe = dict(dense, params_active=5)
        f = {"bytes": 100 * 1024 ** 3, "quant": "q4"}
        self.assertEqual(cost(dense, f, eng, 8192, 1, "fp16", ("vulkan",))["activation_bytes"], int(100 * 1024 ** 3 * 0.05))
        self.assertEqual(cost(moe, f, eng, 8192, 1, "fp16", ("vulkan",))["activation_bytes"], int(100 * 1024 ** 3 * 0.05 * 0.1))

    def test_mla_kv_is_single_latent(self):
        ds = next(m for m in self.cat["models"] if m["id"] == "deepseek-v4-flash")
        self.assertEqual(ds["kv_bytes_per_token"], 43 * 1 * 576 * 2)
        q = next(m for m in self.cat["models"] if m["id"] == "qwen3.8-27b")
        self.assertEqual(q["kv_bytes_per_token"], 2 * 16 * 4 * 256 * 2)
