from __future__ import annotations

import unittest

from core.positions import parse_coords, parse_t_values
from core.types import ExperimentConfig
from experiment import enforce_requested_requirements, requirement_report, run_experiment


class ExperimentSmokeTests(unittest.TestCase):
    def test_all_encoders_smoke(self) -> None:
        cfg = ExperimentConfig(
            encoders=("axial", "spiral", "monster", "f-monster", "ape"),
            dim=120,
            num_vectors=2,
            seed=7,
            theta_base=10_000.0,
            coords_spec=parse_coords("t,x,y"),
            num_directions=3,
            top_delta=1024.0,
            span=2.0 * 3.141592653589793,
            grid_size=3,
            centered_coords=False,
            t_values=parse_t_values("-1,0,1"),
            z_value=0.0,
            position_chunk_size=5,
            save_dir=None,
            save_encoded=False,
            raw_coords="t,x,y",
        )

        checks = requirement_report(cfg)
        enforce_requested_requirements(cfg.encoders, checks)

        artifacts = run_experiment(cfg)
        summary = artifacts.summary

        self.assertEqual(summary["encoders"], ["axial", "spiral", "monster", "f-monster", "ape"])
        self.assertEqual(summary["positions"]["rope_positions"], 27)
        self.assertEqual(summary["positions"]["monster_positions"], 27)

        for name, tensor in artifacts.encoded.items():
            self.assertEqual(tensor.shape, (2, 27, 120), msg=f"Unexpected shape for {name}")
            self.assertEqual(str(tensor.dtype), "float64", msg=f"Unexpected dtype for {name}")

        verification = summary["verification"]
        self.assertLess(verification["axial"]["max_abs_euclidean_norm_error"], 1e-10)
        self.assertLess(verification["spiral"]["max_abs_euclidean_norm_error"], 1e-10)
        self.assertLess(verification["monster"]["max_abs_minkowski_norm_error"], 1e-10)
        self.assertLess(verification["f-monster"]["max_abs_minkowski_norm_error"], 1e-10)
        self.assertLess(verification["ape"]["max_abs_broadcast_consistency_error"], 1e-10)
        self.assertGreater(verification["ape"]["mean_pe_norm"], 0.0)


if __name__ == "__main__":
    unittest.main()
