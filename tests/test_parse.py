from __future__ import annotations

import unittest

from core.cli import merged_defaults
from core.positions import parse_coords, parse_t_values


class ParseTests(unittest.TestCase):
    def test_parse_coords_valid(self) -> None:
        spec = parse_coords("t,x,y")
        self.assertEqual(spec.coords, ("t", "x", "y"))
        self.assertTrue(spec.include_time)
        self.assertEqual(spec.spatial_axes, ("x", "y"))

    def test_parse_coords_rejects_duplicate(self) -> None:
        with self.assertRaises(ValueError):
            parse_coords("x,x")

    def test_parse_t_values_from_string(self) -> None:
        values = parse_t_values("-1,0,2.5")
        self.assertEqual(values.tolist(), [-1.0, 0.0, 2.5])

    def test_parse_t_values_from_list(self) -> None:
        values = parse_t_values([-2, 0, 2])
        self.assertEqual(values.tolist(), [-2.0, 0.0, 2.0])

    def test_merged_defaults_simple_schema(self) -> None:
        defaults = merged_defaults(
            {
                "encoders": ["spiral", "monster"],
                "coords": ["t", "x", "y"],
                "t_values": [-1, 0, 1],
                "encoder_params": {
                    "spiral": {"num_directions": 3},
                    "monster": {"top_delta": 2048, "span": 6.283185307179586},
                },
            }
        )
        self.assertEqual(defaults["encoders"], ["spiral", "monster"])
        self.assertEqual(defaults["coords"], "t,x,y")
        self.assertEqual(defaults["t_values"], "-1,0,1")
        self.assertEqual(defaults["encoder_params"]["spiral"]["num_directions"], 3)
        self.assertEqual(defaults["encoder_params"]["monster"]["top_delta"], 2048)

    def test_merged_defaults_rejects_unknown_keys(self) -> None:
        with self.assertRaises(ValueError):
            merged_defaults({"common": {"dim": 120}})


if __name__ == "__main__":
    unittest.main()
