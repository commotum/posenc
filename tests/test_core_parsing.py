from __future__ import annotations

import unittest

from core.positions import parse_coords, parse_t_values


class ParseHelpersTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
