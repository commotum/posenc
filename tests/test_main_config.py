from __future__ import annotations

import unittest

from core.cli import merged_defaults


class MainConfigTests(unittest.TestCase):
    def test_supports_encoder_list(self) -> None:
        defaults = merged_defaults({
            "encoders": ["monster"],
            "dim": 120,
            "coords": ["t", "x", "y", "z"],
            "t_values": [-1, 0, 1],
        })
        self.assertEqual(defaults["encoders"], ["monster"])
        self.assertEqual(defaults["coords"], "t,x,y,z")
        self.assertEqual(defaults["t_values"], "-1,0,1")

    def test_supports_encoder_override_object(self) -> None:
        defaults = merged_defaults(
            {
                "common": {
                    "encoders": ["all"],
                    "num_directions": None,
                },
                "encoders": {
                    "names": ["spiral", "monster"],
                    "spiral": {"num_directions": 4},
                    "monster": {"top_delta": 2048},
                },
            }
        )
        self.assertEqual(defaults["encoders"], ["spiral", "monster"])
        self.assertEqual(defaults["num_directions"], 4)
        self.assertEqual(defaults["top_delta"], 2048)


if __name__ == "__main__":
    unittest.main()
