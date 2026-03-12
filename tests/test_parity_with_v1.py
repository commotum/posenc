from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class ParityWithV1Tests(unittest.TestCase):
    def _run_json(self, script: str, args: list[str]) -> dict[str, object]:
        output = subprocess.check_output(
            [sys.executable, str(ROOT / script), *args],
            cwd=ROOT,
            text=True,
        )
        json_start = output.find("{")
        self.assertGreaterEqual(json_start, 0, "Expected JSON payload in script output")
        return json.loads(output[json_start:])

    def test_refactor_matches_original_summary(self) -> None:
        args = [
            "--encoders",
            "axial",
            "spiral",
            "--dim",
            "120",
            "--num-vectors",
            "2",
            "--coords",
            "t,x,y",
            "--t-values=-1,0,1",
            "--grid-size",
            "3",
            "--position-chunk-size",
            "5",
        ]

        old = self._run_json("Positional-Encoding-V1.py", args)
        new = self._run_json("main.py", args)

        old.pop("timing_seconds")
        new.pop("timing_seconds")
        new["config"].pop("span", None)
        self.assertEqual(new, old)


if __name__ == "__main__":
    unittest.main()
