from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def maybe_save(
    save_dir: Path | None,
    vectors: np.ndarray,
    metadata: dict[str, object],
    encoded: dict[str, np.ndarray],
    save_encoded: bool,
) -> None:
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "vectors.npy", vectors)
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if save_encoded:
        for name, tensor in encoded.items():
            np.save(save_dir / f"encoded_{name}.npy", tensor)
