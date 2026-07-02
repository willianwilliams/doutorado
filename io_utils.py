from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from config import Config

PROJECT_DIR = Path(__file__).resolve().parent


def load_or_create_jamming(cfg: Config) -> np.ndarray:
    n = int(round(cfg.t_final / cfg.h))
    shape = (n + 1, 35)

    candidates: list[Path] = []
    if cfg.jam_path is not None:
        jam_path = Path(cfg.jam_path)
        if jam_path.is_absolute():
            candidates.append(jam_path)
        else:
            candidates.append(PROJECT_DIR / jam_path)
            candidates.append(jam_path)
    candidates.append(PROJECT_DIR / "JAM.mat")
    candidates.append(cfg.output_dir / "JAM.mat")

    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            data = loadmat(path)
            if "b_noise" in data:
                b_noise = np.asarray(data["b_noise"], dtype=np.float64)
                if b_noise.shape == shape:
                    print(f"Loaded jamming/noise from {path}")
                    return np.ascontiguousarray(b_noise)
                print(f"Ignoring {path}: expected shape {shape}, found {b_noise.shape}")

    rng = np.random.default_rng(cfg.rng_seed)
    b_noise = cfg.noise_scale * rng.standard_normal(shape)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / "JAM.mat"
    savemat(out_path, {"b_noise": b_noise, "meta": {"generator": "numpy.default_rng.standard_normal", "seed": cfg.rng_seed}})
    print(f"Generated Python jamming/noise in {out_path}")
    return np.ascontiguousarray(b_noise)


def fmt(value: float, pattern: str = ".6f") -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return format(float(value), pattern)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
