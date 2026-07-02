from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Config:
    # Original numerical parameters
    freq: float = 60.0
    f: float = 60.0
    h1: float = 0.8
    t_final: float = 120.0 / 60.0
    h: float = 0.0025 / 60.0
    euler_internal_substeps: int = 10

    # Metric horizon and windows
    t_eval: float = 1.920
    final_window_delta: float = 0.100
    epsilon_st: float = 1.0
    epsilon_e5: float = 2.0
    table1_epsilon_st: float = 3.0
    table1_epsilon_e5: float = 3.0

    # Table settings
    k_nominal_proposed: float = 100.0
    k_values: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 1.0, 10.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 1000.0], dtype=np.float64)
    )
    k_reference: np.ndarray = field(default_factory=lambda: 10.0 * np.ones(6, dtype=np.float64))
    theta0_reference: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float64))

    # Message settings used by the secure-communication diagnostics.
    message_word: str = "Secure Transmission"
    n_bits_char: int = 7
    bit_rate: float = 69.0
    message_start_time: float = 0.0
    message_amplitude: float = 1.0
    ber_transient_cut: float = 0.4

    # Reproducible noise/jamming
    rng_seed: int = 1
    noise_scale: float = 2.0

    # sawtooth_style:
    #   0 = MATLAB-like sawtooth(t): range [-1,1], -1 at t=0
    #   1 = fallback formula used in the MATLAB scripts when sawtooth() is absent
    sawtooth_style: int = 0

    # Numerical guard
    abort_abs_state_limit: float = 1e12

    # Paths
    output_dir: Path = Path("results")
    jam_path: Path | None = Path("JAM.mat")

    @property
    def h_euler(self) -> float:
        return self.h / self.euler_internal_substeps

    @property
    def final_window_start(self) -> float:
        return self.t_eval - self.final_window_delta

    @property
    def message_bits(self) -> int:
        return len(self.message_word) * self.n_bits_char

    @property
    def message_duration(self) -> float:
        return self.message_bits / self.bit_rate

    @property
    def ber_windows(self) -> np.ndarray:
        cut = min(self.ber_transient_cut, self.t_eval)
        return np.array(
            [
                [0.0, cut],
                [cut, self.t_eval],
                [self.final_window_start, self.t_eval],
                [0.0, self.t_eval],
            ],
            dtype=np.float64,
        )

    @property
    def epsilon_by_state(self) -> np.ndarray:
        eps = self.epsilon_st * np.ones(6, dtype=np.float64)
        eps[4] = self.epsilon_e5
        return eps

    @property
    def table1_epsilon_by_state(self) -> np.ndarray:
        eps = self.table1_epsilon_st * np.ones(6, dtype=np.float64)
        eps[4] = self.table1_epsilon_e5
        return eps

    def copy_for_smoke(self) -> "Config":
        cfg = Config()
        cfg.__dict__.update(self.__dict__)
        cfg.t_final = 0.01
        cfg.t_eval = 0.008
        cfg.final_window_delta = 0.004
        cfg.k_values = np.array([25.0], dtype=np.float64)
        return cfg
