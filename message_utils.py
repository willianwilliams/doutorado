from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from config import Config


def message_bits(cfg: Config) -> np.ndarray:
    bits: list[int] = []
    for ch in cfg.message_word:
        code = ord(ch)
        if code >= 2**cfg.n_bits_char:
            raise ValueError(f"Character {ch!r} does not fit in {cfg.n_bits_char} bits.")
        bits.extend(int(v) for v in format(code, f"0{cfg.n_bits_char}b"))
    return np.asarray(bits, dtype=np.int8)


def message_signal(n_rows: int, h: float, cfg: Config) -> np.ndarray:
    bits = message_bits(cfg)
    symbols = cfg.message_amplitude * (2.0 * bits.astype(np.float64) - 1.0)
    signal = np.zeros(n_rows, dtype=np.float64)

    for k, symbol in enumerate(symbols):
        t_ini = cfg.message_start_time + k / cfg.bit_rate
        t_end = cfg.message_start_time + (k + 1) / cfg.bit_rate
        i0 = max(0, int(np.ceil(t_ini / h)))
        i1 = min(n_rows, int(np.ceil(t_end / h)))
        if i0 < i1:
            signal[i0:i1] = symbol
    return signal


def compute_ber_from_e5(
    e5: np.ndarray,
    h: float,
    cfg: Config,
    windows: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    bits_tx = message_bits(cfg)
    m = message_signal(e5.size, h, cfg)
    recovered = m - e5
    bits_hat = np.full(bits_tx.shape, np.nan, dtype=np.float64)

    for k in range(bits_tx.size):
        t_ini = cfg.message_start_time + k / cfg.bit_rate
        t_end = cfg.message_start_time + (k + 1) / cfg.bit_rate
        i0 = max(0, int(np.ceil(t_ini / h)))
        i1 = min(e5.size, int(np.ceil(t_end / h)))
        if i0 < i1:
            values = recovered[i0:i1]
            values = values[np.isfinite(values)]
            if values.size:
                bits_hat[k] = 1.0 if values.mean() >= 0.0 else 0.0

    valid = np.isfinite(bits_hat)
    if np.any(valid):
        total_ber = float(np.count_nonzero(bits_hat[valid].astype(np.int8) != bits_tx[valid]) / np.count_nonzero(valid))
    else:
        total_ber = float("nan")

    if windows is None:
        windows = cfg.ber_windows

    rows = np.zeros((windows.shape[0], 5), dtype=np.float64)
    t_bits = cfg.message_start_time + (np.arange(bits_tx.size, dtype=np.float64) + 0.5) / cfg.bit_rate
    for j, (a, b) in enumerate(windows):
        idx = valid & (t_bits >= a) & (t_bits < b)
        n_eval = int(np.count_nonzero(idx))
        if n_eval:
            n_err = int(np.count_nonzero(bits_hat[idx].astype(np.int8) != bits_tx[idx]))
            ber = n_err / n_eval
        else:
            n_err = 0
            ber = float("nan")
        rows[j, :] = [a, b, n_eval, n_err, ber]

    return total_ber, bits_hat, rows


def export_message_info(cfg: Config, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bits = message_bits(cfg)
    path = out_dir / "message_info.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "value"])
        writer.writerow(["message_word", cfg.message_word])
        writer.writerow(["n_chars", len(cfg.message_word)])
        writer.writerow(["n_bits_char", cfg.n_bits_char])
        writer.writerow(["n_bits", bits.size])
        writer.writerow(["bit_rate", cfg.bit_rate])
        writer.writerow(["bit_period", 1.0 / cfg.bit_rate])
        writer.writerow(["message_duration", cfg.message_duration])
        writer.writerow(["message_start_time", cfg.message_start_time])
        writer.writerow(["message_amplitude", cfg.message_amplitude])
        writer.writerow(["metric_T_eval", cfg.t_eval])
        writer.writerow(["metric_final_window_start", cfg.final_window_start])
        writer.writerow(["metric_final_window_end", cfg.t_eval])
        writer.writerow(["ber_transient_cut", cfg.ber_transient_cut])

    bits_path = out_dir / "message_bits.csv"
    with bits_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bit_index", "bit"])
        for i, bit in enumerate(bits, start=1):
            writer.writerow([i, int(bit)])

    n_rows = int(round(cfg.t_final / cfg.h)) + 1
    msg = message_signal(n_rows, cfg.h, cfg)
    signal_path = out_dir / "message_signal.csv"
    with signal_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "message"])
        for i, value in enumerate(msg):
            writer.writerow([i * cfg.h, float(value)])
