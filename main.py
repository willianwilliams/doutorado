from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from config import Config
from io_utils import load_or_create_jamming
from message_utils import export_message_info
from numba_core import NUMBA_AVAILABLE
from tables import compute_table1, compute_table3, export_table1, export_table3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast Python/Numba conversion of RKF.m and euler.m.")
    parser.add_argument("--method", choices=["rkf", "euler", "both"], default="both", help="Which integrator to run.")
    parser.add_argument("--tables", choices=["table1", "table3", "all"], default="all", help="Which tables to compute.")
    parser.add_argument("--out", default="results", help="Output folder.")
    parser.add_argument("--jam", default="", help="Optional MATLAB JAM.mat path with variable b_noise. Defaults to ./JAM.mat.")
    parser.add_argument("--h1", type=float, default=0.8, help="Disturbance/jamming gain h_1.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Settling tolerance epsilon for e1, e2, e3, e4, and e6.")
    parser.add_argument("--epsilon-e5", type=float, default=2.0, help="Settling tolerance epsilon for e5 in Table 3.")
    parser.add_argument("--table1-epsilon", type=float, default=3.0, help="Settling tolerance epsilon for all states in Table 1.")
    parser.add_argument("--table1-epsilon-e5", type=float, default=3.0, help="Settling tolerance epsilon for e5 in Table 1.")
    parser.add_argument("--t-final", type=float, default=2.0, help="Simulation final time.")
    parser.add_argument("--t-eval", type=float, default=1.920, help="Metric horizon T_e.")
    parser.add_argument("--euler-substeps", type=int, default=10, help="Euler substeps per original h.")
    parser.add_argument("--k-values", default="0.1,1,10,25,50,75,100,125,150,200,1000", help="Comma-separated gains for Table 3.")
    parser.add_argument("--message", default="Secure Transmission", help="ASCII message used for secure-communication diagnostics.")
    parser.add_argument("--bit-rate", type=float, default=69.0, help="Message bit rate in bits/s.")
    parser.add_argument("--sawtooth-style", choices=["matlab", "fallback"], default="matlab", help="Sawtooth formula.")
    parser.add_argument("--smoke", action="store_true", help="Run a very short smoke test.")
    parser.add_argument("--allow-slow", action="store_true", help="Allow slow pure-Python fallback when numba is not installed.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.output_dir = Path(args.out)
    if args.jam:
        cfg.jam_path = Path(args.jam)
    cfg.h1 = args.h1
    cfg.epsilon_st = args.epsilon
    cfg.epsilon_e5 = args.epsilon_e5
    cfg.table1_epsilon_st = args.table1_epsilon
    cfg.table1_epsilon_e5 = args.table1_epsilon_e5
    cfg.t_final = args.t_final
    cfg.t_eval = args.t_eval
    cfg.euler_internal_substeps = args.euler_substeps
    cfg.k_values = np.array([float(x.strip()) for x in args.k_values.split(",") if x.strip()], dtype=np.float64)
    cfg.message_word = args.message
    cfg.bit_rate = args.bit_rate
    cfg.sawtooth_style = 0 if args.sawtooth_style == "matlab" else 1
    if args.smoke:
        cfg = cfg.copy_for_smoke()
        cfg.output_dir = Path(args.out)
        if args.jam:
            cfg.jam_path = Path(args.jam)
        cfg.message_word = args.message
        cfg.bit_rate = args.bit_rate
    return cfg


def main() -> int:
    args = parse_args()
    if not NUMBA_AVAILABLE and not args.allow_slow:
        print("ERROR: numba is not installed. This conversion is intended to be fast with Numba.")
        print("Install once with: python -m pip install -r requirements.txt")
        print("For a tiny syntax smoke test only, use --allow-slow --smoke.")
        return 2

    cfg = build_config(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("Fast article-parameter simulation")
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print(f"h={cfg.h:.12g} | h_euler={cfg.h_euler:.12g} | T_final={cfg.t_final:.6g} | T_eval={cfg.t_eval:.6g}")
    print(f"final window=[{cfg.final_window_start:.6g}, {cfg.t_eval:.6g}] | Delta={cfg.final_window_delta:.6g}")
    print(
        f"h1={cfg.h1:g} | T1 epsilon={cfg.table1_epsilon_st:g} for all states | "
        f"T3 epsilon={cfg.epsilon_st:g} for e1-e4,e6 | T3 epsilon_e5={cfg.epsilon_e5:g} | output={cfg.output_dir}"
    )
    print(f"theta0_reference=zeros(8)")
    print(f"message={cfg.message_word!r} | chars={len(cfg.message_word)} | bits={cfg.message_bits} | bit_rate={cfg.bit_rate:g} | duration={cfg.message_duration:.9f} s")
    print("============================================================")

    b_noise = load_or_create_jamming(cfg)
    export_message_info(cfg, cfg.output_dir)
    methods = ["rkf", "euler"] if args.method == "both" else [args.method]

    tic = perf_counter()
    for method in methods:
        if args.tables in {"table1", "all"}:
            t1 = compute_table1(cfg, b_noise, method)
            export_table1(t1, cfg.output_dir)
        if args.tables in {"table3", "all"}:
            t3 = compute_table3(cfg, b_noise, method)
            export_table3(t3, cfg.output_dir)

    print(f"All requested work finished in {perf_counter() - tic:.2f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
