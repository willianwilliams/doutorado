from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from config import Config
from io_utils import fmt, write_text
from metrics import (
    compute_e5_diagnostic,
    compute_e_res,
    compute_max_abs_u,
    compute_rmse_message,
    compute_st_rmse_sse,
    mean_omitnan,
)
from message_utils import compute_ber_from_e5
from numba_core import METHOD_EULER, METHOD_RKF5, simulate_proposed_numba, simulate_reference_numba


def method_settings(method: str, cfg: Config) -> tuple[int, float, str]:
    m = method.lower()
    if m in {"rkf", "rkf5"}:
        return METHOD_RKF5, cfg.h, "RKF5"
    if m == "euler":
        return METHOD_EULER, cfg.h_euler, "Euler"
    raise ValueError(f"Unknown method: {method}")


def ber_from_e5(e: np.ndarray, h: float, cfg: Config) -> dict:
    total, bits_hat, rows = compute_ber_from_e5(e[:, 4], h, cfg, cfg.ber_windows)
    return {
        "total_valid_BER": total,
        "bits_hat": bits_hat,
        "rows": rows,
        "BER_final_window": rows[2, 4] if rows.shape[0] > 2 else float("nan"),
        "BER_0_Teval": rows[-1, 4] if rows.shape[0] else float("nan"),
    }


def epsilon_caption(table: dict) -> str:
    eps = np.asarray(table["epsilon_by_state"], dtype=np.float64)
    if np.allclose(eps, eps[0]):
        return f"$\\varepsilon={eps[0]:.3g}$ for all states"
    return f"$\\varepsilon={table['epsilon']:.3g}$ for $e_1,e_2,e_3,e_4,e_6$ and $\\varepsilon_5={table['epsilon_e5']:.3g}$"


def compute_table1(cfg: Config, b_noise: np.ndarray, method: str) -> dict:
    method_flag, h_int, label = method_settings(method, cfg)
    tic = perf_counter()
    print(f"[Table 1] {label}: proposed k={cfg.k_nominal_proposed:g}")
    e_prop, _u_prop = simulate_proposed_numba(
        method_flag, cfg.t_final, h_int, cfg.h, cfg.freq, cfg.h1, cfg.k_nominal_proposed, b_noise,
        cfg.abort_abs_state_limit, cfg.sawtooth_style
    )
    table1_eps = cfg.table1_epsilon_by_state
    st_p, rmse_p, sse_p = compute_st_rmse_sse(e_prop, h_int, table1_eps, cfg.t_eval)

    print(f"[Table 1] {label}: reference adaptive method")
    e_ref = simulate_reference_numba(
        method_flag, cfg.t_final, h_int, cfg.h, cfg.freq, cfg.h1, cfg.k_reference, cfg.theta0_reference, b_noise,
        cfg.abort_abs_state_limit, cfg.sawtooth_style
    )
    st_r, rmse_r, sse_r = compute_st_rmse_sse(e_ref, h_int, table1_eps, cfg.t_eval)

    table = {
        "method": label,
        "runtime_s": perf_counter() - tic,
        "T_eval": cfg.t_eval,
        "final_window_start": cfg.final_window_start,
        "final_window_delta": cfg.final_window_delta,
        "epsilon": cfg.table1_epsilon_st,
        "epsilon_e5": cfg.table1_epsilon_e5,
        "epsilon_by_state": table1_eps,
        "message_word": cfg.message_word,
        "labels": ["e1", "e2", "e3", "e4", "e5", "e6", "Mean"],
        "proposed": {
            "ST": np.r_[st_p, mean_omitnan(st_p)],
            "RMSE": np.r_[rmse_p, mean_omitnan(rmse_p)],
            "SSE": np.r_[sse_p, mean_omitnan(sse_p)],
            "e5diag": compute_e5_diagnostic(e_prop, h_int, cfg.table1_epsilon_e5, cfg.t_eval, cfg.final_window_start),
            "ber": ber_from_e5(e_prop, h_int, cfg),
        },
        "reference": {
            "ST": np.r_[st_r, mean_omitnan(st_r)],
            "RMSE": np.r_[rmse_r, mean_omitnan(rmse_r)],
            "SSE": np.r_[sse_r, mean_omitnan(sse_r)],
            "e5diag": compute_e5_diagnostic(e_ref, h_int, cfg.table1_epsilon_e5, cfg.t_eval, cfg.final_window_start),
            "ber": ber_from_e5(e_ref, h_int, cfg),
        },
    }
    print(f"[Table 1] {label}: finished in {table['runtime_s']:.2f} s")
    return table


def compute_table3(cfg: Config, b_noise: np.ndarray, method: str) -> dict:
    method_flag, h_int, label = method_settings(method, cfg)
    tic = perf_counter()
    rows = []
    for i, k_gain in enumerate(cfg.k_values, start=1):
        print(f"[Table 3] {label}: k={k_gain:g} ({i}/{len(cfg.k_values)})")
        e, u = simulate_proposed_numba(
            method_flag, cfg.t_final, h_int, cfg.h, cfg.freq, cfg.h1, float(k_gain), b_noise,
            cfg.abort_abs_state_limit, cfg.sawtooth_style
        )
        st, rmse, sse = compute_st_rmse_sse(e, h_int, cfg.epsilon_by_state, cfg.t_eval)
        e_res = compute_e_res(e, h_int, cfg.final_window_start, cfg.t_eval)
        rmse_m = compute_rmse_message(-e[:, 4], h_int, cfg.final_window_start, cfg.t_eval)
        ber = ber_from_e5(e, h_int, cfg)
        rows.append(
            {
                "k": float(k_gain),
                "mean_ST": mean_omitnan(st),
                "mean_RMSE_e": mean_omitnan(rmse),
                "mean_SSE_e": mean_omitnan(sse),
                "E_res_no_msg": e_res,
                "E_res_msg": e_res,
                "E_res_abs_diff": 0.0,
                "max_abs_u": compute_max_abs_u(u, h_int, cfg.t_eval),
                "RMSE_m": rmse_m,
                "BER_total_valid": ber["total_valid_BER"],
                "BER_0_Teval": ber["BER_0_Teval"],
                "BER_final_window": ber["BER_final_window"],
                "BER_windows": ber["rows"],
                "ST_by_state": st,
                "RMSE_by_state": rmse,
                "SSE_by_state": sse,
                "valid_settled_states": int(np.sum(np.isfinite(st))),
            }
        )
    table = {
        "method": label,
        "runtime_s": perf_counter() - tic,
        "rows": rows,
        "T_eval": cfg.t_eval,
        "final_window_start": cfg.final_window_start,
        "final_window_delta": cfg.final_window_delta,
        "message_word": cfg.message_word,
        "epsilon": cfg.epsilon_st,
        "epsilon_e5": cfg.epsilon_e5,
    }
    print(f"[Table 3] {label}: finished in {table['runtime_s']:.2f} s")
    return table


def export_table1(table: dict, out_dir: Path) -> None:
    stem = "RKF" if table["method"].lower().startswith("rkf") else "euler"
    csv_path = out_dir / f"T1_{stem}.csv"
    tex_path = out_dir / f"T1_{stem}.tex"
    article_csv = out_dir / f"T1_article_style_{stem}.csv"
    article_tex = out_dir / f"T1_article_style_{stem}.tex"
    diag_csv = out_dir / f"T1_e5diag_{stem}.csv"
    diag_tex = out_dir / f"T1_e5diag_{stem}.tex"
    ber_csv = out_dir / f"T1_ber_{stem}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["state", "epsilon_used", "proposed_ST_s", "proposed_RMSE", "proposed_SSE", "reference_ST_s", "reference_RMSE", "reference_SSE"])
        for i, label in enumerate(table["labels"]):
            eps_used = "" if label == "Mean" else table["epsilon_by_state"][i]
            writer.writerow([
                label,
                eps_used,
                table["proposed"]["ST"][i], table["proposed"]["RMSE"][i], table["proposed"]["SSE"][i],
                table["reference"]["ST"][i], table["reference"]["RMSE"][i], table["reference"]["SSE"][i],
            ])

    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        f"\\caption{{Comparison of synchronization performance metrics using {table['method']}, {epsilon_caption(table)}, and $T_e={table['T_eval']:.3f}\\,\\mathrm{{s}}$.}}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\begin{tabular}{c|ccc|ccc}",
        "\\hline",
        "State & \\multicolumn{3}{c|}{Proposed method} & \\multicolumn{3}{c}{Reference method} \\\\",
        " & ST (s) & RMSE & SSE & ST (s) & RMSE & SSE \\\\",
        "\\hline",
    ]
    for i, label in enumerate(table["labels"]):
        state = "Mean" if label == "Mean" else f"${label[0]}_{label[1]}$"
        lines.append(
            f"{state} & {fmt(table['proposed']['ST'][i], '.3f')} & {fmt(table['proposed']['RMSE'][i], '.3f')} & {fmt(table['proposed']['SSE'][i], '.3f')} & "
            f"{fmt(table['reference']['ST'][i], '.3f')} & {fmt(table['reference']['RMSE'][i], '.3f')} & {fmt(table['reference']['SSE'][i], '.3f')} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}", ""]
    write_text(tex_path, "\n".join(lines))

    with article_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "state", "proposed_ST_s", "proposed_RMSE", "proposed_SSE",
            "method85_ST_s", "method85_RMSE", "method85_SSE",
            "T_eval_s", "epsilon_used", "note",
        ])
        for i, label in enumerate(table["labels"]):
            eps_used = "" if label == "Mean" else table["epsilon_by_state"][i]
            writer.writerow([
                label,
                table["proposed"]["ST"][i], table["proposed"]["RMSE"][i], table["proposed"]["SSE"][i],
                table["reference"]["ST"][i], table["reference"]["RMSE"][i], table["reference"]["SSE"][i],
                table["T_eval"], eps_used,
                "Digital message disabled; RMSE/SSE are post-settling metrics when ST is defined.",
            ])

    article_lines = [
        "\\begin{table}[!t]",
        "\\centering",
        f"\\caption{{Comparison of synchronization performance metrics using {table['method']} with state-dependent tolerances.}}",
        f"\\label{{tab:t1_article_style_{stem.lower()}}}",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\begin{tabular}{c|ccc|ccc}",
        "\\hline",
        "Error & \\multicolumn{3}{c|}{Proposed Method} & \\multicolumn{3}{c}{Method in [85]} \\\\",
        " & ST (s) & RMSE & SSE & ST (s) & RMSE & SSE \\\\",
        "\\hline",
    ]
    for i, label in enumerate(table["labels"]):
        state = "Mean" if label == "Mean" else f"$e_{label[1]}$"
        article_lines.append(
            f"{state} & {fmt(table['proposed']['ST'][i], '.3f')} & {fmt(table['proposed']['RMSE'][i], '.3f')} & {fmt(table['proposed']['SSE'][i], '.3f')} & "
            f"{fmt(table['reference']['ST'][i], '.3f')} & {fmt(table['reference']['RMSE'][i], '.3f')} & {fmt(table['reference']['SSE'][i], '.3f')} \\\\"
        )
    article_lines += [
        "\\hline",
        "\\end{tabular}",
        "",
        "\\vspace{1mm}",
        "\\begin{minipage}{0.95\\linewidth}",
        "\\footnotesize",
        f"ST denotes settling time. RMSE and SSE were computed over the post-settling interval $[ST_i,T_e]$, with $T_e={table['T_eval']:.3f}\\,\\mathrm{{s}}$. "
        f"The settling tolerance is $\\varepsilon={table['epsilon']:.3g}$ for all error states. "
        "The digital message was disabled for this synchronization-metric table. Entries shown as -- did not satisfy the strict settling rule before $T_e$.",
        "\\end{minipage}",
        "\\end{table}",
        "",
    ]
    write_text(article_tex, "\n".join(article_lines))

    with ber_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "window_start_s", "window_end_s", "evaluated_bits", "bit_errors", "BER", "total_valid_BER"])
        for method_name, key in [("Proposed", "proposed"), ("Reference", "reference")]:
            ber = table[key]["ber"]
            for row in ber["rows"]:
                writer.writerow([
                    method_name,
                    row[0],
                    row[1],
                    int(row[2]),
                    int(row[3]),
                    row[4],
                    ber["total_valid_BER"],
                ])

    with diag_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "ST5_strict_s", "settled_strict", "last_violation_time_s", "abs_e5_Te",
            "max_abs_e5_0Te", "max_abs_e5_final", "min_e5_final", "max_e5_final", "RMSE5_final",
            "SSE5_final", "final_window_start_s", "final_window_end_s", "epsilon", "n_final_points",
        ])
        for method_name, key in [("Proposed", "proposed"), ("Reference", "reference")]:
            d = table[key]["e5diag"]
            writer.writerow([
                method_name, d["ST_strict"], int(bool(d["settled_strict"])), d["last_violation_time"],
                d["abs_e5_Te"], d["max_abs_e5_0Te"], d["max_abs_e5_final"], d["min_e5_final"],
                d["max_e5_final"], d["RMSE5_final"], d["SSE5_final"], d["final_window_start"],
                d["final_window_end"], d["epsilon"], d["n_final_points"],
            ])

    diag_lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        f"\\caption{{Diagnostic evaluation of $e_5$ using {table['method']} with message {table['message_word']}.}}",
        "\\begin{tabular}{lccccccc}",
        "\\hline",
        f"Method & $ST_5$ strict & Settled & Last $|e_5|\\ge {table['epsilon_e5']:.3g}$ & $|e_5(T_e)|$ & $\\max|e_5|$ final & $RMSE_{{5,final}}$ & $SSE_{{5,final}}$ \\\\",
        "\\hline",
    ]
    for method_name, key in [("Proposed", "proposed"), ("Reference", "reference")]:
        d = table[key]["e5diag"]
        settled = "yes" if d["settled_strict"] else "no"
        diag_lines.append(
            f"{method_name} & {fmt(d['ST_strict'], '.3f')} & {settled} & {fmt(d['last_violation_time'], '.3f')} & "
            f"{fmt(d['abs_e5_Te'], '.3f')} & {fmt(d['max_abs_e5_final'], '.3f')} & {fmt(d['RMSE5_final'], '.3f')} & {fmt(d['SSE5_final'], '.3f')} \\\\"
        )
    diag_lines += ["\\hline", "\\end{tabular}", "\\end{table*}", ""]
    write_text(diag_tex, "\n".join(diag_lines))

    print(f"Saved {csv_path.name}, {tex_path.name}, {article_csv.name}, {article_tex.name}, {diag_csv.name}, {diag_tex.name}, {ber_csv.name}")


def export_table3(table: dict, out_dir: Path) -> None:
    stem = "RKF" if table["method"].lower().startswith("rkf") else "euler"
    csv_path = out_dir / f"T3_{stem}.csv"
    tex_path = out_dir / f"T3_{stem}.tex"
    ber_csv = out_dir / f"T3_ber_{stem}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "k", "mean_ST_s", "mean_RMSE_e", "mean_SSE_e", "E_res_no_msg",
            "E_res_msg", "E_res_abs_diff", "max_abs_u", "RMSE_m", "BER_0_Teval",
            "BER_final_window", "BER_total_valid", "settled_states",
        ])
        for row in table["rows"]:
            writer.writerow([
                table["method"], row["k"], row["mean_ST"], row["mean_RMSE_e"], row["mean_SSE_e"],
                row["E_res_no_msg"], row["E_res_msg"], row["E_res_abs_diff"], row["max_abs_u"],
                row["RMSE_m"], row["BER_0_Teval"], row["BER_final_window"], row["BER_total_valid"],
                row["valid_settled_states"],
            ])

    with ber_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "k", "window_start_s", "window_end_s", "evaluated_bits", "bit_errors", "BER"])
        for row in table["rows"]:
            for w in row["BER_windows"]:
                writer.writerow([table["method"], row["k"], w[0], w[1], int(w[2]), int(w[3]), w[4]])

    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        f"\\caption{{Gain sensitivity using {table['method']}, $T_e={table['T_eval']:.3f}\\,\\mathrm{{s}}$, $\\Delta={table['final_window_delta']:.3f}\\,\\mathrm{{s}}$, $\\varepsilon_5={table['epsilon_e5']:.3g}$, and message {table['message_word']}.}}",
        "\\begin{tabular}{c|cccccc}",
        "\\hline",
        "$k$ & $\\overline{ST}$ (s) & $\\overline{RMSE}_e$ & $\\overline{SSE}_e$ & $E_{res}$ & $\\max |u(t)|$ & $RMSE_m$ \\\\",
        "\\hline",
    ]
    for row in table["rows"]:
        lines.append(
            f"{row['k']:g} & {fmt(row['mean_ST'], '.3f')} & {fmt(row['mean_RMSE_e'], '.3f')} & {fmt(row['mean_SSE_e'], '.3f')} & "
            f"{fmt(row['E_res_no_msg'], '.3f')} & {fmt(row['max_abs_u'], '.3f')} & {fmt(row['RMSE_m'], '.3f')} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}", ""]
    write_text(tex_path, "\n".join(lines))
    print(f"Saved {csv_path.name}, {tex_path.name}, {ber_csv.name}")
