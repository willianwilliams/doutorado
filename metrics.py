from __future__ import annotations

import numpy as np


def _trapz_uniform(values: np.ndarray, h: float) -> float:
    if values.size < 2:
        return float("nan")
    if values.size == 2:
        return float(0.5 * h * (values[0] + values[1]))
    return float(h * (0.5 * values[0] + values[1:-1].sum() + 0.5 * values[-1]))


def mean_omitnan(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def metric_indices(n_rows: int, h: float, t_eval: float, t0: float | None = None) -> tuple[int, int]:
    end = min(int(round(t_eval / h)), n_rows - 1)
    if t0 is None:
        start = 0
    else:
        start = max(0, min(int(round(t0 / h)), end))
    return start, end


def _epsilon_vector(epsilon: float | np.ndarray, n_states: int) -> np.ndarray:
    if np.isscalar(epsilon):
        return float(epsilon) * np.ones(n_states, dtype=np.float64)
    eps = np.asarray(epsilon, dtype=np.float64)
    if eps.size != n_states:
        raise ValueError(f"Expected {n_states} epsilon values, received {eps.size}.")
    return eps


def compute_st_rmse_sse(e: np.ndarray, h: float, epsilon: float | np.ndarray, t_eval: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_states = e.shape[1]
    eps = _epsilon_vector(epsilon, n_states)
    st = np.full(n_states, np.nan)
    rmse = np.full(n_states, np.nan)
    sse = np.full(n_states, np.nan)
    _, idx_eval = metric_indices(e.shape[0], h, t_eval)

    finite_rows = np.all(np.isfinite(e[: idx_eval + 1, :]), axis=1)
    if not finite_rows[idx_eval]:
        finite_idx = np.where(finite_rows)[0]
        if finite_idx.size == 0 or finite_idx[-1] < idx_eval:
            return st, rmse, sse

    for i in range(n_states):
        ei = e[: idx_eval + 1, i]
        if not np.all(np.isfinite(ei)):
            continue
        bad = np.where(np.abs(ei) >= eps[i])[0]
        if bad.size == 0:
            st_idx = 0
        elif bad[-1] < idx_eval:
            st_idx = int(bad[-1] + 1)
        else:
            continue

        if st_idx >= idx_eval:
            continue
        duration = (idx_eval - st_idx) * h
        if duration <= 0:
            continue
        st[i] = st_idx * h
        seg = ei[st_idx : idx_eval + 1]
        rmse[i] = np.sqrt(_trapz_uniform(seg * seg, h) / duration)
        sse[i] = _trapz_uniform(np.abs(seg), h) / duration

    return st, rmse, sse


def compute_e_res(e: np.ndarray, h: float, start: float, end: float) -> float:
    i0, i1 = metric_indices(e.shape[0], h, end, start)
    win = e[i0 : i1 + 1, :]
    win = win[np.all(np.isfinite(win), axis=1), :]
    if win.size == 0:
        return float("nan")
    return float(np.max(np.sqrt(np.sum(win * win, axis=1))))


def compute_rmse_message(e5: np.ndarray, h: float, start: float, end: float) -> float:
    i0, i1 = metric_indices(e5.shape[0], h, end, start)
    win = e5[i0 : i1 + 1]
    win = win[np.isfinite(win)]
    if win.size < 2:
        return float("nan")
    return float(np.sqrt(_trapz_uniform(win * win, h) / (end - start)))


def compute_sse_window(e5: np.ndarray, h: float, start: float, end: float) -> float:
    i0, i1 = metric_indices(e5.shape[0], h, end, start)
    win = e5[i0 : i1 + 1]
    win = win[np.isfinite(win)]
    if win.size < 2:
        return float("nan")
    return float(_trapz_uniform(np.abs(win), h) / (end - start))


def compute_max_abs_u(u: np.ndarray, h: float, t_eval: float) -> float:
    _, idx_eval = metric_indices(u.shape[0], h, t_eval)
    vals = u[: idx_eval + 1]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.max(np.abs(vals)))


def compute_e5_diagnostic(e: np.ndarray, h: float, epsilon: float, t_eval: float, final_start: float) -> dict[str, float | bool | int]:
    _, idx_eval = metric_indices(e.shape[0], h, t_eval)
    result: dict[str, float | bool | int] = {
        "ST_strict": float("nan"),
        "settled_strict": False,
        "last_violation_time": float("nan"),
        "T_used": float("nan"),
        "abs_e5_Te": float("nan"),
        "max_abs_e5_0Te": float("nan"),
        "max_abs_e5_final": float("nan"),
        "min_e5_final": float("nan"),
        "max_e5_final": float("nan"),
        "RMSE5_final": float("nan"),
        "SSE5_final": float("nan"),
        "final_window_start": final_start,
        "final_window_end": t_eval,
        "epsilon": epsilon,
        "n_final_points": 0,
    }

    e_until = e[: idx_eval + 1, :]
    finite_rows = np.all(np.isfinite(e_until), axis=1)
    if not finite_rows[idx_eval]:
        return result

    e5 = e_until[:, 4]
    result["T_used"] = idx_eval * h
    bad = np.where(np.abs(e5) >= epsilon)[0]
    if bad.size == 0:
        result["ST_strict"] = 0.0
        result["settled_strict"] = True
    elif bad[-1] < idx_eval:
        result["ST_strict"] = (bad[-1] + 1) * h
        result["settled_strict"] = True
        result["last_violation_time"] = bad[-1] * h
    else:
        result["last_violation_time"] = bad[-1] * h

    result["abs_e5_Te"] = float(abs(e5[-1]))
    result["max_abs_e5_0Te"] = float(np.max(np.abs(e5)))

    i0, i1 = metric_indices(e.shape[0], h, t_eval, final_start)
    e5f = e[i0 : i1 + 1, 4]
    e5f = e5f[np.isfinite(e5f)]
    result["n_final_points"] = int(e5f.size)
    if e5f.size >= 2:
        result["max_abs_e5_final"] = float(np.max(np.abs(e5f)))
        result["min_e5_final"] = float(np.min(e5f))
        result["max_e5_final"] = float(np.max(e5f))
        result["RMSE5_final"] = compute_rmse_message(e[:, 4], h, final_start, t_eval)
        result["SSE5_final"] = compute_sse_window(e[:, 4], h, final_start, t_eval)
    return result
