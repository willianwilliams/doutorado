from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - slow fallback for syntax/smoke tests only
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


METHOD_RKF5 = 0
METHOD_EULER = 1


@njit(cache=True)
def _square_wave(x: float) -> float:
    if math.sin(x) >= 0.0:
        return 1.0
    return -1.0


@njit(cache=True)
def _sawtooth_wave(x: float, style: int) -> float:
    z = x / (2.0 * math.pi)
    if style == 1:
        return 2.0 * (z - math.floor(0.5 + z))
    return 2.0 * (z - math.floor(z)) - 1.0


@njit(cache=True)
def _system_eq_from_state(x: np.ndarray, offset: int, out: np.ndarray) -> None:
    x1 = x[offset + 0]
    x2 = x[offset + 1]
    x3 = x[offset + 2]
    x4 = x[offset + 3]
    x5 = x[offset + 4]
    x6 = x[offset + 5]

    a = 58.0
    b = 2.0
    c = 0.1
    d = 0.1
    epsv = 8.0 / 30.0
    f = 1.0
    g = 8.21
    eta = 24.65

    out[0] = -x1 + a * x2 - b * x4 - c * x5
    out[1] = -d * x2 + d * x1 - d * x1 * x3
    out[2] = -epsv * x3 + d * x1 * x2
    out[3] = -f * x4 + d * x1
    out[4] = -x5 + g * x1 + b * x6
    out[5] = -f * x6 - eta * x4 - d * x5


@njit(cache=True)
def _disturb_master(t: float, style: int, out: np.ndarray) -> None:
    out[0] = 5.0 * (
        0.55 * math.sin(5.0 * math.pi * t)
        + 0.42 * math.sin(10.0 * math.pi * t)
        + 0.18 * _square_wave(15.0 * math.pi * t)
        - 0.1 * _sawtooth_wave(20.0 * math.pi * t, style)
    )
    out[1] = 5.0 * (
        0.06 * math.sin(7.0 * math.pi * t)
        - 0.05 * math.cos(7.0 * math.pi * t)
        - 0.03 * _sawtooth_wave(11.0 * math.pi * t, style)
    )
    out[2] = 5.0 * (
        0.1 * math.cos(8.0 * math.pi * t)
        + 0.055 * math.cos(20.0 * math.pi * t)
        + 0.025 * _square_wave(25.0 * math.pi * t)
    )
    out[3] = 5.0 * (
        0.09 * math.sin(4.0 * math.pi * t)
        - 0.06 * math.cos(9.0 * math.pi * t)
        + 0.04 * math.cos(11.0 * math.pi * t)
    )
    out[4] = 5.0 * (
        1.15 * math.cos(7.0 * math.pi * t)
        - 0.9 * math.cos(12.0 * math.pi * t)
        - 0.44 * _square_wave(15.0 * math.pi * t)
    )
    out[5] = 5.0 * (
        0.95 * math.sin(10.0 * math.pi * t)
        + 0.7 * math.cos(20.0 * math.pi * t)
        + 0.45 * _sawtooth_wave(14.0 * math.pi * t, style)
    )


@njit(cache=True)
def _disturb_slave(t: float, style: int, out: np.ndarray) -> None:
    out[0] = 5.0 * (
        0.6 * math.sin(5.0 * math.pi * t)
        + 0.45 * math.cos(11.0 * math.pi * t)
        + 0.2 * _square_wave(15.0 * math.pi * t)
    )
    out[1] = 5.0 * (
        0.07 * math.sin(6.0 * math.pi * t)
        - 0.04 * math.cos(8.0 * math.pi * t)
        - 0.02 * _sawtooth_wave(11.0 * math.pi * t, style)
    )
    out[2] = 5.0 * (
        0.11 * math.cos(8.0 * math.pi * t)
        + 0.06 * math.cos(22.0 * math.pi * t)
        + 0.05 * _square_wave(26.0 * math.pi * t)
        - 0.03 * _sawtooth_wave(21.0 * math.pi * t, style)
    )
    out[3] = 5.0 * (
        0.12 * math.sin(9.0 * math.pi * t)
        - 0.05 * math.cos(21.0 * math.pi * t)
        - 0.03 * _sawtooth_wave(15.0 * math.pi * t, style)
    )
    out[4] = 5.0 * (
        1.1 * math.sin(4.0 * math.pi * t)
        + 0.82 * math.cos(9.0 * math.pi * t)
        + 0.36 * math.cos(11.0 * math.pi * t)
    )
    out[5] = 5.0 * (
        0.8 * math.cos(6.0 * math.pi * t)
        - 0.75 * math.cos(12.0 * math.pi * t)
        - 0.41 * _square_wave(16.0 * math.pi * t)
    )


@njit(cache=True)
def _channel_jamming(t: float, b_noise: np.ndarray, h_step: float, h1: float, out: np.ndarray) -> None:
    idx = int(math.floor(t / h_step))
    if idx < 0:
        idx = 0
    if idx >= b_noise.shape[0]:
        idx = b_noise.shape[0] - 1
    out[0] = h1 * 0.1 * b_noise[idx, 0]
    out[1] = 0.0
    out[2] = 0.0
    out[3] = 0.0
    out[4] = h1 * 0.5 * b_noise[idx, 4]
    out[5] = 0.0


@njit(cache=True)
def _fill_article_error(x: np.ndarray, t: float, b_noise: np.ndarray, h_step: float, h1: float, out: np.ndarray) -> None:
    idx = int(math.floor(t / h_step))
    if idx < 0:
        idx = 0
    if idx >= b_noise.shape[0]:
        idx = b_noise.shape[0] - 1
    hc0 = h1 * 0.1 * b_noise[idx, 0]
    hc4 = h1 * 0.5 * b_noise[idx, 4]
    out[0] = x[6] - x[0] - hc0
    out[1] = x[7] - x[1]
    out[2] = x[8] - x[2]
    out[3] = x[9] - x[3]
    out[4] = x[10] - x[4] - hc4
    out[5] = x[11] - x[5]


@njit(cache=True)
def _rhs_proposed(
    t: float,
    x: np.ndarray,
    b_noise: np.ndarray,
    h1: float,
    h_step: float,
    freq: float,
    k_gain: float,
    saw_style: int,
    ydot: np.ndarray,
) -> None:
    eq_m = np.empty(6, dtype=np.float64)
    eq_s = np.empty(6, dtype=np.float64)
    hm = np.empty(6, dtype=np.float64)
    hs = np.empty(6, dtype=np.float64)
    hc = np.empty(6, dtype=np.float64)

    _system_eq_from_state(x, 0, eq_m)
    _system_eq_from_state(x, 6, eq_s)
    _disturb_master(t, saw_style, hm)
    _disturb_slave(t, saw_style, hs)
    _channel_jamming(t, b_noise, h_step, h1, hc)

    e0 = x[6] - x[0] - hc[0]
    u0 = -k_gain * e0

    for i in range(6):
        ydot[i] = freq * eq_m[i] + h1 * hm[i]
        ydot[i + 6] = freq * eq_s[i] + h1 * hs[i]
    ydot[6] += freq * u0


@njit(cache=True)
def _control_reference(x: np.ndarray, e: np.ndarray, k_vec: np.ndarray, out: np.ndarray) -> None:
    ahat = x[12]
    bhat = x[13]
    chat = x[14]
    dhat = x[15]
    epshat = x[16]
    fhat = x[17]
    ghat = x[18]
    etahat = x[19]

    xm1 = x[0]
    xm2 = x[1]
    xm3 = x[2]
    ys1 = x[6]
    ys2 = x[7]
    ys3 = x[8]

    out[0] = e[0] - ahat * e[1] + bhat * e[3] + chat * e[4] - k_vec[0] * e[0]
    out[1] = dhat * (e[1] - e[0] + ys1 * ys3 - xm1 * xm3) - k_vec[1] * e[1]
    out[2] = epshat * e[2] - dhat * (ys1 * ys2 - xm1 * xm2) - k_vec[2] * e[2]
    out[3] = fhat * e[3] - dhat * e[0] - k_vec[3] * e[3]
    out[4] = e[4] - ghat * e[0] - bhat * e[5] - k_vec[4] * e[4]
    out[5] = fhat * e[5] - etahat * e[3] + dhat * e[4] - k_vec[5] * e[5]


@njit(cache=True)
def _rhs_reference(
    t: float,
    x: np.ndarray,
    b_noise: np.ndarray,
    h1: float,
    h_step: float,
    freq: float,
    k_vec: np.ndarray,
    saw_style: int,
    ydot: np.ndarray,
) -> None:
    eq_m = np.empty(6, dtype=np.float64)
    eq_s = np.empty(6, dtype=np.float64)
    hm = np.empty(6, dtype=np.float64)
    hs = np.empty(6, dtype=np.float64)
    hc = np.empty(6, dtype=np.float64)
    e = np.empty(6, dtype=np.float64)
    u = np.empty(6, dtype=np.float64)

    _system_eq_from_state(x, 0, eq_m)
    _system_eq_from_state(x, 6, eq_s)
    _disturb_master(t, saw_style, hm)
    _disturb_slave(t, saw_style, hs)
    _channel_jamming(t, b_noise, h_step, h1, hc)

    for i in range(6):
        e[i] = x[i + 6] - x[i] - hc[i]

    _control_reference(x, e, k_vec, u)

    for i in range(6):
        ydot[i] = freq * eq_m[i] + h1 * hm[i]
        ydot[i + 6] = freq * (eq_s[i] + u[i]) + h1 * hs[i]

    ydot[12] = freq * (e[0] * e[1])
    ydot[13] = freq * (-e[0] * e[3])
    ydot[14] = freq * (-e[0] * e[4])
    ydot[15] = freq * (
        -e[1] * e[1]
        + e[0] * (e[1] + e[3])
        - e[1] * (x[6] * x[8] - x[0] * x[2])
        - e[2] * (x[6] * x[7] - x[0] * x[1])
        - e[4] * e[5]
    )
    ydot[16] = freq * (-e[2] * e[2])
    ydot[17] = freq * (-e[3] * e[3] - e[5] * e[5])
    ydot[18] = freq * (e[0] * e[4])
    ydot[19] = freq * (e[3] * e[5])


@njit(cache=True)
def _is_bad_state(x: np.ndarray, limit: float) -> bool:
    for i in range(x.shape[0]):
        if not math.isfinite(x[i]) or abs(x[i]) > limit:
            return True
    return False


@njit(cache=True)
def simulate_proposed_numba(
    method_flag: int,
    t_final: float,
    h_int: float,
    h_step: float,
    freq: float,
    h1: float,
    k_gain: float,
    b_noise: np.ndarray,
    abort_limit: float,
    saw_style: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(round(t_final / h_int))
    x = np.empty(12, dtype=np.float64)
    for i in range(6):
        x[i] = 1.0
        x[i + 6] = 15.0

    e_hist = np.empty((n_steps + 1, 6), dtype=np.float64)
    u_hist = np.empty(n_steps + 1, dtype=np.float64)
    e_tmp = np.empty(6, dtype=np.float64)
    _fill_article_error(x, 0.0, b_noise, h_step, h1, e_tmp)
    for j in range(6):
        e_hist[0, j] = e_tmp[j]
    u_hist[0] = -k_gain * e_tmp[0]

    k1 = np.empty(12, dtype=np.float64)
    k2 = np.empty(12, dtype=np.float64)
    k3 = np.empty(12, dtype=np.float64)
    k4 = np.empty(12, dtype=np.float64)
    k5 = np.empty(12, dtype=np.float64)
    k6 = np.empty(12, dtype=np.float64)
    xt = np.empty(12, dtype=np.float64)
    x_next = np.empty(12, dtype=np.float64)

    for step in range(n_steps):
        t = step * h_int
        if method_flag == METHOD_EULER:
            _rhs_proposed(t, x, b_noise, h1, h_step, freq, k_gain, saw_style, k1)
            for j in range(12):
                x_next[j] = x[j] + h_int * k1[j]
        else:
            _rhs_proposed(t, x, b_noise, h1, h_step, freq, k_gain, saw_style, k1)
            for j in range(12):
                xt[j] = x[j] + h_int * (k1[j] / 4.0)
            _rhs_proposed(t + h_int / 4.0, xt, b_noise, h1, h_step, freq, k_gain, saw_style, k2)
            for j in range(12):
                xt[j] = x[j] + h_int * (3.0 * k1[j] / 32.0 + 9.0 * k2[j] / 32.0)
            _rhs_proposed(t + 3.0 * h_int / 8.0, xt, b_noise, h1, h_step, freq, k_gain, saw_style, k3)
            for j in range(12):
                xt[j] = x[j] + h_int * (1932.0 * k1[j] / 2197.0 - 7200.0 * k2[j] / 2197.0 + 7296.0 * k3[j] / 2197.0)
            _rhs_proposed(t + 12.0 * h_int / 13.0, xt, b_noise, h1, h_step, freq, k_gain, saw_style, k4)
            for j in range(12):
                xt[j] = x[j] + h_int * (439.0 * k1[j] / 216.0 - 8.0 * k2[j] + 3680.0 * k3[j] / 513.0 - 845.0 * k4[j] / 4104.0)
            _rhs_proposed(t + h_int, xt, b_noise, h1, h_step, freq, k_gain, saw_style, k5)
            for j in range(12):
                xt[j] = x[j] + h_int * (-8.0 * k1[j] / 27.0 + 2.0 * k2[j] - 3544.0 * k3[j] / 2565.0 + 1859.0 * k4[j] / 4104.0 - 11.0 * k5[j] / 40.0)
            _rhs_proposed(t + h_int / 2.0, xt, b_noise, h1, h_step, freq, k_gain, saw_style, k6)
            for j in range(12):
                x_next[j] = x[j] + h_int * (
                    16.0 * k1[j] / 135.0
                    + 6656.0 * k3[j] / 12825.0
                    + 28561.0 * k4[j] / 56430.0
                    - 9.0 * k5[j] / 50.0
                    + 2.0 * k6[j] / 55.0
                )

        if _is_bad_state(x_next, abort_limit):
            for r in range(step + 1, n_steps + 1):
                for c in range(6):
                    e_hist[r, c] = np.nan
                u_hist[r] = np.nan
            return e_hist, u_hist

        for j in range(12):
            x[j] = x_next[j]
        _fill_article_error(x, (step + 1) * h_int, b_noise, h_step, h1, e_tmp)
        for j in range(6):
            e_hist[step + 1, j] = e_tmp[j]
        u_hist[step + 1] = -k_gain * e_tmp[0]

    return e_hist, u_hist


@njit(cache=True)
def simulate_reference_numba(
    method_flag: int,
    t_final: float,
    h_int: float,
    h_step: float,
    freq: float,
    h1: float,
    k_reference: np.ndarray,
    theta0: np.ndarray,
    b_noise: np.ndarray,
    abort_limit: float,
    saw_style: int,
) -> np.ndarray:
    n_steps = int(round(t_final / h_int))
    x = np.empty(20, dtype=np.float64)
    for i in range(6):
        x[i] = 1.0
        x[i + 6] = 15.0
    for i in range(8):
        x[i + 12] = theta0[i]

    e_hist = np.empty((n_steps + 1, 6), dtype=np.float64)
    e_tmp = np.empty(6, dtype=np.float64)
    _fill_article_error(x, 0.0, b_noise, h_step, h1, e_tmp)
    for j in range(6):
        e_hist[0, j] = e_tmp[j]

    k1 = np.empty(20, dtype=np.float64)
    k2 = np.empty(20, dtype=np.float64)
    k3 = np.empty(20, dtype=np.float64)
    k4 = np.empty(20, dtype=np.float64)
    k5 = np.empty(20, dtype=np.float64)
    k6 = np.empty(20, dtype=np.float64)
    xt = np.empty(20, dtype=np.float64)
    x_next = np.empty(20, dtype=np.float64)

    for step in range(n_steps):
        t = step * h_int
        if method_flag == METHOD_EULER:
            _rhs_reference(t, x, b_noise, h1, h_step, freq, k_reference, saw_style, k1)
            for j in range(20):
                x_next[j] = x[j] + h_int * k1[j]
        else:
            _rhs_reference(t, x, b_noise, h1, h_step, freq, k_reference, saw_style, k1)
            for j in range(20):
                xt[j] = x[j] + h_int * (k1[j] / 4.0)
            _rhs_reference(t + h_int / 4.0, xt, b_noise, h1, h_step, freq, k_reference, saw_style, k2)
            for j in range(20):
                xt[j] = x[j] + h_int * (3.0 * k1[j] / 32.0 + 9.0 * k2[j] / 32.0)
            _rhs_reference(t + 3.0 * h_int / 8.0, xt, b_noise, h1, h_step, freq, k_reference, saw_style, k3)
            for j in range(20):
                xt[j] = x[j] + h_int * (1932.0 * k1[j] / 2197.0 - 7200.0 * k2[j] / 2197.0 + 7296.0 * k3[j] / 2197.0)
            _rhs_reference(t + 12.0 * h_int / 13.0, xt, b_noise, h1, h_step, freq, k_reference, saw_style, k4)
            for j in range(20):
                xt[j] = x[j] + h_int * (439.0 * k1[j] / 216.0 - 8.0 * k2[j] + 3680.0 * k3[j] / 513.0 - 845.0 * k4[j] / 4104.0)
            _rhs_reference(t + h_int, xt, b_noise, h1, h_step, freq, k_reference, saw_style, k5)
            for j in range(20):
                xt[j] = x[j] + h_int * (-8.0 * k1[j] / 27.0 + 2.0 * k2[j] - 3544.0 * k3[j] / 2565.0 + 1859.0 * k4[j] / 4104.0 - 11.0 * k5[j] / 40.0)
            _rhs_reference(t + h_int / 2.0, xt, b_noise, h1, h_step, freq, k_reference, saw_style, k6)
            for j in range(20):
                x_next[j] = x[j] + h_int * (
                    16.0 * k1[j] / 135.0
                    + 6656.0 * k3[j] / 12825.0
                    + 28561.0 * k4[j] / 56430.0
                    - 9.0 * k5[j] / 50.0
                    + 2.0 * k6[j] / 55.0
                )

        if _is_bad_state(x_next, abort_limit):
            for r in range(step + 1, n_steps + 1):
                for c in range(6):
                    e_hist[r, c] = np.nan
            return e_hist

        for j in range(20):
            x[j] = x_next[j]
        _fill_article_error(x, (step + 1) * h_int, b_noise, h_step, h1, e_tmp)
        for j in range(6):
            e_hist[step + 1, j] = e_tmp[j]

    return e_hist
