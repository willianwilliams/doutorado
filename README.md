# Fast Python conversion of `RKF.m` and `euler.m`

This folder converts the two MATLAB scripts to Python with a compiled Numba
core. It uses the article-consistent parameter set selected for the revised
tables:

- `T_e = 1.920 s`
- final window `[1.820, 1.920] s`
- `Delta = 0.1 s`
- Table 1: `epsilon = 3` for `e1`, `e2`, `e3`, `e4`, `e5`, and `e6`
- Table 3: `epsilon = 3` for `e1`, `e2`, `e3`, `e4`, `e5`, and `e6`
- `h_1 = 0.8`
- `theta0_reference = zeros(8)`
- message `"Secure Transmission"` with 19 characters and 133 bits
- bit rate `69 bit/s`
- bundled MATLAB `JAM.mat`, generated as `2*wgn(Tam,35,0)`
- Table 3 gains `k = 0.1, 1, 10, 25, 50, 75, 100, 125, 150, 200, 1000`

It exports:

- `T1_RKF.csv`, `T1_RKF.tex`
- `T1_article_style_RKF.csv`, `T1_article_style_RKF.tex`
- `T1_e5diag_RKF.csv`, `T1_e5diag_RKF.tex`
- `T1_ber_RKF.csv`
- `T3_RKF.csv`, `T3_RKF.tex`
- `T3_ber_RKF.csv`
- `T1_euler.csv`, `T1_euler.tex`
- `T1_article_style_euler.csv`, `T1_article_style_euler.tex`
- `T1_e5diag_euler.csv`, `T1_e5diag_euler.tex`
- `T1_ber_euler.csv`
- `T3_euler.csv`, `T3_euler.tex`
- `T3_ber_euler.csv`
- `message_info.csv`
- `message_bits.csv`
- `message_signal.csv`

## Install

Open a terminal in this folder and run:

```bat
python -m pip install -r requirements.txt
```

Then run:

```bat
python main.py --method both --tables all
```

The first run includes Numba compilation time. Later runs are faster because
Numba caches compiled functions.

## Useful commands

Run only RKF:

```bat
python main.py --method rkf
```

Run only Euler:

```bat
python main.py --method euler
```

Run only Table 3:

```bat
python main.py --method both --tables table3
```

Run a tiny smoke test:

```bat
python main.py --smoke --allow-slow
```

## Exact jamming/noise comparison with MATLAB

The local release ZIP includes `JAM.mat`. By default, the program loads this
file before trying to generate any new jamming/noise realization. If you are
using a GitHub clone where `JAM.mat` is not present, copy it from the delivered
ZIP or pass another MATLAB `JAM.mat` with `--jam`.

If you want to pass another MATLAB `JAM.mat` containing the variable `b_noise`,
use:

```bat
python main.py --method both --jam "C:\path\to\JAM.mat"
```

If no `JAM.mat` is supplied, Python creates `results/JAM.mat` using NumPy's
random generator. That is reproducible inside Python, but it is not bit-for-bit
the same random stream as MATLAB `wgn`.

## Message convention

The configured message is `"Secure Transmission"`:

```text
19 characters * 7 bits/character = 133 bits
T_msg = 133 / 69 = 1.927536232 s
```

The synchronization tables are evaluated up to `T_e = 1.920 s`. The message
therefore lasts slightly longer than the metric horizon, which is intentional
for this revised parameter set.

The `T1_article_style_*.tex` files are additional Table 1 exports formatted
like the manuscript table: Proposed Method versus Method in [85], one table
for RKF5 and one for Euler. Existing `T1_*.tex` files are kept unchanged.

BER is computed with the same convention used in the MATLAB menu project:
`m_hat(t) = m(t) - e5(t)`. The exported BER windows are `[0, 0.4] s`,
`[0.4, T_e] s`, `[T_e - 0.1, T_e] s`, and `[0, T_e] s`.

## Important performance note

Python is fast here only because the numerical core is compiled with Numba.
Pure Python loops would be slower than MATLAB for this problem.
