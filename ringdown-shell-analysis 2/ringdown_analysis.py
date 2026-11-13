"""
RINGDOWN OF A BLACK HOLE SURROUNDED BY A THIN SHELL OF MATTER
Reanalysis of the public data of Laeuger, Weller, Li, Chen, arXiv:2506.00367
(data: Zenodo, DOI 10.5281/zenodo.15586889)

For three shell configurations the program:
  1. FIXED-MODE FIT  - fits the ringdown with damped sinusoids of FIXED frequency
                       (vacuum Schwarzschild modes, or the shell modes in the data files)
                       and computes the mismatch;
  2. FREE FIT        - fits K damped sinusoids with FREE frequency and damping and checks
                       which set of modes they are closest to;
  3. BIAS TEST       - repeats the free fit on a vacuum-only synthetic signal: any
                       "shell-like" result there is an artifact of the method;
  4. LIGO NOISE TEST - adds colored noise from the Advanced LIGO design sensitivity (PyCBC)
                       at a given SNR and checks whether the shell modes still fit better
                       than the vacuum modes (noise-weighted chi^2 fits, for a black
                       hole of M_PHYS solar masses). The paper itself has no detector noise.
Output: plots/*.png and a LaTeX report, report/ringdown_report.tex, with all tables and figures.
        ringdown_report.zip can be uploaded as it is to Overleaf (New Project -> Upload Project);
        if pdflatex is installed the PDF is compiled too.

Units: the data use M = 0.5 in code units. Times are given in units of M,
frequencies in code units and as M*omega.

Install:  pip install numpy scipy matplotlib pandas pycbc
Run:      python3 ringdown_analysis.py [data_folder]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2 as chi2_dist

# np.trapezoid exists from numpy 2.0; older versions (e.g. with Python 3.9) call it np.trapz
trapz = getattr(np, "trapezoid", None) or np.trapz

# ------------------------------------------------------------------ settings
M = 0.5                                  # black-hole mass in code units
CASES = {                                # dM, R in code units; window in units of M after the peak
    "Case 1": dict(dM="0.25", R="3.0", window=(10, 200), label="heavy, close shell (dM = M/2, R = 6M)"),
    "Case 2": dict(dM="0.005", R="15.0", window=(10, 135), label="light, distant shell (dM = M/100, R = 30M)"),
    "Case 3": dict(dM="0.25", R="15.0", window=(10, 135), label="heavy, distant shell (dM = M/2, R = 30M) - echo"),
}
PAPER_WINDOW = (10, 100)                 # window of the paper's mismatch, units of M
N_MODES = [1, 2, 3, 4]                   # modes in the fixed-mode fits
K_FREE = 2                               # damped sinusoids in the free fit
SNR_LIST = [10, 30, 100, 300, 1000]
N_NOISE = 100                            # noise realizations per SNR
RNG = np.random.default_rng(54321)
# LIGO noise test: the simulation is in units of M, so a physical mass is needed to go to seconds
M_PHYS = 60                              # solar masses (about the remnant of GW150914)
F_LOW = 20.0                             # Hz, low-frequency cutoff of the detector
MSUN_S = 4.925491e-6                     # G·M_sun/c³ in seconds
SOGLIA_LIGO = 0.9                        # "distinguishable": shell modes better in at least 90% of realizations
N_FFT = 2 ** 14                          # FFT length for the noise-weighted inner product

# Schwarzschild l=2 QNMs, M*omega, n = 0..3 (Berti, Cardoso & Starinets, CQG 26 (2009) 163001)
VACUUM_M_OMEGA = np.array([0.373672 - 0.088962j, 0.346711 - 0.273915j,
                           0.301053 - 0.478277j, 0.251505 - 0.705148j])


# ------------------------------------------------------------------ data
def load_case(data, dM, R):
    """Waveform (real part), time of the peak and the shell QNMs listed in the data files."""
    w = np.load(data / f"infall_waveform_coupled_M0.5_deltaM{dM}_ell2_vs0.1_rshell{R}_rstart5M.npz")
    t, h = w["t_vals"], w["waveform_no_filter"].real
    shell = np.load(data / f"PolarQNMlist_M0.5_deltaM{dM}_ell2_vs0.1_rshell{R}.npz")["QNMs"]
    return t, h, t[np.argmax(np.abs(h))], shell


def vacuum_modes(n):
    """First n vacuum modes in code units."""
    return VACUUM_M_OMEGA[:n] / M


def alpha(dM, R):
    """Redshift factor between the inside of the shell and infinity (eq. 17 of the paper)."""
    dM, R = float(dM), float(R)
    return np.sqrt((1 - 2 * M / R) / (1 - 2 * (M + dM) / R))


def least_damped(modes, n):
    """The n least damped modes (the paper does not say which shell modes it used)."""
    return modes[np.argsort(np.abs(modes.imag))][:n]


# ------------------------------------------------------------------ fits
def basis(tt, modes):
    """Each mode w = omega - i*gamma gives two real functions: e^(-gamma t) cos(omega t), e^(-gamma t) sin(omega t)."""
    return np.column_stack([f(w.real * tt) * np.exp(w.imag * tt) for w in modes for f in (np.cos, np.sin)])


def fit_fixed(t, h, modes, t0, t1):
    """Fixed frequencies, free amplitudes: linear least squares on [t0, t1].
    Returns the model on the whole time axis (zero before t0)."""
    win = (t >= t0) & (t <= t1)
    amp = np.linalg.lstsq(basis(t[win] - t0, modes), h[win], rcond=None)[0]
    model = np.zeros_like(h)
    model[t >= t0] = basis(t[t >= t0] - t0, modes) @ amp
    return model


def mismatch(t, h, model, t0, t1):
    """1 - <h,model>/sqrt(<h,h><model,model>), with <a,b> = integral of a*b on [t0, t1] (eqs. 30-31)."""
    win = (t >= t0) & (t <= t1)
    ip = lambda a, b: trapz(a[win] * b[win], t[win])
    return 1 - ip(h, model) / np.sqrt(ip(h, h) * ip(model, model))


def ringdown(t, *p):
    """Sum of damped sinusoids A e^(-gamma t) cos(omega t + phi); p = (A, omega, gamma, phi) * K."""
    return sum(A * np.exp(-g * t) * np.cos(w * t + phi) for A, w, g, phi in np.reshape(p, (-1, 4)))


def fit_free(t, h, t0, t1, guesses):
    """Free frequencies and damping rates. Tries every starting point in `guesses`
    (a list of arrays of complex frequencies) and keeps the best fit.
    Returns parameters and errors (rows: A, omega, gamma, phi; strongest mode first) and the model."""
    win = (t >= t0) & (t <= t1)
    tt, y = t[win] - t0, h[win]
    scale = np.max(np.abs(y))
    best = None
    for g in guesses:
        p0 = np.ravel([[scale / len(g), w.real, -w.imag, 0.0] for w in g])
        bounds = ([-np.inf, 0.01, 0, -np.inf] * len(g), [np.inf, 5, 5, np.inf] * len(g))
        try:
            p, cov = curve_fit(ringdown, tt, y / scale, p0=p0, bounds=bounds, x_scale="jac", max_nfev=20000)
        except RuntimeError:
            continue
        chi2 = np.sum((ringdown(tt, *p) - y / scale) ** 2)
        if best is None or chi2 < best[2]:
            best = (p, cov, chi2)
    p, cov, _ = best
    p, err = p.reshape(-1, 4), np.sqrt(np.abs(np.diag(cov))).reshape(-1, 4)
    p[:, 0] *= scale
    err[:, 0] *= scale
    model = np.zeros_like(h)
    model[t >= t0] = ringdown(t[t >= t0] - t0, *p.ravel())
    order = np.argsort(-np.abs(p[:, 0]))
    return p[order], err[order], model


def distance(w, modes):
    """Distance in the complex plane from w to the closest mode of a set."""
    return np.min(np.abs(modes - w))


# ------------------------------------------------------------------ LIGO noise (PyCBC)
def ligo_tools(dt_s):
    """Advanced LIGO design PSD (zero-detuned high power) from PyCBC, a colored-noise generator and the
    noise-weighted transform: with X = whiten(x), the inner product is <x,y> = 4·df·Re Σ X·conj(Y),
    i.e. 4 Re ∫ x~(f) y~*(f) / S(f) df above F_LOW."""
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.noise import noise_from_psd
    df = 1 / (N_FFT * dt_s)
    psd = aLIGOZeroDetHighPower(N_FFT // 2 + 1, df, F_LOW)
    S, f = psd.numpy(), psd.sample_frequencies.numpy()
    w = np.zeros_like(S)
    ok = (f >= F_LOW) & (S > 0)
    w[ok] = np.sqrt(4 * df / S[ok])

    def whiten(x):                                   # works on a vector or on the columns of a matrix
        return np.fft.rfft(x, N_FFT, axis=0) * dt_s * (w if np.ndim(x) == 1 else w[:, None])

    def noise(n, seed):
        return noise_from_psd(n, dt_s, psd, seed=seed).numpy()
    return whiten, noise


def inner(X, Y):
    return np.real(np.vdot(Y, X)) if np.ndim(X) == 1 else np.real(X.conj().T @ Y)


def fit_fixed_ligo(tt, d, modes, whiten):
    """Noise-weighted least squares: minimizes <d - model, d - model> with the LIGO inner product."""
    B = basis(tt, modes)
    WB, Wd = whiten(B), whiten(d)
    amp = np.linalg.lstsq(inner(WB, WB), inner(WB, Wd), rcond=None)[0]
    return B @ amp


def chi2_ligo(d, model, whiten):
    """chi^2 = <d - model, d - model>: noise-weighted sum of squared residuals."""
    R = whiten(d - model)
    return inner(R, R)


def n_eff(n, whiten, noise, draws=300):
    """Effective number of data points: the mean of <n,n> over pure-noise realizations
    (for unit-normalized Gaussian noise E[chi^2] equals the number of independent points)."""
    return float(np.mean([inner(whiten(x), whiten(x)) for x in (noise(n, int(RNG.integers(1 << 31)))
                                                                    for _ in range(draws))]))


def mismatch_ligo(d, model, whiten):
    D, Mo = whiten(d), whiten(model)
    return 1 - inner(D, Mo) / np.sqrt(inner(D, D) * inner(Mo, Mo))


# ------------------------------------------------------------------ analysis
def analyse(data):
    tab = {k: [] for k in ["Cases", "Modes", "Fixed-mode fits", "Free-frequency fit", "Bias test",
                           "LIGO noise test"]}
    plots = []
    Path("plots").mkdir(exist_ok=True)

    for name, c in CASES.items():
        t, h, tp, shell = load_case(data, c["dM"], c["R"])
        al = alpha(c["dM"], c["R"])
        t0, t1 = tp + c["window"][0] * M, tp + c["window"][1] * M
        p0, p1 = tp + PAPER_WINDOW[0] * M, tp + PAPER_WINDOW[1] * M
        sets = {"vacuum": vacuum_modes(4), "shell (as listed)": shell, "shell / alpha": shell / al}
        print(f"\n=== {name}: {c['label']}   alpha = {al:.4f}")
        tab["Cases"].append(dict(Case=name, Description=c["label"], **{"dM / M": float(c["dM"]) / M,
                                 "R / M": float(c["R"]) / M, "alpha": al, "Peak time [M]": tp / M,
                                 "Window [M]": f"{c['window'][0]}-{c['window'][1]}",
                                 "Shell modes listed": len(shell)}))
        for s, modes in sets.items():
            for k, w in enumerate(modes):
                tab["Modes"].append({"Case": name, "Set": s, "Mode": k, "Re ω (code)": w.real,
                                     "Im ω (code)": w.imag, "M·Re ω": M * w.real, "M·Im ω": M * w.imag})

        # 1. fixed-mode fits
        last = {}
        for n in N_MODES:
            for s, modes in sets.items():
                use = modes[:n] if s == "vacuum" else least_damped(modes, n)
                model = fit_fixed(t, h, use, t0, t1)
                mm = mismatch(t, h, model, t0, t1)
                tab["Fixed-mode fits"].append({"Case": name, "Mode set": s, "N requested": n, "N used": len(use),
                                               "Mismatch (fit window)": mm,
                                               "Mismatch (10M-100M)": mismatch(t, h, model, p0, p1)})
                last[s] = (model, mm)
            print(f"  N = {n}: " + "  ".join(f"{r['Mode set']} {r['Mismatch (fit window)']:.2e}"
                                             for r in tab["Fixed-mode fits"][-3:]))

        # 2. free fit, from several starting points
        v1 = vacuum_modes(1)
        guesses = [vacuum_modes(K_FREE),
                   np.concatenate([v1, least_damped(shell, K_FREE - 1)]),
                   least_damped(shell, K_FREE) if len(shell) >= K_FREE else vacuum_modes(K_FREE),
                   np.concatenate([v1, least_damped(shell, K_FREE - 1) / al])]
        p, e, free_model = fit_free(t, h, t0, t1, guesses)
        mm_free = mismatch(t, h, free_model, t0, t1)
        for k, ((A, w, g, _), (sA, sw, sg, _)) in enumerate(zip(p, e)):
            d = {s: distance(w - 1j * g, modes) for s, modes in sets.items()}
            tab["Free-frequency fit"].append({"Case": name, "Mode": k + 1, "ω (code)": w, "σ ω": sw,
                                              "γ (code)": g, "σ γ": sg, "|A|": abs(A), "σ A": sA,
                                              "M·ω": M * w, "M·γ": M * g,
                                              **{f"Distance to {s}": v for s, v in d.items()},
                                              "Closest set": min(d, key=d.get), "Mismatch of free fit": mm_free})
            print(f"  free mode {k + 1}: omega = {w:.4f} ± {sw:.4f}, gamma = {g:.4f} ± {sg:.4f}"
                  f"  -> closest: {min(d, key=d.get)}")

        # 3. bias test: same free fit on a signal made only of vacuum modes
        pb, eb, _ = fit_free(t, fit_fixed(t, h, vacuum_modes(4), t0, t1), t0, t1, guesses)
        for k, ((_, w, g, _), (_, sw, sg, _)) in enumerate(zip(pb, eb)):
            v = vacuum_modes(4)[np.argmin(np.abs(vacuum_modes(4) - (w - 1j * g)))]
            tab["Bias test"].append({"Case": name, "Mode": k + 1, "ω fitted": w, "σ ω": sw, "γ fitted": g,
                                     "σ γ": sg, "ω nearest vacuum": v.real, "γ nearest vacuum": -v.imag,
                                     "Distance": abs(v - (w - 1j * g))})

        # 4. LIGO noise test: vacuum vs shell modes with Advanced LIGO colored noise, in physical units
        n = max(N_MODES)
        vac, sh = vacuum_modes(n), least_damped(shell, n) / al
        m_s = M_PHYS * MSUN_S                          # 1 M in seconds
        win = (t >= t0) & (t <= t1)
        tt_s = (t[win] - t0) / M * m_s                 # time in seconds from the start of the window
        dt_s = tt_s[1] - tt_s[0]
        whiten, noise = ligo_tools(dt_s)
        hw = h[win]
        rho = np.sqrt(inner(whiten(hw), whiten(hw)))   # optimal SNR of the unscaled waveform
        vac_s, sh_s = vac * M / m_s, sh * M / m_s      # frequencies in rad/s
        neff = n_eff(len(hw), whiten, noise)           # effective number of points of the noise-weighted fit
        nd_v, nd_s = neff - 2 * len(vac_s), neff - 2 * len(sh_s)   # degrees of freedom (2 amplitudes per mode)
        for snr in SNR_LIST:
            hs = hw * snr / rho                        # signal rescaled to the requested SNR
            diff, cv, cs = [], [], []
            for _ in range(N_NOISE):
                d = hs + noise(len(hs), int(RNG.integers(1 << 31)))
                mv, ms = fit_fixed_ligo(tt_s, d, vac_s, whiten), fit_fixed_ligo(tt_s, d, sh_s, whiten)
                diff.append(mismatch_ligo(d, mv, whiten) - mismatch_ligo(d, ms, whiten))
                cv.append(chi2_ligo(d, mv, whiten))
                cs.append(chi2_ligo(d, ms, whiten))
            diff, cv, cs = np.array(diff), np.array(cv), np.array(cs)
            dchi = cv - cs                             # > 0: the shell modes fit better
            frac = np.mean(dchi > 0)                   # shell preferred: lower chi^2 of the fit
            tab["LIGO noise test"].append({"Case": name, "SNR": snr, "M (M_sun)": M_PHYS,
                                           "f of vacuum n=0 (Hz)": vac[0].real * M / m_s / (2 * np.pi),
                                           "Window (ms)": 1e3 * tt_s[-1],
                                           "Mean (mm vac - mm shell)": diff.mean(), "Std": diff.std(),
                                           "Fraction shell better": frac,
                                           "σ fraction (binomial)": np.sqrt(frac * (1 - frac) / N_NOISE),
                                           "N_eff": neff,
                                           "χ²/ndof vacuum": cv.mean() / nd_v,
                                           "p vacuum": float(chi2_dist.sf(cv.mean(), nd_v)),
                                           "χ²/ndof shell": cs.mean() / nd_s,
                                           "p shell": float(chi2_dist.sf(cs.mean(), nd_s)),
                                           "Δχ² mean": dchi.mean(), "Δχ² std": dchi.std(),
                                           "Fraction lower mismatch": np.mean(diff > 0)})
        print(f"  LIGO noise ({M_PHYS} M_sun, f = {tab['LIGO noise test'][-1]['f of vacuum n=0 (Hz)']:.0f} Hz): "
              + ", ".join(f"SNR {r['SNR']}: shell better in {100 * r['Fraction shell better']:.0f}%"
                          for r in tab["LIGO noise test"][-len(SNR_LIST):]))

        plots += make_plots(name, c["label"], t, h, tp, t0, t1, last, free_model, mm_free, tab)
    return tab, plots


# ------------------------------------------------------------------ plots
def make_plots(name, label, t, h, tp, t0, t1, last, free_model, mm_free, tab):
    stem = "plots/" + name.replace(" ", "_").lower()
    sel = (t >= t0 - 5 * M) & (t <= t1)
    x = (t[sel] - tp) / M

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, h[sel], color="black", lw=1.6, label="signal")
    for (s, (model, mm)), col in zip(last.items(), ["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(x, model[sel], lw=1, color=col, label=f"{s} ({mm:.1e})")
    ax.plot(x, free_model[sel], "--", color="red", lw=1, label=f"free fit ({mm_free:.1e})")
    ax.set(xlabel="t - t_peak [M]", ylabel="h (real part)", title=f"{name}: {label}")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(stem + "_fits.png", dpi=130)

    fx = pd.DataFrame(tab["Fixed-mode fits"]).query("Case == @name")
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, mk in [("vacuum", "o"), ("shell (as listed)", "s"), ("shell / alpha", "^")]:
        r = fx[fx["Mode set"] == s]
        ax.plot(r["N requested"], r["Mismatch (fit window)"], "-", color="black", lw=1)
        ax.plot(r["N requested"], r["Mismatch (fit window)"], mk, color="red", label=s)
    ax.set(yscale="log", xlabel="number of modes N", ylabel="mismatch (fit window)",
           title=f"{name}: mismatch vs number of modes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(stem + "_mismatch.png", dpi=130)

    fig, ax = plt.subplots(figsize=(6, 4))
    nz = pd.DataFrame(tab["LIGO noise test"]).query("Case == @name")
    ax.plot(nz["SNR"], 100 * nz["Fraction shell better"], "-", color="black", lw=1)
    ax.plot(nz["SNR"], 100 * nz["Fraction shell better"], "o", color="red")
    ax.axhline(50, color="gray", ls="--", lw=0.8)
    ax.set(xscale="log", ylim=(-5, 105), xlabel="SNR", ylabel="shell fit better than vacuum (%)",
           title=f"{name}: Advanced LIGO noise ({M_PHYS} M☉)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(stem + "_noise.png", dpi=130)
    plt.close("all")
    return [stem + "_fits.png", stem + "_mismatch.png", stem + "_noise.png"]


# ------------------------------------------------------------------ Excel
SET_TEX = {"vacuum": "vacuum", "shell (as listed)": "shell (as listed)", "shell / alpha": r"shell / $\alpha$"}


def tex(x):
    """Plain text -> LaTeX-safe text."""
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("≥", r"$\geq$")]:
        x = x.replace(a, b)
    return x


def sci(x, d=2):
    """1.08e-04 -> $1.08\times10^{-4}$"""
    m, e = f"{x:.{d}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$" if int(e) != 0 else f"${m}$"


def table(header, rows, caption, align=None):
    align = align or "l" + "c" * (len(header) - 1)
    out = [r"\begin{table}[H]", r"\centering\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    out += [" & ".join(map(str, r)) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", r"\end{table}", ""]
    return "\n".join(out)


def figure(files, caption, width):
    out = [r"\begin{figure}[H]", r"\centering"]
    out += [rf"\includegraphics[width={width}\linewidth]{{{f}}}" for f in files]
    out += [rf"\caption{{{caption}}}", r"\end{figure}", ""]
    return "\n".join(out)


def write_report(tab, folder="report"):
    """Writes report/ringdown_report.tex with all tables and figures, copies the plots next to it,
    zips the folder for Overleaf (New Project -> Upload Project) and, if pdflatex is installed,
    compiles the PDF."""
    import shutil
    import subprocess
    out = Path(folder)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree("plots", out / "plots")
    fx, fr, bs, lg, cs = (pd.DataFrame(tab[k]) for k in
                          ["Fixed-mode fits", "Free-frequency fit", "Bias test", "LIGO noise test", "Cases"])
    n = max(N_MODES)
    L = [r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,graphicx,booktabs,float,hyperref}
\title{Ringdown of a black hole surrounded by a thin shell of matter:\\ a reanalysis with Advanced LIGO noise}
\author{Darin La Rocca\\ \small University of Milano-Bicocca}
\date{\today}
\begin{document}
\maketitle

\section{Goal}
This note reanalyses the public waveforms of Laeuger, Weller, Li \& Chen, \emph{The ringdown of a black hole
surrounded by a thin shell of matter}, arXiv:2506.00367 (data: Zenodo, DOI 10.5281/zenodo.15586889).
The question is whether a fit of the ringdown can tell a black hole with a shell of matter apart from a
black hole in vacuum, and whether this still holds with realistic detector noise. The paper itself does not
include detector noise; the last test (Advanced LIGO noise) is an extension of this work.

\section{Data and method}
Three shell configurations are studied (Table~\ref{tab:cases}). The code uses $M=0.5$; below, times are in
units of $M$ and frequencies are given as $M\omega$. Each fit uses a window that starts $10M$ after the peak
of the signal.
"""]
    L.append(table(["Case", "Shell", r"$\delta M/M$", "$R/M$", r"$\alpha$", "Window [$M$]", "Shell modes"],
                   [[r["Case"], tex(r["Description"]), f"{r['dM / M']:g}", f"{r['R / M']:g}", f"{r['alpha']:.4f}",
                     r["Window [M]"], r["Shell modes listed"]] for _, r in cs.iterrows()],
                   r"Shell configurations. $\alpha=\sqrt{(1-2M/R)/(1-2(M+\delta M)/R)}$ is the redshift factor "
                   r"between the inside of the shell and infinity (eq.~17 of the paper).\label{tab:cases}",
                   "llccccc").replace(r"\small", r"\footnotesize"))
    L.append(r"""
\paragraph{1. Fixed-mode fits.} The signal is fitted with $N=1,\dots,""" + str(n) + r"""$ damped sinusoids
whose complex frequencies $\omega_k$ are fixed,
\begin{equation}
h(t)\simeq\sum_k e^{\mathrm{Im}\,\omega_k t}\left[a_k\cos(\mathrm{Re}\,\omega_k t)+b_k\sin(\mathrm{Re}\,\omega_k t)\right],
\end{equation}
and the amplitudes $a_k,b_k$ are found by linear least squares. Three sets of frequencies are compared:
the Schwarzschild $\ell=2$ modes $n=0,\dots,3$ (Berti, Cardoso \& Starinets 2009), the shell modes as listed in
the data files, and the shell modes divided by $\alpha$. As in the paper, at most four modes are used, the
least damped ones for the shell. The quality of a fit is the mismatch (eqs.~30--31 of the paper)
\begin{equation}
\mathcal{M}=1-\frac{\langle h,h_\mathrm{fit}\rangle}{\sqrt{\langle h,h\rangle\langle h_\mathrm{fit},h_\mathrm{fit}\rangle}},
\qquad \langle a,b\rangle=\int_{t_0}^{t_1}a\,b\,dt .
\end{equation}

\paragraph{2. Free fit.} """ + str(K_FREE) + r""" damped sinusoids $A\,e^{-\gamma t}\cos(\omega t+\varphi)$ with
free $\omega$ and $\gamma$ are fitted (several starting points, best fit kept). Each fitted frequency
$\omega-i\gamma$ is compared with the closest mode of each set.

\paragraph{3. Bias test.} The same free fit is repeated on a synthetic signal built only from vacuum modes.
Any deviation from the vacuum modes found here is produced by the method, not by the shell.

\paragraph{4. Advanced LIGO noise.} To use detector noise, a physical mass is needed: $M=""" + str(M_PHYS)
             + r"""\,M_\odot$, so that $1M=""" + f"{M_PHYS * MSUN_S * 1e3:.2f}" + r"""$\,ms and the fundamental vacuum
mode is at $""" + f"{lg['f of vacuum n=0 (Hz)'].iloc[0]:.0f}" + r"""$\,Hz. Gaussian noise is drawn from the Advanced LIGO
design sensitivity (PyCBC, \texttt{aLIGOZeroDetHighPower}, $f>""" + f"{F_LOW:g}" + r"""$\,Hz). The signal on the
fit window is rescaled to an optimal SNR $\rho=\sqrt{\langle h,h\rangle}$, and both the fits and the mismatch
use the noise-weighted inner product
\begin{equation}
\langle a,b\rangle=4\,\mathrm{Re}\int_{f_\mathrm{low}}^{\infty}\frac{\tilde a(f)\,\tilde b^*(f)}{S_n(f)}\,df .
\end{equation}
For each SNR, """ + str(N_NOISE) + r""" noise realizations are generated and the fraction in which the shell
modes (divided by $\alpha$) give a lower $\chi^2$ than the vacuum modes is recorded (the fraction with a lower
noise-weighted mismatch is also saved, and gives the same result).

\section{Results}
\subsection{Fixed-mode fits}
""")
    rows = []
    for name in CASES:
        r = fx[(fx.Case == name) & (fx["N requested"] == n)].set_index("Mode set")
        best = r["Mismatch (fit window)"].idxmin()
        rows.append([name] + [(r"\textbf{%s}" if s == best else "%s") % sci(r.loc[s, "Mismatch (fit window)"])
                              + f" ({r.loc[s, 'N used']})" for s in SET_TEX]
                    + [sci(r.loc[s, "Mismatch (10M-100M)"]) for s in ["vacuum", "shell / alpha"]])
    L.append(table(["Case"] + [SET_TEX[s] for s in SET_TEX] + [r"vacuum, $10$--$100M$", r"shell / $\alpha$, $10$--$100M$"],
                   rows, rf"Mismatch with $N={n}$ requested modes (number actually used in brackets) on the fit window, "
                   r"and on the window of the paper ($10M$--$100M$). Best set in bold."))
    L.append(figure([f"plots/case_{i}_mismatch.png" for i in (1, 2, 3)],
                    "Mismatch as a function of the number of modes.", 0.32))
    L.append(figure([f"plots/case_{i}_fits.png" for i in (1, 2, 3)],
                    rf"Signal and best fits with $N={n}$ modes (mismatch in the legend).", 0.85))
    L.append(r"\subsection{Free fit and bias test}" + "\n")
    L.append(table(["Case", "Mode", r"$M\omega$", r"$M\gamma$", "Closest set", r"$\mathcal{M}$ of free fit"],
                   [[r["Case"], r["Mode"], f"{M * r['ω (code)']:.4f}", f"{M * r['γ (code)']:.4f}",
                     SET_TEX[r["Closest set"]], sci(r["Mismatch of free fit"])] for _, r in fr.iterrows()],
                   "Free fit: fitted frequency and damping, and the closest set of modes."))
    L.append(table(["Case", "Mode", r"$M\omega$ fitted", r"$M\gamma$ fitted", r"$M\omega$ vacuum", r"$M\gamma$ vacuum",
                    r"Distance ($M|\Delta\omega|$)"],
                   [[r["Case"], r["Mode"], f"{M * r['ω fitted']:.4f}", f"{M * r['γ fitted']:.4f}",
                     f"{M * r['ω nearest vacuum']:.4f}", f"{M * r['γ nearest vacuum']:.4f}", f"{M * r['Distance']:.4f}"]
                    for _, r in bs.iterrows()],
                   "Bias test: free fit on a vacuum-only signal, compared with the nearest vacuum mode."))
    L.append(r"\subsection{Advanced LIGO noise}" + "\n" + rf"""For each case and SNR, {N_NOISE} independent
realizations of Advanced LIGO noise are added to the signal and both fits (vacuum modes and shell modes) are
repeated on each of them. Table~\ref{{tab:ligo}} gives the fraction of realizations in which the shell modes give the
lower $\chi^2$. A value of 100\% means that the shell modes always describe the data better, so the shell is
clearly identified; a value around 50\% means that the outcome is random, as for a coin toss, so the two
hypotheses cannot be distinguished; a value below 50\% means that the vacuum modes are preferred. In the summary,
the shell is called distinguishable when its modes win in at least {100 * SOGLIA_LIGO:.0f}\% of the realizations.
""" + "\n")
    snrs = sorted(lg["SNR"].unique())
    L.append(table(["Case"] + [f"SNR {x}" for x in snrs],
                   [[name] + ["{:.0f} $\\pm$ {:.0f}\\%".format(*(100 * lg[(lg.Case == name) & (lg.SNR == x)][
                       ["Fraction shell better", "σ fraction (binomial)"]].iloc[0].values)) for x in snrs]
                    for name in CASES],
                   rf"Fraction of the {N_NOISE} noise realizations in which the shell modes fit better than the "
                   "vacuum modes. Values near 50\\% mean that the two cannot be distinguished.\\label{tab:ligo}"))
    L.append(r"""The fit with fixed modes is a noise-weighted least-squares fit, so its minimum is
$\chi^2=\langle d-h_\mathrm{fit},d-h_\mathrm{fit}\rangle$. The number of degrees of freedom is
$N_\mathrm{eff}-2N_\mathrm{modes}$, where $N_\mathrm{eff}$ is the mean of $\langle n,n\rangle$ over 300 pure-noise
realizations (the effective number of independent points). Table~\ref{tab:chi2} gives, for each SNR, the mean
$\chi^2/\mathrm{ndof}$ of the two fits with its $p$-value, and the difference $\Delta\chi^2=\chi^2_\mathrm{vac}-
\chi^2_\mathrm{shell}$. Since both models have the same number of parameters, $\Delta\chi^2$ is twice the
log-likelihood ratio: positive values favour the shell. The two models are not nested, so no $p$-value is attached
to $\Delta\chi^2$; its mean and spread over the noise realizations are reported instead. The errors on the
fractions are binomial, $\sqrt{f(1-f)/N}$ (zero when $f$ is exactly 0 or 1).
""" + "\n")
    rows = []
    for name in CASES:
        for x in snrs:
            r = lg[(lg.Case == name) & (lg.SNR == x)].iloc[0]
            rows.append([name, x, f"{r['χ²/ndof vacuum']:.2f}", f"{r['p vacuum']:.2g}", f"{r['χ²/ndof shell']:.2f}",
                         f"{r['p shell']:.2g}", f"{r['Δχ² mean']:.1f} $\\pm$ {r['Δχ² std']:.1f}",
                         f"{100 * r['Fraction lower mismatch']:.0f}\\%"])
    L.append(table(["Case", "SNR", r"$\chi^2/$ndof vac.", "$p$ vac.", r"$\chi^2/$ndof shell", "$p$ shell",
                    r"$\Delta\chi^2$ (mean $\pm$ std)", r"lower $\mathcal{M}$"], rows,
                   rf"Goodness of fit with Advanced LIGO noise (means over {N_NOISE} realizations; "
                   rf"$N_\mathrm{{eff}}\approx{lg['N_eff'].mean():.0f}$).\label{{tab:chi2}}"))
    L.append(figure([f"plots/case_{i}_noise.png" for i in (1, 2, 3)],
                    rf"Advanced LIGO noise test ({M_PHYS}\,$M_\odot$).", 0.32))
    L.append(r"\section{Summary}" + "\n" + r"\begin{itemize}")
    for name, c in CASES.items():
        won = [x for x in snrs if lg[(lg.Case == name) & (lg.SNR == x)]["Fraction shell better"].iloc[0] >= SOGLIA_LIGO]
        first = next((x for x in snrs if all(y in won for y in snrs if y >= x)), None)
        r = fx[(fx.Case == name) & (fx["N requested"] == n)].set_index("Mode set")["Mismatch (fit window)"]
        verdict = (f"with Advanced LIGO noise the shell modes win in at least {100 * SOGLIA_LIGO:.0f}\\% of the "
                   f"realizations from SNR {first} on" if first
                   else "with Advanced LIGO noise the shell cannot be told apart from vacuum at any SNR studied")
        if first and snrs.index(first) > 0:
            prev = lg[(lg.Case == name) & (lg.SNR == snrs[snrs.index(first) - 1])].iloc[0]
            verdict += (f" (at SNR {prev['SNR']}: {100 * prev['Fraction shell better']:.0f} $\\pm$ "
                        f"{100 * prev['σ fraction (binomial)']:.0f}\\%)")
        L.append(rf"\item \textbf{{{name}}} ({tex(c['label'])}): best fixed-mode set {SET_TEX[r.idxmin()]}; {verdict}.")
    L.append(r"""\end{itemize}

\section{Open questions}
Which shell modes were used in the paper's fits, and are the listed shell frequencies already as seen from
infinity, or should they be divided by $\alpha$? The results with detector noise depend on the chosen mass,
which moves the ringdown frequency along the sensitivity curve; the waveform is also cut to the fit window,
so this is a simplified version of a real ringdown analysis.

\end{document}
""")
    (out / "ringdown_report.tex").write_text("\n".join(L), encoding="utf-8")
    shutil.make_archive("ringdown_report", "zip", out)
    if shutil.which("pdflatex"):
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "ringdown_report.tex"], cwd=out,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    data = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else here
    if not list(data.glob("infall_waveform_coupled_*.npz")):
        data = data.parent                       # also works from a subfolder of the data folder
    import os
    os.chdir(here)
    tab, _ = analyse(data)
    write_report(tab)
    print("\nReport saved in report/ringdown_report.tex (and .pdf if LaTeX is installed);"
          "\nringdown_report.zip is ready to upload on Overleaf (New Project -> Upload Project).")
