"""probe_error_masking.py — the parity error is masked in integrated quantities.

Two distinct errors are compared on the same footing:

  PARITY error       DP-naive  vs  DP-correct   (𝒲(r) vs 𝒲(r)−𝒲(−r))
                     — a *bug*: wrong branch of the closed form
  TRUNCATION error   DP-correct vs OZ+MSA        (first order vs exact MSA)
                     — the method's *honest* limitation

and evaluated on three observables of increasing integration:

  1. pointwise  c_ij(r)          — no integration
  2. ĉ_ij(0) = 4π∫r²c dr         — weight r²
  3. βF_ex/V                     — compressibility route, ĉ(0) integrated
                                    twice over density

The point: the parity error is 1–2 orders of magnitude *larger* than the
truncation error pointwise, but *smaller* than it in ĉ(0) and F_ex.  An integrated
scalar therefore cannot detect the bug — it hides inside the first-order
truncation error and would be dismissed as such.

Compressibility route (fixed composition x, total density ρ, ρ_i = x_i ρ):

    d²(βf_ex)/dρ² = −Ψ(ρ),      Ψ(ρ) = Σ_ij x_i x_j ĉ_ij(0; ρ)
    ⇒  βf_ex(ρ) = −∫₀^ρ (ρ − ρ'')·Ψ(ρ'') dρ''      (f_ex(0)=f_ex'(0)=0)
"""

from __future__ import annotations
import sys
import warnings
import io
import contextlib
import numpy as np

try:                      # numpy >= 2.0 renamed trapz -> trapezoid
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── uniform figure typography: everything at 18 pt ───────────────────────────
plt.rcParams.update({
    "font.size":         18,
    "axes.titlesize":    18,
    "axes.labelsize":    18,
    "xtick.labelsize":   18,
    "ytick.labelsize":   18,
    "legend.fontsize":   18,
    "figure.titlesize":  18,
})

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

from fmsa_double_prop import FMSA_double_prop, px_eval
from fmsa_ga_matrix_mix import FMSA_GA_matrix_mix
from oz_numerical import OZSolverMSA

SIGS = np.array([1.0, 2.0])
EPS  = np.array([1.0, 0.5])
X    = np.array([0.25, 0.75])
T    = 1.0
RHO0 = 0.139
PAIRS  = [(0, 0), (0, 1), (1, 1)]
LABELS = ["(1,1)", "(1,2)", "(2,2)"]

def dp_curves(dp, r):
    """(c_correct, c_naive) totals on r, hs_ref='py' (the consistent reference)."""
    N = dp.N
    cc = np.zeros((N, N, len(r)))
    cn = np.zeros((N, N, len(r)))
    for i in range(N):
        for j in range(N):
            W = dp.closed_form_pieces(i, j)
            pref = 2.0 * np.pi * np.sqrt(dp.rho[i] * dp.rho[j])
            R = float(dp.R_ij[i, j])
            Wp = np.array([px_eval(W, float(x)) for x in r])
            Wm = np.array([px_eval(W, -float(x)) for x in r])
            c1c, c1n = (Wp - Wm) / (pref * r), Wp / (pref * r)
            hs = np.array([dp.get_HS_PY(float(x), i, j) if x < R else 0.0 for x in r])
            tail = np.zeros_like(r)
            out = r >= R
            for t in range(dp.n_tails):
                z = float(dp.z_poles[t, i, j]); K = float(dp.K[t, i, j])
                tail[out] += K * np.exp(-z * (r[out] - R))
            tail[out] /= (pref * r[out])
            cc[i, j] = np.where(r < R, hs + c1c, tail)
            cn[i, j] = np.where(r < R, hs + c1n, tail)
    return cc, cn


def chat0_from(r, c):
    return float(4.0 * np.pi * _trapz(r**2 * c, r))


def oz_solve(rho_star):
    """Converged OZ+MSA solver at this density (None if it fails)."""
    ga = FMSA_GA_matrix_mix(SIGS, EPS, X, rho_star, T, n_tails=2, verbose=False)
    s = OZSolverMSA(ga, n_r=1024)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        try:
            s.solve(method='diis', closure='msa', tol=1e-6, max_iter=300,
                    alpha=0.2, n_coupling_steps=3)
        except Exception:
            return None
    return s if s._c_converged is not None else None


def oz_chat0(rho_star):
    s = oz_solve(rho_star)
    if s is None:
        return None
    return np.array([[chat0_from(s.r, s._c_converged[i, j]) for j in range(2)]
                     for i in range(2)])


def psi(chat):
    """Ψ = Σ_ij x_i x_j ĉ_ij(0)."""
    return float(np.sum(np.outer(X, X) * chat))


def f_ex(rhos, psis):
    """βf_ex(ρ) = −∫₀^ρ (ρ−ρ'')Ψ(ρ'')dρ'' on the grid."""
    out = np.zeros_like(rhos)
    for n, rho in enumerate(rhos):
        m = rhos <= rho
        out[n] = -_trapz((rho - rhos[m]) * psis[m], rhos[m])
    return out


def run() -> None:
    print("=" * 78)
    print("  Parity error vs first-order truncation error, across three observables")
    print("=" * 78)

    r = np.linspace(0.01, 6.0, 5000)

    # ── 1. pointwise, at the base state point ─────────────────────────
    dp0 = FMSA_double_prop(SIGS, EPS, X, RHO0, T, n_tails=2, verbose=False)
    cc0, cn0 = dp_curves(dp0, r)
    s0 = oz_solve(RHO0)
    if s0 is None:
        raise RuntimeError("OZ+MSA did not converge at the base state point")

    print(f"\n  1. POINTWISE  c_ij(r), inner core (ρ*={RHO0}, T*={T}):")
    print(f"  {'pair':>6}  {'parity |Δc|':>12}  {'truncation |Δc|':>16}  {'ratio':>8}")
    print("  " + "-" * 50)
    pw = {}
    for (i, j), lbl in zip(PAIRS, LABELS):
        R = float(dp0.R_ij[i, j])
        inner = r < R
        e_par = float(np.max(np.abs(cn0[i, j] - cc0[i, j])[inner]))
        c_oz = np.interp(r, s0.r, s0._c_converged[i, j])
        e_tru = float(np.max(np.abs(cc0[i, j] - c_oz)[inner]))
        pw[lbl] = (e_par, e_tru)
        print(f"  {lbl:>6}  {e_par:12.4f}  {e_tru:16.4f}  {e_par/e_tru:8.1f}×")

    # ── 2. ĉ(0) ───────────────────────────────────────────────────────
    print(f"\n  2. ĉ_ij(0):")
    print(f"  {'pair':>6}  {'correct':>9}  {'naive':>9}  {'OZ+MSA':>9}  "
          f"{'parity %':>9}  {'trunc %':>8}")
    print("  " + "-" * 60)
    ch_oz0 = oz_chat0(RHO0)
    ch0 = {}
    for (i, j), lbl in zip(PAIRS, LABELS):
        vc = chat0_from(r, cc0[i, j]); vn = chat0_from(r, cn0[i, j])
        vo = float(ch_oz0[i, j])
        ch0[lbl] = (abs(vn-vc)/abs(vc)*100, abs(vc-vo)/abs(vo)*100)
        print(f"  {lbl:>6}  {vc:9.3f}  {vn:9.3f}  {vo:9.3f}  "
              f"{ch0[lbl][0]:9.2f}  {ch0[lbl][1]:8.2f}")

    # ── 3. F_ex via the compressibility route ─────────────────────────
    print(f"\n  3. βF_ex/V (compressibility route, double ρ-integral of ĉ(0)):")
    rhos = np.linspace(0.005, 0.16, 16)
    psi_c, psi_n, psi_o = [], [], []
    for rs in rhos:
        d = FMSA_double_prop(SIGS, EPS, X, rs, T, n_tails=2, verbose=False)
        cc, cn = dp_curves(d, r)
        psi_c.append(psi(np.array([[chat0_from(r, cc[i, j]) for j in range(2)]
                                   for i in range(2)])))
        psi_n.append(psi(np.array([[chat0_from(r, cn[i, j]) for j in range(2)]
                                   for i in range(2)])))
        o = oz_chat0(rs)
        psi_o.append(psi(o) if o is not None else np.nan)
    psi_c, psi_n, psi_o = map(np.array, (psi_c, psi_n, psi_o))
    F_c, F_n, F_o = f_ex(rhos, psi_c), f_ex(rhos, psi_n), f_ex(rhos, psi_o)

    print(f"  {'ρ*':>7}  {'F correct':>10}  {'F naive':>9}  {'F OZ+MSA':>9}  "
          f"{'parity %':>9}  {'trunc %':>8}")
    print("  " + "-" * 62)
    fex = {}
    for n in range(0, len(rhos), 3):
        pe = abs(F_n[n]-F_c[n])/abs(F_c[n])*100 if abs(F_c[n]) > 1e-12 else 0
        te = abs(F_c[n]-F_o[n])/abs(F_o[n])*100 if abs(F_o[n]) > 1e-12 else 0
        print(f"  {rhos[n]:7.4f}  {F_c[n]:10.4f}  {F_n[n]:9.4f}  {F_o[n]:9.4f}  "
              f"{pe:9.2f}  {te:8.2f}")
    ip = np.argmin(np.abs(rhos - RHO0))
    fex['parity'] = abs(F_n[ip]-F_c[ip])/abs(F_c[ip])*100
    fex['trunc']  = abs(F_c[ip]-F_o[ip])/abs(F_o[ip])*100
    print(f"\n  at ρ*={rhos[ip]:.3f}:  parity {fex['parity']:.2f}%  vs  "
          f"truncation {fex['trunc']:.2f}%  → "
          f"{'MASKED' if fex['parity'] < fex['trunc'] else 'visible'}")
    print("  (βF_ex/V crosses zero near ρ*≈0.11, so relative errors spike there;")
    print("   the figure plots the absolute difference, which is spike-free —")
    print("   parity stays ~4–8× below truncation at every density.)")

    # ── figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22.0, 7.0))
    # ⚠ NO suptitle: this figure is placed in the paper and `figure_split --pub`
    # strips panel titles into the caption, where a suptitle would be duplicated.
    # ⚠⚠ The colours used to be `#c23b22` and `#2a78d6` -- OZ-HNCB's red and
    # OZ+MSA's blue.  Neither series here is a model, so those were two model
    # colours naming things that are not models.  `figure_style.ERR` instead: the
    # truncation error IS FMSA-DP's, so it is FMSA-DP's orange; the parity error
    # is a bug belonging to no method, so it is a dark neutral.
    import figure_style as S
    C_PAR, C_TRU = S.ERR['bug'], S.ERR['method']

    # panel 1: pointwise
    ax = axes[0]
    xpos = np.arange(3); w = 0.36
    ax.bar(xpos - w/2, [pw[l][0] for l in LABELS], w, color=C_PAR, label="parity (bug)")
    ax.bar(xpos + w/2, [pw[l][1] for l in LABELS], w, color=C_TRU, label="truncation (honest)")
    ax.set_yscale("log"); ax.set_xticks(xpos); ax.set_xticklabels(LABELS)
    ax.set_ylabel(r"max $|\Delta c_{ij}(r)|$, inner core")
    ax.set_title("1. pointwise $c_{ij}(r)$ — bug dominates")
    # ⚠ No "16x / 328x / 87x" over the bars.  The caption already states those three
    # ratios, so on the plot they were the same numbers twice -- and on a log axis
    # the bar pair shows the ratio directly, which is the point of the panel.

    # panel 2: chat(0)
    ax = axes[1]
    ax.bar(xpos - w/2, [ch0[l][0] for l in LABELS], w, color=C_PAR, label="parity (bug)")
    ax.bar(xpos + w/2, [ch0[l][1] for l in LABELS], w, color=C_TRU, label="truncation (honest)")
    ax.set_xticks(xpos); ax.set_xticklabels(LABELS)
    ax.set_ylabel(r"relative error in $\hat c_{ij}(0)$  [%]")
    ax.set_title(r"2. $\hat c_{ij}(0)$ ($r^2$ weight) — bug now smaller")

    # panel 3: F_ex
    # absolute error: βF_ex/V crosses zero near ρ*≈0.11, so a *relative* error
    # there is a spurious 0/0 spike — the absolute difference is the honest measure.
    ax = axes[2]
    ax.semilogy(rhos, np.abs(F_n - F_c) + 1e-16, "o-", color=C_PAR, lw=2, ms=4,
                label="parity (bug)")
    ax.semilogy(rhos, np.abs(F_c - F_o) + 1e-16, "s-", color=C_TRU, lw=2, ms=4,
                label="truncation (honest)")
    ax.axvline(RHO0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"$\rho^*$"); ax.set_ylabel(r"$|\Delta(\beta F_{ex}/V)|$")
    ax.set_title(r"3. $F_{ex}$ (double $\rho$-integral) — bug fully masked")

    plt.tight_layout()
    fname = "numerical_notes/results/fmsa_dp_error_masking.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  figure → {fname}")

    # ⭐ DRAW FIRST, THEN PERSIST.  Every number the paper's caption quotes lives
    # here, so the figure and the text cannot drift apart the way a printed-only
    # table can.
    import json
    rec = {'rho_star': RHO0, 'T_star': T, 'pairs': LABELS,
           'pointwise': {l: {'parity': pw[l][0], 'truncation': pw[l][1]} for l in LABELS},
           'chat0_pct': {l: {'parity': ch0[l][0], 'truncation': ch0[l][1]} for l in LABELS},
           'fex_pct_at_rho0': fex,
           'fex_curve': {'rho': list(map(float, rhos)),
                         'correct': list(map(float, F_c)),
                         'naive': list(map(float, F_n)),
                         'ozmsa': list(map(float, F_o))}}
    jname = "numerical_notes/results/fmsa_dp_error_masking.json"
    with open(jname, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"  data   → {jname}")

    # ONE key for three panels.  Each used to carry its own copy of the same two
    # entries; at 2-up print size that is the same six words three times.  Both
    # encodings in one handle -- the bars of (a),(b) and the marked lines of (c).
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import figure_legend
    h = [(Patch(facecolor=c, edgecolor="none"),
          Line2D([], [], color=c, marker=m, ms=6.0, lw=1.8))
         for c, m in ((C_PAR, "o"), (C_TRU, "s"))]
    figure_legend.emit(h, ["parity error (a bug)", "first-order truncation (the method)"],
                       "fmsa_dp_error_masking", out_dir="numerical_notes/results",
                       handlelength=3.4, block_ncol=1,
                       handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)})


if __name__ == "__main__":
    run()
