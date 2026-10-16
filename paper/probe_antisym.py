"""probe_antisym.py — the antisymmetrization step, isolated.

Compares the two branches of the FMSA-DP closed form, and nothing else:

    NAIVE      c_ij(r) = 𝒲_ij(r)                  / (2π√(ρᵢρⱼ)·r)
    CORRECT    c_ij(r) = [𝒲_ij(r) − 𝒲_ij(−r)]     / (2π√(ρᵢρⱼ)·r)

`𝒲 = 𝒬⁻ ⋆ ℬ ⋆ (𝒬⁻)ᵀ` is the raw triple convolution.  It is supported on
[−R_ij, ∞) because the Baxter kernel 𝒬⁻ carries q_ij(−t) on [−R_ij, −λ_ij], i.e.
the convolution spills onto negative separations.  The physical object
𝒞_ij(r) = 2π√(ρᵢρⱼ)·r·c^(1)_ij(r) does not: r·c(r) is ODD (odd r × even c(|r|)),
and the 3-D radial transform is a SINE transform, which pairs only with the odd
part.  So the k-space assembly fixes 𝒲's odd part and leaves its even part
completely unconstrained — antisymmetrizing is the projection onto the physical
(odd / one-sided) sector, not a choice.

Diagnostic signature the naive branch produces, and why it is easy to miss:
outside the core 𝒲(−r) = 0, so the two branches coincide exactly (a one-sided
function has equal even and odd parts); inside the core they differ, with the
error growing toward r → 0 where the reflected image overlaps most.
"""

from __future__ import annotations
import sys
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

SIGS = np.array([1.0, 2.0])
EPS  = np.array([1.0, 0.5])
X    = np.array([0.25, 0.75])
RHO, T = 0.139, 1.0
PAIRS  = [(0, 0), (0, 1), (1, 1)]
LABELS = ["(1,1)", "(1,2)", "(2,2)"]

def branches(dp, i, j, r):
    """(correct, naive) c^(1)_ij(r) arrays."""
    W = dp.closed_form_pieces(i, j)
    pref = 2.0 * np.pi * np.sqrt(dp.rho[i] * dp.rho[j])
    Wp = np.array([px_eval(W, float(x)) for x in r])
    Wm = np.array([px_eval(W, -float(x)) for x in r])
    return (Wp - Wm) / (pref * r), Wp / (pref * r)


def run() -> None:
    print("=" * 76)
    print("  Antisymmetrization isolated:  𝒞 = 𝒲(r) − 𝒲(−r)   vs   naive 𝒞 = 𝒲(r)")
    print("=" * 76)

    dp = FMSA_double_prop(SIGS, EPS, X, RHO, T, n_tails=2, verbose=False)

    # ── the decisive anchor: N=1 reduction to the Tang closed form ─────
    from FMSA_pure import FMSA_pure
    dp1 = FMSA_double_prop(np.array([1.0]), np.array([1.0]), np.array([1.0]),
                           0.30, 1.5, n_tails=2, input_type='bh', verbose=False)
    pure = FMSA_pure(1.0, 1.0, 0.30, 1.5, input_type='bh')
    rr = np.linspace(0.08, 0.93, 300)
    cc, cn = branches(dp1, 0, 0, rr)
    tang = np.array([pure.get_rc1r(x) / (2*np.pi*0.30*x) for x in rr])
    m = float(np.max(np.abs(tang)))
    print("\n  N=1 anchor — which branch reproduces the Tang single-component closed form?")
    print(f"    CORRECT (antisymmetrized): rel err = {float(np.max(np.abs(cc-tang)))/m:.2e}")
    print(f"    NAIVE   (raw 𝒲)          : rel err = {float(np.max(np.abs(cn-tang)))/m:.2e}")

    # ── mixture: pointwise + ĉ(0) ──────────────────────────────────────
    r = np.linspace(0.01, 6.0, 6000)
    print(f"\n  2YK mixture (σ={SIGS.tolist()}, ρ*={RHO}, T*={T}):")
    print(f"  {'pair':>6}  {'R_ij':>6}  {'inner max|Δc|':>13}  {'inner rel':>10}  "
          f"{'outer max|Δc|':>13}  {'ĉ(0) correct':>12}  {'ĉ(0) naive':>11}")
    print("  " + "-" * 84)
    data = {}
    for (i, j), lbl in zip(PAIRS, LABELS):
        cc, cn = branches(dp, i, j, r)
        R = float(dp.R_ij[i, j])
        inner, outer = r < R, (r >= R) & (r < 4*R)
        dmax = float(np.max(np.abs(cn - cc)[inner]))
        rel  = dmax / float(np.max(np.abs(cc[inner])))
        dout = float(np.max(np.abs(cn - cc)[outer]))
        # ĉ(0): inner from each branch, outer identical (bare tail) for both
        tail = np.zeros_like(r)
        rho_ij = np.sqrt(dp.rho[i]*dp.rho[j])
        for t in range(dp.n_tails):
            z = float(dp.z_poles[t, i, j]); K = float(dp.K[t, i, j])
            tail[outer] += K*np.exp(-z*(r[outer]-R))
        tail[outer] /= (2*np.pi*rho_ij*r[outer])
        hs = np.array([dp.get_HS_PY(float(x), i, j) if x < R else 0.0 for x in r])
        tot_c = np.where(r < R, hs + cc, tail)
        tot_n = np.where(r < R, hs + cn, tail)
        k_c = float(4*np.pi*_trapz(r**2 * tot_c, r))
        k_n = float(4*np.pi*_trapz(r**2 * tot_n, r))
        data[lbl] = (cc, cn, R, abs(float(dp.lambda_ij[i, j])))
        print(f"  {lbl:>6}  {R:6.3f}  {dmax:13.4f}  {rel:10.2f}  {dout:13.2e}  "
              f"{k_c:12.3f}  {k_n:11.3f}")

    print("\n  outer max|Δc| ≈ 0: the branches coincide exactly for r > R_ij, where")
    print("  𝒲(−r)=0 (a one-sided function has equal even and odd parts) — which is")
    print("  why the missing antisymmetrization hides outside and only bites in-core.")

    # ── figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20.0, 10.5))
    fig.suptitle(r"FMSA: antisymmetrization $\mathcal{C}=\mathcal{W}(r)-\mathcal{W}(-r)$ "
                 r"vs naive $\mathcal{C}=\mathcal{W}(r)$   "
                 rf"($\sigma$={SIGS.tolist()}, $\rho^*$={RHO}, $T^*$={T})")

    for col, ((i, j), lbl) in enumerate(zip(PAIRS, LABELS)):
        cc, cn, R, lam = data[lbl]
        ax = axes[0, col]
        ax.plot(r, cc, color=C_CORRECT, lw=2.2, label=r"correct: $\mathcal{W}(r)-\mathcal{W}(-r)$")
        ax.plot(r, cn, color=C_NAIVE, lw=1.8, ls="--", label=r"naive: $\mathcal{W}(r)$")
        ax.axvline(R, color="gray", lw=0.7, ls=":", label=r"$R_{ij}$" if col == 0 else None)
        if lam > 1e-9:
            ax.axvline(lam, color="gray", lw=0.7, ls="-.",
                       label=r"$\lambda_{ij}$" if col == 0 else None)
        sel = (r > 0.02) & (r < 1.6*R)
        lo, hi = float(np.min(cc[sel])), float(np.max(cc[sel]))
        pad = 0.35*(hi-lo)
        ax.set_xlim(0, 1.6*R); ax.set_ylim(lo-pad, hi+pad)
        ax.set_title(f"pair {lbl},  $R_{{ij}}$={R:.2f}")
        # y-label on EVERY panel: figure_split writes one PDF each and the .tex
        # places them independently, so a panel cannot lean on its neighbour.
        # The pair goes in the AXIS LABEL, as in Fig. 2 -- not a corner box.
        ax.set_ylabel(f"$c^{{(1)}}_{{{i + 1}{j + 1}}}(r)$")
        # No per-panel legend.  Three panels showing the same two curves carried
        # the same two-entry key three times, and at panel size it sat on the
        # data.  A shared strip is written below instead.

        ax = axes[1, col]
        err = np.abs(cn - cc)
        ax.semilogy(r, err + 1e-18, color="#7b3fa0", lw=1.8)
        ax.axvline(R, color="gray", lw=0.7, ls=":")
        if lam > 1e-9:
            ax.axvline(lam, color="gray", lw=0.7, ls="-.")
        ax.axvspan(0, R, color=C_NAIVE, alpha=0.05)
        ax.set_xlim(0, 1.6*R); ax.set_ylim(1e-17, max(err.max()*3, 1e-15))
        ax.set_xlabel("$r$")
        if col == 0:
            ax.set_ylabel(r"$|c_{\rm naive}-c_{\rm correct}|$")
        ax.set_title("error: grows as $r\\to0$, vanishes for $r>R_{ij}$")

    plt.tight_layout()
    fname = "numerical_notes/results/fmsa_dp_antisym.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  figure → {fname}")



# ⚠ ONE definition of the two curve colours, used by BOTH the panels and the shared
# key.  They were written out twice and drifted: the key said matplotlib's default
# blue #1f77b4 and red #d62728 while the panels drew #1baf7a and #c23b22, so the key
# named the correct branch in a colour that appears nowhere in those panels -- and
# worse, blue IS a colour in this figure, panel (a)'s W(-r), so the key did not merely
# mismatch, it pointed at the wrong object.
C_CORRECT = "#1baf7a"   # the odd part -- matches C_ODD in figure_antisym_explain.py
C_NAIVE = "#c23b22"     # W(r) kept whole


def _legend_strip(path="numerical_notes/results/fmsa_dp_antisym_legend.png"):
    """Shared key for the per-pair panels, since none carries its own."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_CORRECT, lw=2.2, label="correct (odd part)"),
         Line2D([], [], color=C_NAIVE, lw=1.8, ls="--", label="naive"),
         Line2D([], [], color="gray", lw=0.9, ls=":", label=r"$R_{ij}$ (contact)")]
    f = plt.figure(figsize=(7.06, 0.40))
    f.legend(handles=h, loc="center", ncol=len(h), frameon=False, fontsize=8,
             handlelength=2.2, columnspacing=1.6, handletextpad=0.5)
    f.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(f)
    print("  wrote", path)


if __name__ == "__main__":
    run()
    _legend_strip()
