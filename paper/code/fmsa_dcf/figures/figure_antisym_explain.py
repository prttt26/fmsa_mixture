"""figure_antisym_explain.py — explanatory figure for the antisymmetrization step.

Three panels, one per logical step:

  A.  SUPPORT.  𝒲_C1 lives on (−R_ij, +∞); its mirror 𝒲_C1(−r) lives on
      (−∞, +R_ij).  The two overlap exactly on (−R_ij, R_ij) — the core.  This is
      why the contamination is confined to r < R_ij and vanishes beyond it.

  B.  BOTH PARITIES PRESENT.  On the overlap 𝒲(r) and 𝒲(−r) are independent
      functions, so 𝒲 carries an even part as well as an odd one.

  C.  PARITY EXTRACTION.  odd(𝒲) = ½[𝒲(r)−𝒲(−r)] is exactly ½ × the physical
      𝒞₁; even(𝒲) = ½[𝒲(r)+𝒲(−r)] is unconstrained convolution bookkeeping and
      is discarded.  Antisymmetrizing = projecting onto the physical sector.

Note on wording: 𝒲 is *not* a constant-coefficient linear combination
a·𝒞₁(r) + b·𝒞₁(−r) — measured 𝒲/𝒞₁ runs 24.6 → 3.55 → 14.6 across the core
(it → 1 only outside).  The exact statement is the parity one: 𝒲 and 𝒞₁ share
their odd part, and nothing else.
"""

from __future__ import annotations
import sys
import numpy as np
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

from fmsa_double_prop import FMSA_double_prop, px_eval

SIGS = np.array([1.0, 2.0])
EPS  = np.array([1.0, 0.5])
X    = np.array([0.25, 0.75])
RHO, T = 0.139, 1.0
RHO1, T1 = 0.2, 1.0           # panel C: one-component 2YK state, bh form (= Fig. 2(a))
I, J = 0, 1                    # unlike pair — has an interior λ knot too

C_W    = "#c23b22"   # 𝒲(r)
C_MIR  = "#2a78d6"   # 𝒲(−r)
C_C1   = "#1baf7a"   # C^(1)(r) = W(r) - W(-r)  (same green as probe_antisym.C_CORRECT)
C_EVEN = "#7b3fa0"   # even branch [W(r)+W(-r)]/2 (same as probe_antisym.C_EVEN)
LS_EVEN = (0, (4, 1.5, 1, 1.5))


def main() -> None:
    dp = FMSA_double_prop(SIGS, EPS, X, RHO, T, n_tails=2, verbose=False)
    W  = dp.closed_form_pieces(I, J)
    R  = float(dp.R_ij[I, J])
    lam = abs(float(dp.lambda_ij[I, J]))
    pref = 2.0 * np.pi * np.sqrt(dp.rho[I] * dp.rho[J])

    t = np.linspace(-1.7 * R, 1.9 * R, 4000)
    Wt  = np.array([px_eval(W, float(x)) for x in t])          # 𝒲(t)
    Wmt = np.array([px_eval(W, -float(x)) for x in t])         # 𝒲(−t)

    r = np.linspace(1e-3, 1.9 * R, 4000)
    Wp = np.array([px_eval(W, float(x)) for x in r])
    Wm = np.array([px_eval(W, -float(x)) for x in r])
    odd, even = 0.5 * (Wp - Wm), 0.5 * (Wp + Wm)
    phys_half = 0.5 * pref * r * np.array([dp.get_c1_inner(float(x), I, J) for x in r])

    fig, axes = plt.subplots(1, 3, figsize=(23.0, 7.4))
    fig.suptitle(r"Why $\mathcal{C}_1(r)=\mathcal{W}_{C1}(r)-\mathcal{W}_{C1}(-r)$   "
                 r"— the antisymmetrization step   "
                 rf"(2YK unlike pair, $R_{{ij}}$={R:.2f})")

    # ── A. support overlap ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, Wt,  color=C_W,   lw=2.2, label=r"$\mathcal{W}_{C1}(r)$")
    ax.plot(t, Wmt, color=C_MIR, lw=2.0, ls="--", label=r"$\mathcal{W}_{C1}(-r)$")
    ax.axvspan(-R, R, color="gray", alpha=0.13, zorder=0)
    for x in (-R, R):
        ax.axvline(x, color="k", lw=0.9, ls=":")
    ymin, ymax = float(np.min(Wt)), float(np.max(Wt))
    pad = 0.35 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + 1.35 * pad)
    yb = ymax + 0.55 * pad
    ax.annotate("", xy=(1.85*R, yb), xytext=(-R, yb),
                arrowprops=dict(arrowstyle="-", color=C_W, lw=4, alpha=0.75))
    ax.annotate("", xy=(-1.65*R, yb + 0.30*pad), xytext=(R, yb + 0.30*pad),
                arrowprops=dict(arrowstyle="-", color=C_MIR, lw=4, alpha=0.75))
    ax.text(0.55*R, yb + 0.09*pad, r"supp $\mathcal{W}(r)=(-R_{ij},\infty)$", color=C_W, ha="center")
    ax.text(-0.55*R, yb + 0.39*pad, r"supp $\mathcal{W}(-r)=(-\infty,R_{ij})$", color=C_MIR, ha="center")
    ax.text(0.0, ymin + 0.35*pad, r"overlap $(-R_{ij},R_{ij})$", ha="center", color="#444444")
    ax.set_xticks([-R, 0, R]); ax.set_xticklabels([r"$-R_{ij}$", "0", r"$R_{ij}$"])
    ax.set_xlabel("$r$"); ax.set_ylabel(r"$\mathcal{W}_{C1}$")
    ax.set_title("A.  both cover the core")
    # no legend: panels A and B carry the SAME two entries as each other and
    # as the companion figure's support panel, so all three share one strip.
    # See figure_rdf_support_gap.legend_strip().

    # ── B. both parity sectors present ────────────────────────────────
    ax = axes[1]
    ax.plot(r, Wp, color=C_W,   lw=2.2, label=r"$\mathcal{W}_{C1}(r)$")
    ax.plot(r, Wm, color=C_MIR, lw=2.0, ls="--", label=r"$\mathcal{W}_{C1}(-r)$")
    ax.axvspan(0, R, color="gray", alpha=0.13, zorder=0)
    ax.axvline(R, color="k", lw=0.9, ls=":")
    ax.axvline(lam, color="gray", lw=0.8, ls="-.")
    sel = r < 1.85 * R
    lo, hi = float(np.min(Wp[sel])), float(np.max(Wp[sel]))
    pad = 0.30 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.text(0.5*R, hi + 0.45*pad, "both nonzero\n⇒ two parity sectors", ha="center", color="#444444")
    ax.text(1.45*R, hi + 0.45*pad, r"$\mathcal{W}(-r)=0$" "\n⇒ one-sided", ha="center", color=C_MIR)
    ax.set_xlabel("$r$"); ax.set_ylabel(r"$\mathcal{W}_{C1}$")
    ax.set_title(r"B.  on the overlap $\mathcal{W}(r)$ and $\mathcal{W}(-r)$"
                 "\nare independent")
    # no legend: panels A and B carry the SAME two entries as each other and
    # as the companion figure's support panel, so all three share one strip.
    # See figure_rdf_support_gap.legend_strip().

    # ── C. the three objects of the main-text figure ──────────────────
    # Same three curves, colours and line styles as the shared key under the
    # per-pair panels (probe_antisym._legend_strip), so this panel has no legend:
    #   C^(1)(r) = W(r) - W(-r)           solid green   (the reconstruction)
    #   W(r)                              dashed red    (the raw representative)
    #   Even Branch = W(r) + W(-r)        dash-dot purple
    # FULL sum, the partner of the FULL difference C^(1) = W(r) - W(-r): outside the
    # core W(-r) = 0, so all three curves coincide there, and inside it the even
    # branch is the part the projection removes.  W = (C^(1) + Even Branch)/2.  ALL THREE are divided by 2 pi sqrt(rho_i rho_j) r,
    # as in panels (b)-(d), so the axis is c^(1)_12(r) and the spurious 1/r divergence
    # of W and of the even branch at r -> 0 is visible (the weight would hide it).
    # The y-range is set by the FINITE curve c^(1) only; the divergent ones run off it.
    # Panel C is the ONE-COMPONENT hard-core two-Yukawa fluid, in the 'bh' form with
    # the hard core at sigma = 1 and the standard 2YK tail parameters -- exactly the
    # system of Fig. 2(a) (figure_incore_split.SYS_2YK, rho* = 0.2, T* = 1) -- not the
    # unlike pair of the binary used by panels A and B.  Built separately for that
    # reason.
    dp1 = FMSA_double_prop(np.array([1.0]), np.array([1.0]), np.array([1.0]),
                           RHO1, T1, n_tails=2, z=np.array([2.9637, 14.0167]),
                           k=np.array([2.1714, -2.1714]), input_type='bh',
                           verbose=False)
    W1 = dp1.closed_form_pieces(0, 0)
    R1 = float(dp1.R_ij[0, 0])
    pref1 = 2.0 * np.pi * float(dp1.rho[0])
    r1 = np.linspace(1e-3, 1.9 * R1, 4000)
    Wp1 = np.array([px_eval(W1, float(x)) for x in r1])
    Wm1 = np.array([px_eval(W1, -float(x)) for x in r1])
    c1 = Wp1 - Wm1
    even1 = Wp1 + Wm1
    wt = pref1 * r1
    ax = axes[2]
    # Draw order matters: W(r) coincides with C^(1)(r) outside the core and runs
    # close to the even branch inside it, so it goes on TOP (dashes let the curves
    # below show through, exactly as in panels (b)-(d)).
    ax.plot(r1, even1 / wt, color=C_EVEN, lw=2.0, ls=LS_EVEN)
    ax.plot(r1, c1 / wt, color=C_C1, lw=2.2)
    ax.plot(r1, Wp1 / wt, color=C_W, lw=1.8, ls="--")
    ax.axvline(R1, color="k", lw=0.9, ls=":")
    fin = (r1 > 0.02) & (r1 < 1.85 * R1)
    lo, hi = float(np.min((c1 / wt)[fin])), float(np.max((c1 / wt)[fin]))
    pad = 0.35 * (hi - lo)
    ax.set_xlim(0, 1.85 * R1)
    ax.set_ylim(lo - pad, hi + pad)
    # Checks, printed for the caption and not drawn.
    phys1 = 0.5 * pref1 * r1 * np.array([dp1.get_c1_inner(float(x), 0, 0) for x in r1])
    err = float(np.max(np.abs(c1 - 2.0 * phys1)))
    out = r1 > R1
    print(f"  panel C (one-component, rho*={RHO1}, T*={T1}, R={R1:.4f}):")
    print(f"  panel C: max |C1 - 2*physical half| = {err:.2e}")
    print(f"  panel C: max |W - (C1 + Even)/2|   = {float(np.max(np.abs(Wp1 - 0.5*(c1 + even1)))):.2e}")
    print(f"  panel C: outside the core max |Even - C1| = {float(np.max(np.abs(even1 - c1)[out])):.2e}")
    ax.set_xlabel("$r$"); ax.set_ylabel(r"$c^{(1)}(r)$")
    ax.set_title(rf"one component,  $R$={R1:.2f}")

    plt.tight_layout()
    fname = "numerical_notes/results/fmsa_dp_antisym_explain.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"figure → {fname}")
    print(f"  odd-vs-physical max diff = {err:.2e}  (they coincide identically)")
    print(f"  even part at r=0.2R: {float(np.interp(0.2*R, r, even)):+.5f}  vs "
          f"physical ½𝒞 {float(np.interp(0.2*R, r, phys_half)):+.5f}  → unrelated")


if __name__ == "__main__":
    main()
