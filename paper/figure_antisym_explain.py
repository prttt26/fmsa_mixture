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

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from fmsa_double_prop import FMSA_double_prop, px_eval

SIGS = np.array([1.0, 2.0])
EPS  = np.array([1.0, 0.5])
X    = np.array([0.25, 0.75])
RHO, T = 0.139, 1.0
I, J = 0, 1                    # unlike pair — has an interior λ knot too

C_W    = "#c23b22"   # 𝒲(r)
C_MIR  = "#2a78d6"   # 𝒲(−r)
C_ODD  = "#1baf7a"   # odd part (physical)
C_EVEN = "#7b3fa0"   # even part (discarded)


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

    # ── C. parity extraction ──────────────────────────────────────────
    ax = axes[2]
    ax.plot(r, odd, color=C_ODD, lw=3.0, label="odd")
    ax.plot(r, phys_half, color="k", lw=1.2, ls=(0, (1, 2)),
            label=r"physical $\mathcal{C}_1$")
    ax.plot(r, even, color=C_EVEN, lw=2.0, ls="--", label="even")
    ax.axvspan(0, R, color="gray", alpha=0.13, zorder=0)
    ax.axvline(R, color="k", lw=0.9, ls=":")
    lo = float(min(np.min(odd[sel]), np.min(even[sel])))
    hi = float(max(np.max(odd[sel]), np.max(even[sel])))
    pad = 0.30 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    err = float(np.max(np.abs(odd - phys_half)))
    # The "odd vs physical: max diff" annotation is removed from the artwork.
    # It is a caption fact, not a plot element, and it sat in the lower right
    # where the curves run.  Still printed, so it can be quoted in the caption.
    print(f"  panel C: max |odd - physical| = {err:.2e}")
    ax.set_xlabel("$r$"); ax.set_ylabel("parity components")
    ax.set_title("C.  odd branch $=$ physical $\\mathcal{C}_1$;\neven branch is junk")
    # 'best' rather than a fixed corner, with tight line spacing: the two curves
    # cross the upper left, which is where this legend used to sit.
    # Narrower and shorter: it cleared the curves by a hair at handlelength 1.7,
    # so the handles, the gaps and the font all come down a notch.  fontsize is
    # relative to the rcParam so --pub scaling still applies.
    ax.legend(loc="best", labelspacing=0.0, borderpad=0.15, handlelength=0.8,
              handletextpad=0.25, borderaxespad=0.2, framealpha=0.9,
              fontsize=plt.rcParams["legend.fontsize"] * 0.85)

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
