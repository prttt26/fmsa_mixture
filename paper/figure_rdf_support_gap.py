"""figure_rdf_support_gap.py — the RDF counterpart of `figure_antisym_explain.py`.

Companion figure.  `fmsa_dp_antisym_explain.png` shows why the *DCF* needs the
odd-part extraction; this one shows why the *RDF* does not, and the reason is a
single geometric fact about where each support starts.

  A.  DISJOINT SUPPORTS.  Because the hard-core condition h_ij(r) = −1 is exact,
      h⁽¹⁾ vanishes identically inside the core, so supp(r·h⁽¹⁾) = [R_ij, ∞) is
      bounded *away* from the origin and its mirror (−∞, −R_ij] is disjoint from
      it.  A gap of width 2R_ij separates them.  The Wiener–Hopf projection
      1_{[R_ij,∞)} therefore annihilates the partner EXACTLY.

  B.  WHERE THE SUPPORT STARTS is the whole difference.  r·c⁽¹⁾ is nonzero right
      down to r → 0⁺, so the fold lands on the core and the two parity sectors
      mix there.  r·h⁽¹⁾ is flat zero until contact.  Same machinery, same
      reflection, opposite fate — and nothing about the DCF calculation is at
      fault.

  C.  EXACT, NOT MERELY SMALL.  The RDF's core content is a quadrature artifact
      of the contact jump: it falls off with the transform cutoff k_max and
      converges to zero.  The DCF's does not — it is a closed-form O(1) value
      with no cutoff in it at all.  This is what distinguishes "the projection
      kills it" from "the projection makes it small".

⚠ `get_g_fourier` masks r < R_ij to zero by hand.  That mask is *assumed*, not
measured, so this script re-does the inversion without it — otherwise panel C
would be plotting its own premise.

Formalised as `FMSA.ReflectedSupports.parity_partners_causal_projection_zero`
(supp W ⊆ (−∞,R) ⟹ 1_{[R,∞)}·W = 0), axiom-clean.
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import dst

# ── two sizings, and the difference is not cosmetic ──────────────────────────
#
# SCREEN: 23 in wide at 18 pt — what the companion figure uses, readable on its own.
# PUB:    sized to the PAGE.  revtex4-2 aip/jcp gives \textwidth = 468 pt = 6.48 in
#         in `preprint`, and 510 pt = 7.06 in across both columns in `reprint`.
#         A 23-in figure at width=\textwidth scales by 0.28, so 18 pt text prints at
#         5.1 pt — and 2.7 pt in a reprint column.  Building at the printed width
#         instead makes the scale factor 1 and matplotlib pt equal page pt.
#
# A 3-panel figure cannot live in a 3.40 in reprint column (1.1 in/panel), so this
# one is a `figure*`: build at 7.06 in, which also renders at 0.92 in preprint.
PUB_WIDTH_IN = 7.06          # reprint \textwidth, i.e. figure* across both columns
PUB_FONT_PT  = 9.0           # -> ~8.3 pt after the 0.92 preprint scale

AP = argparse.ArgumentParser()
AP.add_argument("--pub", action="store_true",
                help="page-sized vector output for the paper (no suptitle; "
                     "panel titles reduce to (a)/(b)/(c) — the prose belongs in \\caption)")
ARGS, _ = AP.parse_known_args()

if ARGS.pub:
    plt.rcParams.update({
        "font.size": PUB_FONT_PT, "axes.titlesize": PUB_FONT_PT,
        "axes.labelsize": PUB_FONT_PT, "xtick.labelsize": PUB_FONT_PT - 1,
        "ytick.labelsize": PUB_FONT_PT - 1, "legend.fontsize": PUB_FONT_PT - 1,
        "figure.titlesize": PUB_FONT_PT,
        "pdf.fonttype": 42, "ps.fonttype": 42,     # embed as TrueType, not Type 3
        "savefig.transparent": False,
    })
    FIGSIZE, LW, FS_SM = (PUB_WIDTH_IN, PUB_WIDTH_IN / 3.05), 0.65, PUB_FONT_PT - 1.5
else:
    plt.rcParams.update({
        "font.size":         18,
        "axes.titlesize":    18,
        "axes.labelsize":    18,
        "xtick.labelsize":   18,
        "ytick.labelsize":   18,
        "legend.fontsize":   18,
        "figure.titlesize":  18,
    })
    FIGSIZE, LW, FS_SM = (23.0, 7.4), 1.0, 15.0


def panel_title(pub_short: str, screen_long: str) -> str:
    """Panel title: a bare (a)/(b)/(c) in print, the full sentence on screen.

    In the paper the descriptive half moves into \\caption{}, where it is set in the
    document font and can be edited without regenerating anything.
    """
    return pub_short if ARGS.pub else screen_long

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from fmsa_double_prop import FMSA_double_prop

SIGS = np.array([1.0, 2.0])
EPS  = np.array([1.0, 0.5])
X    = np.array([0.25, 0.75])
RHO, T = 0.139, 1.0
I, J = 0, 1                    # same unlike pair as the companion figure

C_RDF  = "#c23b22"   # r·h⁽¹⁾(r)      — matches C_W  of the companion
C_MIR  = "#2a78d6"   # its mirror     — matches C_MIR
C_DCF  = "#7b3fa0"   # r·c⁽¹⁾(r)      — matches C_EVEN (the object that needed work)
C_OK   = "#1baf7a"   # "exactly zero" — matches C_ODD

KMAX_MAIN = 4000.0
KMAX_SCAN = (250.0, 500.0, 1000.0, 2000.0, 4000.0)


def invert_first_order_h(dp, kmax: float, n_k: int = 200000):
    """r ↦ h⁽¹⁾_ij(r) by sine transform of the closed-form ĥ⁽¹⁾(k) — NO core mask.

    Deliberately not `get_g_fourier`, which zeroes r < R_ij by hand; the whole
    point here is to let the core value be measured.
    """
    dk = kmax / n_k
    k = dk * np.arange(1, n_k + 1)
    hk = np.atleast_3d(dp.get_h1hat_k(k))
    r = np.pi / ((n_k + 1) * dk) * np.arange(1, n_k + 1)
    h = (0.5 * dk) * dst(k[:, None, None] * hk, type=1, axis=0) \
        / (2.0 * np.pi ** 2 * r[:, None, None])
    return r, h[:, I, J]



def legend_strip():
    """The shared key for Fig. 1(a)-(c), spanning BOTH scripts of that figure.

    Those three panels plot three different functions -- W_C1 twice and r*h^(1)
    once -- with ONE encoding: red solid is the function, blue dashed is its
    mirror.  The colours already agree across the two scripts (see C_RDF/C_MIR
    above), so the key is stated by ROLE.

    ⚠ A concrete key would be ambiguous, which is why this one is not.  Listing
    `W_C1(r)` and `r*h^(1)(r)` as separate entries gives two entries the same red
    solid handle, since panels (a),(b) and panel (c) reuse the styles for
    different functions.  The caption names which function belongs to which panel.

    STRIP, not block: Fig. 1 is a full 2x2 with no empty slot, so the key needs a
    row of its own.  See figure_legend.py.
    """
    from matplotlib.lines import Line2D

    import figure_legend
    h = [Line2D([], [], color=C_RDF, lw=2.4, label=r'$f(r)$'),
         Line2D([], [], color=C_MIR, lw=2.2, ls='--', label=r'its mirror, $f(-r)$')]
    figure_legend.emit(h, [x.get_label() for x in h], 'fmsa_dp_rdf_support_gap',
                       out_dir='numerical_notes/results', handlelength=2.6)


def main() -> None:
    dp = FMSA_double_prop(SIGS, EPS, X, RHO, T, n_tails=2, verbose=False)
    R = float(dp.R_ij[I, J])

    r, h1 = invert_first_order_h(dp, KMAX_MAIN)
    keep = r < 2.7 * R
    r, rh1 = r[keep], (r * h1)[keep]

    # DCF inner core, closed form — same per-pair normalisation as h⁽¹⁾
    rc = np.linspace(1e-3, R, 1200)
    rc1 = rc * np.array([dp.get_c1_inner(float(x), I, J) for x in rc])

    core = (r > 0.08) & (r < 0.92 * R)
    rdf_core = float(np.max(np.abs(rh1[core])))
    dcf_core = float(np.max(np.abs(rc1[(rc > 0.08) & (rc < 0.92 * R)])))

    scan = []
    for km in KMAX_SCAN:
        rs, hs = invert_first_order_h(dp, km)
        sel = (rs > 0.08) & (rs < 0.92 * R)
        scan.append(float(np.max(np.abs((rs * hs)[sel]))))

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    if not ARGS.pub:
        fig.suptitle(r"Why the RDF needs no parity correction: "
                     r"$\mathrm{supp}\,(r\,h^{(1)})=[R_{ij},\infty)$ is bounded away from $0$   "
                     rf"(2YK unlike pair, $R_{{ij}}$={R:.2f})")

    # ── A. the two supports are disjoint ──────────────────────────────
    ax = axes[0]
    ax.plot(r, rh1, color=C_RDF, lw=2.2*LW, label=r"$r\,h^{(1)}(r)$")
    ax.plot(-r, -rh1, color=C_MIR, lw=2.0*LW, ls="--", label=r"mirror, $-r\,h^{(1)}(-r)$")
    ax.axvspan(-R, R, color="gray", alpha=0.13, zorder=0)
    for x in (-R, R):
        ax.axvline(x, color="k", lw=0.9, ls=":")
    lo, hi = float(np.min(rh1)), float(np.max(rh1))
    pad = 0.42 * (hi - lo)
    ax.set_ylim(lo - 1.15 * pad, hi + 1.5 * pad)
    yb = hi + 0.70 * pad
    ax.annotate("", xy=(2.6 * R, yb), xytext=(R, yb),
                arrowprops=dict(arrowstyle="-", color=C_RDF, lw=4*LW, alpha=0.75))
    ax.annotate("", xy=(-2.6 * R, yb + 0.32 * pad), xytext=(-R, yb + 0.32 * pad),
                arrowprops=dict(arrowstyle="-", color=C_MIR, lw=4*LW, alpha=0.75))
    ax.text(1.80 * R, yb - 0.26 * pad, r"$[R_{ij},\infty)$", color=C_RDF, ha="center",
            fontsize=FS_SM)
    ax.text(-1.80 * R, yb + 0.58 * pad, r"$(-\infty,-R_{ij}]$", color=C_MIR, ha="center",
            fontsize=FS_SM)
    # axes-fraction, not data coords: the two sizings have very different aspect
    # ratios, and a data-coordinate label tuned for one strikes through the curves
    # in the other.
    ax.text(0.5, 0.60, "gap $2R_{ij}$\nDISJOINT", transform=ax.transAxes,
            ha="center", va="center", color="#444444", fontweight="bold", fontsize=FS_SM)
    ax.set_xticks([-R, 0, R]); ax.set_xticklabels([r"$-R_{ij}$", "0", r"$R_{ij}$"])
    ax.set_xlabel("$r$"); ax.set_ylabel(r"$r\,h^{(1)}$")
    ax.set_title(panel_title("(a)", "A.  neither reaches the origin"))
    # no legend -- shared strip, see legend_strip()

    # ── B. where the support starts is the whole difference ───────────
    # Scaled to the CORE, not to the RDF: outside contact r·h⁽¹⁾ swings ±3 and
    # would squash r·c⁽¹⁾ (max 0.34) into the axis, hiding the one thing the
    # panel exists to show.
    ax = axes[1]
    selB = r <= R
    ax.plot(rc, rc1, color=C_DCF, lw=2.6*LW, label=r"$r\,c^{(1)}(r)$  (DCF)")
    ax.plot(r[selB], rh1[selB], color=C_RDF, lw=3.0*LW, label=r"$r\,h^{(1)}(r)$  (RDF)")
    ax.axvspan(0, R, color="gray", alpha=0.13, zorder=0)
    ax.axvline(R, color="k", lw=0.9, ls=":")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    lo = float(np.min(rc1)); hi = float(np.max(rc1))
    pad = 0.38 * (hi - lo)
    ax.set_xlim(-0.04 * R, 1.16 * R)
    ax.set_ylim(lo - 1.5 * pad, hi + 1.4 * pad)
    if not ARGS.pub:
        ax.annotate("nonzero as $r\\to0^+$\n⇒ the fold lands on the core",
                    xy=(0.03 * R, float(rc1[0])), xytext=(0.09 * R, hi + 0.30 * pad),
                    color=C_DCF, ha="left",
                    arrowprops=dict(arrowstyle="->", color=C_DCF, lw=1.6*LW))
    ax.text(0.50 * R, 0.42 * hi, r"$h^{(1)}\equiv0$" if ARGS.pub
            else r"$h^{(1)}\equiv0$ on the whole core",
            ha="center", color=C_RDF, fontsize=FS_SM)
    if not ARGS.pub:
        ax.annotate("switches on\nat contact", xy=(R, 0.0), xytext=(0.94 * R, lo - 1.05 * pad),
                    color=C_RDF, ha="center", fontsize=FS_SM,
                    arrowprops=dict(arrowstyle="->", color=C_RDF, lw=1.6*LW))
    ax.set_xticks([0, R]); ax.set_xticklabels(["0", r"$R_{ij}$"])
    ax.set_xlabel("$r$"); ax.set_ylabel("first-order correction")
    ax.set_title(panel_title("(b)", "B.  same reflection, opposite fate:\nonly the DCF support reaches $0$"))
    # This one STAYS: its two entries (DCF against RDF) appear in no other
    # panel, so the shared strip cannot carry them.  Shrunk instead --
    # tight spacing and short handles -- because at full size it sat on the
    # r*c^(1) curve, which is the one thing the panel exists to show.
    ax.legend(loc="lower left", fontsize=FS_SM * 0.85, labelspacing=0.15,
              borderpad=0.25, handlelength=1.1, handletextpad=0.4,
              borderaxespad=0.25, framealpha=0.85)

    # ── C. exact, not merely small ────────────────────────────────────
    ax = axes[2]
    ax.loglog(KMAX_SCAN, scan, "o-", color=C_OK, lw=2.6*LW, ms=9*LW,
              label=r"RDF: $\max|r\,h^{(1)}|$ in core")
    ref = scan[0] * (np.array(KMAX_SCAN) / KMAX_SCAN[0]) ** -1.0
    ax.loglog(KMAX_SCAN, ref, color="#888888", lw=1.4*LW, ls=":",
              label=r"$\propto 1/k_{\max}$ (guide)")
    ax.axhline(dcf_core, color=C_DCF, lw=2.6*LW, ls="--",
               label=r"DCF: $\max|r\,c^{(1)}|$ in core")
    ax.set_xlabel(r"transform cutoff $k_{\max}$")
    ax.set_ylabel("core content")
    ax.set_title(panel_title("(c)", "C.  the RDF's core content is quadrature\nand refines away; the DCF's does not"))
    ax.legend(loc="upper right", fontsize=FS_SM)
    if not ARGS.pub:
        ax.text(0.03, 0.13, f"{dcf_core/scan[-1]:.0f}× apart at $k_{{\\max}}$={KMAX_SCAN[-1]:.0f},"
                            "\nand still separating",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_SM, color="#444444")

    plt.tight_layout(pad=0.4 if ARGS.pub else 1.08)
    if ARGS.pub:
        # ⚠ NO bbox_inches="tight" here.  It crops to the drawn content, so the
        # output width is whatever the annotations happened to need — measured at
        # 8.07 in for a figure declared 7.06 in.  That silently reintroduces an
        # unknown scale factor, which is the one thing this mode exists to remove.
        # Save at exactly `figsize` and let tight_layout do the packing instead.
        fname = "numerical_notes/results/fmsa_dp_rdf_support_gap.pdf"
        plt.savefig(fname)
    else:
        fname = "numerical_notes/results/fmsa_dp_rdf_support_gap.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    legend_strip()

    print(f"figure → {fname}")
    print(f"  R_ij = {R:.4f}   gap width 2R_ij = {2*R:.4f}")
    print(f"  RDF core max|r h1| : {rdf_core:.3e}  at k_max={KMAX_MAIN:.0f}")
    print(f"  DCF core max|r c1| : {dcf_core:.3e}  (closed form, no cutoff)")
    print("  k_max scan (core residual, RDF):")
    for km, v in zip(KMAX_SCAN, scan):
        print(f"    k_max={km:7.0f}  {v:.3e}")
    rate = np.polyfit(np.log(KMAX_SCAN), np.log(scan), 1)[0]
    print(f"  fitted decay exponent: {rate:+.2f}  (−1 ⇒ pure contact-jump quadrature)")


if __name__ == "__main__":
    main()
