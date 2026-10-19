#!/usr/bin/env python3
"""Panel (a) of Figure 4: FMSA-DP reproduces FMSA_pure, at N=1 and at N=2.

Two independent reductions of the mixture construction onto the same target:

  FMSA_pure              a separately written single-component implementation
  FMSA-DP, N = 1         the mixture code with one species
  FMSA-DP, N = 2 equal   the mixture code with TWO IDENTICAL species

The second is a limit; the third is a degeneracy.  They test different things.
N = 1 exercises the scalar path.  N = 2 with identical species exercises the FULL
matrix path -- the Baxter factor is 2x2, the propagator table is built over all
four pairs, and every pair DCF must come out equal to the scalar answer.  A bug
in the mixing, the pair loops or the propagator assembly shows up in the third
curve and not the second.

⚠ Equal diameters only.  FMSA_pure has a known units bug for d != 1, so a
comparison at unequal diameters would be measuring that bug, not this agreement.
The panel therefore uses one sigma for both species, which is what "equal 2YK"
means here.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from FMSA_pure import FMSA_pure
from fmsa_double_prop import FMSA_double_prop

SIGMA, EPSILON = 1.0, 1.0
RHOSTAR, TSTAR = 0.7, 1.5
OUT = 'numerical_notes/results/fmsa_dp_pure_limit.png'
OUT_G = 'numerical_notes/results/fmsa_dp_pure_limit_rdf.png'

# ── the independent single-component RDF, straight from the source paper ─────
#
# Tang & Lu, J. Chem. Phys. 99, 9828 (1993), Eqs. (32)-(34).  This is the paper
# FMSA_pure is built on, and it carries the first-order Yukawa RDF in CLOSED
# LAPLACE FORM -- which is why panel (b) has an independent curve at all.
#
# ⚠ The long-standing note here said "FMSA_pure implements no g(r), so this is
# an internal consistency check".  The first half is still true (the class stops
# at get_c_total/get_chat0; its `get_S` is the Baxter polynomial numerator, NOT
# the structure factor S(k) -- a false friend).  The second half did not follow:
# the THEORY has the formula even though that class never implemented it, so an
# independent pure-fluid RDF was always writable.  It is five lines.
#
# Implemented from the paper alone -- no FMSA_pure call, no FMSA_double_prop
# call -- so agreement is two independent derivations meeting, not a tautology.

def tanglu93_gL(s, eta, K, z, R=1.0):
    r"""L[r·g(r)](s) for the one-Yukawa fluid at first order, hard diameter R.

        g^L      = g0^L + h1^L                          (the FMSA split)
        g0^L(s)  = s·tau0(s)·e^{-s} / [1 - 12*eta*q0(s)]            Eq. (34), K=0
        h1^L(s)  = K·e^{-s} / ([1-12*eta*q0(s)]^2 [1-12*eta*q0(z)]^2 (s+z))  Eq. (32)
        q0(t)    = a0/t^3 + b0/t^2 - (a0/2+b0)/t
                   - (a0/t^3 + (a0+b0)/t^2)·e^{-t}                  Eq. (33)

    with a0 = (1+2eta)/(1-eta)^2 and b0 = -(3/2)eta/(1-eta)^2, and K = beta*epsilon
    -- the same K this code uses, so no convention factor intervenes.  Tang & Lu
    state this coincides with the Hoye-Blum MSA solution.

    ⚠ `K` and `z` may be ARRAYS (multi-tail).  At first order the tails enter
    LINEARLY -- no cross-pole mixing occurs for a single-component fluid, which
    is the same statement FMSA_pure.get_rc1r makes for r·c1(r) -- so h1^L is a
    plain sum of Eq. (32) over tails.  That is an extension of the paper's
    single-tail result, so it is GATED numerically in main(), not assumed.

    ⚠ `R` != 1 (Barker-Henderson diameter) needs the paper's R=1 result RESCALED,
    and the amplitude does not scale the way the length does.  With u = r/R,

        L[r·g(r)](s) = R^2 · (Tang & Lu at  s->sR,  z->zR,  K -> K/R),

    because c1(r) = K e^{-z(r-R)}/r becomes (K/R)·e^{-zR(u-1)}/u in reduced units.
    Verified: K/R gives ratio 1.000000 against get_gL_laplace where K·R and K
    drift to 1.2e-3 and 6e-4 by s=3 -- close enough to pass a loose eyeball test
    and wrong, so this is gated exactly rather than by inspection.
    """
    K = np.atleast_1d(K) / R
    z = np.atleast_1d(z) * R
    s = np.asarray(s) * R
    a0 = (1.0 + 2.0 * eta) / (1.0 - eta) ** 2
    b0 = -1.5 * eta / (1.0 - eta) ** 2

    def q0(t):
        return (a0 / t ** 3 + b0 / t ** 2 - (a0 / 2 + b0) / t
                - (a0 / t ** 3 + (a0 + b0) / t ** 2) * np.exp(-t))

    tau0 = a0 * (1.0 / s ** 3 + 1.0 / s ** 2) + b0 / s ** 2
    denom = (1.0 - 12.0 * eta * q0(s)) ** 2
    g0L = s * tau0 * np.exp(-s) / (1.0 - 12.0 * eta * q0(s))
    h1L = np.zeros_like(np.asarray(s, dtype=complex))
    for Kt, zt in zip(K, z):
        h1L = h1L + (Kt * np.exp(-s)
                     / (denom * (1.0 - 12.0 * eta * q0(zt)) ** 2 * (s + zt)))
    return (g0L + h1L) * R ** 2


def tanglu93_g_r(r_grid, eta, K, z, R=1.0, M: int = 24, tol: float = 1e-9):
    """Invert `tanglu93_gL` on the SAME de Hoog contour `get_g_laplace` uses.

    Reusing FMSA_double_prop._dehoog is deliberate: with one inverter on both
    sides, anything visible in the panel is a difference in the ANALYTIC input,
    which is what the comparison is about.  A second inverter would have added
    its own error and blurred exactly the thing being measured.
    """
    r_grid = np.atleast_1d(np.asarray(r_grid, dtype=float))
    out = np.zeros(len(r_grid))
    for n, t in enumerate(r_grid):
        if t <= 0:
            continue
        T = 2.0 * t
        a = -np.log(tol) / (2.0 * T)
        s = a + 1j * np.arange(0, 2 * M + 1) * np.pi / T
        out[n] = FMSA_double_prop._dehoog(
            tanglu93_gL(s, eta, K, z, R), t, T, M, a) / t
    return out


def tanglu93_params(dp: FMSA_double_prop):
    """(eta, K_phys, z, R) for a one-component DP object, read OFF the object.

    `dp.K` carries the code's internal 2*pi*rho factor; dividing it out recovers
    the physical amplitude of c1(r) = K e^{-z(r-R)}/r, which is exactly Tang &
    Lu's Eq. (29) convention (their K = beta*epsilon).  Read rather than
    reconstructed, so the comparison cannot drift from what DP actually solved."""
    rho = float(np.ravel(dp.rho)[0])
    return (float(dp.eta),
            np.ravel(dp.K).astype(float) / (2.0 * np.pi * rho),
            np.ravel(dp.z_poles).astype(float),
            float(np.ravel(dp.d_bh)[0]))


def main() -> None:
    r = np.linspace(0.05, 3.0, 600)

    pure = FMSA_pure(sigma=SIGMA, epsilon=EPSILON, rhostar=RHOSTAR, tstar=TSTAR)
    c_pure = np.array([pure.get_c_total(np.array([x]))[0] for x in r]) \
        if np.ndim(pure.get_c_total(r)) == 0 else pure.get_c_total(r)

    dp1 = FMSA_double_prop(np.array([SIGMA]), np.array([EPSILON]),
                           np.array([1.0]), RHOSTAR, TSTAR, n_tails=2)
    # 'fmt', NOT 'py': FMSA_pure.get_c_total is documented as "hard-sphere FMT
    # reference + first-order Yukawa perturbation".  Comparing against hs_ref='py'
    # measures the PY-vs-FMT reference gap (0.75 in the core) and reads as a
    # disagreement between the two implementations, which it is not.
    c_dp1 = dp1.get_c_total(r, hs_ref='fmt')[0, 0]

    dp2 = FMSA_double_prop(np.array([SIGMA, SIGMA]), np.array([EPSILON, EPSILON]),
                           np.array([0.5, 0.5]), RHOSTAR, TSTAR, n_tails=2)
    c_dp2 = dp2.get_c_total(r, hs_ref='fmt')

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(r, c_pure, color='#222222', lw=2.4, label='Tang 2003')
    ax.plot(r, c_dp1, color='#1f77b4', lw=1.6, ls='--',
            label='DP, $N=1$')
    for (i, j), st in ((( 0, 0), (0, (5, 2))), ((0, 1), (0, (1, 1.6)))):
        ax.plot(r, c_dp2[i, j], color='#d62728', lw=1.4, ls=st,
                label=f'DP, $N=2$ eq., $({i+1},{j+1})$')

    ax.axvline(dp1.R_ij[0, 0], color='gray', lw=0.8, ls=':')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('$r/\\sigma$')
    ax.set_ylabel('$c(r)$')
    ax.set_title('equal 2YK: the mixture code reduces to the pure one')
    ax.legend(fontsize=6.5, loc='lower right', framealpha=0.85,
              handlelength=1.8, borderpad=0.35, labelspacing=0.3)

    # The comparison is only meaningful if BOTH sides carry the SAME hard-sphere
    # block -- otherwise a difference in the reference could cancel one in the
    # tail and the totals would agree for the wrong reason.  Verified, not
    # assumed: d_bh matches to 2e-16 and the FMT block to 2e-14, so whatever
    # residual survives below is the first-order Yukawa part, which is the claim.
    inside = r < float(dp1.diag_r[0])
    hs_p = np.array([pure.get_c0_fmtHS_ref(x) for x in r[inside]])
    hs_d = np.array([dp1.get_HS_FMT(x, 0, 0) for x in r[inside]])
    d_hs = float(np.max(np.abs(hs_p - hs_d)))
    d_bh = abs(pure.d_bh - float(dp1.diag_r[0]))
    print(f'  d_bh                        = {d_bh:.3e}')
    print(f'  FMT hard-sphere block       = {d_hs:.3e}   (same reference, verified)')
    assert d_hs < 1e-10, f'hard-sphere references differ by {d_hs:.2e}'

    d1 = float(np.max(np.abs(c_dp1 - c_pure)))
    d2 = float(np.max(np.abs(c_dp2[0, 0] - c_pure)))
    d3 = float(np.max(np.abs(c_dp2[0, 1] - c_dp2[0, 0])))
    print(f'  max |DP(N=1) - pure|        = {d1:.3e}')
    print(f'  max |DP(N=2,(1,1)) - pure|  = {d2:.3e}')
    print(f'  max |DP(N=2,(1,2)) - (1,1)| = {d3:.3e}')

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print('  wrote', OUT)

    # ---- the RDF companion -------------------------------------------------
    # ⚠ THIS PANEL IS NOW AN INDEPENDENT CHECK, and the caption must say so.
    # It used to be internal-only, on the grounds that "FMSA_pure implements no
    # g(r)".  That class still doesn't (it stops at get_c_total/get_chat0, and
    # its `get_S` is the Baxter polynomial numerator, NOT the structure factor).
    # But the PAPER it is built on -- Tang & Lu 1993 -- gives the first-order
    # Yukawa RDF in closed Laplace form, so an independent curve was always
    # writable and nobody had written it.  `tanglu93_gL` above is it.
    #
    # The panel therefore now carries BOTH tests at once:
    #   DP N=1 vs N=2 equal   the matrix path collapsing onto the scalar one
    #   DP    vs Tang & Lu    two independent derivations of the same RDF
    eta_tl, K_tl, z_tl, R_tl = tanglu93_params(dp1)

    # GATE, in Laplace space, where the comparison is EXACT -- no inversion, so
    # no inverter error can hide a real disagreement or manufacture one.  This
    # also gates the multi-tail sum, which is an extension of the paper's
    # single-tail Eq. (32) and so must be checked rather than asserted.
    s_gate = np.array([0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
    tlL = np.array([complex(tanglu93_gL(s, eta_tl, K_tl, z_tl, R_tl)) for s in s_gate])
    dpL = np.array([np.ravel(dp1.get_gL_laplace(s))[0] for s in s_gate])
    d_lap = float(np.max(np.abs(dpL / tlL - 1.0)))
    print(f'  L[r g](s): DP vs Tang & Lu 1993 = {d_lap:.3e}   (exact, no inversion)')
    assert d_lap < 1e-10, f'Tang & Lu Laplace form disagrees by {d_lap:.2e}'

    rg = np.linspace(0.9, 3.0, 400)
    g1 = dp1.get_g_r(rg, route='fourier')[0, 0]
    g2 = dp2.get_g_r(rg, route='fourier')
    g_tl = tanglu93_g_r(rg, eta_tl, K_tl, z_tl, R_tl)

    fig2, ax2 = plt.subplots(figsize=(6.4, 4.4))
    # Tang & Lu drawn FIRST and widest, so the three DP curves land on top of it
    # -- the reading order of the claim: the literature result is the target.
    # `\&` only in a usetex build; with the default mathtext renderer the
    # backslash prints literally, which is how it first came out.
    lab_tl = ('Tang \\& Lu 1993, Eq. (32)' if plt.rcParams['text.usetex']
              else 'Tang & Lu 1993, Eq. (32)')
    ax2.plot(rg, g_tl, color='#222222', lw=3.0, alpha=0.30, label=lab_tl)
    ax2.plot(rg, g1, color='#1f77b4', lw=2.0, label='DP, $N=1$')
    ax2.plot(rg, g2[0, 0], color='#d62728', lw=1.5, ls=(0, (5, 2)),
             label='DP, $N=2$ eq., $(1,1)$')
    ax2.plot(rg, g2[0, 1], color='#2ca02c', lw=1.3, ls=(0, (1, 1.6)),
             label='DP, $N=2$ eq., $(1,2)$')
    ax2.axhline(1, color='k', lw=0.5)
    ax2.set_xlabel('$r/\\sigma$')
    ax2.set_ylabel('$g(r)$')
    ax2.set_title('equal 2YK: against the independent literature RDF')
    ax2.legend(fontsize=6.5, loc='best', framealpha=0.85,
               handlelength=1.8, borderpad=0.35, labelspacing=0.3)
    # ⚠ Quote the real-space agreement AWAY FROM CONTACT, and say why.  On the
    # jump at r = R the de Hoog inversion returns the mid-value and rings over a
    # band ~1/M wide, so |TL - DP| peaks at 1.6e-01 at r = R+ -- an inversion
    # artefact of BOTH curves, not a disagreement between them.  The exact
    # statement is the Laplace one above; these are inverter residuals:
    #   vs DP's LAPLACE route (same inverter)  2.1e-05  -- inverter-identical
    #   vs DP's FOURIER route                  7.8e-04  -- the known route gap
    # Reporting the 1.6e-01 as "the disagreement" would be reporting the inverter.
    far = rg >= 1.1 * R_tl
    g1L = dp1.get_g_laplace(rg)[0, 0]
    print(f'  max |g DP - Tang & Lu|, r > 1.1R = '
          f'{float(np.max(np.abs(g_tl - g1L)[far])):.3e} (same inverter), '
          f'{float(np.max(np.abs(g_tl - g1)[far])):.3e} (vs Fourier route)')
    print(f'    [at contact it is {float(np.max(np.abs(g_tl - g1))):.2e} -- de Hoog '
          f'ringing on the jump, in BOTH curves, not a disagreement]')
    dg = float(np.max(np.abs(g2[0, 0] - g1)))
    dg2 = float(np.max(np.abs(g2[0, 1] - g2[0, 0])))
    print(f'  max |g DP(N=2,(1,1)) - DP(N=1)|  = {dg:.3e}')
    print(f'  max |g DP(N=2,(1,2)) - (1,1)|    = {dg2:.3e}')
    fig2.tight_layout()
    fig2.savefig(OUT_G, dpi=150)
    # ⚠ The panel PDF is NOT written here.  `figure_split.py` produces it, and
    # it does two things a plain savefig does not: crops to the axes (dropping
    # the title, which belongs in the \caption) and rescales for print.  A
    # hand-rolled 6.4x4.4 in PDF placed in the same \panel slot as (a) came out
    # visibly smaller with unreadable tick labels, because (a) is splitter-made
    # at 234x146 pt and the hand-rolled one was 461x317 pt of mostly margin.
    print('  wrote', OUT_G, '  (panel PDF: run figure_split.py on this script)')


if __name__ == '__main__':
    main()
