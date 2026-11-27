#!/usr/bin/env python3
r"""Fig. 14 — the pure fluid, and why the exact MSA cannot close its own binodal.

Exact MSA and FMSA-DP are the same speed (Fig. 7) and agree on structure to a few
per cent, so on the evidence of the benchmarks there is nothing to choose between
them.  This figure is the thing that does choose: what happens on a subcritical
isotherm, where a binodal is actually built.

Four panels, one per claim, on the HSY fluid (`z = 1.8`):

  (a) THE FOLD IS NESTED STRICTLY INSIDE THE SPINODAL.  Two curves, both from the
      exact solution: the compressibility spinodal `a = 0`, and the fold, where
      the Blum-Hoye system stops having a real root at all.  Apexes 1.0331 and
      1.0176.  Below the fold apex there is an interval of density at which the
      theory returns NOTHING.

  (b) IN THE BAND BETWEEN THE TWO APEXES THE SOLUTION EXISTS EVERYWHERE, AND
      THERE IS STILL NO LOOP.  This is the panel that makes the point that
      absence of a solution is not the whole problem.  The isotherm is uncut, it
      passes through its own spinodal interval, and `βp` rises monotonically
      across it: `∂βp/∂ρ = (A/2π)²` is a PERFECT SQUARE, so it cannot be
      negative, and the minimum measured on this isotherm is `+2.45e-06`.  A van
      der Waals loop is not merely hard to find here; the route forbids it.

  (c) BELOW THE FOLD APEX THE ISOTHERM IS CUT, AND THE TWO PIECES GO THE WRONG
      WAY.  Past each spinodal a physical isotherm must FALL.  These rise --
      again because the square cannot change sign -- and then die at the fold.
      What is left is two disconnected arcs at pressures a factor of 8 apart with
      nothing in between, which no tangent construction can read.

  (d) FMSA-DP, THE SAME CONSTRUCTION, HAS THE LOOP.  Truncating at first order in
      `K` destroys the perfect square, so `a` goes genuinely negative, the
      isotherm is never cut, and the equal-area/common-tangent construction has
      something to act on.  Nothing is bridged, completed or fitted.

⚠⚠ (b)/(c) AND (d) ARE NOT ON A SHARED REFERENCE and must not be read as a
numerical comparison.  FMSA-DP's hard-sphere reference is FMT (Carnahan-Starling
contact); the exact-MSA Baxter route's is Percus-Yevick.  Only the SHAPE of the
isotherm is being compared -- whether a loop exists -- and that is a structural
property of each route, not a number.  For the same reason the reduced
temperatures differ (0.985 and 0.974 of the exact-MSA compressibility `T_c`,
0.880 of FMSA-DP's) and it does not matter: `(A/2π)² ≥ 0` holds at EVERY
temperature, so no colder isotherm would give the exact route a loop.

⚠ The dotted grey curve in (c) is the quintic-Hermite BRIDGE across the gap --
one of the four completions of `msa_exact_hybrids_compare`.  It is drawn because
it is the only way a compressibility-route binodal is obtained here at all, and
labelled a completion because that is what it is: the loop is put there, not
found.  The flanks' pressures are anchored to it at the spinodals, which is why
the two arcs in (c) are on one common pressure scale.

⚠ NO EXPLANATORY TEXT INSIDE THE AXES.  Every number and every encoding these
panels used to spell out -- the two apexes, the min of dbeta_p/drho, which shaded
band is which, what the open circles and the dashed tie line are -- now lives in
the `\caption{}`.  Four panels at `--pub` width are 3.53 in each; prose sized to
be legible there is prose competing with the curves, and the same sentence set in
the caption is both larger on the page and searchable.  The panels carry data,
the caption carries the argument.

Sources (all pre-computed; this script only draws):
    numerical_notes/results/msa_exact/msa_exact_phase_lines.json      (a)
    numerical_notes/results/msa_exact/msa_exact_vdw_loop.json         (b), (c)
    .../compare_energy_compress/phase_lines/fmsa_vdw_loop.json        (d)

Regenerate the inputs with `msa_exact_phase_lines.py`, `msa_exact_vdw_loop.py`
and `fmsa_vdw_loop.py`; this script is seconds.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PL = 'numerical_notes/results/msa_exact/msa_exact_phase_lines.json'
VD = 'numerical_notes/results/msa_exact/msa_exact_vdw_loop.json'
FD = ('numerical_notes/results/compare_energy_compress/phase_lines/'
      'fmsa_vdw_loop.json')
OUT = 'numerical_notes/results/msa_exact/msa_fold_vdw.png'

# ONE palette -- `figure_style.py`.  The spinodal was OZ+MSA's blue; it is a
# construction line of the purple curve, not a fifth model, so it is neutral now.
import figure_style as S                                        # noqa: E402
C_EX = S.color('exact MSA')
C_DP = S.color('FMSA-DP')
C_SP = S.AUX['spinodal']  # the compressibility spinodal, a = 0
C_BR = S.AUX['bridge']    # a completion across the gap, NOT a measurement
SHADE = S.AUX['shade']    # where no real root exists

ETA2RHO = 6.0 / np.pi     # sigma = 1


def _xy(rows, key):
    return (np.array([r['eta'] for r in rows]) * ETA2RHO,
            np.array([r[key] for r in rows]))


def main() -> None:
    pl = json.loads(pathlib.Path(PL).read_text())['HSY']
    vd = json.loads(pathlib.Path(VD).read_text())
    fd = json.loads(pathlib.Path(FD).read_text())['HSY']

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))
    (a1, a2), (a3, a4) = axes

    # ── (a) the two lines, and the fact that one is inside the other ─────────
    rs, Ts = _xy(pl['spinodal'], 'T')
    rf, Tf = _xy(pl['fold'], 'T')
    o = np.argsort(rs)
    rs, Ts = rs[o], Ts[o]
    o = np.argsort(rf)
    rf, Tf = rf[o], Tf[o]
    # The no-solution region is the inside of the fold curve.  Filling between
    # the fold and the axis would claim the whole low-T strip; filling UNDER the
    # fold curve only is what "no real root at this (rho, T)" means.
    a1.fill_between(rf, Tf, 0.0, color=SHADE, zorder=0)
    a1.plot(rs, Ts, '-', color=C_SP, lw=2.0, zorder=3)
    a1.plot(rf, Tf, '--', color=C_EX, lw=2.0, zorder=3)
    for T, sty in ((vd['band']['T'], ':'), (vd['below']['T'], '-.')):
        a1.axhline(T, color='k', lw=0.8, ls=sty, zorder=2)
    i = int(np.argmax(Ts))
    j = int(np.argmax(Tf))
    a1.plot([rs[i]], [Ts[i]], '*', ms=12, color=C_SP, zorder=5)
    a1.plot([rf[j]], [Tf[j]], '*', ms=12, color=C_EX, zorder=5)
    # The two lines are within 1.5 % of each other in T, so the gap has to be
    # opened somewhere or the panel says only "they coincide".  The inset is the
    # apex at the scale the gap actually lives on.
    ins = a1.inset_axes([0.58, 0.13, 0.38, 0.32])
    ins.fill_between(rf, Tf, 0.0, color=SHADE, zorder=0)
    ins.plot(rs, Ts, '-', color=C_SP, lw=1.6)
    ins.plot(rf, Tf, '--', color=C_EX, lw=1.6)
    for T, sty in ((vd['band']['T'], ':'), (vd['below']['T'], '-.')):
        ins.axhline(T, color='k', lw=0.7, ls=sty)
    ins.set_xlim(0.18, 0.45)
    ins.set_ylim(0.995, 1.038)
    ins.tick_params(labelsize=6.5)
    a1.set_xlim(0.0, 0.95)
    a1.set_ylim(0.55, 1.10)
    a1.set_xlabel(r'$\rho^*$')
    a1.set_ylabel(r'$T^*$')

    # ── (b) the band: uncut, and still monotone ──────────────────────────────
    b = vd['band']
    rb, pb = np.array(b['straight']['rho']), np.array(b['straight']['p'])
    ab = np.array(b['straight']['a'])
    # Cropped to the spinodal neighbourhood: at full range the hard-core rise
    # past rho* = 0.45 sets the y-scale and flattens the only part that carries
    # the claim.  The claim is local -- what the isotherm does between the two
    # spinodal densities -- so the axis is too.
    m = (rb >= 0.10) & (rb <= 0.47)
    a2.plot(rb[m], pb[m], '-', color=C_EX, lw=2.4, zorder=3)
    a2.axvspan(b['rho_sp'][0], b['rho_sp'][1], color=C_SP, alpha=0.10, zorder=0)
    for r in b['rho_sp']:
        a2.axvline(r, color=C_SP, lw=1.0, ls='-', alpha=0.8, zorder=2)
    a2.set_xlabel(r'$\rho^*$')
    a2.set_ylabel(r'$\beta p$')

    # ── (c) below the fold apex: cut, and going the wrong way ────────────────
    c = vd['below']
    br = np.array(c['bridge']['rho'])
    bp = np.array(c['bridge']['p'])
    # Zoomed hard.  The two arcs span rho* 0.247-0.270 and 0.360-0.377; on the
    # full isotherm they are two dots.  What has to be legible is their SLOPE
    # against the bridge's, so the axis is the gap and a little either side.
    m = (br >= 0.215) & (br <= 0.415)
    a3.plot(br[m], bp[m], ':', color=C_BR, lw=1.8, zorder=2)
    a3.axvspan(c['rho_fold'][0], c['rho_fold'][1], color=SHADE, zorder=0)
    for g in c['flank']:
        a3.plot(g['rho'], g['p'], '-', color=C_EX, lw=3.0, zorder=4)
        a3.plot([g['rho'][-1]], [g['p'][-1]], 'o', ms=6.0, color=C_EX,
                mfc='white', mew=1.7, zorder=5)
    for r in c['rho_sp']:
        a3.axvline(r, color=C_SP, lw=1.0, alpha=0.8, zorder=1)
    a3.axhline(c['p_coex'], color='k', lw=0.9, ls='--', zorder=3)
    drop = c['flank'][0]['p'][0] - c['flank'][1]['p'][0]
    a3.set_xlabel(r'$\rho^*$')
    a3.set_ylabel(r'$\beta p$')

    # ── (d) FMSA-DP: the loop, with nothing bridged ──────────────────────────
    R = fd['routes']['C']
    rr, pp = np.array(fd['rho']), np.array(R['p'])
    m = (rr >= 0.05) & (rr <= 0.75)
    a4.plot(rr[m], pp[m], '-', color=C_DP, lw=2.4, zorder=3)
    a4.axhline(R['p_coex'], color='k', lw=0.9, ls='--', zorder=2)
    for r in R['spinodal']:
        a4.axvline(r, color=C_SP, lw=1.0, alpha=0.8, zorder=1)
    for r in (R['rho_g'], R['rho_l']):
        a4.plot([r], [R['p_coex']], 'o', ms=6.5, color=C_DP, zorder=5)
    lo = min(R['spinodal'])
    hi = max(R['spinodal'])
    a4.axvspan(lo, hi, color=C_SP, alpha=0.10, zorder=0)
    # Cropped like (b)/(c): the hard-core rise is not the subject.
    a4.set_xlim(0.03, 0.66)
    a4.set_ylim(-0.05, 0.20)
    a4.set_xlabel(r'$\rho^*$')
    a4.set_ylabel(r'$\beta p$')

    # ── the key ──────────────────────────────────────────────────────────────
    import figure_legend
    # ⚠ SIX entries across \textwidth is ~4 pt of label at print size.  Three
    # columns over two rows costs no page height that matters and doubles the
    # type, so the labels are also cut to the shortest form that stays
    # unambiguous -- the caption carries the qualifications.
    handles = [
        Line2D([], [], color=C_SP, lw=2.0, label='compressibility spinodal'),
        Line2D([], [], color=C_EX, lw=2.0, ls='--', label='exact-MSA fold'),
        Patch(facecolor=SHADE, edgecolor='none', label='no real root exists'),
        Line2D([], [], color=C_EX, lw=2.4, label='exact MSA isotherm'),
        Line2D([], [], color=C_DP, lw=2.4, label='FMSA-DP isotherm'),
        Line2D([], [], color=C_BR, lw=1.6, ls=':',
               label='bridge: a completion, not a result'),
    ]
    figure_legend.emit(handles, [h.get_label() for h in handles],
                       'msa_fold_vdw',
                       out_dir='numerical_notes/results/msa_exact',
                       handlelength=2.8, block_ncol=1, strip_ncol=3,
                       strip_fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print('  wrote', OUT)
    print(f"  (a) spinodal apex {Ts[i]:.4f}  fold apex {Tf[j]:.4f}   "
          f"gap {100 * (Ts[i] - Tf[j]) / Ts[i]:.2f} % in T")
    print(f"  (b) T*={b['T']:.4f}  uncut, min a = {ab.min():.3e} "
          f"(>= 0 by construction)")
    print(f"  (c) T*={c['T']:.4f}  cut over rho = "
          f"{c['rho_fold'][0]:.4f}..{c['rho_fold'][1]:.4f}; spinodals at "
          f"{c['rho_sp'][0]:.4f}, {c['rho_sp'][1]:.4f}")
    for g in c['flank']:
        print(f"      flank {g['rho'][0]:.4f}->{g['rho'][-1]:.4f}: "
              f"beta p {g['p'][0]:.5f} -> {g['p'][-1]:.5f}  (RISES)")
    print(f"  (d) FMSA-DP T*={fd['T']:.4f} (T/Tc={fd['T']/fd['Tc_C']:.3f})  "
          f"min a = {min(R['a']):.4f}  rho_g={R['rho_g']:.4f} "
          f"rho_l={R['rho_l']:.4f}")


if __name__ == '__main__':
    main()
