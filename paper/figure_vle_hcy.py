#!/usr/bin/env python3
r"""figure_vle_hcy.py — vapour/liquid binodals against Shukla's GEMC.

Two panels, each one 0.49\textwidth slot (3.53 in), generated at printed width so the
LaTeX scale factor is 1:

  (a) the coexistence curves, T* against rho*
  (b) liquid-branch deviation from GEMC N=500, in per cent, against T*

THREE LEGS, and the middle one is the point.  GEMC is the measurement; exact MSA is the
MSA CLOSURE alone; FMSA-DP is that closure PLUS the first-order truncation.  So
`exact MSA - GEMC` is what the closure costs and `FMSA-DP - exact MSA` is what
truncating costs.  Neither is readable without the other two.

⚠ BOTH THEORY CURVES SIT ON CARNAHAN-STARLING.  The exact-MSA leg is NOT the stored
`msa_exact_phase_lines.json` binodal, which uses PY-compressibility -- correct there
(its d(beta p)/d rho is a_PY, keeping the K=0 edge consistent with the A=0 spinodal)
and wrong here.  Measured: the hard-sphere reference ALONE moves rho*_L by up to 7 %,
which is larger than the truncation effect this figure exists to display.  Subtracting
two curves with different cores would have attributed the core difference to the
truncation.  See `vle_hcy_shukla.exact_msa`.

⚠ WHY THE REFERENCE GETS TWO SYMBOLS.  Shukla ran N = 500, 216 and 108 and his abstract
calls the system-size dependence "significant".  Plotting one of them as "the data" would
hide a spread that is the same size as the model error.  Both are drawn; N=108 is in the
reference file but is NOT plotted -- its vapour branch is non-monotonic in T* and 8x the
N=216 value at T*=0.8.  In (b) the N=216 set appears as a CURVE, measured against N=500:
that puts the reference's own internal spread on the same axis as the model errors, which
is the only way to see that they are comparable in size.

⚠ ROUTE.  This is the ENERGY route.  The compressibility route is not interchangeable --
at N=1 the exact-MSA route gap moves T*_c by +18.7 % -- so panel (a) is one route's
answer, not "the model's answer".  The caption has to say so.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REF = 'data_benchmark/reference_shukla2000.json'
DP = 'numerical_notes/results/vle_hcy/vle_hcy_fmsadp.json'
EM = 'numerical_notes/results/vle_hcy/vle_hcy_exactmsa.json'
RT = 'numerical_notes/results/vle_hcy/vle_hcy_routes.json'
HB = 'numerical_notes/results/vle_hcy/vle_hcy_hncb.json'
OUT = 'numerical_notes/results/vle_hcy'

# ⭐ ONE palette for the whole paper -- `figure_style.py`.  Two colours here were
# this figure's alone: OZ-HNCB was TEAL (red everywhere else in the paper), and
# the GEMC N=216 series was drawn in `#2a78d6`, which is OZ+MSA's blue.  A
# reference series in a model's colour is the one mistake that changes what a
# panel appears to say, so both simulation series are neutral now.
import figure_style as S                                        # noqa: E402
C_DP = S.color('FMSA-DP')
C_EM = S.color('exact MSA')
C_HB = S.color('OZ+HNCB')
LS_HB = S.linestyle('OZ+HNCB')      # dashed: it is the only iterative leg here
LS_C = S.ROUTE['compressibility']   # dotted: a second route through FMSA-DP
C_500 = S.DATA['primary']
C_216 = S.DATA['secondary']
PANEL_IN = 3.53
FS = 9

plt.rcParams.update({'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS,
                     'xtick.labelsize': FS - 1, 'ytick.labelsize': FS - 1,
                     'pdf.fonttype': 42, 'axes.linewidth': 0.8})


def _curve(path, key='binodal'):
    d = json.load(open(path))[key]
    T = np.array([r['T'] for r in d])
    o = np.argsort(T)
    return T[o], np.array([r['rho_v'] for r in d])[o], np.array([r['rho_l'] for r in d])[o]


def load():
    g = json.load(open(REF))['vle']
    return g, _curve(DP), _curve(EM), _curve(HB)


def tc_estimate(T, rv, rl):
    """T_c from the classical width law: (rho_l - rho_v)^2 is linear in T near T_c.

    Both theories here are mean-field in the relevant sense (beta = 1/2), so the square
    of the width -- not the width -- is what extrapolates linearly.  Fitted on the top
    six points only; further down the curve the law is not supposed to hold.

    ⚠ Applied identically to BOTH models, on curves traced over the same T* grid, so the
    two critical temperatures are comparable.  Quoting one model's T_c from this fit and
    the other's from a separate scan with its own grid and its own hard-sphere reference
    would compare the fits, not the models."""
    w2 = (rl - rv) ** 2
    m = np.argsort(T)[-6:]
    a, b = np.polyfit(T[m], w2[m], 1)
    return -b / a


def close_dome(ax, T, rv, rl, col, ls='-', lw=1.7):
    """Draw the branches and the extrapolated closure to (rho_c, T_c).

    The dome is left OPEN above the last solved point: joining the two branches across
    the top would draw a horizontal tie line that is not a computed result.  The dashed
    pair is the only extrapolated ink in the panel, and `rho_c` comes from the law of
    rectilinear diameters -- (rho_l+rho_v)/2 linear in T."""
    tc = tc_estimate(T, rv, rl)
    ax.plot(rv, T, ls, color=col, lw=lw, zorder=3)
    ax.plot(rl, T, ls, color=col, lw=lw, zorder=3)
    a, b = np.polyfit(T[-6:], 0.5 * (rl + rv)[-6:], 1)
    rho_c = a * tc + b
    ax.plot([rv[-1], rho_c], [T[-1], tc], '--', color=col, lw=1.0, zorder=3)
    ax.plot([rl[-1], rho_c], [T[-1], tc], '--', color=col, lw=1.0, zorder=3)
    ax.plot([rho_c], [tc], '*', color=col, ms=9, zorder=4)
    return tc, rho_c


def dev_vs(T, rl, Td, rd):
    """Liquid-branch deviation, in per cent, of a curve against reference points."""
    m = (Td >= T.min()) & (Td <= T.max())
    return Td[m], 100 * (np.interp(Td[m], T, rl) - rd[m]) / rd[m]


def main() -> None:
    g, (T, rv, rl), (Te, rve, rle), (Th, rvh, rlh) = load()
    d500 = np.array(g['zy_1.8_N500']['data'])
    d216 = np.array(g['zy_1.8_N216']['data'])

    # ── (a) coexistence curves ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN * 0.86))
    # The COMPRESSIBILITY route of the same model, dashed in the same colour: same
    # theory, different thermodynamic route.  It is drawn first, and thinner, so the
    # two solid curves stay the primary reading -- but the spacing to the solid
    # FMSA-DP curve is the ROUTE GAP, and it is larger than the truncation the two
    # solid curves display.  Both legs sit on the SAME WB1/Carnahan-Starling core
    # (checked at K -> 0, where they agree to 1.1e-07), so this spacing is the route
    # and not a hard-sphere reference difference.
    rt = json.load(open(RT))
    cp = rt['compressibility']
    Tc_ = np.array([r['tstar'] for r in cp])
    o = np.argsort(Tc_)
    ax.plot(np.array([r['rho_vap'] for r in cp])[o], Tc_[o], ls=LS_C, color=C_DP,
            lw=1.3, zorder=2)
    ax.plot(np.array([r['rho_liq'] for r in cp])[o], Tc_[o], ls=LS_C, color=C_DP,
            lw=1.3, zorder=2)

    # OZ-HNCB's dome is LEFT OPEN, deliberately.  It converges only to T* = 1.14
    # (three temperatures fail above that, where the dome narrows), so the same
    # width-law closure would extrapolate ~0.10 in T against 0.01-0.02 for the other
    # two -- five times the reach, drawn with the same authority.  Branches only.
    tc_h = tc_estimate(Th, rvh, rlh)
    # zorder ABOVE the two MSA legs.  All three branches sit within a per-cent of
    # each other over most of the dome, so whichever is drawn last is the only one
    # visible; a dashed line on top is the one choice that shows all three.
    ax.plot(rvh, Th, ls=LS_HB, color=C_HB, lw=1.7, zorder=6)
    ax.plot(rlh, Th, ls=LS_HB, color=C_HB, lw=1.7, zorder=6)
    tc_e, rc_e = close_dome(ax, Te, rve, rle, C_EM)
    tc_d, rc_d = close_dome(ax, T, rv, rl, C_DP)

    for d, col, mk in ((d500, C_500, 'o'), (d216, C_216, 's')):
        # 3 % on both densities -- the reference's own quoted error, not a guess
        for rho in (d[:, 1], d[:, 2]):
            ax.errorbar(rho, d[:, 0], xerr=0.03 * rho, fmt=mk, ms=3.0, mfc='none',
                        mec=col, ecolor=col, elinewidth=0.7, capsize=1.5, mew=0.9,
                        ls='none', zorder=2)
    ax.set_xlabel(r'$\rho^*$')
    ax.set_ylabel(r'$T^*$')
    ax.set_title(r'coexistence, $z=1.8$')
    ax.set_xlim(0, 0.92)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'vle_hcy_a.pdf'))
    fig.savefig(os.path.join(OUT, 'vle_hcy_a.png'), dpi=200)
    plt.close(fig)

    # ── (b) liquid-branch deviation from GEMC N=500 ──────────────────────────
    #
    # N=500 is the BASELINE (the zero line), so it carries no curve of its own.  The
    # N=216 curve is the reference's own system-size spread, drawn on the same axis as
    # the model errors on purpose: it is the honest scale against which "how wrong is
    # the theory" has to be read.  The two sets share all 10 temperatures above T*=1.0,
    # so that curve is exact -- no interpolation between reference points.
    fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN * 0.86))
    ax.axhspan(-3, 3, color='0.85', zorder=0)          # the reference's own 3 % band
    ax.axhline(0, color='0.4', lw=0.8, zorder=1)

    shared = np.intersect1d(d500[:, 0], d216[:, 0])
    i5 = np.searchsorted(d500[:, 0], shared)
    i2 = np.searchsorted(d216[:, 0], shared)
    ax.plot(shared, 100 * (d216[i2, 1] - d500[i5, 1]) / d500[i5, 1], 's', ms=3.4,
            mfc='none', mec=C_216, mew=1.0, ls='-', lw=0.9, color=C_216, zorder=2)

    # Each model carries its own marker and linestyle here too, not a shared
    # open circle: with three curves crossing inside a 3 % band the colour alone
    # was doing all the work, and the key promises three encodings.
    for (Tm, rm), model in (((Th, rlh), 'OZ+HNCB'), ((Te, rle), 'exact MSA'),
                            ((T, rl), 'FMSA-DP')):
        td, dev = dev_vs(Tm, rm, d500[:, 0], d500[:, 1])
        col = S.color(model)
        ax.plot(td, dev, marker=S.marker(model), ms=3.6, mfc='none', mec=col,
                mew=1.0, ls=S.linestyle(model), lw=1.1, color=col, zorder=3)
    ax.set_xlabel(r'$T^*$')
    ax.set_ylabel(r'$\rho^*_L$ deviation from $N{=}500$ (\%)'
                  if plt.rcParams['text.usetex']
                  else r'$\rho^*_L$ deviation from $N{=}500$ (%)')
    ax.set_title(r'liquid branch vs GEMC')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'vle_hcy_b.pdf'))
    fig.savefig(os.path.join(OUT, 'vle_hcy_b.png'), dpi=200)
    plt.close(fig)

    # ── shared key ───────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    import figure_legend
    pct = r'GEMC error, 3\%' if plt.rcParams['text.usetex'] \
        else 'GEMC error, 3%'
    h = [Line2D([], [], color=C_DP, lw=1.8, label='FMSA-DP, energy route'),
         Line2D([], [], color=C_DP, lw=1.3, ls=LS_C,
                label='FMSA-DP, compressibility'),
         Line2D([], [], color=C_EM, lw=1.8, label='exact MSA'),
         Line2D([], [], color=C_HB, lw=1.8, ls=LS_HB, label='OZ-HNCB, energy route'),
         Line2D([], [], color=C_500, marker='o', ms=5, mfc='none', mew=1.0, ls='none',
                label=r'GEMC $N=500$'),
         Line2D([], [], color=C_216, marker='s', ms=5, mfc='none', mew=1.0, ls='none',
                label=r'GEMC $N=216$'),
         Line2D([], [], color='0.85', lw=7, label=pct)]
    # TWO ROWS, not one.  Seven entries do not fit across \textwidth, so a
    # one-row strip is 817 pt wide and prints its 8 pt text at 4.6 pt.  ncol=4
    # over 7 entries gives 4 + 3 -- the four theory legs on the top row and the
    # three reference items below -- and `strip_rowmajor` is what makes that
    # split hold, since Matplotlib fills columns downwards.
    figure_legend.emit(h, [x.get_label() for x in h], 'vle_hcy', out_dir=OUT,
                       block_ncol=1, handlelength=2.0,
                       strip_ncol=4, strip_rowmajor=True, columnspacing=1.1)

    # ── what the panels are allowed to claim ─────────────────────────────────
    print(f'  wrote {OUT}/vle_hcy_[ab].pdf')
    dh = dev_vs(Th, rlh, d500[:, 0], d500[:, 1])[1]
    print(f'  OZ-HNCB   vs N=500: {dh.min():+.1f} … {dh.max():+.1f} %  '
          f'(median {np.median(dh):+.1f} %); traced to T* = {Th.max():.2f}, '
          f'width-law T_c would be {tc_h:.3f} -- NOT drawn, extrapolation too long')
    print(f'  T*_c (same width-law fit, same grid):  FMSA-DP {tc_d:.3f}'
          f'   exact MSA {tc_e:.3f}   ({100*(tc_d-tc_e)/tc_e:+.1f} %)')
    print(f'  traced to T* = {T.max():.2f} (FMSA-DP), {Te.max():.2f} (exact MSA)')
    for name, (Tm, rm) in (('FMSA-DP  ', (T, rl)), ('exact MSA', (Te, rle))):
        for key, d in (('N=500', d500), ('N=216', d216)):
            td, dev = dev_vs(Tm, rm, d[:, 0], d[:, 1])
            print(f'  {name} vs {key}: {dev.min():+.1f} … {dev.max():+.1f} %'
                  f'  (median {np.median(dev):+.1f} %)')
    td, dv = dev_vs(T, rl, d500[:, 0], d500[:, 1])
    te, de = dev_vs(Te, rle, d500[:, 0], d500[:, 1])
    print(f'  truncation alone (FMSA-DP - exact MSA, shared HS core):'
          f' {np.min(dv-de):+.1f} … {np.max(dv-de):+.1f} %')
    print(f'  reference spread  (N=216 - N=500):'
          f' {100*np.min((d216[i2,1]-d500[i5,1])/d500[i5,1]):+.1f} …'
          f' {100*np.max((d216[i2,1]-d500[i5,1])/d500[i5,1]):+.1f} %')


if __name__ == '__main__':
    main()
