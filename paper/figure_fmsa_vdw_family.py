#!/usr/bin/env python3
r"""figure_fmsa_vdw_family.py — FMSA-DP's van der Waals loop across temperature,
and the two phase lines it generates.

The positive counterpart of the exact-MSA fold figure.  That one shows a
solution that dies before the instability does; this one shows the truncated
theory doing the ordinary thing, and doing it systematically rather than at one
temperature.

  (a) `βp(ρ)` at six temperatures from `0.80` to `0.99` of `T_c`, with the
      Maxwell tie line on each.  The loop deepens monotonically as the fluid
      cools -- `min ∂βp/∂ρ` runs `-0.035 → -0.946` -- and closes at `T_c`.
  (b) the binodal those tie lines trace, with the spinodal inside it, meeting at
      the critical point.

⭐ THE SPINODAL IN (b) IS THE TURNING POINTS OF THE CURVES IN (a).  Not a
separately computed line: `∂βp/∂ρ = 0` read off the same isotherms.  A spinodal
drawn on its own authority is unfalsifiable -- no simulation measures one, and
this one is known to sit 12--30 % high -- but as the extrema of the panel above
it, it is a construction line and costs nothing.

⭐ (a) CONTAINS Fig. 12(d).  The `0.88 T_c` isotherm is that panel exactly:
`min ∂βp/∂ρ = -0.4872`, coexistence `0.0882 .. 0.5645`.  Keeping it in the
family is what lets this figure absorb the panel rather than add to the count.

⚠ Compressibility route throughout, matching Fig. 12(b)-(d), and `T_c` is that
route's own `1.1845`.  The energy route has its own, and mixing them would put
the tie lines and the dome on different scales.

⚠ No `set_title` on the publication panels -- `\panel{}` in the .tex typesets
the letters, and drawing them here too produced duplicate labels once already.

Run:  python figure_fmsa_vdw_family.py           (composite PNG)
      python figure_fmsa_vdw_family.py --panels  (one PDF+PNG per panel)
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC = ('numerical_notes/results/compare_energy_compress/phase_lines/'
       'fmsa_vdw_loop_family.json')
PL = ('numerical_notes/results/compare_energy_compress/phase_lines/'
      'fmsa_phase_lines.json')
OUT = 'numerical_notes/results/compare_energy_compress/phase_lines'
STEM = 'fmsa_vdw_family'

PUB_W = 7.06 / 2          # one panel of a 2-up full-width figure
# ONE palette -- `figure_style.py`.  The spinodal was drawn in `#2a78d6`, which
# is OZ+MSA's blue; it is not a model, it is a construction line of the orange
# one, so it takes the neutral AUX colour instead.
import figure_style as S                                        # noqa: E402
C_DP = S.color('FMSA-DP')
C_SP = S.AUX['spinodal']
C_TIE = S.AUX['tie']
RHO_OF_ETA = 6.0 / np.pi


def _cmap(n):
    """Cold isotherms dark, warm ones light -- the loop shrinks with the ink."""
    return [plt.cm.autumn(0.05 + 0.62 * i / max(n - 1, 1)) for i in range(n)]


def panel_loops(ax, d):
    iso = d['isotherms']
    cols = _cmap(len(iso))
    # ⚠ The liquid branch climbs to beta*p ~ 1.1 while the loops live inside
    # +-0.2.  Autoscaling on everything drawn therefore FLATTENS THE LOOP, which
    # is the only thing this panel is for -- the same trap `fmsa_vdw_loop.py`
    # records.  So the y-axis is set to the loops and the branch runs off the
    # top, and the label goes at each curve's own minimum rather than at the
    # right edge, which the zoom would push out of frame.
    for row, c in zip(iso, cols):
        rho = np.array(row['rho'])
        p = np.array(row['p'])
        m = rho <= 0.72
        ax.plot(rho[m], p[m], color=c, lw=1.8, zorder=3)
        if 'rho_g' in row:
            ax.plot([row['rho_g'], row['rho_l']],
                    [row['p_coex'], row['p_coex']], color=C_TIE,
                    lw=0.9, ls=(0, (4, 2)), zorder=4)
            ax.plot([row['rho_g'], row['rho_l']],
                    [row['p_coex'], row['p_coex']], 'o', color=c,
                    ms=3.6, mec='k', mew=0.4, zorder=5)
        # The loop's local minimum sits at the UPPER spinodal density -- that is
        # what `dp/drho = 0` turning positive again means.  A plain argmin finds
        # the global minimum instead, which for the warm isotherms is the
        # rho -> 0 end, and stacks every warm label in the bottom-left corner.
        xl = row['spinodal'][1] if row['spinodal'] else rho[m][-1]
        ax.annotate(f"{row['T_frac']:.2f}",
                    xy=(xl, float(np.interp(xl, rho, p))), xytext=(0, -9),
                    textcoords='offset points', fontsize=6.5, color=c,
                    va='top', ha='center')
    ax.axhline(0.0, color='k', lw=0.6, alpha=0.35, zorder=1)
    ax.set_xlabel(r'$\rho^*$')
    ax.set_ylabel(r'$\beta p\,\sigma^3$')
    ax.set_xlim(0, 0.74)
    ax.set_ylim(-0.23, 0.20)
    ax.legend(handles=[
        Line2D([], [], color=cols[0], lw=1.8,
               label=r'isotherms, labelled by $T/T_c$'),
        Line2D([], [], color=C_TIE, lw=0.9, ls=(0, (4, 2)), marker='o',
               ms=3.6, mec='k', mew=0.4, label='Maxwell tie line'),
    ], fontsize=7, loc='upper left')


def panel_lines(ax, d, pl):
    iso = [r for r in d['isotherms'] if 'rho_g' in r]
    T = np.array([r['T'] for r in iso])
    rg = np.array([r['rho_g'] for r in iso])
    rl = np.array([r['rho_l'] for r in iso])
    sp = [(r['T'], r['spinodal']) for r in iso if r['spinodal']]
    Ts = np.array([t for t, _ in sp])
    s_lo = np.array([s[0] for _, s in sp])
    s_hi = np.array([s[1] for _, s in sp])

    Tc = d['Tc']
    rc = pl['HSY']['C']['etac'] * RHO_OF_ETA

    # binodal and spinodal, each closed onto the critical point.  The closure is
    # drawn dotted: it is the one piece not read off an isotherm in (a).
    ax.plot(np.r_[rg, rc, rl[::-1]], np.r_[T, Tc, T[::-1]], '-',
            color=C_DP, lw=2.0, marker='o', ms=4, zorder=4, label='binodal')
    sx = np.r_[s_lo, rc, s_hi[::-1]]
    sy = np.r_[Ts, Tc, Ts[::-1]]
    ax.plot(sx, sy, '--', color=C_SP, lw=1.8, marker='s', ms=3.4, zorder=3,
            label=r'spinodal, from the extrema of (a)')
    # A closed polygon, not fill_between: the path doubles back at the apex, so
    # fill_between against a constant draws a box round the dome instead of the
    # dome.  ax.fill takes the traced curve as the boundary it already is.
    ax.fill(sx, sy, color=C_SP, alpha=0.10, zorder=1, lw=0)
    ax.plot([rc], [Tc], '*', ms=13, color='k', zorder=6)
    ax.annotate(r'$T_c$', xy=(rc, Tc), xytext=(4, 4),
                textcoords='offset points', fontsize=8)
    ax.set_xlabel(r'$\rho^*$')
    ax.set_ylabel(r'$T^*$')
    ax.legend(fontsize=7, loc='lower center')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panels', action='store_true',
                    help='one PDF+PNG per panel, at publication width')
    args = ap.parse_args()

    d = json.loads(pathlib.Path(SRC).read_text())
    pl = json.loads(pathlib.Path(PL).read_text())
    out = pathlib.Path(OUT)

    if args.panels:
        plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9,
                             'xtick.labelsize': 8, 'ytick.labelsize': 8,
                             'pdf.fonttype': 42})
        for letter, fn in (('a', panel_loops), ('b', panel_lines)):
            fig, ax = plt.subplots(figsize=(PUB_W, PUB_W * 0.82))
            fn(ax, d) if letter == 'a' else fn(ax, d, pl)
            fig.tight_layout(pad=0.35)
            for ext in ('pdf', 'png'):
                p = out / f'{STEM}_{letter}.{ext}'
                fig.savefig(p, dpi=200)
                print(f'  {p}')
            plt.close(fig)
        return

    plt.rcParams.update({'font.size': 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    panel_loops(ax1, d)
    panel_lines(ax2, d, pl)
    ax1.set_title('(a) the loop, at six temperatures')
    ax2.set_title('(b) the lines those loops generate')
    fig.suptitle('FMSA-DP: the van der Waals loop is systematic, and the '
                 'spinodal is its own turning points', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = out / f'{STEM}.png'
    fig.savefig(p, dpi=150)
    print(f'  {p}')


if __name__ == '__main__':
    main()
