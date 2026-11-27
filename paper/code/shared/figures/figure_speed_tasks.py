#!/usr/bin/env python3
"""Fig. 7: the speed advantage is a spectrum over tasks, not one number.

The existing speed study prices ONE task -- build and evaluate c_ij(r) on a grid
-- and reports 37x.  That understates the case for everything the closed form is
actually used inside: a free energy, a gradient, a root-find, a phase
equilibrium.  Those are the operations an analytic solution exists for.

Three panels:

  (a) one free energy and its gradient, absolute wall time, four ways.
  (b) why the spread exists.  FMSA-DP costs the same at every state; OZ does not,
      and its cost climbs exactly where the thermodynamics lives.  This is the
      structural claim, and it is the one a single median hides.
  (c) attempts that terminated.

⭐ EXACT MSA IS THE FOURTH LEG, AND IT SPLITS THE FIGURE'S TWO CLAIMS APART.
On (a) and (b) it sits with FMSA-DP: it is analytic, it is the same order of
cost, and it is flat across the states.  So the speed claim is about closed forms
against iteration and NOT about this construction against every alternative --
which is why the models are ordered analytic-then-iterative throughout.

On (c) it is alone.  FMSA-DP cannot fail (an explicit function of the state) and
OZ can fail to CONVERGE, but Blum & Hoye's solution is the real root of a
non-linear algebraic system and can fail to EXIST -- inside the fold there is
nothing to converge to.  Equal speed is not equal usefulness when one of the two
methods is undefined where the phase diagram is decided, and (c) is where that
shows.

Sources.  All four legs of (a)/(b) are timed over the same 12 states (dilute,
moderate, dense, supercritical, and eight coexistence points):
`results/speed/speed_free_energy.json` for the first three,
`results/speed/speed_exact_msa.json` for exact MSA.  Panel (c) reads
`results/speed/solvability_exact_msa.json` and `results/speed/binodal_x085.json`.

⚠ Those four JSONs used to live in a session scratchpad, which is volatile; they
are the only copy of the timings and now sit under `numerical_notes/results/`.

⚠ The end-to-end bubble-point ratio (~40x) is deliberately NOT plotted.  It was
timed while this machine was running an MD batch, so it is contaminated; it is
quoted in the text as approximate and will be re-measured on a quiet machine.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = pathlib.Path('numerical_notes/results/speed')
FE = RES / 'speed_free_energy.json'
EX = RES / 'speed_exact_msa.json'
SOLV = RES / 'solvability_exact_msa.json'
BINO = RES / 'binodal_x085.json'
OUT = 'numerical_notes/results/fmsa_dp_speed_tasks.png'

# Keys are the JSONs'; the hexes must match compare_speed_dp_hncb.py's COL
# exactly, because ONE key is printed for both speed figures and it is built over
# there.  FMSA-DP was #e8a33d here and #e08c36 there -- close enough to look like
# the same orange side by side, far enough apart to be a different colour under
# the shared legend.
#
# ORDER: analytic (FMSA-DP, exact MSA) then iterative (OZ-MSA, OZ-HNCB).
import figure_style as S                                        # noqa: E402
C = {m: S.color(m) for m in ('FMSA-DP', 'MSA-exact', 'OZ-MSA', 'OZ-HNCB')}
MK = {m: S.marker(m) for m in C}      # aliases resolve in figure_style.ALIAS
LS = {m: S.linestyle(m) for m in C}
PRETTY = {'FMSA-DP': 'FMSA-DP', 'MSA-exact': 'exact MSA',
          'OZ-MSA': 'OZ+MSA', 'OZ-HNCB': 'OZ+HNCB'}
MODELS = list(C)


def load_rows() -> list[dict]:
    """The 12 states, with exact MSA joined in on the state name.

    Joined rather than concatenated: the two runs share the state list but not
    its order, and a positional merge would silently pair a dense state with a
    coexisting one.
    """
    rows = json.loads(FE.read_text())
    ex = {r['state']: r for r in json.loads(EX.read_text())}
    for r in rows:
        e = ex.get(r['state'])
        r['MSA-exact'] = ({'A': e['A'], 'mu_p': e['mu_p']} if e and e.get('ok')
                          else {'A': None, 'mu_p': None})
    return rows


def solvability() -> dict:
    """Fraction of attempts that terminated, per model, for the three tasks.

    ⚠ The binodal row is DERIVED from `binodal_x085.json`, not hardcoded.  It
    used to carry FMSA-DP = 20/26 while its own source file holds 17 -- the run
    reached T* = 1.16, lost T* = 1.18 to a non-converged outer Newton, and hit
    the trivial root at 1.20, which is its critical line and not a failure.  A
    number that contradicts the file it came from is worse than a number that is
    partly physics, so it is now read from the file every time.
    """
    # 496-pt grid.  OZ-MSA's figure is not missing: md_grid_dcf.md records that
    # of the 94 failures out of 2480, every one was OZ+HNCB.  So MSA is 2480/2480.
    grid = {'FMSA-DP': 100.0, 'OZ-MSA': 100.0,
            'OZ-HNCB': 100 * (1 - 94 / 2480), 'MSA-exact': float('nan')}
    bub = {'FMSA-DP': 100.0, 'OZ-MSA': 100.0,
           'OZ-HNCB': 100 * 11 / 14, 'MSA-exact': float('nan')}
    if SOLV.exists():
        s = json.loads(SOLV.read_text())
        if 'grid' in s:
            grid['MSA-exact'] = 100.0 * s['grid']['ok'] / s['grid']['n']
        if 'bubble' in s:
            bub['MSA-exact'] = 100.0 * s['bubble']['ok'] / s['bubble']['n']

    n_att = 26                       # T* = 0.80 .. 1.30 by 0.02
    bino = {m: float('nan') for m in MODELS}
    if BINO.exists():
        b = json.loads(BINO.read_text())
        for k, v in b.items():
            if k in bino:
                bino[k] = 100.0 * len(v) / n_att
    if SOLV.exists():
        s = json.loads(SOLV.read_text())
        if 'binodal' in s:
            bino['MSA-exact'] = 100.0 * s['binodal']['ok'] / n_att
    return {'496-pt grid': grid, '14 bubble pts': bub,
            'binodal, 26 $T^*$': bino}


def main() -> None:
    rows = load_rows()
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.4))
    h, off = 0.20, 0.21          # four bars per row, not three

    # ⚠ NO INLINE "N×" LABELS.  With three series they were legible; with four
    # the bars of one row sit within a factor of two of each other on a log
    # axis, so a label anchored past a bar end lands on its neighbour.  The
    # ratios are in the caption instead, as a median with a range rather than
    # eight separate roundings to the nearest whole multiple.

    # ---- (a) by task, absolute wall time, four ways -------------------------
    # Absolute seconds, not a ratio: a ratio can only ever show TWO series, and
    # the comparison wanted here is the two closed forms against both solvers.
    TASKS = [('free energy', 'A'), (r'$\mu_i,\,p$', 'mu_p')]
    y = np.arange(len(TASKS))
    for k, m in enumerate(MODELS):
        med = [float(np.median([r[m][key] for r in rows if r[m][key]]))
               for _, key in TASKS]
        ax.barh(y + (k - 1.5) * off, med, height=h, color=C[m], label=PRETTY[m])
    ax.set_yticks(y)
    ax.set_yticklabels([t for t, _ in TASKS])
    ax.set_xscale('log')
    ax.set_xlim(right=ax.get_xlim()[1] * 1.35)
    ax.set_xlabel('wall time [s]')
    ax.set_title('one free energy, and its gradient\n(median over 12 states)')

    # ---- (b) the same cost, state by state ---------------------------------
    SHORT = {'dilute': 'dilute', 'moderate': 'moderate', 'dense': 'dense',
             'supercritical': 'supercrit.'}
    lbl = [SHORT.get(r['state'],
                     r['state'].replace('coex liquid ', 'L ')
                               .replace('coex vapour ', 'V ')
                               .replace('T*=', ''))
           for r in rows]
    yb = np.arange(len(rows))
    for k, m in enumerate(MODELS):
        ax2.barh(yb + (k - 1.5) * off,
                 [r[m]['A'] if r[m]['A'] else 0.0 for r in rows], height=h,
                 color=C[m], label=PRETTY[m])
    ax2.set_yticks(yb)
    ax2.set_yticklabels(lbl, fontsize=6.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('wall time, one free energy [s]')
    ax2.set_title('flat for both closed forms, not for OZ\n'
                  '($\\times$1.9 and $\\times$2.0 against $\\times$17)')

    # ---- (c) solvability ---------------------------------------------------
    # ⚠ The binodal row counts temperatures traced before the solver stopped
    # ADVANCING, not dome height -- the models have different critical points, so
    # a shorter dome is partly physics.  What is not physics: OZ-HNCB stopped at
    # T*=0.98 against a critical point near 1.30.
    frac = solvability()
    TC = list(frac)
    yc = np.arange(len(TC))
    for k, m in enumerate(MODELS):
        v = [0 if frac[t][m] != frac[t][m] else frac[t][m] for t in TC]
        ax3.barh(yc + (k - 1.5) * off, v, height=h, color=C[m], label=PRETTY[m])
        for yy, t in zip(yc + (k - 1.5) * off, TC):
            if frac[t][m] != frac[t][m]:     # NaN: not measured, NOT zero
                ax3.text(2, yy, 'not measured', va='center', fontsize=6.5,
                         color=C[m], style='italic')
    ax3.set_yticks(yc)
    ax3.set_yticklabels(TC)
    ax3.set_xlim(0, 118)
    ax3.axvline(100, color='k', lw=0.6, ls=':')
    ax3.set_xlabel('attempts that terminated [%]')
    ax3.set_title('only one of the four can fail to EXIST')

    # This figure also has three panels, so its key also lands in an empty
    # fourth slot and costs no height -- which is why it gets one of its own
    # rather than borrowing Fig. 6's.  What was worth removing was the earlier
    # arrangement: two identical FULL-WIDTH strips, each taking a row.
    #
    # Bars only here, so the entries are swatches without the marker half that
    # Fig. 6's scaling panels need.  Same colours, same order, same names.
    from matplotlib.patches import Patch

    import figure_legend
    figure_legend.emit([Patch(facecolor=c, edgecolor='none') for c in C.values()],
                       [PRETTY[m] for m in MODELS], 'fmsa_dp_speed_tasks',
                       handlelength=2.6)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print('  wrote', OUT)
    for m in MODELS:
        v = [r[m]['A'] for r in rows if r[m]['A']]
        print(f'  {PRETTY[m]:10s} {min(v):.3f}-{max(v):.3f} s  '
              f'(x{max(v)/min(v):.1f}, {len(v)}/{len(rows)} states)')
    for m in MODELS:
        rr = [r[m]['A'] / r['FMSA-DP']['A'] for r in rows
              if r[m]['A'] and r['FMSA-DP']['A']]
        print(f'  {PRETTY[m]:10s} / FMSA-DP  median {np.median(rr):6.2f}x   '
              f'range {min(rr):.2f}-{max(rr):.2f}x')
    for t, d in solvability().items():
        print(f'  {t:20s} ' + '  '.join(f'{PRETTY[m]} {d[m]:.1f}%'
                                        for m in MODELS))


if __name__ == '__main__':
    main()
