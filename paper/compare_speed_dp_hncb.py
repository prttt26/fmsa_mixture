#!/usr/bin/env python3
"""How much faster is FMSA-DP than OZ+HNCB, and where does the difference come from.

The two methods do not do the same amount of work, and the reason is structural
rather than an implementation detail:

  OZ+HNCB  solves a nonlinear closure self-consistently.  Every iteration is a
           pair of transforms on an n_r grid, and the number of iterations is a
           property of the state point.  The grid has to be fine enough for the
           ANSWER to be converged, so cost and accuracy are coupled.
  FMSA-DP  evaluates a closed form.  c^(1)(r) is a finite sum of
           polynomial x exponential pieces and chat(k) is an explicit matrix
           expression, so there is no iteration, no grid, and no convergence
           criterion -- cost is whatever you ask to evaluate.

So the honest comparison is not one number.  This measures four things:

  1. one state point, end to end, at the resolution the benchmarks used;
  2. how each scales with the radial resolution n_r;
  3. how each scales with the number of species;
  4. what the speed costs in accuracy, and what it buys in robustness --
     both taken from the multi-state benchmarks rather than re-derived here.

Absolute seconds are NumPy-on-one-machine numbers and will move with the
implementation; the ratios and the scalings are the transferable part.

    python compare_speed_dp_hncb.py               # full run + figure
    python compare_speed_dp_hncb.py --quick
"""

from __future__ import annotations

import argparse
import contextlib
import json
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / 'numerical_notes' / 'results'
FIG = OUT_DIR / 'fmsa_dp_speed.png'

# Encoding shared by BOTH speed figures -- this script's panels and
# figure_speed_tasks.py's.  ONE key is printed for the pair (in fig:speed), so
# the two scripts have to agree exactly; figure_speed_tasks.py repeats these
# hexes with a pointer back here rather than importing, since importing this
# module would drag in the solvers it times.
#
# ORDER IS THE ARGUMENT: analytic first (FMSA-DP, exact MSA), iterative second
# (OZ+MSA, OZ+HNCB).  The gap in these figures is not this construction against
# the alternatives, it is closed forms against iteration -- and exact MSA sits on
# FMSA-DP's side of it.  #7b3fa0 is the purple Figs. 11-13 already use for exact
# MSA, so one model keeps one colour across the paper.
# ⭐ ONE palette for the whole paper -- `figure_style.py`.  This script had
# OZ+MSA as a SQUARE and OZ+HNCB as a CIRCLE while the DCF/RDF figures had them
# the other way round: a reader who learnt one convention was actively wrong in
# the other.  The DCF/RDF assignment wins on figure count and is now the only one.
import figure_style as S                                        # noqa: E402
COL = {m: S.color(m) for m in ('FMSA-DP', 'exact MSA', 'OZ+MSA', 'OZ+HNCB')}
MK = {m: S.marker(m) for m in COL}
LS = {m: S.linestyle(m) for m in COL}

# (label, sigma, eps, x, rho*, T*) -- spread over packing and size ratio, because
# the OZ iteration count is a property of the state point and FMSA-DP's is not.
STATES = [
    ('reference η=0.39',  [1.0, 2.0], [1.0, 0.5], [0.25, 0.75], 0.139, 1.0),
    ('dense η=0.47',      [1.0, 2.0], [1.0, 0.5], [0.25, 0.75], 0.170, 1.0),
    ('dilute η=0.03',     [1.0, 2.0], [1.0, 0.5], [0.25, 0.75], 0.010, 1.0),
    ('ratio 1.5',         [1.0, 1.5], [1.0, 0.5], [0.25, 0.75], 0.312, 1.0),
    ('ratio 3.0',         [1.0, 3.0], [1.0, 0.5], [0.25, 0.75], 0.0425, 1.0),
    ('x1=0.9 η=0.40',     [1.0, 2.0], [1.0, 0.5], [0.90, 0.10], 0.400, 1.0),
    ('equal size',        [1.0, 1.0], [1.0, 0.5], [0.50, 0.50], 0.800, 1.5),
    ('hot T*=3',          [1.0, 2.0], [1.0, 0.5], [0.25, 0.75], 0.139, 3.0),
]


def timed(fn, repeats=1):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        best = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = fn()
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
    return best, out


def build_dp(sig, eps, x, rho, T):
    from fmsa_double_prop import FMSA_double_prop
    return FMSA_double_prop(np.array(sig), np.array(eps), np.array(x), rho, T,
                            n_tails=2, verbose=False)


def build_ga(sig, eps, x, rho, T):
    from fmsa_ga_matrix_mix import FMSA_GA_matrix_mix
    return FMSA_GA_matrix_mix(np.array(sig), np.array(eps), np.array(x), rho, T,
                              n_tails=2, verbose=False)


def solve_oz(ga, n_r, closure='hncb'):
    """One converged OZ solve.  `closure` selects hncb or msa.

    Both are timed, because the paper's comparison is three-way: quoting only
    OZ+HNCB leaves the reader unable to tell how much of the gap is the ITERATION
    and how much is that particular closure's difficulty.
    """
    from oz_numerical import OZSolverMSA
    s = OZSolverMSA(ga, n_r=n_r)
    try:
        s.solve(method='diis', closure=closure, tol=1e-6, max_iter=500,
                alpha=0.2, n_coupling_steps=5)
    except Exception:
        pass
    return s


def solve_hncb(ga, n_r):
    return solve_oz(ga, n_r, 'hncb')


def solve_msa(ga, n_r):
    return solve_oz(ga, n_r, 'msa')


def solve_exact(dp):
    """One converged exact-MSA solution, from the SAME Yukawa system.

    `msa_exact_mix.from_fmsa` reads σ_ij, z_ij and K_ij off the FMSA object, so
    this times the same physics the other three legs are solving, not a
    re-parameterised look-alike.

    ⚠ WHAT THIS DOES NOT INCLUDE, and it matters for reading panel (a): the
    real-space c_ij(r).  Blum & Høye's solution is delivered as the Baxter
    factor and its Laplace transforms, and `msa_exact_mix` implements it that
    way; there is no r-space evaluator to time.  So the bar is build + root-find
    -- a converged SOLUTION -- while FMSA-DP's is build + a curve on 1000 points
    and OZ's is build + a curve on the whole grid.  It is therefore a LOWER
    bound on the like-for-like cost, and the conclusion it supports (the two
    analytic methods are the same order, both far under the iterative ones)
    survives that: the missing piece is an evaluation, and evaluation is the
    part FMSA-DP is already being charged for.
    """
    import msa_exact_mix as X
    return X.solve_bh_mix(X.from_fmsa(dp))


# --------------------------------------------------------------------------
def sec_per_state(states, n_r=4096, n_pts=1000):
    """One state point, end to end: a converged c_ij(r) either way."""
    import msa_exact_mix as X
    print('=' * 92)
    print(f'  1.  ONE STATE POINT, END TO END   (OZ at n_r = {n_r}; '
          f'FMSA-DP evaluated on {n_pts} r points)')
    print('=' * 92)
    print(f"  {'state':<20} {'DP build':>9} {'DP c(r)':>9} {'DP total':>9} | "
          f"{'exact tot':>9} | {'OZ build':>9} {'OZ solve':>9} {'iters':>6} "
          f"{'OZ total':>9} | {'speed-up':>9}")
    rows = []
    for label, sig, eps, x, rho, T in states:
        tb, dp = timed(lambda: build_dp(sig, eps, x, rho, T))
        r = np.linspace(1e-3, 5.0 * float(np.max(dp.R_ij)), n_pts)
        tc, _ = timed(lambda: dp.get_c_total(r, hs_ref='py'))
        te, sol = timed(lambda: solve_exact(dp))
        # ⭐ CHARGE THE k -> r INVERSION.  FMSA-DP's bar is build + a curve on
        # n_pts points; exact MSA's used to be the SOLVE alone, because there was
        # no r-space evaluator to time.  `msa_exact_mix.c_of_r` is that evaluator,
        # so the bar is now like-for-like.  n_r = 4096 is the grid the OZ legs run
        # at and the resolution at which the inverted outer tail is right to
        # 2.6e-3; asking for the 1000 arbitrary r that FMSA-DP is charged for
        # would understate it, since a transform delivers a grid, not a function.
        ti = float('nan')
        if sol is not None:
            ti, _ = timed(lambda: X.c_of_r(sol, n_r=n_r))
        if sol is None:
            te = float('nan')               # no real root: absence, not slowness
        tg, ga = timed(lambda: build_ga(sig, eps, x, rho, T))
        ts, s = timed(lambda: solve_hncb(ga, n_r))
        tm, sm = timed(lambda: solve_msa(ga, n_r))          # three-way, not two
        it = getattr(s, '_n_iters', None)
        ok = getattr(s, '_c_converged', None) is not None
        itm = getattr(sm, '_n_iters', None)
        dp_tot, oz_tot, msa_tot = tb + tc, tg + ts, tg + tm
        rows.append((label, tb, tc, dp_tot, tg, ts, it, oz_tot, oz_tot / dp_tot, ok,
                     msa_tot, msa_tot / dp_tot, itm, te + ti, te, ti))
        print(f'  {label:<20} {tb:9.3f} {tc:9.3f} {dp_tot:9.3f} | '
              f'{te + ti:9.3f} | '
              f'{tg:9.3f} {ts:9.3f} {str(it) if ok else "FAIL":>6} {oz_tot:9.3f} | '
              f'{oz_tot / dp_tot:8.1f}x  msa {msa_tot:7.3f} {msa_tot / dp_tot:6.1f}x')
    import json as _json, pathlib as _pl
    _pl.Path('numerical_notes/results/fmsa_dp_speed_states.json').write_text(_json.dumps(
        [{'state': r[0], 'FMSA-DP': r[3], 'exact MSA': r[13],
          'exact MSA solve': r[14], 'exact MSA k->r': r[15],
          'OZ-HNCB': r[7], 'OZ-MSA': r[10]} for r in rows], indent=1))
    sp = np.array([r[8] for r in rows])
    spm = np.array([r[11] for r in rows])
    ex = np.array([r[13] / r[3] for r in rows], dtype=float)
    print(f'  median speed-up vs MSA  {np.median(spm):.1f}x   '
          f'range {spm.min():.1f}-{spm.max():.1f}x')
    fin = ex[np.isfinite(ex)]
    if fin.size:
        print(f'  exact MSA / FMSA-DP     {np.median(fin):.2f}x   '
              f'range {fin.min():.2f}-{fin.max():.2f}x   '
              f'({fin.size}/{ex.size} states returned a solution)')
        so = np.array([r[14] for r in rows], float)
        iv = np.array([r[15] for r in rows], float)
        g = np.isfinite(so) & np.isfinite(iv)
        print(f'     of which  solve {np.median(so[g]):.3f} s   '
              f'k->r inversion {np.median(iv[g]):.3f} s   '
              f'(inversion is {100*np.median(iv[g]/(so[g]+iv[g])):.0f}% of the bar)')
    print(f'\n  median speed-up {np.median(sp):.1f}x   range {sp.min():.1f}-{sp.max():.1f}x')
    print('  The spread is the OZ iteration count, which is a property of the state')
    print('  point; FMSA-DP does the same work everywhere.')
    return rows


def sec_grid(state, n_r_list=(1024, 2048, 4096, 8192, 16384)):
    """Radial resolution: the axis where the two diverge fastest."""
    label, sig, eps, x, rho, T = state
    print('\n' + '=' * 92)
    print(f'  2.  SCALING WITH RADIAL RESOLUTION   ({label})')
    print('=' * 92)
    import msa_exact_mix as X
    _, dp = timed(lambda: build_dp(sig, eps, x, rho, T))
    _, ga = timed(lambda: build_ga(sig, eps, x, rho, T))
    # ⭐ The exact-MSA SOLVE never sees n_r -- its unknowns are the 2N^2*omega
    # amplitudes, not samples of a function.  But the solve is not a c_ij(r):
    # Blum & Hoye deliver a Baxter factor whose chat(k) is closed form, and
    # getting to r-space needs the k -> r INVERSION, which does scale with the
    # grid.  Charging only the solve drew a flat line for a quantity nobody can
    # use, so the panel now carries both: the solve (flat, correct) and
    # solve + inversion (the like-for-like cost of one c_ij(r)).
    te, sol = timed(lambda: solve_exact(dp), repeats=3)
    X._gate_c_of_r(sol)
    print(f"  {'n_r':>7} {'OZ+HNCB solve':>15} {'iters':>6} | "
          f"{'DP c(r) on n_r pts':>19} {'ratio':>8} | "
          f"{'exact k->r':>11} {'+solve':>9}")
    out = []
    for n_r in n_r_list:
        ts, s = timed(lambda: solve_hncb(ga, n_r))
        tmsa, _ = timed(lambda: solve_msa(ga, n_r))
        r = np.linspace(1e-3, 5.0 * float(np.max(dp.R_ij)), n_r)
        tc, _ = timed(lambda: dp.get_c_total(r, hs_ref='py'))
        ti, _ = timed(lambda: X.c_of_r(sol, n_r=n_r), repeats=3)
        it = getattr(s, '_n_iters', None)
        out.append((n_r, ts, tc, it, tmsa, te, ti))
        print(f'  {n_r:7d} {ts:15.3f} {str(it):>6} | {tc:19.3f} {ts / tc:8.1f}x'
              f' | {ti:11.3f} {te + ti:9.3f}')
    print(f'  exact MSA solve alone: {te:.3f} s at EVERY n_r -- the SOLVE has no grid')
    print('\n  All three r-space legs are linear in n_r, so the ratios are roughly flat')
    print('  -- but the three n_r mean different things.  The OZ answer is only as good')
    print('  as its grid (the benchmark had to run at 4096; at 1024 the reference-state')
    print('  chat(0) error was 2.2% instead of 0.6%).  FMSA-DP is exact at every r it')
    print('  is asked for, so n_r is a plotting choice, not an accuracy knob.  Exact')
    print('  MSA is in between: its SOLVE has no grid, but its c_ij(r) is a transform')
    print('  of chat(k) and inherits one -- see the `outer tail` entry of the gate')
    print('  above, which is ringing from the contact jump and falls with n_r.')
    return out


def sec_species(n_list=(2, 3, 4), eta=0.35, T=1.0, n_r=4096):
    """Number of species -- the axis a mixture study actually grows along.

    Density is set by PACKING FRACTION, not by rho*: adding a bigger species at
    fixed rho* drives eta past 1 and the comparison then measures an impossible
    state rather than a method.
    """
    print('\n' + '=' * 92)
    print('  3.  SCALING WITH THE NUMBER OF SPECIES')
    print('=' * 92)
    print(f'  every N at the same packing fraction eta = {eta}')
    print(f"  {'N':>3} {'rho*':>7} {'DP build':>9} {'DP c(r)':>9} {'DP total':>9} | "
          f"{'OZ solve':>9} {'iters':>6} | {'speed-up':>9}")
    out = []
    for N in n_list:
        sig = list(np.linspace(1.0, 1.0 + 0.5 * (N - 1), N))
        eps = [1.0] * N
        x = [1.0 / N] * N
        rho = eta / ((np.pi / 6.0) * float(np.sum(np.array(x) * np.array(sig) ** 3)))
        try:
            tb, dp = timed(lambda: build_dp(sig, eps, x, rho, T))
            r = np.linspace(1e-3, 5.0 * float(np.max(dp.R_ij)), 1000)
            tc, _ = timed(lambda: dp.get_c_total(r, hs_ref='py'))
            _, ga = timed(lambda: build_ga(sig, eps, x, rho, T))
            ts, s = timed(lambda: solve_hncb(ga, n_r))
            tm, _ = timed(lambda: solve_msa(ga, n_r))
            # This is the axis where the two analytic methods part company.
            # FMSA-DP's build grows with N; exact MSA's UNKNOWN COUNT does,
            # 2N^2*omega with omega = n_tails*N(N+1)/2 -- 48 at N=2, 216 at
            # N=3, 640 at N=4 -- and a dense root-find on that is a different
            # curve, not a steeper version of the same one.
            try:
                import msa_exact_mix as X
                te, sol = timed(lambda: solve_exact(dp))
                # Same charge as panels (a) and (b): a c_ij(r), not a solution.
                # The inversion is nearly N-independent, so it shifts N = 2 by
                # roughly a tenth and N = 4 not at all -- which is itself the
                # point, the growth here is the UNKNOWN COUNT and not the grid.
                ti = 0.0 if sol is None else timed(lambda: X.c_of_r(sol, n_r=n_r))[0]
                if sol is None:
                    te = float('nan')
            except Exception:                                    # noqa: BLE001
                te, ti = float('nan'), float('nan')
            it = getattr(s, '_n_iters', None)
            ok = getattr(s, '_c_converged', None) is not None
            out.append((N, tb + tc, ts, it, ok, tm, te + ti, te, ti))
            print(f'  {N:3d} {rho:7.3f} {tb:9.3f} {tc:9.3f} {tb + tc:9.3f} | '
                  f'{ts:9.3f} {str(it) if ok else "FAIL":>6} | {ts / (tb + tc):8.1f}x'
                  f'   exact {te + ti:8.3f}  (solve {te:.3f} + k->r {ti:.3f})')
        except Exception as e:                                   # noqa: BLE001
            print(f'  {N:3d}  not supported by one of the two: {type(e).__name__}')
    return out


def sec_products(state, n_r=4096):
    """The other things each method is asked for, priced separately."""
    label, sig, eps, x, rho, T = state
    print('\n' + '=' * 92)
    print(f'  4.  WHAT ELSE EACH ONE COSTS   ({label})')
    print('=' * 92)
    _, dp = timed(lambda: build_dp(sig, eps, x, rho, T))
    _, ga = timed(lambda: build_ga(sig, eps, x, rho, T))
    ts, s = timed(lambda: solve_hncb(ga, n_r))
    k = np.linspace(1e-3, 50.0, 1000)
    tk, _ = timed(lambda: dp.get_chat_k(k))
    tgf, _ = timed(lambda: dp.get_g_fourier(k_max=1500.0, n_k=200000))
    rq = np.array([1.05, 1.5, 2.0]) * float(dp.R_ij[0, 0])
    tgl, _ = timed(lambda: dp.get_g_laplace(rq))
    print(f'  chat_ij(k), 1000 wavevectors     FMSA-DP {tk * 1e3:8.1f} ms   '
          f'({tk / 1000 * 1e6:.1f} us per k)')
    print(f'                                   OZ+HNCB  — only on its own k grid, '
          f'after the {ts:.2f} s solve')
    print(f'  g_ij(r), whole curve             FMSA-DP {tgf:8.3f} s   (k-space route)')
    print(f'                                   OZ+HNCB {ts:8.3f} s   (free with the solve)')
    print(f'  g_ij(r), 3 points only           FMSA-DP {tgl * 1e3:8.1f} ms  '
          f'({tgl / 3 * 1e3:.1f} ms per point, Laplace route)')
    print(f'                                   OZ+HNCB {ts:8.3f} s   (no cheaper for 3 than for 4096)')
    print('\n  This is where the structural difference shows: FMSA-DP can be asked for')
    print('  three values and charge for three.  An iterative solver cannot.')
    return dict(chat_k=tk, g_fourier=tgf, g_laplace=tgl, oz=ts)


def sec_context():
    print('\n' + '=' * 92)
    print('  5.  WHAT THE SPEED COSTS, AND WHAT IT BUYS')
    print('=' * 92)
    print("""  Measured elsewhere, not here -- quoted so the trade is visible:

  accuracy (multi-state benchmarks, median NRMSE, lower is better)
      DCF inner core     FMSA-DP(py) 0.097   OZ+HNCB 0.050        benchmark/all_samples
      RDF whole range    FMSA-DP     0.124   OZ+HNCB 0.052        benchmark_rdf/all_samples
      RDF first peak     FMSA-DP     10.4 %  OZ+HNCB  3.7 %
      HCTY contact       FMSA-DP     18.0 %  OZ+HNCB  3.0 %       benchmark_rdf/literature

  robustness (same runs)
      OZ+HNCB failed to converge on 7 of 186 DCF state points and 7 of 168 RDF
      ones -- all at high packing with a small-species-rich composition, i.e.
      exactly where it is also slowest.  FMSA-DP is closed form: 0 failures, and
      the cost does not depend on the state point at all.

  So the trade is roughly a factor 20-30 in time and a guarantee of termination,
  against a factor 2-3 in accuracy.  Which side of that is worth taking depends
  on whether the answer is the end product or an inner loop.""")


# --------------------------------------------------------------------------
def legend_strip():
    """The one key for BOTH speed figures.

    Every entry carries both encodings at once: the bar swatch of the barh
    panels and the marked line of the scaling panels.  Two separate strips said
    the same model names twice, and -- because each was built from thick colour
    bars -- neither of them ever mentioned the markers that panels (b) and (c)
    distinguish the curves by.

    Four entries now, ordered analytic-then-iterative, and that order is the
    figure's argument rather than a layout choice: exact MSA belongs beside
    FMSA-DP on cost and opposite it on solvability.

    Emitted in BOTH shapes -- see `figure_legend.py` for why the choice belongs
    to the .tex and not here.  Fig. 6 has three panels, so today the block goes
    in the empty fourth slot and the strip is unused; a fourth panel would flip
    that without touching this script.

    Written on its own so it can be refreshed without re-timing anything:

        python compare_speed_dp_hncb.py --legend-only
    """
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    import figure_legend

    # ndivide=None splits the handle box in two -- swatch left, marked line
    # right.  The default (1) would draw them on top of each other.
    # The marked line now carries the model's LINESTYLE too, so the key states
    # the solid-analytic / dashed-iterative rule instead of leaving the panels
    # to imply it.
    handles = [(Patch(facecolor=c, edgecolor='none'),
                Line2D([], [], color=c, marker=MK[m], ls=LS[m], ms=6.5, lw=1.8))
               for m, c in COL.items()]
    return figure_legend.emit(
        handles, list(COL), 'fmsa_dp_speed', out_dir=str(OUT_DIR),
        handlelength=3.4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)})


def figure(per_state, grid, species, products):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 11,
                         'axes.labelsize': 10.5, 'legend.fontsize': 9})
    DP, OZ, MSA = COL['FMSA-DP'], COL['OZ+HNCB'], COL['OZ+MSA']
    EX = COL['exact MSA']
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9))
    fig.suptitle('two closed forms against two converged OZ closures — '
                 'cost of one $c_{ij}(r)$', y=1.02)

    # ⚠ NO INLINE "N×" LABELS ANYWHERE IN THIS FIGURE.  They were readable with
    # three series and are not with four: on a log axis the bars of one row sit
    # within a factor of two of each other, so a label anchored just past a bar
    # end lands on the neighbouring bar.  The ratios they carried are quoted in
    # the caption, where they are legible at print size and can be given as a
    # median with a range instead of eight separate roundings to the nearest 1×.
    ax = axes[0]
    labels = [r[0] for r in per_state]
    y = np.arange(len(labels))
    h, off = 0.20, 0.21
    ax.barh(y - 1.5 * off, [r[3] for r in per_state], height=h, color=DP)
    ax.barh(y - 0.5 * off, [r[13] for r in per_state], height=h, color=EX)
    ax.barh(y + 0.5 * off, [r[10] for r in per_state], height=h, color=MSA)
    ax.barh(y + 1.5 * off, [r[7] for r in per_state], height=h, color=OZ)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel('wall time per state point [s]')
    ax.set_title('one state point, end to end\n(build + a full $c_{ij}(r)$)')
    ax.set_xlim(right=ax.get_xlim()[1] * 1.35)
    # legend moved to a shared strip -- see legend_strip()

    ax = axes[1]
    nr = [g[0] for g in grid]
    # ⭐ ONE COLOUR PER MODEL, ONE MARKED LINE PER MODEL.  The shared key promises
    # exactly that, so exact MSA's extra information -- that part of its cost is a
    # gridless solve and part is the transform -- is NOT a second purple series.
    # Two purple lines with the same diamond, separated only by a dash pattern,
    # were illegible at print size and broke the key's own contract.  It is a
    # SHADED BAND instead: the band's lower edge is the solve, its upper edge is
    # the marked curve, and the band's HEIGHT is the inversion.  The quantity the
    # panel was asked to add is the thing the eye now measures.
    if grid and len(grid[0]) > 6 and np.isfinite(grid[0][6]):
        solve = np.array([g[5] for g in grid], float)
        total = np.array([g[5] + g[6] for g in grid], float)
        ax.fill_between(nr, solve, total, color=EX, alpha=0.30, lw=0, zorder=1)
        # The lower edge carries NO marker: it is a level, not a sampled series.
        ax.loglog(nr, solve, '-', color=EX, lw=0.9, alpha=0.55, zorder=2)
    # Dashed = an ITERATIVE solve, solid = a closed form.  The linestyle carries
    # that distinction in every panel of the paper now, so the two OZ legs here
    # are dashed where they used to be solid.
    ax.loglog(nr, [g[1] for g in grid], **S.line('OZ+HNCB', lw=2, ms=7, zorder=4),
              label='OZ+HNCB solve')
    ax.loglog(nr, [g[4] for g in grid], **S.line('OZ+MSA', lw=2, ms=6, zorder=4),
              label='OZ+MSA solve')
    ax.loglog(nr, [g[2] for g in grid], **S.line('FMSA-DP', lw=2, ms=7, zorder=4),
              label='FMSA-DP, $c(r)$ on $n_r$ points')
    if grid and len(grid[0]) > 6 and np.isfinite(grid[0][6]):
        ax.loglog(nr, total, **S.line('exact MSA', lw=2, ms=6, zorder=5),
                  label='exact MSA, $c(r)$ on $n_r$ points')
    ax.set_xlabel('$n_r$')
    ax.set_ylabel('wall time [s]')
    ax.set_title('scaling with radial resolution\n$n_r$ is an accuracy knob for OZ, '
                 'a plotting choice for DP')
    # ONE label, on the band, naming what the band IS.  Anchored at the right end,
    # where it is thickest.  No label on the lower edge: "solve" is what is left
    # when the band is taken away, and the caption says so in words.
    if grid and len(grid[0]) > 6 and np.isfinite(grid[0][6]):
        ax.annotate(r'$k\to r$', xy=(nr[-1], float(np.sqrt(total[-1] * solve[-1]))),
                    xytext=(-6, 0), textcoords='offset points', color=EX,
                    fontsize=8, ha='right', va='center')

    ax = axes[2]
    if species:
        N = [s[0] for s in species]
        ax.plot(N, [s[1] for s in species], **S.line('FMSA-DP', lw=2, ms=8),
                label='FMSA-DP')
        ax.plot(N, [s[2] for s in species], **S.line('OZ+HNCB', lw=2, ms=8),
                label='OZ+HNCB')
        ax.plot(N, [s[5] for s in species], **S.line('OZ+MSA', lw=2, ms=7),
                label='OZ+MSA')
        ex = np.array([s[6] if len(s) > 6 else float('nan') for s in species],
                      dtype=float)
        m = np.isfinite(ex)
        if m.any():
            ax.plot(np.array(N)[m], ex[m], **S.line('exact MSA', lw=2, ms=6.5),
                    label='exact MSA')
        # An N with no exact-MSA point is a no-solution or a non-convergence, and
        # a line that simply stops there reads as "not plotted".  Say which.
        for n, has in zip(N, m):
            if not has:
                ax.annotate('no solution', (n, ax.get_ylim()[0]),
                            textcoords='offset points', xytext=(0, 12),
                            ha='center', fontsize=7.5, color=EX, style='italic')
        ax.set_yscale('log')
        ax.set_xlim(min(N) - 0.25, max(N) + 0.45)   # the N=4 label was clipped
        ax.set_xticks(N)
        ax.set_xlabel('number of species')
        ax.set_ylabel('wall time [s]')
        ax.set_title('scaling with the number of species')
    legend_strip()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {FIG}')


RAW = REPO / 'numerical_notes' / 'results' / 'speed' / 'fmsa_dp_speed_raw.json'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--no-figure', action='store_true')
    ap.add_argument('--legend-only', action='store_true',
                    help='rewrite the shared key and stop -- it is pure '
                         'encoding, so it must not cost a full re-timing')
    ap.add_argument('--replot', action='store_true',
                    help='redraw from the last run\'s measurements instead of '
                         're-timing -- see below')
    args = ap.parse_args()
    if args.legend_only:
        legend_strip()
        return 0

    # ⭐ WHY --replot EXISTS.  Splitting this figure into panels re-runs the
    # script, and the script is a stopwatch: a second run measures a second set
    # of numbers, so the panels would disagree with the ratios quoted in the
    # caption and in panels.md by however much the machine drifted between them
    # (26.1x -> 22.9x across two runs here, entirely load).  Persisting the
    # measurement and redrawing from it makes the figure and its prose come from
    # ONE timing, and makes a layout fix cost seconds instead of ten minutes.
    if args.replot:
        with open(RAW) as fh:
            d = json.load(fh)
        figure(d['per_state'], d['grid'], d['species'], d.get('products'))
        return 0

    states = STATES[:3] if args.quick else STATES

    per_state = sec_per_state(states)
    grid = sec_grid(states[0], (1024, 2048, 4096) if args.quick
                    else (1024, 2048, 4096, 8192, 16384))
    species = sec_species((2, 3) if args.quick else (2, 3, 4))
    products = sec_products(states[0])
    sec_context()
    # ⚠ DRAW FIRST, then persist.  The dump is the cheap half and the timing is
    # the expensive half; if serialising ever raises -- a numpy scalar in the
    # iteration count would do it, hence `default=float` -- it must not be able
    # to destroy ten minutes of measurement before the figure exists.
    if not args.no_figure:
        figure(per_state, grid, species, products)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW, 'w') as fh:
        json.dump({'per_state': per_state, 'grid': grid, 'species': species,
                   'products': products}, fh, indent=1, default=float)
    print(f'\nwrote {RAW}  (redraw with --replot, no re-timing)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
