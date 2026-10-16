#!/usr/bin/env python3
"""Does the exact MSA return an answer?  The same three tasks Fig. 7(c) prices.

Fig. 7(c) counts attempts that TERMINATED, for FMSA-DP, OZ+MSA and OZ+HNCB.  Exact
MSA belongs in that panel for a reason none of the other three share: it is the one
method whose answer can fail to EXIST.  FMSA-DP is an explicit function of the state
and OZ is an iteration that may or may not converge, but Blum & Høye's solution is
the real root of a non-linear algebraic system --- `2N^2 omega` unknowns, `omega`
tails --- and inside the fold there is no real root to find.  A count of terminations
is therefore the natural place to show it, and it is not a criticism of the
implementation: a failure here is the theory having nothing to return.

Three tasks, matching the panel's rows:

    496-pt grid     one converged solution at each state of the gridded MD batch
                    (`data_benchmark/md/states.build_grid`), which is where the 94
                    OZ+HNCB non-convergences were counted.
    14 bubble pts   the Potoff reference rows of `phase_equilibrium`.
    binodal 26 T*   the x1L = 0.85 binodal sweep, T* = 0.80 .. 1.30 by 0.02.

⚠ The system is read off the FMSA object through `msa_exact_mix.from_fmsa`, never
rebuilt from the physical parameters -- see that function for why rebuilding
silently produces a DIFFERENT system (3.4 % in z, 12.2 % in k_bare) and would make
this a comparison of two models rather than of two solution methods.

    python solvability_exact_msa.py grid
    python solvability_exact_msa.py vle          # bubble points + binodal
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'data_benchmark' / 'md'))

OUT = REPO / 'numerical_notes' / 'results' / 'speed' / 'solvability_exact_msa.json'


def _quiet(fn):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        return fn()


def solve_state(sigs, eps, x, rho, T, n_tails=2, **mkw):
    """-> (ok, seconds).  `ok` is False for BOTH no-root and a bad root.

    `symmetry_defect` is checked because `solve_bh_mix` can return a converged
    root of the residual system that is not a physical solution; the benchmark
    legs use the same 1e-6 screen, so the counts stay comparable.
    """
    import msa_exact_mix as X
    from fmsa_double_prop import FMSA_double_prop
    t0 = time.perf_counter()
    try:
        dp = _quiet(lambda: FMSA_double_prop(np.asarray(sigs, float),
                                             np.asarray(eps, float),
                                             np.asarray(x, float), rho, T,
                                             n_tails=n_tails, verbose=False, **mkw))
        sol = _quiet(lambda: X.solve_bh_mix(X.from_fmsa(dp)))
        if sol is None:
            return False, time.perf_counter() - t0
        sd = sol.symmetry_defect
        sd = sd() if callable(sd) else sd
        ok = bool(np.isfinite(float(sd))) and abs(float(sd)) <= 1e-6
    except Exception:                                            # noqa: BLE001
        ok = False
    return ok, time.perf_counter() - t0


# ─── task 1: the gridded MD batch ─────────────────────────────────────────────

def task_grid() -> dict:
    import states as S
    from ljhs_mapping import model_inputs
    from run_md import rho_of_eta

    grid = S.build_grid()
    n_ok = 0
    fails = []
    t0 = time.time()
    for q, (label, spec) in enumerate(sorted(grid.items()), 1):
        ratio, x1, T = float(spec['ratio']), float(spec['x1']), float(spec['T'])
        rho = rho_of_eta(float(spec['eta']), x1, ratio)
        sigs, eps, mkw = model_inputs(spec.get('kind', 'lj'), ratio,
                                      spec['eps_ratio'], T)
        ok, dt = solve_state(sigs, eps, [x1, 1.0 - x1], rho, T, **mkw)
        n_ok += ok
        if not ok:
            fails.append(label)
        if q % 50 == 0 or q == len(grid):
            print(f'  {q:4d}/{len(grid)}  ok={n_ok}  '
                  f'[{time.time() - t0:6.1f}s]', flush=True)
    return {'n': len(grid), 'ok': n_ok, 'failures': fails}


# ─── tasks 2 and 3: the two coexistence sweeps ────────────────────────────────

def task_vle() -> dict:
    import phase_equilibrium as pe

    out = {}

    # (2) the 14 reference bubble points.
    # ⚠ SEEDED FROM THE MEASUREMENT, exactly as `pe.compute` does for the other
    # three models.  A cold seed shared by all 14 rows makes 3 of them fail, and
    # those failures would be the seed rather than the method -- the panel counts
    # solvability, so every leg has to start from the same place.
    rows = pe._table_I()
    n_ok = 0
    for q, row in enumerate(rows, 1):
        guess = (row['ref_rho_liq'], row['ref_rho_gas'] * row['ref_y1'],
                 row['ref_rho_gas'] * (1.0 - row['ref_y1']))
        r = pe.bubble_point(float(row['T']), float(row['x1']), 'MSA-exact', guess)
        ok = r is not None and not r.get('trivial')
        n_ok += ok
        print(f"  bubble {q:2d}/{len(rows)}  T*={row['T']:.3f} "
              f"x1={row['x1']:.3f}  {'ok' if ok else 'FAIL'}", flush=True)
    out['bubble'] = {'n': len(rows), 'ok': n_ok}

    # (3) the x1L = 0.85 binodal, marching UP from the wide end -- the same
    # direction and seed binodal_sweep.py used for the other three models, so a
    # shorter dome here is the method and not the march.
    X1L, guess = 0.85, (0.72, 0.020 * 0.9, 0.020 * 0.1)
    tlist = np.arange(0.80, 1.32, 0.02)
    pts = []
    for T in tlist:
        r = pe.bubble_point(float(T), X1L, 'MSA-exact', guess)
        if r is None:
            print(f'  binodal T*={T:.3f}  no solution', flush=True)
            continue
        if r.get('trivial'):
            print(f'  binodal T*={T:.3f}  trivial -> above its critical line, stop',
                  flush=True)
            break
        pts.append({'T': float(T), **{k: r[k] for k in
                                      ('p', 'y1', 'rho_liq', 'rho_gas')}})
        guess = (r['rho_liq'], r['rho_gas'] * r['y1'],
                 r['rho_gas'] * (1.0 - r['y1']))
        print(f"  binodal T*={T:.3f}  rho_L={r['rho_liq']:.4f} "
              f"rho_V={r['rho_gas']:.4f}", flush=True)
    out['binodal'] = {'n': len(tlist), 'ok': len(pts), 'points': pts}
    return out


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'grid'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if which in ('grid', 'all'):
        res['grid'] = task_grid()
        print('  grid:', res['grid']['ok'], '/', res['grid']['n'])
        OUT.write_text(json.dumps(res, indent=1))
    if which in ('vle', 'all'):
        res.update(task_vle())
        OUT.write_text(json.dumps(res, indent=1))
    print('wrote', OUT)
