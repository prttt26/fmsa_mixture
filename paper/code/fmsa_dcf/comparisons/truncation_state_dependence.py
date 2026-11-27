#!/usr/bin/env python3
"""FMSA against EXACT MSA on a (rho, T) grid: how the first-order truncation error depends on the state.

For each state the exact multi-Yukawa MSA (Blum-Hoye, msa_exact_mix.from_fmsa) is read off the
FMSA object itself, so both solve one system and only first order against all orders differs.
Both use the same Percus-Yevick hard-sphere reference (hs_ref='py'), as in SI Sec. "Truncation error
at physical coupling".

Per state it records
  a_exact, a_fmsa   1 - rho sum_ij x_i x_j chat_ij(0)  = 1/S(0) at fixed composition (the distance to the
                    spinodal: a -> 0 is the spinodal; a_id = 1 for the ideal gas)
  det_exact/fmsa    det(I - sqrt(rho_i rho_j) chat_ij(0))  (concentration + density instability)
  chat0_rel_ij      |chat_F(0) - chat_E(0)| / |chat_E(0)| per pair
  core residual     |c_E - c_F| over the core r < R_ij, per pair, in three normalisations:
                    L2 / L2 of the first-order correction c1 = c_F - c_HS, L2 / L2 of the total core c_E,
                    and the r^2-weighted versions of both
and T_c(exact) from the grid by the interpolated T at which min_rho a_exact reaches zero.

    python code/run.py code/fmsa_dcf/comparisons/truncation_state_dependence.py --system binary --workers 6
    python code/run.py code/fmsa_dcf/comparisons/truncation_state_dependence.py --system pure
Output: numerical_notes/results/truncation/truncation_<system>.csv
"""
from __future__ import annotations

import argparse
import csv
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

OUT = Path('numerical_notes/results/truncation')
SYSTEMS = {
    'binary': dict(sig=[1., 2.], eps=[1., .5], x=[.25, .75]),      # the SI reference binary / GCMC mixture
    'pure':   dict(sig=[1.], eps=[1.], x=[1.]),                    # one-component LJ -> 2YK
}
PAIRS = {'binary': [(0, 0), (0, 1), (1, 1)], 'pure': [(0, 0)]}
FIELDS = ['system', 'T', 'rho', 'eta', 'ok', 'a_exact', 'a_fmsa', 'det_exact', 'det_fmsa']


def one_state(args):
    name, T, rho = args
    warnings.simplefilter('ignore')
    import msa_exact_mix as X
    from fmsa_double_prop import FMSA_double_prop
    s = SYSTEMS[name]
    sig, eps, x = map(np.array, (s['sig'], s['eps'], s['x']))
    row = dict(system=name, T=T, rho=rho, ok=0)
    try:
        dp = FMSA_double_prop(sig, eps, x, rho, T, n_tails=2, verbose=False)
        row['eta'] = dp.eta
        ex = X.solve_bh_mix(X.from_fmsa(dp, 1.0))
        hs = X.solve_bh_mix(X.from_fmsa(dp, 0.0))
        cf0 = dp.get_chat_k(np.array([1e-7]))[0]
        rv = dp.rho
        sq = np.sqrt(np.outer(rv, rv))
        n = len(rv)
        a = lambda C: 1.0 - rv.sum() * sum(x[i] * x[j] * C[i, j] for i in range(n) for j in range(n))
        det = lambda C: float(np.linalg.det(np.eye(n) - sq * C))
        row['a_fmsa'], row['det_fmsa'] = a(cf0), det(cf0)
        if ex is None or hs is None:
            return row                      # beyond the exact fold: no exact reference
        ce0 = X.chat0(ex)
        row['a_exact'], row['det_exact'] = a(ce0), det(ce0)
        r, c = X.c_of_r(ex, 4096)
        _, chs = X.c_of_r(hs, 4096)
        cF = dp.get_c_total(r, hs_ref='py')
        c1 = cF - chs
        for (i, j) in PAIRS[name]:
            tag = f'{i + 1}{j + 1}'
            m = r < dp.R_ij[i, j]
            d = (c[i, j] - cF[i, j])[m]
            w = r[m] ** 2
            L = np.linalg.norm
            row[f'chat0_rel_{tag}'] = abs(cf0[i, j] - ce0[i, j]) / abs(ce0[i, j])
            row[f'res_c1_{tag}'] = L(d) / L(c1[i, j][m])
            row[f'res_tot_{tag}'] = L(d) / L(c[i, j][m])
            row[f'res_c1_r2_{tag}'] = np.sqrt(np.sum(w * d ** 2) / np.sum(w * c1[i, j][m] ** 2))
            row[f'res_tot_r2_{tag}'] = np.sqrt(np.sum(w * d ** 2) / np.sum(w * c[i, j][m] ** 2))
            row[f'max_c1_{tag}'] = np.max(np.abs(d)) / np.max(np.abs(c1[i, j][m]))
            row[f'max_tot_{tag}'] = np.max(np.abs(d)) / np.max(np.abs(c[i, j][m]))
        row['ok'] = 1
    except Exception as e:                      # a state the solver cannot do is a gap, not an error
        row['err'] = type(e).__name__
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default='binary', choices=list(SYSTEMS))
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--nT', type=int, default=26)
    ap.add_argument('--nrho', type=int, default=22)
    ap.add_argument('--Tmin', type=float, default=0.55)
    ap.add_argument('--Tmax', type=float, default=3.0)
    ap.add_argument('--rhomax', type=float, default=None)
    ap.add_argument('--tag', default='', help='suffix of the output file (refined grids)')
    a = ap.parse_args()
    rhomax = a.rhomax or (0.16 if a.system == 'binary' else 0.75)
    Ts = np.geomspace(a.Tmin, a.Tmax, a.nT)
    rhos = np.linspace(0.01 * rhomax, rhomax, a.nrho)
    jobs = [(a.system, float(T), float(r)) for T in Ts for r in rhos]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one_state, jobs, chunksize=4))
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    p = OUT / f'truncation_{a.system}{a.tag}.csv'
    with p.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} states, {sum(r["ok"] for r in rows)} with an exact reference, {time.time() - t0:.0f}s -> {p}')


if __name__ == '__main__':
    main()
