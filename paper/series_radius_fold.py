#!/usr/bin/env python3
"""Is the real fold the nearest singularity of the coupling series?  (SI, Convergence)

For each state the physical branch is continued from the PY seed in real tau to
locate its fold tau_fold (if any).  Three readings follow:

  * monodromy on the circles |tau| = 0.90, 0.97, 1.03 tau_fold: a zero gap means
    the circle encloses no branch point;
  * the sign pattern of the Taylor coefficients of each amplitude component,
    n = 1..25, read from the 0.90 tau_fold circle (one sign = Pringsheim's
    condition, which puts the nearest singularity on the positive real axis);
  * for states without a real fold, the fitted radius R_c and monodromy on a
    ladder of circles, with the real-axis solution checked at tau = 1.

Writes numerical_notes/results/series_radius_fold.json.

    python series_radius_fold.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import msa_series_radius as S

FOLD_CASES = [('HSY z=1.8', [1.8], [1.0]), ('HSY z=5', [5.0], [1.0]),
              ('2YK LJ', [2.9637, 14.0167], [2.1714, -2.1714])]
COMPETE = ('competing z=(1.8,12) w=(1,-6)', [1.8, 12.0], [1.0, -6.0])
ETAS = (0.10, 0.20, 0.30, 0.40)


def sign_pattern(c, floor, tf, n_comp):
    out = []
    k = np.arange(1, 26)
    for comp in range(n_comp):
        v = np.real(c[1:26, comp]) * tf ** k
        v = v[np.abs(v) > 30.0 * floor[1:26] * tf ** k]
        s = np.sign(v)
        out.append({'n_resolved': int(len(s)),
                    'one_sign': bool(len(s) and np.all(s == s[0]))})
    return out


def fold_rows():
    rows = []
    for name, z, kv in FOLD_CASES:
        z, kv = np.array(z, float), np.array(kv, float)
        for xi in ETAS:
            tf, tmax, _ = S.fold_in_tau(xi, z, kv)
            row = {'model': name, 'eta': xi, 'tau_fold': tf}
            if np.isfinite(tf):
                gaps = {}
                for f in (0.90, 0.97, 1.03):
                    _, gap, _ = S.walk_circle(xi, z, kv, f * tf, M_pts=512)
                    gaps[f'{f:.2f}'] = gap if np.isfinite(gap) else None
                smp, _, _ = S.walk_circle(xi, z, kv, 0.9 * tf, M_pts=512)
                c, floor = S.coefficients(smp, 0.9 * tf, n_max=40)
                row['monodromy_gap'] = gaps
                row['coefficient_signs'] = sign_pattern(c, floor, tf, 2 * len(z))
            else:
                row['traced_to'] = tmax
            rows.append(row)
            print(row, flush=True)
    return rows


def compete_rows():
    name, z, kv = COMPETE
    z, kv = np.array(z, float), np.array(kv, float)
    rows = []
    for xi in ETAS:
        tf, tmax, _ = S.fold_in_tau(xi, z, kv)
        Rc, _, _ = S.study('', xi, z, kv, [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                           verbose=False)
        gaps = {}
        for r in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0):
            _, gap, _ = S.walk_circle(xi, z, kv, r, M_pts=1024)
            gaps[f'{r}'] = gap if np.isfinite(gap) else None
        row = {'model': name, 'eta': xi, 'tau_fold': tf, 'traced_to': tmax,
               'R_c_fit': Rc, 'monodromy_gap': gaps,
               'real_solution_at_tau_1': S.exact_at(xi, z, kv, 1.0) is not None}
        rows.append(row)
        print(row, flush=True)
    return rows


def main() -> int:
    out = {'fold_cases': fold_rows(), 'competing_tails': compete_rows()}
    path = os.path.join('numerical_notes', 'results', 'series_radius_fold.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1, default=float)
    print('wrote', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
