#!/usr/bin/env python3
"""Two supporting coexistence pairs already at second coupling order.  (SI, Finite coupling orders)

Scalar compressibility route on the PY hard-sphere reference, one attractive
Yukawa tail of range z, unit amplitude, tau = 1/T*.  The strict second-order jet

    a_[2] = d(beta p)/d rho = c0 + c1 tau + c2 tau^2

has the closed coefficients of Lean_code/results/order_thresholds (derived there
by SymPy from the PY/Blum-Hoye residual).  For each z the script

  * scans T* and counts the supporting common tangents of beta f/V(rho) as the
    non-trivial segments of its lower convex hull on 0 < rho* < 1.80;
  * refines one example by solving equal pressure and chemical potential with
    adaptive quadrature, and checks that each tangent lies below beta f/V on a
    grid 1e-7 <= rho* <= 1.85 evaluated by the same quadrature.

Writes numerical_notes/results/finite_order_two_tangents.json.

    python finite_order_two_tangents.py
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

RHO = np.concatenate([np.geomspace(1e-10, 1e-3, 4000, endpoint=False),
                      np.linspace(1e-3, 1.80, 40000)])
ZS = (1.0, 1.8, 3.0, 5.0, 10.0, 20.0)
TS = np.geomspace(0.005, 0.2, 300)
EXAMPLE = {'z': 5.0, 'T': 0.02}


def coeffs(eta, z):
    E = np.exp(-z)
    w = 1 - eta
    J = E * (z + 2) + z - 2
    K = -E * (z ** 2 + 2 * z + 2) + 2
    H0 = E * (2 * z + 2) + z ** 2 - 2
    P = 18 * eta ** 2 * J + 6 * eta * w * H0 + z ** 3 * w ** 2
    L = 2 * eta + z * w + 1
    H = 6 * eta * J + w * H0
    c0 = (1 + 2 * eta) ** 2 / w ** 4
    c1 = -24 * eta * z ** 4 * (1 + 2 * eta) * L / P ** 2
    c2 = 144 * eta ** 2 * z ** 11 * w ** 7 * H / P ** 5
    return c0, c1, c2


def a2(rho, z, T):
    c0, c1, c2 = coeffs(np.pi * rho / 6, z)
    return c0 + c1 / T + c2 / T ** 2


def grid_free_energy(z, T):
    a = a2(RHO, z, T)
    g = (a - 1) / RHO
    mu = np.log(RHO) + np.concatenate([[0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(RHO))])
    p = np.concatenate([[0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(RHO))]) + RHO[0] * a[0]
    return a, RHO * mu - p


def tangent_segments(x, y, minw=2e-3):
    hv = np.sort(ConvexHull(np.column_stack([x, y])).vertices)
    x, y = x[hv], y[hv]
    h = []
    for i in range(len(x)):
        while len(h) >= 2:
            i1, i2 = h[-2], h[-1]
            if (x[i2] - x[i1]) * (y[i] - y[i1]) - (y[i2] - y[i1]) * (x[i] - x[i1]) <= 0:
                h.pop()
            else:
                break
        h.append(i)
    return [(float(x[a]), float(x[b])) for a, b in zip(h[:-1], h[1:]) if x[b] - x[a] > minw]


def scan(z):
    two = []
    for T in TS:
        _, f = grid_free_energy(z, T)
        if len(tangent_segments(RHO, f)) >= 2:
            two.append(float(T))
    return {'z': z, 'T_scanned': [float(TS[0]), float(TS[-1])],
            'T_two_pairs': [min(two), max(two)] if two else None, 'n_T_two_pairs': len(two)}


def refine(z, T):
    a = lambda r: a2(r, z, T)
    p = lambda r: quad(a, 0, r, limit=1000, epsabs=1e-15, epsrel=1e-14)[0]
    mu = lambda r: np.log(r) + quad(lambda s: (a(s) - 1) / s, 0, r, limit=1000,
                                    epsabs=1e-15, epsrel=1e-14)[0]
    _, f = grid_free_energy(z, T)
    rr = np.unique(np.concatenate([np.geomspace(1e-7, 1e-2, 600), np.linspace(1e-2, 1.85, 3000)]))
    fr = np.array([r * mu(r) - p(r) for r in rr])
    sgn = np.sign(a(rr))
    spinodals = [float(rr[i]) for i in np.where(np.diff(sgn) != 0)[0]]
    pairs = []
    for g in tangent_segments(RHO, f):
        F = lambda x: [p(x[0]) - p(x[1]), mu(x[0]) - mu(x[1])]
        x = fsolve(F, list(g), xtol=1e-15)
        P, M = p(x[0]), mu(x[0])
        gap = fr - (M * rr - P)
        pairs.append({'rho': [float(x[0]), float(x[1])],
                      'eta': [float(np.pi * x[0] / 6), float(np.pi * x[1] / 6)],
                      'beta_p': float(P), 'beta_mu': float(M),
                      'residual': float(np.abs(F(x)).max()),
                      'min_f_minus_tangent': float(gap.min())})
    return {'z': z, 'T': T, 'spinodals_rho': spinodals, 'pairs': pairs}


def main() -> int:
    out = {'scan': [], 'example': None}
    for z in ZS:
        row = scan(z)
        out['scan'].append(row)
        print(row, flush=True)
    out['example'] = refine(EXAMPLE['z'], EXAMPLE['T'])
    print(out['example'])
    path = os.path.join('numerical_notes', 'results', 'finite_order_two_tangents.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
