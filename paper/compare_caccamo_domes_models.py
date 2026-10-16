#!/usr/bin/env python3
"""Energy-route demixing domes for three models on Caccamo's symmetric mixture.

FMSA-DP's dome is in `demixing_energy_route.json` (from
`compare_caccamo_demixing_dome.py`).  This adds the two that bracket it in kind:

    exact MSA   the SAME closure without the first-order truncation
    OZ-HNCB     a DIFFERENT closure, solved numerically

so the spacing FMSA-DP <-> exact MSA is the truncation and exact MSA <-> OZ-HNCB
is the closure, exactly as Figure 11 does for the pure fluid.

TWO CANCELLATIONS MAKE THIS CHEAP, AND BOTH NEED EQUAL DIAMETERS
----------------------------------------------------------------
1. The exchange potential is  dmu = ln(chi/(1-chi)) + (mu_1^ex - mu_2^ex).  For
   EQUAL diameters the hard-sphere excess is species-independent, so the whole
   hard-sphere block drops out of dmu -- verified to 0.0 at chi = 0.3 as well as
   at 1/2.  No hard-sphere reference enters, and none can bias the comparison.

2. `mu_1^ex - mu_2^ex = dA/drho_1 - dA/drho_2` is the derivative ALONG (1,-1),
   i.e. changing composition at FIXED total density.  So it needs TWO free-energy
   evaluations, not four:

       mu_1^ex - mu_2^ex = [A(chi+d) - A(chi-d)] / (2 rho d)

   which halves an OZ-HNCB sweep from ~2 h to ~50 min.

⚠ THERE IS DELIBERATELY NO COMPRESSIBILITY-ROUTE DOME HERE.  For this model
chat(0) is not integrable -- d(chat_11)/d(rho_2) - d(chat_12)/d(rho_1) is ~20 %
of the terms themselves -- so its local Hessian (det M = 0, T_cr = 0.650 at
rho* = 0.60) and its ray-integrated free energy (T_cr = 0.396) disagree by 64 %,
and the ray-integrated one has no non-trivial root of dmu = 0 anywhere in the
region the data occupy.  A "compressibility-route dome" would be a choice between
two constructions 64 % apart, not a result.  See `lambda_line.md`.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.optimize import brentq

import msa_exact_mix as X
from energy_compress_core import oz_point

NU, Z = 0.7, 2.45
OUT = 'numerical_notes/results/mixture_caccamo'
EPS_IJ = np.array([[1.0, NU], [NU, 1.0]])
CHI_EPS = 2e-3
DCHI = 5e-3                      # composition step for the (1,-1) derivative


# ─── exact MSA ────────────────────────────────────────────────────────────────

def dmu_exact(rho_t: float, T: float, chi: float) -> float:
    """mu_1 - mu_2, exact MSA.  Equal diameters => SIG_ij = 1 and one shared z,
    which is Blum & Hoye's covered case directly -- no pair-specific tails."""
    st = X.HCYMix(np.array([1.0, 1.0]),
                  np.array([rho_t * chi, rho_t * (1.0 - chi)]),
                  np.array([Z]), (EPS_IJ / T)[None, :, :])
    sol = X.solve_bh_mix(st)
    if sol is None:            # no MSA solution this deep -- not an error
        return None
    m = np.asarray(X.chem_pot_excess(sol), dtype=float)
    return float(np.log(chi / (1.0 - chi)) + (m[0] - m[1]))


# ─── OZ-HNCB ──────────────────────────────────────────────────────────────────

def _sysd(chi: float) -> dict:
    return dict(key=f'cacc|nu{NU}|z{Z}|{chi:.9f}',
                sig=np.array([1.0, 1.0]), eps=np.array([1.0, 1.0]),
                x=np.array([chi, 1.0 - chi]), nt=1,
                z=np.array([Z]), k=np.array([1.0]),
                eps_ij=EPS_IJ, input_type='bh')


def dmu_hncb(rho_t: float, T: float, chi: float, closure: str = 'hncb'):
    """mu_1 - mu_2 by the (1,-1) directional derivative of A at fixed total rho.

    ⚠ The step must stay inside (0,1).  The root bracket starts at chi = 2e-3, so a
    fixed 5e-3 step reaches a NEGATIVE mole fraction there, and FMSA_double_prop
    fails deep inside `_P_kernel` on `sqrt(rho_i rho_m)` rather than at the call --
    a domain error a long way from its cause.  The step is therefore shrunk to fit.
    """
    d = min(DCHI, 0.4 * chi, 0.4 * (1.0 - chi))
    if d <= 0.0:
        return None
    a = oz_point(_sysd(chi + d), rho_t, T, closure)
    b = oz_point(_sysd(chi - d), rho_t, T, closure)
    if a is None or b is None:
        return None
    return float(np.log(chi / (1.0 - chi))
                 + (a['A'] - b['A']) / (2.0 * rho_t * d))


# ─── domes ────────────────────────────────────────────────────────────────────

def _bisect(f, a, b, fa, fb, tol=1e-5, maxit=60):
    """Bisection that tolerates a FAILED evaluation instead of crashing.

    `brentq` calls `np.isnan` on whatever the function returns, so a `None` from a
    non-converging solve raises a TypeError from inside SciPy rather than being
    handled.  Both solvers here legitimately return None in part of the domain --
    exact MSA has no solution at the coldest temperatures, OZ-HNCB none near its
    own spinodal -- so the root finder has to cope with holes.  A midpoint that
    cannot be evaluated is retried with a small jitter; if that fails too the
    bracket is abandoned and None returned, which costs one point rather than the
    whole sweep.
    """
    for _ in range(maxit):
        if b - a < tol:
            return 0.5 * (a + b)
        m = 0.5 * (a + b)
        fm = f(m)
        if fm is None:
            for j in (0.25, -0.25, 0.4, -0.4):
                m2 = m + j * (b - a)
                fm = f(m2)
                if fm is not None:
                    m = m2
                    break
            if fm is None:
                return None
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def _root(fn, rho_t, T):
    """Non-trivial root of dmu = 0 on (0, 1/2), or None if the state is mixed."""
    hi = 0.5 - 1e-3
    fa, fb = fn(rho_t, T, CHI_EPS), fn(rho_t, T, hi)
    if fa is None or fb is None or fa * fb > 0:
        return None
    x = _bisect(lambda c: fn(rho_t, T, c), CHI_EPS, hi, fa, fb, tol=1e-6)
    return None if x is None or x > 0.5 - 5e-3 else float(x)


def tcr(fn, rho_t, lo=0.35, hi=2.2):
    """Consolute critical T*: where d(dmu)/dchi at chi = 1/2 changes sign."""
    h = 0.01
    def sl(T):
        a, b = fn(rho_t, T, 0.5 + h), fn(rho_t, T, 0.5 - h)
        return None if a is None or b is None else (a - b) / (2.0 * h)
    # SCAN for the bracket rather than trusting the endpoints: the exact-MSA
    # solve has no solution at the lowest temperatures, so sl(lo) can be None.
    vals = [(float(T), sl(float(T))) for T in np.linspace(lo, hi, 26)]
    vals = [(T, v) for T, v in vals if v is not None]
    for (Ta, va), (Tb, vb) in zip(vals, vals[1:]):
        if va * vb < 0:
            r = _bisect(sl, Ta, Tb, va, vb, tol=1e-5)
            if r is not None:
                return float(r)
    return None


def dome(fn, rho_t, n=10):
    Tc = tcr(fn, rho_t)
    if Tc is None:
        return [], None
    pts = []
    for T in np.linspace(Tc * 0.99, Tc * 0.62, n):
        c = _root(fn, rho_t, float(T))
        if c is not None:
            pts.append({'tstar': float(T), 'chi': c})
    return pts, Tc


if __name__ == '__main__':
    lam = json.load(open(os.path.join(OUT, 'lambda_line.json')))
    isos = [r['rho'] for r in lam['isochores']]
    res = {}
    for name, fn, n in (('exact MSA', dmu_exact, 12), ('OZ-HNCB', dmu_hncb, 8)):
        t0 = time.time()
        res[name] = {'domes': {}, 'tcr': {}}
        for rho in isos:
            d, Tc = dome(fn, rho, n)
            res[name]['domes'][f'{rho:.2f}'] = d
            res[name]['tcr'][f'{rho:.2f}'] = Tc
            print(f"  {name:<10} rho*={rho:.2f}  T_cr="
                  f"{'None' if Tc is None else f'{Tc:.4f}'}  {len(d)} dome pts"
                  f"   [{time.time()-t0:6.1f}s]")
    with open(os.path.join(OUT, 'domes_models.json'), 'w') as fh:
        json.dump({'nu': NU, 'z': Z, 'models': res}, fh, indent=2)
    print(f"\nwrote {OUT}/domes_models.json")
