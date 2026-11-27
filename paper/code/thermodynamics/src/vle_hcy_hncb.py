#!/usr/bin/env python3
r"""The OZ-HNCB energy-route binodal for the hard-core Yukawa fluid of Figure 11.

WHAT THIS LEG ADDS
------------------
Figure 11 already separates, on one system with NO potential-mapping error:

    potential / model-class error   zero -- Shukla's Hamiltonian IS ours at N=1
    OZ solver error                 ~zero -- OZ+MSA reproduces the EXACT MSA
                                    solution to 0.1 pp on the 145-point pool
    the MSA closure                 exact MSA  minus  GEMC
    first-order truncation          FMSA-DP    minus  exact MSA
    thermodynamic route             energy     minus  compressibility

The one leg missing is WHAT A BETTER CLOSURE BUYS.  OZ-HNCB is this project's
best-scoring closure on structure (3.0 % against MSA's 15 % on the HCTY contact
pool), so this is the natural candidate.

⚠ AND THE ANSWER IS NOT OBVIOUS, WHICH IS THE REASON TO RUN IT.  On that same
HCTY pool HNCB was best on contact g by 5x and SIMULTANEOUSLY WORST on internal
energy (16.4 % against exact MSA's 12.3 %, losing in three cases of four).  The
energy route integrates exactly that internal energy.  So HNCB may well produce a
WORSE binodal than MSA despite better structure -- in which case the finding is
that the closure which best reproduces structure is not the one that best predicts
coexistence.

⚠ SAME HARD-SPHERE CORE AS EVERY OTHER LEG.  `_betaf_hs_raw` is White Bear I
(BMCSL), which at N = 1 is Carnahan-Starling -- the same expression
`FMSAFreeEnergy.betaf_hs` uses.  Without that the spacing between this curve and
the others would carry a hard-sphere reference difference worth up to 7 % in
rho*_L, comparable to the effects being measured.

ROUTE.  Energy route: beta f_ex = beta f_HS + A, with A the coupling-constant
integral of the tail over the converged OZ-HNCB h(r) at five Gauss-Legendre
lambda nodes.  P and mu follow by differentiating f in rho.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import root

from energy_compress_core import _betaf_hs_raw, oz_point

Z_HCY = 1.8
KV = 1.0
OUT = 'numerical_notes/results/vle_hcy'
DRHO = 2e-3                       # density step for dP/drho and dmu/drho

# `input_type='bh'` takes sigma as the HARD diameter -- there is no Barker-
# Henderson softening to apply to a hard core -- so the FMT diameter is 1.
SYSD = dict(key=f'hcy|z{Z_HCY}|k{KV}', sig=np.array([1.0]), eps=np.array([1.0]),
            x=np.array([1.0]), nt=1, z=np.array([Z_HCY]), k=np.array([KV]),
            input_type='bh')
DIAM = np.array([1.0])

_F_CACHE: dict = {}


def f_ex(rho: float, tstar: float, closure: str = 'hncb') -> float | None:
    """beta f_ex = beta f_HS(WB1) + A(tail), or None if the OZ solve failed."""
    key = (closure, round(float(rho), 10), round(float(tstar), 10))
    if key in _F_CACHE:
        return _F_CACHE[key]
    p = oz_point(SYSD, float(rho), float(tstar), closure)
    val = None if p is None else (
        _betaf_hs_raw(np.array([float(rho)]), DIAM) + p['A'])
    _F_CACHE[key] = val
    return val


def thermo(rho: float, tstar: float, closure: str = 'hncb'):
    """(beta p, beta mu) including the ideal terms, or None if any solve failed."""
    f0 = f_ex(rho, tstar, closure)
    fp = f_ex(rho + DRHO, tstar, closure)
    fm = f_ex(rho - DRHO, tstar, closure)
    if f0 is None or fp is None or fm is None:
        return None
    g = (fp - fm) / (2.0 * DRHO)
    return rho + rho * g - f0, float(np.log(rho)) + g


def coexist(tstar: float, guess, closure: str = 'hncb'):
    def res(u):
        rv, rl = np.exp(u)
        if not (1e-7 < rv < 1.3) or not (1e-7 < rl < 1.3) or rl <= rv:
            return [1e3, 1e3]
        a, b = thermo(rv, tstar, closure), thermo(rl, tstar, closure)
        if a is None or b is None:
            return [1e3, 1e3]
        return [a[0] - b[0], a[1] - b[1]]
    sol = root(res, np.log(np.array(guess)), method='hybr', tol=1e-9)
    if not sol.success:
        return None
    rv, rl = np.exp(sol.x)
    return None if rl - rv < 5e-3 else (float(rv), float(rl))


if __name__ == '__main__':
    import time
    temps = np.round(np.arange(1.20, 0.799, -0.02), 3)
    out, prev, t0 = [], (0.03, 0.78), time.time()
    for T in temps:
        s = coexist(float(T), prev)
        if s is None:
            print(f"  T*={T:.2f}  no solution")
            continue
        prev = s
        out.append({'T': float(T), 'rho_v': s[0], 'rho_l': s[1]})
        print(f"  T*={T:.2f}  rho_v={s[0]:.4f}  rho_l={s[1]:.4f}"
              f"   [{time.time()-t0:6.1f}s]")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'vle_hcy_hncb.json'), 'w') as fh:
        json.dump({'model': 'OZ-HNCB', 'z': Z_HCY, 'closure': 'hncb', 'route': 'energy',
                   'binodal': out}, fh, indent=2)
    print(f"\n{len(out)}/{len(temps)} temperatures, "
          f"{time.time()-t0:.0f} s total -> {OUT}/vle_hcy_hncb.json")
