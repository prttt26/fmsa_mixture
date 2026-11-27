#!/usr/bin/env python3
"""The demixing DOME, and the second route to the lambda line.

`compare_caccamo_lambda_line.py` gives the lambda line from `det M = 0`, i.e. from
chat_ij(0) -- the DCF route.  This module gives the same physics from the FREE ENERGY,
and the two do not agree.

THE DOME NEEDS NO DOUBLE-TANGENT SEARCH
---------------------------------------
At fixed (rho, T) the exchange potential

    dmu(chi) = mu_1(chi) - mu_2(chi)

is ODD about chi = 1/2, because the two species are related by exchange symmetry:
mu_1(1-chi) = mu_2(chi).  Two-phase equilibrium between compositions chi* and 1-chi*
requires mu_1 and mu_2 each equal across the phases, which collapses to the single
condition `dmu(chi*) = 0`; and since f(chi) = f(1-chi), equal pressure holds
automatically at fixed total density.  So the coexisting pair is just the NON-TRIVIAL
root of dmu = 0, and the stability of the mixed state is the sign of d(dmu)/dchi at
chi = 1/2.  This is the fixed-total-density ensemble Caccamo's SGCMC actually used, so
the comparison is like-for-like.

⚠ THIS IS THE ENERGY ROUTE, AND IT DISAGREES WITH THE DCF ROUTE IN SIGN
-----------------------------------------------------------------------
At rho* = 0.60, T* = 0.70:  d(dmu)/dchi = -0.796 (demixed) while lam_- = +0.238
(mixed).  Across all five isochores the DCF route runs 12-22 % BELOW the measured
consolute temperatures and the energy route 14-17 % ABOVE, so the measurement sits
between them.  Neither is "the" answer; the pair is a bracket, exactly as the
exact-MSA compressibility work concluded for the pure fluid.  Both must therefore be
drawn, labelled, in Figure 12 -- a dome from this module plotted against a lambda line
from the other one would silently mix routes across panels.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import brentq

from compare_mkanya_mixture import build
from fmsa_free_energy import FMSAFreeEnergy
from compare_caccamo_lambda_line import NU, Z, lambda_T

OUT = 'numerical_notes/results/mixture_caccamo'
EPS = 2e-3          # keep off the chi -> 0 logarithmic wall


def dmu(rho: float, T: float, chi: float) -> float:
    """mu_1 - mu_2 at composition chi, energy route."""
    f = build(rho, T, Z, NU, x=np.array([chi, 1.0 - chi]))
    m = np.ravel(FMSAFreeEnergy(f).get_chemical_potentials_total(incore='lj'))
    return float(m[0] - m[1])


def dmu_slope(rho: float, T: float, h: float = 0.01) -> float:
    """d(dmu)/dchi at chi = 1/2.  > 0 stable against demixing, < 0 unstable."""
    return (dmu(rho, T, 0.5 + h) - dmu(rho, T, 0.5 - h)) / (2.0 * h)


def tcr_energy(rho: float, lo: float = 0.35, hi: float = 3.0) -> float | None:
    """Consolute critical T* by the energy route: the root of d(dmu)/dchi = 0."""
    try:
        if dmu_slope(rho, lo) * dmu_slope(rho, hi) > 0:
            return None
        return float(brentq(lambda T: dmu_slope(rho, T), lo, hi, xtol=1e-6))
    except ValueError:
        return None


def coexisting_chi(rho: float, T: float) -> float | None:
    """The non-trivial root of dmu = 0 on (0, 1/2), or None if the state is mixed."""
    hi = 0.5 - 1e-3
    a, b = dmu(rho, T, EPS), dmu(rho, T, hi)
    if a * b > 0:                       # no sign change -> no demixing
        return None
    x = float(brentq(lambda c: dmu(rho, T, c), EPS, hi, xtol=1e-8))
    return None if x > 0.5 - 5e-3 else x


def dome(rho: float, n: int = 14) -> list[dict]:
    """chi*(T) along one isochore, from just under T_cr down."""
    Tc = tcr_energy(rho)
    if Tc is None:
        return []
    out = []
    for T in np.linspace(Tc * 0.995, Tc * 0.62, n):
        c = coexisting_chi(rho, float(T))
        if c is not None:
            out.append({'tstar': float(T), 'chi': c})
    return out


if __name__ == '__main__':
    lam = json.load(open(os.path.join(OUT, 'lambda_line.json')))
    isos = [r['rho'] for r in lam['isochores']]

    print('consolute critical temperature, two routes')
    print(f"{'rho*':>6} {'Caccamo fit':>12} {'DCF':>9} {'dev%':>7} "
          f"{'energy':>9} {'dev%':>7}")
    rows = []
    for r in lam['isochores']:
        Te = tcr_energy(r['rho'])
        rows.append({'rho': r['rho'], 'tcr_fit': r['tcr_fit'],
                     'tcr_dcf': r['tcr_dp'], 'tcr_energy': Te})
        print(f"{r['rho']:>6.2f} {r['tcr_fit']:>12.4f} {r['tcr_dp']:>9.4f} "
              f"{100*(r['tcr_dp']-r['tcr_fit'])/r['tcr_fit']:>+7.1f} "
              f"{Te:>9.4f} {100*(Te-r['tcr_fit'])/r['tcr_fit']:>+7.1f}")

    print('\ndemixing domes (energy route)')
    domes = {}
    for rho in isos:
        d = dome(rho)
        domes[f'{rho:.2f}'] = d
        if d:
            print(f"  rho*={rho:.2f}: {len(d)} points, "
                  f"T* {d[-1]['tstar']:.3f}..{d[0]['tstar']:.3f}, "
                  f"chi {d[-1]['chi']:.3f}..{d[0]['chi']:.3f}")

    print('\nenergy-route lambda line on a grid')
    grid = []
    for rho in np.arange(0.40, 1.101, 0.04):
        Te = tcr_energy(float(rho))
        if Te is not None:
            grid.append({'rho': float(rho), 'tstar': Te})
    print(f"  {len(grid)} points, rho* {grid[0]['rho']:.2f}..{grid[-1]['rho']:.2f}")

    with open(os.path.join(OUT, 'demixing_energy_route.json'), 'w') as fh:
        json.dump({'nu': NU, 'z': Z, 'isochores': rows,
                   'domes': domes, 'lambda_line_energy': grid}, fh, indent=2)
    print(f"\nwrote {OUT}/demixing_energy_route.json")
