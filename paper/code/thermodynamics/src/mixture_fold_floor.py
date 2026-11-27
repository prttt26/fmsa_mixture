#!/usr/bin/env python3
"""Does the exact MSA have a no-solution region in the MIXTURE too?

The pure fluid has one: below the fold apex there is an interval of density at
which Blum & Hoye's algebraic system has no real root (Fig. 14a).  The obvious
question is whether that is a one-component accident or a property of the exact
solution, and if it is the latter, whether it reaches the demixing dome that
Fig. 13 is built from.

This measures the floor directly on Caccamo's symmetric mixture (nu = 0.7,
z = 2.45, equal diameters): at fixed total density and composition, walk T down
in steps of 0.01 and record the last temperature at which `solve_bh_mix` returns
anything.  No continuation, no seeding tricks -- the same cold call the dome
sweep makes, so the floor found here is the floor the dome would hit.

The composition axis is scanned as well as the density one, because the dome's
cold end is at SMALL chi, not at chi = 1/2, and the floor is not flat in chi.

    python mixture_fold_floor.py
"""

from __future__ import annotations

import json
import os

import numpy as np

import msa_exact_mix as X

NU, Z = 0.7, 2.45
EPS_IJ = np.array([[1.0, NU], [NU, 1.0]])
OUT = 'numerical_notes/results/mixture_caccamo'
CHIS = (0.5, 0.35, 0.25, 0.15, 0.05)


def has_solution(rho_t: float, T: float, chi: float) -> bool:
    st = X.HCYMix(np.array([1.0, 1.0]),
                  np.array([rho_t * chi, rho_t * (1.0 - chi)]),
                  np.array([Z]), (EPS_IJ / T)[None, :, :])
    return X.solve_bh_mix(st) is not None


def floor_T(rho_t: float, chi: float, t_hi: float = 2.0,
            t_lo: float = 0.05, step: float = 0.01) -> float | None:
    """Lowest T* with a real root, walking DOWN from t_hi.

    Walking down and stopping at the first failure -- rather than scanning the
    whole range and taking the minimum success -- is deliberate: it reports the
    boundary of the connected region that contains the physical high-T solution,
    which is what a dome sweep marching down would actually meet.
    """
    last = None
    for T in np.arange(t_hi, t_lo, -step):
        if not has_solution(rho_t, float(T), chi):
            break
        last = float(T)
    return last


if __name__ == '__main__':
    lam = json.load(open(os.path.join(OUT, 'lambda_line.json')))
    isos = [r['rho'] for r in lam['isochores']]
    dm = json.load(open(os.path.join(OUT, 'domes_models.json')))
    tcr = dm['models']['exact MSA']['tcr']

    res = {'nu': NU, 'z': Z, 'chis': list(CHIS), 'isochores': []}
    hdr = '  rho*   T_cr    ' + ''.join(f'chi={c:<7.2f}' for c in CHIS)
    print(hdr)
    for rho in isos:
        row = {'rho': rho, 'tcr': tcr[f'{rho:.2f}'], 'floor': {}}
        cells = []
        for chi in CHIS:
            f = floor_T(rho, chi)
            row['floor'][f'{chi:.2f}'] = f
            cells.append('  none  ' if f is None else f'{f:7.3f} ')
        res['isochores'].append(row)
        print(f'  {rho:4.2f}  {row["tcr"]:6.3f}  ' + ''.join(cells), flush=True)

    print('\n  ratio floor(chi=1/2) / T_cr:')
    for r in res['isochores']:
        print(f"    rho*={r['rho']:.2f}   {r['floor']['0.50'] / r['tcr']:.3f}")

    # The one dome point the sweep lost, and why.  `dome()` asks for 12 points
    # from 0.99*T_cr down to 0.62*T_cr; rho* = 0.60 returned 11.
    d = dm['models']['exact MSA']['domes']['0.60']
    tc = tcr['0.60']
    want = np.linspace(tc * 0.99, tc * 0.62, 12)
    got = {round(p['tstar'], 6) for p in d}
    miss = [float(t) for t in want if round(float(t), 6) not in got]
    res['rho060_missing'] = miss
    print(f"\n  rho* = 0.60 dome: asked 12, got {len(d)};  missing "
          f"{['%.4f' % m for m in miss]}")
    print(f"  floor there at small chi: {res['isochores'][0]['floor']['0.05']}")

    with open(os.path.join(OUT, 'fold_floor.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print(f'\nwrote {OUT}/fold_floor.json')
