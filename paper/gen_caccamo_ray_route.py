r"""gen_caccamo_ray_route.py — the demixing line and dome by the COMPRESSIBILITY
route, taken through the constant-composition density ray.

The third line for Fig. 13 and the third dome for Fig. 14.  Measured in
`probe_mixture_routes_ray.py` (and in `lambda_line.md`'s 2026-09-02 section) to be
the most accurate of the three on every isochore -- +5.4 to +9.0 % against
Caccamo, where the local-Hessian line runs -12 to -22 % and the energy route
+14 to +17 % -- which is why it is worth drawing rather than merely recording.

⭐ SAME CONSTRUCTION AS THE PROBE, IMPORTED NOT COPIED.  Everything numerical here
comes from `probe_mixture_routes_ray`; this module only chooses the grids and
writes the JSON the figure reads.  A second implementation of the ray integral
would be a second thing to keep in step with the table in `lambda_line.md`.

Run:  python3 gen_caccamo_ray_route.py
"""
from __future__ import annotations

import json
import warnings

warnings.filterwarnings('ignore')

import numpy as np

import probe_mixture_routes_ray as P

OUT = 'numerical_notes/results/mixture_caccamo/ray_route.json'
# The lambda line is smooth and nearly linear in rho (0.79, 0.95, 1.03, 1.12,
# 1.47 at the five measured isochores), so a 14-point grid draws it; the existing
# local-Hessian line uses 41 points because it is cheap and this one is not.
RHO_LINE = np.round(np.arange(0.45, 1.101, 0.05), 3)
ISOCHORES = (0.60, 0.70, 0.75, 0.80, 1.00)
T_FRAC = (0.985, 0.96, 0.92, 0.88, 0.83, 0.78, 0.72, 0.66)


def main() -> None:
    rec = {'nu': P.NU, 'z': P.Z, 'route': 'compressibility, constant-composition ray'}

    print('lambda line')
    line = []
    for rho in RHO_LINE:
        try:
            t = P.tcr(float(rho), 'C')
        except Exception:
            t = None
        if t:
            line.append({'rho': float(rho), 'tstar': float(t)})
            print(f'  rho* = {rho:.2f}   T_cr = {t:.4f}', flush=True)
        else:
            print(f'  rho* = {rho:.2f}   no root', flush=True)
    rec['lambda_line_ray'] = line
    with open(OUT, 'w') as fh:      # write early: the line alone draws Fig. 13
        json.dump(rec, fh, indent=1)

    print('\ndomes')
    tcr = {}
    domes = {}
    for rho in ISOCHORES:
        t = next((p['tstar'] for p in line if abs(p['rho'] - rho) < 1e-9), None)
        if t is None:
            t = P.tcr(rho, 'C')
        tcr[f'{rho:.2f}'] = float(t)
        pts = []
        for frac in T_FRAC:
            T = frac * t
            try:
                c = P.coexisting_chi(rho, T, 'C')
            except Exception:
                c = None
            if c is not None:
                pts.append({'tstar': float(T), 'chi': float(c)})
            print(f'  rho* = {rho:.2f}  T/T_cr = {frac:.3f}  T* = {T:.4f}  '
                  f"chi = {('%.4f' % c) if c else 'none'}", flush=True)
        domes[f'{rho:.2f}'] = pts
        with open(OUT, 'w') as fh:
            json.dump(rec | {'tcr': tcr, 'domes': domes}, fh, indent=1)
    rec['tcr'], rec['domes'] = tcr, domes

    with open(OUT, 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f'\n[saved] {OUT}')


if __name__ == '__main__':
    main()
