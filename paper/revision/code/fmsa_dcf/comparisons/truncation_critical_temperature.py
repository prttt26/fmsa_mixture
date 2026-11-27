#!/usr/bin/env python3
"""Critical temperature of the EXACT multi-Yukawa MSA and of FMSA on the same system, by minimising a(rho).

a = 1 - rho sum_ij x_i x_j chat_ij(0) = 1/S(0) at fixed composition.  The critical temperature is where
min_rho a reaches zero; near it min_rho a is linear in T (mean-field), so the exact value is obtained by
minimising a over rho at several T above T_c and extrapolating linearly to zero.  FMSA's a goes through
zero and negative, so its T_c is a bracketed root.  The exact MSA has a >= 0 identically and no solution
inside the spinodal (the unreachable band), which is why it needs the extrapolation.

    python code/run.py code/fmsa_dcf/comparisons/truncation_critical_temperature.py
Output: numerical_notes/results/truncation/truncation_critical_temperature.csv
"""
import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, brentq

warnings.simplefilter('ignore')
import msa_exact_mix as X
from fmsa_double_prop import FMSA_double_prop

OUT = Path('numerical_notes/results/truncation/truncation_critical_temperature.csv')
SYS = {'pure': dict(sig=[1.], eps=[1.], x=[1.], rho0=0.27, span=(0.18, 0.36), Tex=(0.975, 0.985, 0.995, 1.005, 1.02)),
       'binary': dict(sig=[1., 2.], eps=[1., .5], x=[.25, .75], rho0=0.045, span=(0.030, 0.062), Tex=(0.535, 0.545, 0.555, 0.565, 0.58))}


def a_of(name, T, rho, exact):
    s = SYS[name]
    sig, eps, x = map(np.array, (s['sig'], s['eps'], s['x']))
    dp = FMSA_double_prop(sig, eps, x, rho, T, n_tails=2, verbose=False)
    rv = dp.rho
    n = len(rv)
    if exact:
        sol = X.solve_bh_mix(X.from_fmsa(dp, 1.0))
        if sol is None:
            return np.nan
        C = X.chat0(sol)
    else:
        C = dp.get_chat_k(np.array([1e-7]))[0]
    return 1.0 - rv.sum() * sum(x[i] * x[j] * C[i, j] for i in range(n) for j in range(n))


def amin(name, T, exact):
    lo, hi = SYS[name]['span']
    f = lambda r: (lambda v: 1e3 if not np.isfinite(v) else v)(a_of(name, T, r, exact))
    r = minimize_scalar(f, bounds=(lo, hi), method='bounded', options={'xatol': 1e-4})
    return float(r.fun), float(r.x)


def main():
    rows = []
    for name, s in SYS.items():
        # FMSA: bracketed root of min_rho a
        Tf = brentq(lambda T: amin(name, T, False)[0], 0.8 * s['Tex'][0], 1.35 * s['Tex'][-1] + 0.2, xtol=2e-4)
        # exact: linear extrapolation of min_rho a(T) to zero from T > T_c
        pts = [(T,) + amin(name, T, True) for T in s['Tex']]
        T_arr = np.array([p[0] for p in pts])
        a_arr = np.array([p[1] for p in pts])
        # quadratic in T (min a is not exactly linear); a straight line through the two lowest points is the check
        c2 = np.polyfit(T_arr, a_arr, 2)
        roots = np.roots(c2)
        roots = roots[np.isreal(roots)].real
        Te = float(roots[np.argmin(np.abs(roots - T_arr[0]))])
        k1, b1 = np.polyfit(T_arr[:2], a_arr[:2], 1)
        Te_lin2 = -b1 / k1
        k, b = np.polyfit(T_arr, a_arr, 1)
        resid = float(np.max(np.abs(np.polyval(c2, T_arr) - a_arr)))
        rows.append(dict(system=name, Tc_fmsa=Tf, Tc_exact=Te, Tc_exact_two_lowest=Te_lin2, ratio=Tf / Te, rel_diff_pct=100 * (Tf / Te - 1),
                         extrap_resid=resid, slope=k,
                         **{f'amin_T{T:.2f}': a for T, a in zip(T_arr, a_arr)},
                         **{f'rho_T{T:.2f}': p[2] for T, p in zip(T_arr, pts)}))
        print(name, f'Tc(FMSA)={Tf:.4f}  Tc(exact MSA, quadratic)={Te:.4f} (two lowest points, linear: {Te_lin2:.4f})  FMSA/exact-1={100 * (Tf / Te - 1):+.1f}%  fit resid={resid:.1e}')
        for T, p in zip(T_arr, pts):
            print(f'   T={T:.3f}  min a={p[1]:.5f} at rho={p[2]:.4f}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    with OUT.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
