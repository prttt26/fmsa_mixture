#!/usr/bin/env python3
"""tau-scaling of the first-order DCF against the EXACT multi-Yukawa MSA, pair by pair, for unequal-diameter mixtures.

The exact MSA at coupling tau (all Yukawa amplitudes scaled by tau, rates fixed) is read off the FMSA object
(msa_exact_mix.from_fmsa) and its real-space core is evaluated by grid-free adaptive quadrature
(brk_completeness.machinery).  FMSA at coupling tau is c_HS^PY + tau c^(1) with the closed forms of FMSA_double_prop.
The residual  c_MSA(tau) - c_HS - tau c^(1)  must scale as tau^2 on every pair.  The check is the max norm over the
core r < R_ij of the residual divided by the max norm of tau c^(1) (so it must fall as tau), and the fitted slope of
log(max residual) against log tau for tau <= 1/4.

    python code/run.py code/fmsa_dcf/validation/verify_tau_scaling.py
"""
import warnings
import numpy as np

warnings.simplefilter('ignore')
import msa_exact_mix as X
from fmsa_double_prop import FMSA_double_prop
from brk_completeness import machinery

TAUS = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
SYSTEMS = {
    'binary (sigma 1:2)': dict(sig=[1., 2.], eps=[1., .5], x=[.25, .75], rho=0.139, T=1.0),
    'ternary (sigma 1:1.6:2.3)': dict(sig=[1., 1.6, 2.3], eps=[1., .7, .5], x=[.3, .3, .4], rho=0.12, T=1.2),
}


def residuals(dp, i, j, n=240):
    R = float(dp.R_ij[i, j])
    L = abs(float(dp.lambda_ij[i, j]))
    rs = np.linspace(1e-3, R * 0.999, n)
    if L > 0:                                   # never sample the knot itself
        rs = rs[np.abs(rs - L) > 1e-3]
    c1 = np.array([dp.get_c1_inner(float(r), i, j) for r in rs])
    chs = np.array([float(dp.get_HS_PY(float(r), i, j)) for r in rs])
    out = []
    for tau in TAUS:
        st = X.from_fmsa(dp, tau)
        cfun, *_ = machinery(sigma=list(st.sigma), rho=list(st.rho), K=np.asarray(st.K, float), z=list(np.asarray(st.z, float)))
        ce = np.array([cfun(i, j, float(r)) / (2.0 * np.pi * float(r)) for r in rs])
        res = ce - chs - tau * c1
        out.append((tau, float(np.max(np.abs(res))), float(np.max(np.abs(tau * c1)))))
    return out


def main():
    for name, s in SYSTEMS.items():
        dp = FMSA_double_prop(np.array(s['sig']), np.array(s['eps']), np.array(s['x']), s['rho'], s['T'],
                              n_tails=2, verbose=False)
        print(f'\n{name}: rho*={s["rho"]}, T*={s["T"]}')
        for i in range(dp.N):
            for j in range(i, dp.N):
                o = residuals(dp, i, j)
                taus = np.array([a for a, _, _ in o]); res = np.array([b for _, b, _ in o])
                slope = np.polyfit(np.log(taus[1:]), np.log(res[1:]), 1)[0]      # tau <= 1/4
                rel = ' '.join(f'{b / c:.2e}' for _, b, c in o)
                print(f'  pair ({i + 1},{j + 1}): max|resid|/max|tau c1| at tau=1/2..1/32: {rel};  slope {slope:.3f}')
                assert 1.85 < slope < 2.15, (name, i, j, slope)


if __name__ == '__main__':
    main()
