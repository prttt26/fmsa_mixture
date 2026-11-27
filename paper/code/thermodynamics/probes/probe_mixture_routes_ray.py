r"""probe_mixture_routes_ray.py — the demixing line by the COMPRESSIBILITY and
VIRIAL routes, both through the constant-composition density ray.

`lambda_line.md` (2026-08-21) concluded there is "NO compressibility-route demixing
dome", on two grounds: chat(0) is not a Hessian (~20 % defect), and the
ray-integrated Delta-mu "has no non-trivial root anywhere in 0.45 <= T* <= 0.72".

⚠⚠ THE SECOND GROUND IS AN ARTEFACT OF THE SCAN WINDOW.  That same note records
the ray route's own consolute temperature as T_cr = 0.396 at rho* = 0.60 -- BELOW
the 0.45 the scan started at.  A dome cannot exist above its own critical point,
so finding none in [0.45, 0.72] says nothing.  This re-runs it from far below.

The constant-composition ray is the standard, reproducible convention for getting
a free energy out of a mixture chat(0), and the question worth answering is not
whether it is legitimate but whether it LANDS anywhere near the measurement.

Both routes here are FMSA-DP -- the model both lambda lines of Fig. 13 are drawn
from, and one of the two legs of Fig. 14.

  betaf_ex(rho, chi) = -Int_0^rho (rho - rho') C(rho', chi) drho'      [compressibility]
                     = rho Int_0^rho p_ex(rho', chi)/rho'^2 drho'      [virial]

  C     = Sum_ij x_i x_j chat_ij(0)          (the ray-direction second derivative)
  p_ex  = betap_virial - rho

  Delta-mu(chi) = ln(chi/(1-chi)) + (1/rho) d(betaf_ex)/dchi |_rho

Demixing spinodal: d(Delta-mu)/dchi = 0 at chi = 1/2.  Dome: the non-trivial root
of Delta-mu = 0.

Run:  python3 probe_mixture_routes_ray.py
"""
from __future__ import annotations

import json
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from scipy.optimize import brentq

import energy_compress_core as EC
import msa_exact_mix as X
from compare_mkanya_mixture import build
from compare_caccamo_lambda_line import NU, Z

OUT = 'numerical_notes/results/mixture_caccamo/routes_ray.json'
NQ = 48                     # Gauss-Legendre nodes on the density ray
RHO = 0.60                  # the isochore lambda_line.md reports on
CACCAMO = {0.60: 0.7390, 0.70: 0.8729, 0.75: 0.9500, 0.80: 1.0306, 1.00: 1.3953}


def model(rho: float, T: float, chi: float):
    return build(rho, T, Z, NU, x=np.array([chi, 1.0 - chi]))


# ── the two ray integrands ───────────────────────────────────────────────────
def C_ray(rho: float, T: float, chi: float) -> float:
    """`Sum_ij x_i x_j chat_ij(0)` -- so that `d2(betaf_ex)/drho^2 = -C`."""
    m = model(rho, T, chi)
    hs, c1 = EC.dp_chat0(m)
    x = np.array([chi, 1.0 - chi])
    return float(x @ (hs + c1) @ x)


def p_virial(rho: float, T: float, chi: float, h: float = 1e-3) -> float:
    """`betap` by the virial route, the mixture form of `msa_exact_mix.pressure_virial`.

    ⚠ The Yukawa amplitudes and rates are read off `from_fmsa`, not rebuilt: that
    is the only way the virial pressure and the rest of the paper see one system.
    `g` is FMSA-DP's own, so this is FMSA-DP's virial route and not a hybrid.
    """
    m = model(rho, T, chi)
    st = X.from_fmsa(m)
    rr = st.rho[:, None] * st.rho[None, :]
    gL = np.array([np.real(m.get_gL_laplace(np.array([complex(z)]))[0]) for z in st.z])
    dg = np.array([(np.real(m.get_gL_laplace(np.array([complex(z + h)]))[0])
                    - np.real(m.get_gL_laplace(np.array([complex(z - h)]))[0])) / (2 * h)
                   for z in st.z])
    J = (2 * np.pi / 3) * float(np.sum(
        rr[None] * st.K * np.exp(st.z[:, None, None] * st.SIG[None])
        * (st.z[:, None, None] * dg - gL)))
    g = np.array([[m.get_g_contact(i, j) for j in range(st.N)] for i in range(st.N)])
    hard = (2 * np.pi / 3) * float(np.sum(rr * st.SIG ** 3 * g))
    return float(np.sum(st.rho)) + hard + J


# ── ray integration ──────────────────────────────────────────────────────────
_GL = np.polynomial.legendre.leggauss(NQ)


def betaf_ex(rho: float, T: float, chi: float, route: str) -> float:
    u, w = _GL
    r = 0.5 * rho * (u + 1.0)
    wt = 0.5 * rho * w
    if route == 'C':
        # -Int (rho - rho') C(rho') drho'
        return float(-np.sum(wt * (rho - r) * np.array([C_ray(x, T, chi) for x in r])))
    # virial:  rho * Int p_ex/rho'^2 ;  p_ex/rho'^2 -> B_2 finite as rho' -> 0
    v = np.array([(p_virial(x, T, chi) - x) / x ** 2 for x in r])
    return float(rho * np.sum(wt * v))


def betaf_ex_energy(rho: float, T: float, chi: float) -> float:
    """⭐ GATE LEG.  The energy route needs no ray at all -- its free energy comes
    from charging in the coupling K, which has an exact endpoint at K = 0 -- so it
    is the one leg whose answer is already known independently
    (`compare_caccamo_demixing_dome.tcr_energy` gives 0.8432 at rho* = 0.60).
    Running it through the SAME chi-derivative and bisection as the two ray legs
    is what says whether this scaffolding is right; a route disagreeing with the
    record is only interesting once the machinery has reproduced something."""
    from fmsa_free_energy import FMSAFreeEnergy
    return float(FMSAFreeEnergy(model(rho, T, chi)).get_free_energy('lj'))


def dmu(rho: float, T: float, chi: float, route: str, h: float = 5e-3) -> float:
    # ⚠ Clamp the step: at chi = 2e-3 a fixed h = 5e-3 asks for a NEGATIVE partial
    # density, and `_P_kernel` dies in `sqrt(rho_i rho_m)` rather than returning
    # anything a caller can test.
    h = min(h, 0.4 * chi, 0.4 * (1.0 - chi))
    f = betaf_ex_energy if route == 'E' else (lambda r, T_, c: betaf_ex(r, T_, c, route))
    d = (f(rho, T, chi + h) - f(rho, T, chi - h)) / (2 * h)
    return float(np.log(chi / (1.0 - chi)) + d / rho)


def slope(rho: float, T: float, route: str, h: float = 0.02) -> float:
    """`d(Delta-mu)/dchi` at chi = 1/2.  Negative = unstable to demixing."""
    return (dmu(rho, T, 0.5 + h, route) - dmu(rho, T, 0.5 - h, route)) / (2 * h)


def tcr(rho: float, route: str, lo: float = 0.10, hi: float = 2.60) -> float | None:
    # ⚠ hi = 2.60, not 1.60.  At rho* = 1.00 the energy route's T_cr is 1.6216 and
    # the virial route's 1.8645, so a 1.60 ceiling returned "no root" for both and
    # the table read as though those routes FAILED at the highest density -- the
    # one density that decides whether a line degrades.  The bracket has to clear
    # the worst route, not the reference.
    a, b = slope(rho, lo, route), slope(rho, hi, route)
    if a * b > 0:
        return None
    return float(brentq(lambda T: slope(rho, T, route), lo, hi, xtol=2e-4))


def coexisting_chi(rho: float, T: float, route: str) -> float | None:
    """The non-trivial root of Delta-mu = 0 on (0, 1/2)."""
    lo, hi = 2e-3, 0.5 - 1e-3
    a, b = dmu(rho, T, lo, route), dmu(rho, T, hi, route)
    if a * b > 0:
        return None
    return float(brentq(lambda c: dmu(rho, T, c, route), lo, hi, xtol=1e-6))


ROUTES = (('E', 'energy (charging)'), ('C', 'compressibility (ray)'),
          ('V', 'virial (ray)'))
LOCAL_HESSIAN = {0.60: 0.6500, 0.70: 0.7584, 0.75: 0.8125,
                 0.80: 0.8667, 1.00: 1.0834}      # lambda_line.md, det M = 0


def main():
    rec = {'system': f'Caccamo symmetric HCY mixture, nu={NU}, z={Z}',
           'reference': CACCAMO, 'local_hessian': LOCAL_HESSIAN}

    print('S0  gate: the virial pressure at K -> 0 must be the CS/BMCSL virial EOS')
    for rho in (0.30, 0.60):
        eta = np.pi / 6 * rho
        cs = rho * (1 + eta + eta ** 2 - eta ** 3) / (1 - eta) ** 3
        pv = p_virial(rho, 50000.0, 0.5)
        print(f'   rho*={rho:.2f}  virial {pv:.8f}   CS {cs:.8f}   rel {abs(pv/cs-1):.2e}')

    print('\nS1  gate: this scaffolding must reproduce the RECORDED energy-route T_cr')
    from compare_caccamo_demixing_dome import tcr_energy
    a, b = tcr(0.60, 'E'), tcr_energy(0.60)
    print(f'   here {a:.4f}   recorded {b:.4f}   rel {abs(a/b-1):.2e}')
    rec['gate_energy'] = dict(here=a, recorded=b)

    print('\nS2  consolute temperature, every isochore, every route')
    print(f"   {'rho*':>5} {'Caccamo':>8} {'det M=0':>9} "
          f"{'energy':>9} {'compr(ray)':>11} {'virial(ray)':>12}")
    rows = []
    for rho in sorted(CACCAMO):
        r = {'rho': rho, 'ref': CACCAMO[rho], 'hessian': LOCAL_HESSIAN[rho]}
        for key, _ in ROUTES:
            try:
                r[key] = tcr(rho, key)
            except Exception:
                r[key] = None
        rows.append(r)
        f = lambda v: f'{v:9.4f}' if v else '     none'
        print(f"   {rho:5.2f} {r['ref']:8.4f} {r['hessian']:9.4f} "
              f"{f(r['E'])} {f(r['C']):>11} {f(r['V']):>12}")
    print(f"\n   {'rho*':>5} {'det M=0':>9} {'energy':>9} {'compr(ray)':>11} {'virial(ray)':>12}"
          '   [ % against Caccamo ]')
    for r in rows:
        pc = lambda v: f'{100*(v/r["ref"]-1):+8.1f}%' if v else '    none'
        print(f"   {r['rho']:5.2f} {100*(r['hessian']/r['ref']-1):+8.1f}% "
              f"{pc(r['E'])} {pc(r['C']):>11} {pc(r['V']):>12}")
    rec['tcr'] = rows

    print('\nS3  the dome below each route\'s own T_cr, at rho* = 0.60')
    doms = {}
    for key, name in ROUTES:
        tc = next(r[key] for r in rows if r['rho'] == RHO)
        if not tc:
            continue
        out = []
        for frac in (0.98, 0.90, 0.80, 0.70):
            try:
                c = coexisting_chi(RHO, frac * tc, key)
            except Exception:
                c = None
            out.append(dict(T=frac * tc, T_frac=frac, chi=c))
            print(f"   {name:22s} T/T_cr={frac:.2f}  T*={frac*tc:.4f}  "
                  f"chi = {('%.4f' % c) if c else 'none'}")
        doms[name] = out
    rec['dome'] = doms

    with open(OUT, 'w') as fh:
        json.dump(rec, fh, indent=1)
    print(f'\n[saved] {OUT}')


if __name__ == '__main__':
    main()
