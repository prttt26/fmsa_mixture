#!/usr/bin/env python3
r"""The COMPRESSIBILITY-route HCY binodal, to sit beside the energy-route one.

Figure 11 currently shows one route only (energy), so the caption has to *assert*
that "the compressibility route is a different curve, not a check on this one".
This module computes that other curve so the claim can be shown instead.

THE ROUTE
---------
The compressibility route hands you the second derivative of the free energy,
`d2(beta f_ex)/drho^2 = -chat(0)`, so integrating along the density ray gives, for
one component,

    beta mu_ex(rho) = -Int_0^rho chat(0; rho') drho'
    beta p(rho)     = rho - Int_0^rho rho' chat(0; rho') drho'
    beta mu(rho)    = ln rho + beta mu_ex(rho)

(the second line is the integral of `d beta P/d rho = 1 - rho chat(0)`).

⚠ SAME HARD-SPHERE REFERENCE ON BOTH ROUTES, OR THE GAP IS NOT THE GAP.  The
energy-route leg sits on WB1/Carnahan-Starling through `FMSAFreeEnergy`, so the
hard-sphere block of `chat(0)` here is taken from the SAME FMT identity
(`energy_compress_core.chat0_hs_fmt`), not from PY.  The hard-sphere reference alone
moves rho*_L by up to 7 % -- larger than the route gap this exists to display -- so
mixing references would attribute a core difference to the route.  Gate A checks that
the two routes agree at K = 0, which is exactly the statement that the HS blocks match.

⚠ PATH DEPENDENCE IS REAL.  The ray integral is exact only if chat(0) really is a
second derivative of something.  No approximate closure satisfies that identically,
so at N = 1 the integral is unambiguous (one variable, no path freedom) but the
resulting f is not guaranteed to equal the energy-route f -- and that difference IS
the route gap, not an error.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import root

from fmsa_double_prop import FMSA_double_prop
from fmsa_free_energy import FMSAFreeEnergy
from energy_compress_core import chat0_hs_fmt

Z_HCY = 1.8
KV = 1.0
OUT = 'numerical_notes/results/vle_hcy'
NQ = 64                       # Gauss-Legendre nodes on the density ray


def build(rho: float, tstar: float) -> FMSA_double_prop:
    return FMSA_double_prop(
        sig_pure=np.array([1.0]), eps_pure=np.array([1.0]),
        x=np.array([1.0]), rhostar=rho, tstar=tstar,
        n_tails=1, z=np.array([Z_HCY]), k=np.array([KV]),
        input_type='bh', verbose=False)


def chat0(rho: float, tstar: float, k: float = 1e-5) -> float:
    """Total chat(0) = FMT hard-sphere block + closed-form Yukawa block."""
    f = build(rho, tstar)
    hs = float(np.ravel(chat0_hs_fmt(f))[0])
    c1 = float(np.real(np.ravel(f.get_c1hat_k(np.array([k])))[0]))
    return hs + c1


_NODES, _WTS = np.polynomial.legendre.leggauss(NQ)


def thermo_compr(rho: float, tstar: float) -> tuple[float, float]:
    """(beta p, beta mu) by the compressibility route, one ray quadrature."""
    u = 0.5 * (_NODES + 1.0)          # [0,1]
    w = 0.5 * _WTS
    rp = rho * u
    c = np.array([chat0(float(r), tstar) for r in rp])
    mu_ex = -rho * float(np.dot(w, c))
    p = rho - rho * rho * float(np.dot(w, u * c))
    return p, float(np.log(rho)) + mu_ex


def thermo_energy(rho: float, tstar: float) -> tuple[float, float]:
    """(beta p, beta mu) by the energy route -- the existing Figure 11 leg."""
    fe = FMSAFreeEnergy(build(rho, tstar))
    return (float(fe.get_pressure(incore='lj')),
            float(np.ravel(fe.get_chemical_potentials_total(incore='lj'))[0]))


def coexist(tstar: float, guess, route) -> tuple[float, float] | None:
    def res(u):
        rv, rl = np.exp(u)
        if not (1e-9 < rv < 1.4 and 1e-9 < rl < 1.4) or rl <= rv:
            return [1e3, 1e3]
        pv, mv = route(rv, tstar)
        pl, ml = route(rl, tstar)
        return [pv - pl, mv - ml]
    sol = root(res, np.log(np.array(guess)), method='hybr', tol=1e-10)
    if not sol.success:
        return None
    rv, rl = np.exp(sol.x)
    return None if rl - rv < 1e-3 else (float(rv), float(rl))


def gate_hs_limit() -> None:
    """At K -> 0 the two routes must give the SAME pressure: both are the WB1 core."""
    print("gate A: at K = 0 the two routes share the hard-sphere reference")
    worst = 0.0
    for rho in (0.2, 0.4, 0.6):
        T = 1e6                       # K = KV/T* -> 0
        pc, _ = thermo_compr(rho, T)
        pe, _ = thermo_energy(rho, T)
        Zc, Ze = pc / rho, pe / rho
        worst = max(worst, abs(Zc - Ze) / Ze)
        print(f"  rho*={rho}  Z compr={Zc:.8f}  Z energy={Ze:.8f}  "
              f"rel={abs(Zc-Ze)/Ze:.2e}")
    assert worst < 5e-5, f"HS references differ between routes: {worst:.2e}"
    print(f"  -> PASS, worst rel {worst:.1e}: the route gap below is the ROUTE,\n"
          f"     not a hard-sphere core difference.\n")


def binodal(temps, route, label: str) -> list[dict]:
    out, prev = [], (0.02, 0.70)
    for T in temps:
        s = coexist(float(T), prev, route)
        if s is None:
            continue
        prev = s
        out.append({'tstar': float(T), 'rho_vap': s[0], 'rho_liq': s[1]})
    print(f"  {label}: {len(out)} points, T* {out[-1]['tstar']:.3f}"
          f"..{out[0]['tstar']:.3f}" if out else f"  {label}: none")
    return out


if __name__ == '__main__':
    gate_hs_limit()
    temps = np.round(np.arange(1.20, 0.799, -0.02), 3)
    print("binodals:")
    en = binodal(temps, thermo_energy, 'energy route')
    cp = binodal(temps, thermo_compr, 'compressibility route')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'vle_hcy_routes.json'), 'w') as fh:
        json.dump({'z': Z_HCY, 'energy': en, 'compressibility': cp}, fh, indent=2)

    ce = {r['tstar']: r for r in en}
    print(f"\n{'T*':>6} | {'rho_l en':>9} {'rho_l cp':>9} {'dev %':>7} | "
          f"{'rho_v en':>9} {'rho_v cp':>9} {'dev %':>7}")
    devs = []
    for r in cp:
        e = ce.get(r['tstar'])
        if not e:
            continue
        dl = 100 * (r['rho_liq'] - e['rho_liq']) / e['rho_liq']
        dv = 100 * (r['rho_vap'] - e['rho_vap']) / e['rho_vap']
        devs.append(dl)
        print(f"{r['tstar']:>6.2f} | {e['rho_liq']:>9.4f} {r['rho_liq']:>9.4f} "
              f"{dl:>+7.2f} | {e['rho_vap']:>9.4f} {r['rho_vap']:>9.4f} {dv:>+7.2f}")
    if devs:
        print(f"\n  liquid-branch route gap: median {np.median(devs):+.2f} %, "
              f"range {min(devs):+.2f} to {max(devs):+.2f} %")
    print(f"\nwrote {OUT}/vle_hcy_routes.json")
