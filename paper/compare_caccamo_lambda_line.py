#!/usr/bin/env python3
"""The demixing (lambda) line of the symmetric HCY mixture, from det M = 0.

Reference
---------
C. Caccamo, D. Costa and G. Pellicane, J. Chem. Phys. 109, 4498 (1998).
Symmetric equal-diameter hard-core Yukawa mixture, nu = K_12/K_11 = 0.7,
z = 2.45.  Table II gives SGCMC liquid-liquid coexistence: 31 points on five
isochores, each a T*-scan of the two coexisting liquid compositions.
Transcribed in `data_benchmark/reference_caccamo1998.json`.

THE CRITERION
-------------
The stability matrix of a binary mixture is the Hessian of the free-energy
density in the densities,

    M_ij = delta_ij / rho_i - chat_ij(0),

and the spinodal is det M = 0.  For the SYMMETRIC mixture at x = 1/2,
M = [[a, b], [b, a]] with a = 2/rho - chat_11(0) and b = -chat_12(0), so it
diagonalises exactly in the (sum, difference) basis:

    lam_+ = a + b   eigenvector (1,1)/sqrt2   DENSITY   channel -> liquid-vapour
    lam_- = a - b   eigenvector (1,-1)/sqrt2  CONCENTRATION channel -> DEMIXING

det M = lam_+ * lam_-, so det M = 0 fires on whichever channel goes first.  The
lambda line is specifically the lam_- = 0 branch.  Both are reported, because
that split is the whole point:

  ⚠ lam_+ NEVER VANISHES anywhere in the demixing region (it runs +4.6 to +128
    over the states here).  A mechanical-stability criterion -- dP/drho, or any
    `mech` test -- therefore reports "no transition" across the entire lambda
    line, silently.  Demixing is a COMPOSITION instability at fixed total
    density, with mechanical stability fully intact.

THE HARD-SPHERE REFERENCE DROPS OUT
-----------------------------------
For EQUAL diameters the Lebowitz hard-sphere DCF is species-independent, so
chat^HS_11(0) = chat^HS_12(0) exactly and the HS block CANCELS in
lam_- = 2/rho - chat_11(0) + chat_12(0).  Verified numerically below to 0.0.
Consequence: the lambda line is a pure statement about the Yukawa part c^(1),
free of the PY-vs-FMT hard-sphere ambiguity that afflicts every other line in
this project.  It is the cleanest object here.

WHAT IS COMPARED
----------------
At each density the consolute critical point sits at chi = 1/2 by symmetry, and
there the binodal and spinodal touch.  So solving lam_-(rho, T) = 0 at x = 1/2
yields exactly the theory's consolute critical temperature, directly comparable
to the one fitted from Table II.

⚠ Caccamo et al. do not PRINT T_cr per isochore -- they show the line in their
Fig. 2 and print only the crossover point.  The reference values below are
therefore OUR fit of THEIR stated procedure: Delta_chi ~ |T - T_cr|^beta with
beta = 0.32 held fixed (their Eq. 8 and the sentence fixing beta), applied to
each isochore of Table II.  Labelled `fit`, never `published`.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from compare_mkanya_mixture import build
from energy_compress_core import chat0_hs_fmt

REF = 'data_benchmark/reference_caccamo1998.json'
OUT = 'numerical_notes/results/mixture_caccamo'

NU = 0.7          # Caccamo's cross-to-like ratio for the LL data
Z = 2.45          # Caccamo's inverse range
BETA_CRIT = 0.32  # the nonclassical exponent THEY hold fixed


# ─── the stability matrix ─────────────────────────────────────────────────────

def stability_eigs(rho: float, T: float, nu: float = NU, z: float = Z,
                   k: float = 1e-5) -> tuple[float, float]:
    """(lam_+, lam_-) of M_ij = delta_ij/rho_i - chat_ij(0) at x = 1/2.

    chat(0) = HS block (FMT identity) + Yukawa block (closed-form transform at
    k -> 0; flat to 1e-8 below k = 1e-4, per `energy_compress_core.dp_chat0`).
    """
    f = build(rho, T, z, nu)
    c0 = np.real(np.asarray(chat0_hs_fmt(f))) + \
        np.real(np.asarray(f.get_c1hat_k(np.array([k])))[0])
    a = 1.0 / f.rho[0] - c0[0, 0]
    b = -c0[0, 1]
    return float(a + b), float(a - b)


def det_M(rho: float, T: float, **kw) -> float:
    lp, lm = stability_eigs(rho, T, **kw)
    return lp * lm


def lambda_T(rho: float, nu: float = NU, z: float = Z,
             lo: float = 0.25, hi: float = 4.0) -> float | None:
    """Consolute critical T* at this density: the root of lam_- = 0."""
    g = lambda T: stability_eigs(rho, T, nu, z)[1]
    try:
        if g(lo) * g(hi) > 0:
            return None
        return float(brentq(g, lo, hi, xtol=1e-10, rtol=1e-12))
    except (ValueError, FloatingPointError):
        return None


# ─── Caccamo's consolute points, fitted from Table II ─────────────────────────

def caccamo_consolute() -> list[dict]:
    """T_cr per isochore by fitting Delta_chi = A |T_cr - T|^0.32 to Table II.

    For fixed T_cr the amplitude A is linear, so it is eliminated by least
    squares and only T_cr is searched.
    """
    ref = json.load(open(REF))
    L = ref['lle']
    ic = {c: i for i, c in enumerate(L['columns'])}
    by_rho: dict[float, list[tuple[float, float]]] = {}
    for r in L['data']:
        rho, T = r[ic['rho']], r[ic['tstar']]
        dchi = abs(r[ic['chi2']] - r[ic['chi1']])
        by_rho.setdefault(rho, []).append((T, dchi))

    out = []
    for rho in sorted(by_rho):
        pts = sorted(by_rho[rho])
        T = np.array([p[0] for p in pts])
        d = np.array([p[1] for p in pts])

        def sse(Tc: float) -> float:
            if Tc <= T.max() + 1e-9:
                return 1e6
            b = (Tc - T) ** BETA_CRIT
            A = float(np.dot(b, d) / np.dot(b, b))
            return float(np.sum((d - A * b) ** 2))

        res = minimize_scalar(sse, bounds=(T.max() + 1e-6, T.max() + 2.0),
                              method='bounded',
                              options={'xatol': 1e-8})
        Tc = float(res.x)
        b = (Tc - T) ** BETA_CRIT
        A = float(np.dot(b, d) / np.dot(b, b))
        rms = float(np.sqrt(np.mean((d - A * b) ** 2)))
        out.append({'rho': rho, 'tcr_fit': Tc, 'amp': A, 'rms': rms,
                    'n': len(pts), 'tmax': float(T.max())})
    return out


# ─── gates ────────────────────────────────────────────────────────────────────

def gate_hs_cancels() -> None:
    """For equal diameters the HS block must cancel out of the demixing channel."""
    print("gate A: the hard-sphere reference cancels in the concentration channel")
    worst = 0.0
    for rho, T in ((0.60, 0.80), (0.80, 1.00), (1.00, 1.20)):
        f = build(rho, T, Z, NU)
        chs = np.real(np.asarray(chat0_hs_fmt(f)))
        d = abs(chs[0, 0] - chs[0, 1])
        worst = max(worst, d)
        print(f"  rho*={rho} T*={T}  chat_HS_11(0) - chat_HS_12(0) = {d:.3e}")
    assert worst < 1e-12, f"HS block does NOT cancel: {worst:.3e}"
    print(f"  -> PASS: the lambda line is free of the PY/FMT HS ambiguity "
          f"(worst {worst:.1e})\n")


def gate_alpha_one_no_demixing() -> None:
    """At nu = 1 the species are identical and there must be NO demixing."""
    print("gate B: nu = 1 has no concentration instability at any T")
    bad = []
    for rho in (0.6, 0.8, 1.0):
        for T in (0.4, 0.6, 1.0, 2.0):
            lm = stability_eigs(rho, T, nu=1.0)[1]
            if lm <= 0:
                bad.append((rho, T, lm))
    assert not bad, f"nu=1 shows a spurious demixing instability at {bad}"
    print("  -> PASS: lam_- > 0 everywhere at nu = 1, as identical species demand\n")


def gate_mech_blind() -> None:
    """The density channel must stay positive right through the lambda line."""
    print("gate C: the DENSITY channel never fires where demixing does")
    rows = []
    for rho in (0.60, 0.70, 0.75, 0.80, 1.00):
        Tc = lambda_T(rho)
        if Tc is None:
            continue
        lp, lm = stability_eigs(rho, Tc)
        rows.append((rho, Tc, lp, lm))
        print(f"  rho*={rho:.2f}  on the lambda line T*={Tc:.4f}: "
              f"lam_-={lm:+.2e} (zero), lam_+={lp:+.3f} (density channel)")
    assert all(r[2] > 1.0 for r in rows), "density channel came close to zero"
    print("  -> PASS: lam_+ stays > 1 throughout, so a mechanical/dP-drho test")
    print("     returns 'no transition' across the WHOLE lambda line.\n")


# ─── run ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    cac = caccamo_consolute()
    rows = []
    for c in cac:
        Tc = lambda_T(c['rho'])
        rows.append({**c, 'tcr_dp': Tc,
                     'dev': None if Tc is None else Tc - c['tcr_fit'],
                     'dev_pct': None if Tc is None
                     else 100.0 * (Tc - c['tcr_fit']) / c['tcr_fit']})
    # the line itself, on a grid, for plotting / crossover work
    grid = []
    for rho in np.arange(0.30, 1.101, 0.02):
        Tc = lambda_T(float(rho))
        if Tc is not None:
            grid.append({'rho': float(rho), 'tstar': Tc})
    return {'nu': NU, 'z': Z, 'beta_crit': BETA_CRIT,
            'isochores': rows, 'lambda_line': grid}


# ─── the crossover point (CEP) ────────────────────────────────────────────────

def lv_coexist(T: float, guess: tuple[float, float]) -> tuple[float, float] | None:
    """Liquid-vapour coexistence of the MIXTURE at x = 1/2, equal P and mu.

    By the sigma_1 = sigma_2, K_11 = K_22 symmetry both phases stay equimolar --
    which is not an assumption but a fact of the reference data: chi_v = chi_l =
    0.50 (1) in every row of Caccamo's Table I.  So the two conditions
    P_v = P_l and mu_v = mu_l in the two unknowns (rho_v, rho_l) close the
    problem, exactly as for one component.  Solved in log(rho) so the vapour
    root cannot walk through zero.
    """
    from scipy.optimize import root
    from compare_mkanya_mixture import thermo as mix_thermo

    def res(u):
        rv, rl = np.exp(u)
        if not (1e-9 < rv < 1.4 and 1e-9 < rl < 1.4) or rl <= rv:
            return [1e3, 1e3]
        a = mix_thermo(rv, T, Z, NU)
        b = mix_thermo(rl, T, Z, NU)
        return [a['p'] - b['p'], a['mu'] - b['mu']]

    sol = root(res, np.log(np.array(guess)), method='hybr', tol=1e-10)
    if not sol.success:
        return None
    rv, rl = np.exp(sol.x)
    if rl - rv < 1e-3:
        return None                     # collapsed onto the critical point
    return float(rv), float(rl)


def crossover_point(t_hi: float = 0.80, t_lo: float = 0.40,
                    n: int = 25) -> dict:
    """CEP = where the lambda line meets the LIQUID branch of the LV binodal.

    Walks the binodal down in T*, warm-starting each solve from the previous
    root, and brackets the sign change of  g(T) = T - T_lambda(rho_liquid(T)).
    Above the CEP the liquid is stable against demixing (g > 0); below it the
    liquid branch has entered the demixing region (g < 0).
    """
    branch, prev = [], (0.02, 0.80)
    for T in np.linspace(t_hi, t_lo, n):
        sol = lv_coexist(float(T), prev)
        if sol is None:
            continue
        prev = sol
        Tl = lambda_T(sol[1])
        branch.append({'tstar': float(T), 'rho_vap': sol[0], 'rho_liq': sol[1],
                       'tlambda_at_rho_liq': Tl,
                       'g': None if Tl is None else float(T) - Tl})
    # Walk UP from the seed too, so the dome closes.  Descending alone left the
    # two branches stopping short of each other, which reads as an unfinished
    # curve; the critical point is only ~0.06 above the seed.
    # Seed from the TOP OF THE CONVERGED BRANCH, not from t_hi: the descending
    # walk's first few temperatures fail from a cold guess, so t_hi itself has no
    # solution to continue from.  Warm-started, those same temperatures converge.
    if not branch:
        return {'branch': branch, 'cep': None, 'critical': None}
    top = max(branch, key=lambda b: b['tstar'])
    prev = (top['rho_vap'], top['rho_liq'])
    for T in np.arange(top['tstar'] + 0.01, top['tstar'] + 0.30, 0.01):
        sol = lv_coexist(float(T), prev)
        if sol is None:
            break
        prev = sol
        Tl = lambda_T(sol[1])
        branch.append({'tstar': float(T), 'rho_vap': sol[0], 'rho_liq': sol[1],
                       'tlambda_at_rho_liq': Tl,
                       'g': None if Tl is None else float(T) - Tl})
    branch.sort(key=lambda b: b['tstar'])

    good = [b for b in branch if b['g'] is not None]
    cep = None
    for u, v in zip(good, good[1:]):   # branch is sorted ascending in T
        if u['g'] * v['g'] < 0:
            # linear interpolation in T of the sign change
            w = u['g'] / (u['g'] - v['g'])
            Tc = u['tstar'] + w * (v['tstar'] - u['tstar'])
            rc = u['rho_liq'] + w * (v['rho_liq'] - u['rho_liq'])
            cep = {'tstar': float(Tc), 'rho': float(rc)}
            break
    # Critical point of the LV dome, by the same width law Figure 11 uses:
    # (rho_l - rho_v)^2 is linear in T near T_c, and rho_c by rectilinear diameters.
    crit = None
    if len(branch) >= 6:
        T = np.array([b['tstar'] for b in branch])
        w = np.array([b['rho_liq'] - b['rho_vap'] for b in branch])
        d = np.array([0.5 * (b['rho_liq'] + b['rho_vap']) for b in branch])
        m = slice(-6, None)
        a, b0 = np.polyfit(T[m], (w ** 2)[m], 1)
        tc = float(-b0 / a)
        ad, bd = np.polyfit(T[m], d[m], 1)
        crit = {'tstar': tc, 'rho': float(ad * tc + bd),
                'extrapolation': tc - float(T.max())}
    return {'branch': branch, 'cep': cep, 'critical': crit}


def gate_caccamo_fit_reproduces_their_cep(iso: list[dict]) -> None:
    """Our fit of THEIR consolute line must extrapolate to THEIR printed CEP.

    Caccamo et al. print rho*_e ~= 0.57, T*_e ~= 0.7 (journal p. 4504) as the
    crossover, reached by 'a very short extrapolation' of the consolute line to
    the binodal.  Extrapolating the line we fitted from Table II to rho* = 0.57
    must land on T* ~= 0.7, or the fit is not reproducing their procedure.
    """
    print("gate D: our fit of Caccamo's consolute line vs THEIR printed CEP")
    pts = sorted(iso, key=lambda r: r['rho'])[:2]
    (r0, t0), (r1, t1) = ((p['rho'], p['tcr_fit']) for p in pts)
    slope = (t1 - t0) / (r1 - r0)
    t_at = t0 + slope * (0.57 - r0)
    print(f"  fitted line through rho*={r0:.2f} (T*={t0:.4f}) and "
          f"rho*={r1:.2f} (T*={t1:.4f}), slope {slope:.3f}")
    print(f"  extrapolated to rho* = 0.57 -> T* = {t_at:.4f}   "
          f"(Caccamo print T*_e ~= 0.70)")
    assert abs(t_at - 0.70) < 0.05, f"fit does not reproduce their CEP: {t_at:.3f}"
    print(f"  -> PASS, |difference| = {abs(t_at - 0.70):.4f}: the Table II fit "
          f"reproduces a number they DID print.\n")


def report(res: dict) -> None:
    print("=== consolute critical temperature, nu = 0.7, z = 2.45 " + "=" * 12)
    print(f"{'rho*':>6} {'n':>3} {'T_max':>7} | {'Caccamo fit':>12} {'rms':>7} | "
          f"{'FMSA-DP':>9} | {'dev':>8} {'dev %':>8}")
    devs = []
    for r in res['isochores']:
        dp = '   --   ' if r['tcr_dp'] is None else f"{r['tcr_dp']:9.4f}"
        dv = '    --  ' if r['dev'] is None else f"{r['dev']:+8.4f}"
        dpc = '    --  ' if r['dev_pct'] is None else f"{r['dev_pct']:+8.2f}"
        print(f"{r['rho']:>6.2f} {r['n']:>3} {r['tmax']:>7.2f} | "
              f"{r['tcr_fit']:>12.4f} {r['rms']:>7.4f} | {dp} | {dv} {dpc}")
        if r['dev_pct'] is not None:
            devs.append(r['dev_pct'])
    if devs:
        print(f"\n  mean deviation {np.mean(devs):+.1f} %, "
              f"range {min(devs):+.1f} to {max(devs):+.1f} %")
    print("\n  'Caccamo fit' = OUR fit of THEIR procedure (Delta_chi ~ |T-T_cr|^0.32,")
    print("  beta fixed at 0.32) to each isochore of their Table II. The paper prints")
    print("  no per-isochore T_cr; it shows the line in Fig. 2. Not a published number.")


if __name__ == '__main__':
    gate_hs_cancels()
    gate_alpha_one_no_demixing()
    gate_mech_blind()
    res = run()
    gate_caccamo_fit_reproduces_their_cep(res['isochores'])
    report(res)
    print("\n=== crossover point (CEP): lambda line meets the LV binodal " + "=" * 6)
    cx = crossover_point()
    res['crossover'] = cx
    print(f"{'T*':>7} {'rho_vap':>9} {'rho_liq':>9} {'T_lambda(rho_l)':>16} {'g=T-T_lam':>11}")
    for b in cx['branch']:
        tl = '   --  ' if b['tlambda_at_rho_liq'] is None else f"{b['tlambda_at_rho_liq']:16.4f}"
        g = '   --  ' if b['g'] is None else f"{b['g']:+11.4f}"
        print(f"{b['tstar']:>7.4f} {b['rho_vap']:>9.4f} {b['rho_liq']:>9.4f} {tl} {g}")
    if cx['cep'] is None:
        print("\n  no sign change bracketed -- the lambda line does not meet the")
        print("  liquid branch in the scanned window.")
    else:
        c = cx['cep']
        print(f"\n  FMSA-DP crossover:  rho* = {c['rho']:.3f}   T* = {c['tstar']:.3f}")
        print(f"  Caccamo (GEMC+SGCMC): rho*_e ~= 0.57   T*_e ~= 0.70")
        print(f"  deviation:          rho* {100*(c['rho']-0.57)/0.57:+.1f} %   "
              f"T* {100*(c['tstar']-0.70)/0.70:+.1f} %")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'lambda_line.json'), 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}/lambda_line.json")
