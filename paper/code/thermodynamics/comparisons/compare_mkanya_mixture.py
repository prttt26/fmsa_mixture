#!/usr/bin/env python3
"""FMSA-DP against Mkanya et al. Table 1 -- symmetric hard-core Yukawa MIXTURE.

Reference
---------
A. Mkanya, G. Pellicane, C. Caccamo and D. Pini, J. Phys.: Condens. Matter 29,
365102 (2017), Table 1: seven single-phase states of the symmetric equimolar
HCY mixture at alpha = eps_12/eps_11 = 0.75, each carrying MC simulation AND
HRT AND MHNC for U/N, P and mu.  Transcribed in
`data_benchmark/reference_mkanya2017.json`.

Why this benchmark
------------------
It is the cheapest mixture test in the literature set: prescribed (z, T*, rho*),
no coexistence solve, no free-energy integration along a path, no inversion.
And it lands a new model in a THREE-WAY comparison for free.

The model is exactly this project's N=1 contact normalisation promoted to a
matrix -- sigma_1 = sigma_2 = 1, K_11 = K_22 = 1/T*, K_12 = alpha/T*, one z for
every pair.  Until 2026-08-19 it was unreachable: FMSA_double_prop hard-wired
Lorentz-Berthelot, which for equal diameters with eps_11 = eps_22 forces
alpha == 1 -- the degenerate one-component limit.  The `eps_ij` constructor
argument added that day is what makes alpha = 0.75 expressible.

ROUTE.  Energy route throughout, matching `vle_hcy_shukla.py`: `input_type='bh'`
(sigma IS the hard diameter, no Barker-Henderson softening), n_tails=1, and
`incore='lj'` in FMSAFreeEnergy.

U/N IS CLOSED FORM HERE, NOT A QUADRATURE.  For a Yukawa tail with sigma = 1,

    U_ex/N = (rho/2) sum_ij x_i x_j INT u_ij(r) g_ij(r) 4 pi r^2 dr
           = (rho/2) sum_ij x_i x_j * (-4 pi eps_ij e^z) * L[r g_ij](z)

because u_ij(r) = -eps_ij e^{-z(r-1)}/r kills the 1/r against the r^2 and leaves
exactly the Laplace transform of r*g_ij evaluated AT THE POLE s = z, which
`get_gL_laplace` returns in closed form.  No inversion, no de Hoog, no
contact-jump ringing.  Gate B below checks it against the thermodynamic
derivative U = d(beta f_ex)/d beta, which is a genuinely independent route.
"""

from __future__ import annotations

import json
import os

import numpy as np

from fmsa_double_prop import FMSA_double_prop
from fmsa_free_energy import FMSAFreeEnergy

REF = 'data_benchmark/reference_mkanya2017.json'
OUT = 'numerical_notes/results/mixture_mkanya'


# ─── model ────────────────────────────────────────────────────────────────────

def build(rhostar: float, tstar: float, z: float, alpha: float,
          x: np.ndarray | None = None) -> FMSA_double_prop:
    """Symmetric equal-diameter binary HCY mixture at (rho*, T*, z, alpha)."""
    if x is None:
        x = np.array([0.5, 0.5])
    eps_ij = np.array([[1.0, alpha], [alpha, 1.0]])
    return FMSA_double_prop(
        sig_pure=np.array([1.0, 1.0]), eps_pure=np.array([1.0, 1.0]),
        x=np.asarray(x, float), rhostar=rhostar, tstar=tstar,
        n_tails=1, z=np.array([z]), k=np.array([1.0]),
        eps_ij=eps_ij, input_type='bh', verbose=False)


def u_per_particle(f: FMSA_double_prop) -> float:
    """U_ex/(N eps_11), closed form: Laplace transform of r*g_ij at s = z.

    Valid only for sigma = 1 and a SINGLE Yukawa tail, both true here.
    """
    z = float(np.ravel(f.z_poles)[0])
    gL = np.asarray(f.get_gL_laplace(np.array([z], dtype=complex)))[0]
    gL = np.real(gL)
    x, rho = f.x, f.rhostar
    w = np.outer(x, x) * f.eps_mix                      # x_i x_j eps_ij
    return float(0.5 * rho * np.sum(w * (-4.0 * np.pi * np.exp(z) * gL)))


def thermo(rhostar: float, tstar: float, z: float, alpha: float) -> dict:
    """(u, p, mu) in Mkanya's units: U/(N eps), P sigma^3/eps, mu/eps."""
    f = build(rhostar, tstar, z, alpha)
    fe = FMSAFreeEnergy(f)
    bp = float(fe.get_pressure(incore='lj'))            # beta P
    bmu = np.ravel(fe.get_chemical_potentials_total(incore='lj'))
    return {
        'u': u_per_particle(f),
        'p': bp * tstar,                                # beta P -> P sigma^3/eps
        'mu': float(bmu[0]) * tstar,                    # beta mu -> mu/eps
        'mu_species_spread': float(np.max(bmu) - np.min(bmu)) * tstar,
        # mu_ex = mu - kT ln(rho_i): the part free of the de Broglie/standard-state
        # convention, so it can be compared even if the reference's mu is offset.
        'mu_ex': (float(bmu[0]) - float(np.log(f.rho[0]))) * tstar,
    }


# ─── gates ────────────────────────────────────────────────────────────────────

def gate_alpha_one() -> None:
    """alpha = 1 must reproduce the ONE-COMPONENT HCY exactly.

    At alpha = 1 with equal diameters the two species are energetically
    indistinguishable, so U/N and P must equal the N = 1 result at the same
    (rho*, T*).  This is the strongest available check that `eps_ij` did not
    perturb anything it should not have.

    mu IS THE EXCEPTION, AND MUST NOT BE ASSERTED EQUAL.  The species remain
    distinguishable LABELS, so mu_i keeps its ideal mixing term: with
    rho_i = rho/2, mu_i = kT ln(rho/2) + mu_ex against the pure fluid's
    kT ln(rho) + mu_ex.  The difference is exactly -kT ln 2 -- not an error, and
    the reduction to check is mu_mix + T* ln 2 == mu_pure.  (Asserting plain
    equality here fails by 0.749 at T* = 1.08 and 0.520 at T* = 0.75, both
    equal to T* ln 2 to every printed digit; that is what flagged this.)
    """
    print("gate A: alpha = 1 reduces to the one-component fluid")
    worst = 0.0
    for rho, T, z in ((0.45, 1.08, 1.8), (0.55, 1.08, 1.8), (0.50, 0.75, 3.0)):
        mix = thermo(rho, T, z, alpha=1.0)
        pure_f = FMSA_double_prop(
            sig_pure=np.array([1.0]), eps_pure=np.array([1.0]),
            x=np.array([1.0]), rhostar=rho, tstar=T,
            n_tails=1, z=np.array([z]), k=np.array([1.0]),
            input_type='bh', verbose=False)
        pure_fe = FMSAFreeEnergy(pure_f)
        pu = u_per_particle(pure_f)
        pp = float(pure_fe.get_pressure(incore='lj')) * T
        pmu = float(np.ravel(pure_fe.get_chemical_potentials_total(incore='lj'))[0]) * T
        mu_reduced = mix['mu'] + T * np.log(2.0)        # undo the ideal mixing term
        for name, a, b in (('u', mix['u'], pu), ('p', mix['p'], pp),
                           ('mu+T*ln2', mu_reduced, pmu)):
            d = abs(a - b)
            worst = max(worst, d)
            print(f"  rho*={rho} T*={T} z={z}  {name:9s} mix={a:+.10f} pure={b:+.10f}  |d|={d:.2e}")
    assert worst < 1e-9, f"alpha=1 does NOT reduce to N=1: worst |d| = {worst:.3e}"
    print(f"  -> PASS, worst |difference| = {worst:.2e}\n")


def gate_energy_routes(delta: float = 2e-3) -> None:
    """U/N from the Laplace pole must equal d(beta f_ex)/d beta / rho.

    Independent: the first reads the RDF at s = z, the second differentiates the
    free energy the coupling-constant integration produced.  A normalisation
    error in either shows up here.
    """
    print("gate B: U/N, Laplace-at-the-pole vs the thermodynamic derivative")
    worst = 0.0
    for rho, T, z, al in ((0.45, 1.08, 1.8, 0.75), (0.50, 0.85, 2.4, 0.75),
                          (0.40, 0.75, 3.0, 0.75)):
        u_lap = u_per_particle(build(rho, T, z, al))
        # U/V = d(beta f_ex)/d beta at fixed rho;  beta = 1/T*
        bp, bm = 1.0 / T + delta, 1.0 / T - delta
        fp = FMSAFreeEnergy(build(rho, 1.0 / bp, z, al)).get_free_energy('lj')
        fm = FMSAFreeEnergy(build(rho, 1.0 / bm, z, al)).get_free_energy('lj')
        u_thermo = (fp - fm) / (2.0 * delta) / rho
        d = abs(u_lap - u_thermo)
        worst = max(worst, d)
        print(f"  rho*={rho} T*={T} z={z}  laplace={u_lap:+.6f} "
              f"d(bf)/dbeta={u_thermo:+.6f}  |d|={d:.2e}")
    assert worst < 5e-4, f"the two energy routes disagree: worst |d| = {worst:.3e}"
    print(f"  -> PASS, worst |difference| = {worst:.2e} "
          f"(finite-difference truncation, O(delta^2) with delta={delta})\n")


# ─── the benchmark ────────────────────────────────────────────────────────────

def run() -> dict:
    ref = json.load(open(REF))
    t = ref['thermo']
    cols = t['columns']
    ix = {c: i for i, c in enumerate(cols)}
    alpha = t['alpha']

    rows = []
    for r in t['data']:
        z, T, rho = r[ix['z']], r[ix['tstar']], r[ix['rho']]
        dp = thermo(rho, T, z, alpha)
        rows.append({
            'z': z, 'tstar': T, 'rho': rho,
            'u': {'mc': r[ix['u_mc']], 'mc_err': r[ix['u_mc_err']],
                  'hrt': r[ix['u_hrt']], 'mhnc': r[ix['u_mhnc']], 'dp': dp['u']},
            'p': {'mc': r[ix['p_mc']], 'mc_err': r[ix['p_mc_err']],
                  'hrt': r[ix['p_hrt']], 'mhnc': r[ix['p_mhnc']], 'dp': dp['p']},
            'mu': {'mc': r[ix['mu_mc']], 'mc_err': r[ix['mu_mc_err']],
                   'hrt': r[ix['mu_hrt']], 'mhnc': r[ix['mu_mhnc']], 'dp': dp['mu']},
            'mu_species_spread': dp['mu_species_spread'],
        })
    return {'alpha': alpha, 'rows': rows}


def report(res: dict) -> None:
    rows = res['rows']
    # The paper's own outlier: MC disagrees with BOTH theories far more here
    # than anywhere else, and the state sits near the LV boundary.
    def is_outlier(r):
        return (r['z'], r['tstar'], r['rho']) == (1.8, 1.04, 0.55)

    for q, label in (('u', 'U/(N eps)'), ('p', 'P sigma^3/eps'), ('mu', 'mu/eps')):
        print(f"\n=== {label} " + "=" * (62 - len(label)))
        print(f"{'z':>4} {'T*':>5} {'rho*':>5} | {'MC':>10} {'HRT':>9} {'MHNC':>9} "
              f"{'FMSA-DP':>9} | {'DP-MC':>8} {'HRT-MC':>8} {'MHNC-MC':>8}")
        acc = {'dp': [], 'hrt': [], 'mhnc': []}
        acc_no = {'dp': [], 'hrt': [], 'mhnc': []}
        for r in rows:
            d = r[q]
            e = {m: d[m] - d['mc'] for m in ('dp', 'hrt', 'mhnc')}
            flag = ' *' if is_outlier(r) else '  '
            print(f"{r['z']:>4} {r['tstar']:>5} {r['rho']:>5} | {d['mc']:>10.3f} "
                  f"{d['hrt']:>9.3f} {d['mhnc']:>9.3f} {d['dp']:>9.3f} | "
                  f"{e['dp']:>+8.3f} {e['hrt']:>+8.3f} {e['mhnc']:>+8.3f}{flag}")
            for m in acc:
                acc[m].append(e[m])
                if not is_outlier(r):
                    acc_no[m].append(e[m])
        rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
        mad = lambda v: float(np.max(np.abs(v)))
        print(f"{'':>4} {'':>5} {'':>5} | {'RMS  all 7':>10} {'':>9} {'':>9} {'':>9} | "
              f"{rms(acc['dp']):>8.3f} {rms(acc['hrt']):>8.3f} {rms(acc['mhnc']):>8.3f}")
        print(f"{'':>4} {'':>5} {'':>5} | {'RMS  no *':>10} {'':>9} {'':>9} {'':>9} | "
              f"{rms(acc_no['dp']):>8.3f} {rms(acc_no['hrt']):>8.3f} {rms(acc_no['mhnc']):>8.3f}")
        print(f"{'':>4} {'':>5} {'':>5} | {'max|.| no *':>10} {'':>9} {'':>9} {'':>9} | "
              f"{mad(acc_no['dp']):>8.3f} {mad(acc_no['hrt']):>8.3f} {mad(acc_no['mhnc']):>8.3f}")
    print("\n  * = z=1.8, T*=1.04, rho*=0.55, the reference's own outlier state:")
    print("    MC disagrees with HRT and MHNC by far more there than anywhere else,")
    print("    and it sits near the liquid-vapour boundary (CEP at T* = 0.965).")
    spread = max(r['mu_species_spread'] for r in rows)
    print(f"\n  symmetry check: max |mu_1 - mu_2| over all states = {spread:.2e}")
    print("    (must vanish -- the two species are related by exchange symmetry)")


if __name__ == '__main__':
    gate_alpha_one()
    gate_energy_routes()
    res = run()
    report(res)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'mkanya_table1.json'), 'w') as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {OUT}/mkanya_table1.json")
