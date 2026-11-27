#!/usr/bin/env python3
"""OZ-MSA, OZ-HNCB and FMSA-DP against Yu & Jin's EXACT-model-class MC data.

Every other benchmark in this repo is full Lennard-Jones while the models solve
hard spheres plus two Yukawa tails, so every score carries a shared mapping
error that could not be measured.  Yu & Jin, J. Chem. Phys. 128, 014901 (2008),
simulated the **hard-core two-Yukawa mixture itself** -- our model class exactly
-- and tabulated g_ij(sigma_ij), U/NkT and Z at 145 state points.  Scoring
against it isolates the CLOSURE error with nothing else in the way.

The potential convention
------------------------
The caption strings were transcribed verbatim ('lam1=1.8, lam2,1=4.0',
'k1=k2=-6.0') and cannot be decoded by reading.  It was pinned numerically
instead (numerical_notes/results/benchmark_rdf/literature/hcty_exact_benchmark.md
sec. 1): every alternative reading is 40-800 % off the table, this one is 2-13 %.

    u_ij(r) = - eps_ij (sigma_ij/r) [ K1 exp(-lam1 (r/sigma_ij - 1))
                                    + K2 exp(-lam2 (r/sigma_ij - 1)) ],  r > sigma_ij
    u_ij(r) = infinity,                                                  r <= sigma_ij

with **K1 = 1** (the caption omits it -- in case 1 K2 is also 1, which is how
'k1=k2=1.0' arose) and (lam1, lam2, K2) as printed:

    case 1  (1.8, 4.0, +1.0)        case 2  (1.8, 8.0,  -6.0)
    case 3  (1.8, 8.0, -12.0)       case 4  (2.8647, 13.5485, -1.4466)

Case 4 is the classical two-Yukawa representation of Lennard-Jones, which is an
independent check on the reading.  The sign convention (K > 0 = attraction) and
the per-pair scaling of lambda are fixed by the same group's single-Yukawa paper,
You, Yu & Gao, JCP 123, 114705 (2005), which prints the potential explicitly.

Unit and species mapping (they label the LARGE species 1, we label it 2)
------------------------------------------------------------------------
    sigma_ours = [1, 1/ratio],  eps_ours = [1, 1/eps_ratio],  x1_ours = 1 - x1
    rho*_ours  = rho* * ratio^3,  T*_ours = T* / eps_ratio
    their (1,1) is our (2,2); their (2,2) is our (1,1); (1,2) is unchanged.

    python compare_models_hcty.py --workers 8     # -> hcty_metrics.csv
    python compare_models_hcty.py --report
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

TABLE = REPO / 'data_benchmark' / 'literature_direct' / 'rdf_contact' / 'yu_jin_2008.csv'
OUT_DIR = REPO / 'numerical_notes' / 'results' / 'benchmark_rdf' / 'literature'
METRICS = OUT_DIR / 'hcty_metrics.csv'

# (lam1, lam2, K2); K1 = 1 always.  See the module docstring.
CASE = {1: (1.8, 4.0, 1.0), 2: (1.8, 8.0, -6.0), 3: (1.8, 8.0, -12.0),
        4: (2.8647, 13.5485, -1.4466)}
CASE_OF_TABLE = {'I': 1, 'II': 2, 'III': 3, 'IV': 1, 'V': 2, 'VI': 1, 'VII': 2, 'VIII': 4}
MODELS = ['OZ+MSA', 'OZ+HNCB', 'FMSA-DP', 'MSA-exact']
N_R = 4096

FIELDS = ['table', 'case', 'row', 'sigma_ratio', 'eps_ratio', 'x1', 'T_star', 'rho_star',
          'eta', 'model', 'converged', 'g11', 'g11_ref', 'g22', 'g22_ref',
          'g12', 'g12_ref', 'U', 'U_ref', 'Z', 'Z_ref']


def rows() -> list[dict]:
    return [r for r in csv.DictReader(l for l in TABLE.open() if not l.startswith('#'))]


def state_of(row: dict) -> dict:
    ratio = float(row['sigma2_over_sigma1'])
    epsr = float(row['eps2_over_eps1'])
    x1 = float(row['x1'])
    lam1, lam2, K2 = CASE[CASE_OF_TABLE[row['table']]]
    sig = np.array([1.0, 1.0 / ratio])
    x = np.array([1.0 - x1, x1])
    rho = float(row['rho_star']) * ratio ** 3
    return dict(sig=sig, eps=np.array([1.0, 1.0 / epsr]), x=x, rho=rho,
                T=float(row['T_star']) / epsr, z=[lam1, lam2], k=[1.0, K2],
                eta=float((np.pi / 6.0) * rho * np.sum(x * sig ** 3)))


def beta_u(r, sij, eij, T, z, k):
    out = np.zeros_like(r)
    for lam, K in zip(z, k):
        out += -K * (eij / T) * (sij / r) * np.exp(-lam * (r / sij - 1.0))
    return out


def beta_dudr(r, sij, eij, T, z, k):
    """d(beta u)/dr for the same form."""
    out = np.zeros_like(r)
    for lam, K in zip(z, k):
        e = np.exp(-lam * (r / sij - 1.0))
        out += K * (eij / T) * sij * e * (1.0 / r ** 2 + lam / (sij * r))
    return out


def thermo(st, r, g, contacts):
    """U/NkT and Z by the energy and virial routes, in THEIR reduced units.

    U/NkT   = (rho/2) sum_ij x_i x_j Int 4 pi r^2 beta u_ij g_ij dr
    Z       = 1 + (2 pi rho/3) sum_ij x_i x_j [ sigma_ij^3 g_ij(sigma+)
                                                - Int r^3 (beta u_ij)' g_ij dr ]
    Both are dimensionless and invariant under the length/energy rescaling, so
    no unit conversion back to their frame is needed.
    """
    sig, x, rho, T = st['sig'], st['x'], st['rho'], st['T']
    U = 0.0
    vir = 0.0
    for i in range(2):
        for j in range(2):
            sij = 0.5 * (sig[i] + sig[j])
            eij = float(np.sqrt(st['eps'][i] * st['eps'][j]))
            m = r > sij
            rr, gg = r[m], g[i, j][m]
            U += x[i] * x[j] * _trapz(4 * np.pi * rr ** 2
                                      * beta_u(rr, sij, eij, T, st['z'], st['k']) * gg, rr)
            vir += x[i] * x[j] * (sij ** 3 * contacts[(min(i, j), max(i, j))]
                                  - _trapz(rr ** 3 * beta_dudr(rr, sij, eij, T,
                                                               st['z'], st['k']) * gg, rr))
    return 0.5 * rho * U, 1.0 + (2.0 * np.pi * rho / 3.0) * vir


def exact_msa_mix(st: dict):
    """The HCTY state as a Blum-Hoye multi-Yukawa mixture, solved EXACTLY.

    Two things have to line up with `msa_exact_mix.HCYMix`, whose tails carry one
    rate `z_nu` SHARED by every pair:

    1. THE RATE IS PAIR-SPECIFIC HERE.  Yu & Jin write the tail as
       `exp(-lam (r/sigma_ij - 1))`, i.e. a decay `z_ij = lam/sigma_ij` in r,
       which differs across pairs whenever the diameters do.  That is Cummings'
       generalisation, not Blum-Hoye's covered case -- but it needs no new
       theory: a pair-specific rate is a tail whose amplitude matrix is
       SUPPORTED ON ONE PAIR, exactly as `msa_exact_mix.pair_specific_z` does
       for the single-tail case.  Two tails x three pairs = omega = 6 tails,
       2*N^2*omega = 48 unknowns.  (That construction is gated against
       FMSA_double_prop's native pair-specific rates to 1e-10 in that module.)

    2. THE AMPLITUDE IS IN THE CONTACT NORMALISATION.  MSA sets c = -beta u, and
       with `u_ij = -eps_ij K_nu (sigma_ij/r) exp(-z_ij (r-sigma_ij))` their
       Eq. (3) form `c_ij = K^nu_ij e^{-z(r-sigma_ij)}/r` gives
       `K^nu_ij = sigma_ij eps_ij K_nu / T*`.

    ⚠ `pair_specific_z` requires DISTINCT diameters -- equal sigma collapses the
    rates and the tails become linearly dependent.  Guarded below.
    """
    import msa_exact_mix as X
    sig, x, rho_t, T = st['sig'], st['x'], st['rho'], st['T']
    n = len(sig)
    SIG = 0.5 * (sig[:, None] + sig[None, :])
    EPS = np.sqrt(st['eps'][:, None] * st['eps'][None, :])
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    rates = {SIG[i, j] for i, j in pairs}
    if len(rates) < len(pairs):
        raise ValueError('equal diameters collapse the pair-specific rates')
    zs, Ks = [], []
    for lam, Kamp in zip(st['z'], st['k']):
        for (i, j) in pairs:
            K = np.zeros((n, n))
            K[i, j] = K[j, i] = SIG[i, j] * EPS[i, j] * Kamp / T
            zs.append(lam / SIG[i, j])
            Ks.append(K)
    return X.solve_bh_mix(X.HCYMix(sig, x * rho_t, np.array(zs), np.array(Ks)))


def contact_from_curve(r, g_ij, R):
    m = (r > R) & (r < 1.25 * R)
    rr, gg = r[m][:6], g_ij[m][:6]
    if rr.size < 2:
        return float('nan')
    return float(np.polyval(np.polyfit(rr, gg, 1), R))


def score_row(row: dict) -> list[dict]:
    from oz_yukawa_mixture import OZYukawaMixture
    from fmsa_double_prop import FMSA_double_prop
    from oz_solver import OZSolver
    from compare_models_dcf_multistate import _c_is_physical

    st = state_of(row)
    meta = dict(table=row['table'], case=CASE_OF_TABLE[row['table']],
                row=f"{row['table']}_{row['T_star']}_{row['rho_star']}_{row['x1']}"
                    f"_{row['sigma2_over_sigma1']}_{row['eps2_over_eps1']}",
                sigma_ratio=1.0 / float(row['sigma2_over_sigma1']),
                eps_ratio=float(row['eps2_over_eps1']), x1=float(row['x1']),
                T_star=float(row['T_star']), rho_star=float(row['rho_star']),
                eta=round(st['eta'], 4))
    ref = dict(g11_ref=float(row['g11']), g22_ref=float(row['g22']),
               g12_ref=float(row['g12']), U_ref=float(row['U_NkT']),
               Z_ref=float(row['Z']))

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        ga = OZYukawaMixture(st['sig'], st['eps'], st['x'], st['rho'], st['T'],
                                n_tails=2, z=st['z'], k=st['k'], input_type='bh',
                                verbose=False)
        dp = FMSA_double_prop(st['sig'], st['eps'], st['x'], st['rho'], st['T'],
                              n_tails=2, z=st['z'], k=st['k'], input_type='bh',
                              verbose=False)
    R = np.asarray(ga.R_ij, float)

    out = []
    for name in MODELS:
        try:
            if name == 'MSA-exact':
                import msa_exact_mix as X
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                    warnings.simplefilter('ignore')
                    sol = exact_msa_mix(st)
                sd = sol.symmetry_defect
                sd = sd() if callable(sd) else sd
                if not np.isfinite(float(sd)) or abs(float(sd)) > 1e-6:
                    out.append(dict(meta, **ref, model=name, converged=0))
                    continue
                c = np.asarray(sol.contact, dtype=float)
                # U is the ENERGY route in closed form (their Eq. 42), no quadrature.
                # Z is NOT recorded: the exact excess pressure is an excess over the
                # PY hard-sphere reference, so quoting Z would require choosing a
                # hard-sphere EOS -- reintroducing exactly the reference ambiguity
                # the exact solution exists to avoid.  Left blank on purpose.
                out.append(dict(meta, **ref, model=name, converged=1,
                                g11=c[1, 1], g22=c[0, 0], g12=c[0, 1],
                                U=float(X.excess_energy(sol)) / st['rho'],
                                Z=float('nan')))
                continue
            if name == 'FMSA-DP':
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                    warnings.simplefilter('ignore')
                    r, g = dp.get_g_fourier(k_max=1500.0, n_k=200000, return_r=True)
                # the contact value is algebraic for FMSA-DP -- no inversion at all
                contacts = {(i, j): dp.get_g_contact(i, j) for (i, j) in
                            ((0, 0), (0, 1), (1, 1))}
            else:
                closure = 'msa' if name == 'OZ+MSA' else 'hncb'
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                    warnings.simplefilter('ignore')
                    s = OZSolver(ga, n_r=N_R)
                    try:
                        s.solve(method='diis', closure=closure, tol=1e-6, max_iter=400,
                                alpha=0.2, n_coupling_steps=5)
                    except Exception:
                        pass
                h = getattr(s, '_h_converged', None)
                if h is None or not _c_is_physical(getattr(s, '_c_converged', None)) \
                        or not np.all(np.isfinite(h)):
                    out.append(dict(meta, **ref, model=name, converged=0))
                    continue
                r, g = s.r, 1.0 + h
                contacts = {(i, j): contact_from_curve(r, g[i, j], float(R[i, j]))
                            for (i, j) in ((0, 0), (0, 1), (1, 1))}
            U, Z = thermo(st, np.asarray(r), np.asarray(g), contacts)
            # their (1,1) is our (2,2)
            out.append(dict(meta, **ref, model=name, converged=1,
                            g11=contacts[(1, 1)], g22=contacts[(0, 0)],
                            g12=contacts[(0, 1)], U=U, Z=Z))
        except Exception:
            out.append(dict(meta, **ref, model=name, converged=0))
    return out


def _worker(row):
    try:
        return row['table'], score_row(row), None
    except Exception:
        return row['table'], [], traceback.format_exc(limit=3)


def report() -> int:
    import pandas as pd
    pd.set_option('display.width', 200)
    d = pd.read_csv(METRICS)
    ok = d[d.converged == 1].copy()
    for q in ('g11', 'g22', 'g12', 'U', 'Z'):
        ok[f'e_{q}'] = (ok[q] - ok[f'{q}_ref']).abs() / ok[f'{q}_ref'].abs() * 100
        ok[f's_{q}'] = (ok[q] - ok[f'{q}_ref']) / ok[f'{q}_ref'].abs() * 100
    ok['e_gmean'] = ok[['e_g11', 'e_g22', 'e_g12']].mean(axis=1)

    def blk(t):
        print('\n' + '=' * 78 + f'\n  {t}\n' + '=' * 78)

    blk('0.  POOL — Yu & Jin (2008), hard-core two-Yukawa MIXTURE, MC')
    print(f'  {len(d.row.unique())} state points, {len(d)} model rows')
    print('  by case:', d.groupby('case').row.nunique().to_dict())
    nc = d[d.converged == 0]
    print(f'  non-converged: {len(nc)} rows',
          nc.groupby(['model', 'case']).size().to_dict() if len(nc) else '')

    blk('1.  CONTACT RDF  g_ij(sigma_ij) — median relative error %\n'
        '      NO model-class error here: the models solve exactly this Hamiltonian.')
    t = ok.pivot_table(index='model', values=['e_g11', 'e_g22', 'e_g12', 'e_gmean'],
                       aggfunc='median')
    print(t[['e_g11', 'e_g22', 'e_g12', 'e_gmean']].round(1).to_string())
    print('\n  signed (is the model high or low?):')
    print(ok.pivot_table(index='model', values=['s_g11', 's_g22', 's_g12'],
                         aggfunc='median').round(1).to_string())

    blk('2.  BY POTENTIAL CASE — the closure meets four different tails')
    for q, lbl in (('e_gmean', 'contact RDF, mean over pairs'),
                   ('e_U', 'internal energy U/NkT'), ('e_Z', 'compressibility factor Z')):
        print(f'\n  -- {lbl}, median relative error %')
        print(ok.pivot_table(index='model', columns='case', values=q,
                             aggfunc='median').round(1).to_string())
    print('\n  case 1 = (1.8, 4.0, +1.0) both tails attractive')
    print('  case 2 = (1.8, 8.0, -6.0)   attractive long + repulsive short')
    print('  case 3 = (1.8, 8.0, -12.0)  same, twice the repulsion')
    print('  case 4 = (2.8647, 13.5485, -1.4466) = the two-Yukawa Lennard-Jones')

    blk('3.  BY DENSITY AND TEMPERATURE (contact RDF, mean over pairs)')
    print(ok.pivot_table(index='model', columns='rho_star', values='e_gmean',
                         aggfunc='median').round(1).to_string())
    print('\n  by T* (their reduced units):')
    print(ok.pivot_table(index='model', columns='T_star', values='e_gmean',
                         aggfunc='median').round(1).to_string())

    blk('4.  BY SIZE RATIO AND COMPOSITION')
    print(ok.pivot_table(index='model', columns='sigma_ratio', values='e_gmean',
                         aggfunc='median').round(1).to_string())
    print('\n  by x1 (THEIR x1 = mole fraction of the LARGE species):')
    print(ok.pivot_table(index='model', columns='x1', values='e_gmean',
                         aggfunc='median').round(1).to_string())

    blk('5.  THERMODYNAMICS')
    print('  U/NkT and Z, median relative error % (all cases)')
    print(ok.pivot_table(index='model', values=['e_U', 'e_Z'], aggfunc='median')
          .round(1).to_string())
    print('\n  NOTE  U/NkT passes through zero inside cases 2 and 3, so a relative')
    print('  error there is meaningless; median |U_model - U_ref| in absolute units:')
    ok['a_U'] = (ok.U - ok.U_ref).abs()
    print(ok.pivot_table(index='model', columns='case', values='a_U',
                         aggfunc='median').round(3).to_string())

    blk('6.  HOW OFTEN IS EACH MODEL CLOSEST (contact RDF)')
    per = ok.groupby(['row', 'model'], as_index=False).e_gmean.mean()
    win = per.loc[per.groupby('row').e_gmean.idxmin()]
    print('   ', win.model.value_counts().to_dict())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()
    if args.report:
        return report()

    data = rows()[:args.limit] if args.limit else rows()
    print(f'scoring {len(data)} Yu & Jin state points with {args.workers} workers')
    out, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, r): n for n, r in enumerate(data)}
        for n, fut in enumerate(as_completed(futs), 1):
            tbl, res, err = fut.result()
            if err:
                failed.append(err)
            else:
                out.extend(res)
            if n % 20 == 0:
                print(f'  [{n}/{len(data)}]')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(out)
    print(f'wrote {METRICS} ({len(out)} rows)')
    if failed:
        print(f'{len(failed)} rows failed:\n{failed[0]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
