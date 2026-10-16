#!/usr/bin/env python3
"""OZ-MSA, OZ-HNCB and FMSA-DP against the whole benchmark DCF pool.

Every earlier DCF comparison in this repo ran at ONE state point (sigma=[1,2],
eps=[1,0.5], x=[0.25,0.75], rho*=0.139, T*=1.0, eta=0.455 -- the MC_DATA table).
The inner-core closure ranking, the outer ranking, the contact-peak ranking and
the FMSA-DP vs OZ+MSA truncation check are all settled there and are not redone.
What is new here is the sweep: `data_benchmark/dcf_pool/pool_index.csv` carries
~100 state points spanning eta, T*, eps22/eps11, composition and -- through our
own MD -- size ratio, and this scores the three models across all of them.

Two paths, both resolved by pair and by region
----------------------------------------------
pointwise      c_ij(r) point by point:  RMSE, MAE, max|dc|, and NRMSE = RMSE
               divided by the rms of the benchmark inside the same region, which
               is what makes state points with wildly different c scales
               comparable.  Plus the contact peak, the observable that
               discriminated the closures at the reference state.
compressibility  the integrated content, chat_ij(0) = 4 pi Int r^2 c dr, split at
               R_ij into inner and outer, and the aggregate
               chi_T/chi_id = [1 - rho sum_ij x_i x_j chat_ij(0)]^-1 and
               det(I - X^ C^(0)).  Note this is chat(0) RESTRICTED TO THE
               BENCHMARK'S OWN r RANGE, not the true k -> 0 limit; models are
               truncated to the same range so the comparison is like-for-like.

Two reference variants, side by side
------------------------------------
`published`   the benchmark exactly as it stands.
`c22_mfphi`   c22 beyond sigma22 replaced by the rigorous -beta*phi limit.
numerical_notes/rootcause/c22_outer_inversion_artifact.md showed the TN 1331
outer c22 is an artifact of that report's own OZ inversion and that correcting
c22 ALONE moves chi_T/chi_id from 0.10 (an outlier) to 0.058 (where all four
closures sit).  Reporting both is the honest option: on the low-eta tables the
two variants coincide, which is itself the check that the correction is not
doing violence to good data.

Read before trusting a number in the outer column
-------------------------------------------------
OZ+MSA fixes c(r > R) = -beta*u.  FMSA-DP's tail is the same -beta*u by
construction.  Their outer pointwise errors are therefore IDENTICAL, and equal
to the error of the mean-field tail itself; only OZ+HNCB has an independent
outer region.  This is a property of the methods, not a bug, and the outer
tables should be read as a two-way comparison.

One scoring run, three write-ups
-------------------------------
The metric tables are written once, over the whole pool, to
numerical_notes/results/benchmark/all_samples/.  `--report SCOPE` then slices
them for each write-up under numerical_notes/results/benchmark/:

    literature -> literature/literature_dcf_benchmark.md   (TN 1331, Yu & Jin)
    md         -> LAMMPS_MD/md_dcf_benchmark.md            (our own LAMMPS MD)
    all        -> all_samples/dcf_multistate_comparison.md (pooled, plus the
                  cross-source agreement check that only exists there)

Usage
-----
    python compare_models_dcf_multistate.py                  # full pool -> CSVs
    python compare_models_dcf_multistate.py --limit 6        # a quick taste
    python compare_models_dcf_multistate.py --label tn1331dcf_T1_eps0.50_x0.2500
    python compare_models_dcf_multistate.py --validate       # reference-state regression
    python compare_models_dcf_multistate.py --report md      # one scope's tables
    python compare_models_dcf_multistate.py --workers 6      # be kind to a running MD

CAUTION: a run without --report REWRITES the metric CSVs with only the state
points it scored, so `--limit`/`--label` runs clobber the full tables.  Re-run
the full pool afterwards.
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

try:                      # numpy >= 2.0 renamed trapz -> trapezoid
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'data_benchmark'))

from convert import read_table                                   # noqa: E402
from c22_outer_diagnosis import eps_of, phi                      # noqa: E402

POOL_INDEX = REPO / 'data_benchmark' / 'dcf_pool' / 'pool_index.csv'
SUSPECT = REPO / 'data_benchmark' / 'tn1331' / 'suspect.csv'
BENCH_DIR = REPO / 'numerical_notes' / 'results' / 'benchmark'
OUT_DIR = BENCH_DIR / 'all_samples'
METRICS_CSV = OUT_DIR / 'dcf_multistate_metrics.csv'
SCALARS_CSV = OUT_DIR / 'dcf_multistate_scalars.csv'

PAIRS = [(0, 0), (0, 1), (1, 1)]
PAIR_TAG = {(0, 0): '11', (0, 1): '12', (1, 1): '22'}
LABELS = {(0, 0): '(1,1)', (0, 1): '(1,2)', (1, 1): '(2,2)'}
MODELS = ['OZ+MSA', 'OZ+HNCB', 'FMSA-DP(fmt)', 'FMSA-DP(py)']
REFS = ['published', 'c22_mfphi', 'lowk']
N_R = 4096

METRIC_FIELDS = [
    'label', 'source', 'potential', 'sigma_ratio', 'eps_ratio', 'T_star', 'x1',
    'rho_star', 'eta', 'fidelity', 'outer_trust', 'ref', 'model', 'pair',
    'region', 'n_pts', 'rmse', 'mae', 'maxe', 'nrmse', 'rms_bench',
    'chat0_model', 'chat0_bench', 'contact_model', 'contact_bench',
]
SCALAR_FIELDS = [
    'label', 'source', 'sigma_ratio', 'eps_ratio', 'T_star', 'x1', 'rho_star',
    'eta', 'fidelity', 'outer_trust', 'ref', 'model', 'converged', 'iters',
    'chat0_11', 'chat0_12', 'chat0_22', 'chi_ratio', 'det_spinodal',
]

def sig_mix_of(ratio: float) -> np.ndarray:
    s = np.array([1.0, float(ratio)])
    return 0.5 * (s[:, None] + s[None, :])


# --------------------------------------------------------------------------
# benchmark side
# --------------------------------------------------------------------------
def load_suspects() -> dict[tuple[str, str], set]:
    """(file, column) -> the set of R* values flagged as having lost a digit."""
    out: dict[tuple[str, str], set] = {}
    if not SUSPECT.exists():
        return out
    for row in csv.DictReader(SUSPECT.open()):
        out.setdefault((row['file'], row['column']), set()).add(round(float(row['R_star']), 4))
    return out


def lowk_path(entry: dict) -> Path | None:
    """Where dcf_lowk.py put this state's repaired c, if it repaired it."""
    c = Path(entry['c_path'])
    if entry['source'] == 'tn1331_dcf':
        p = REPO / 'data_benchmark' / 'tn1331' / 'dcf_lowk' / c.name
    else:
        p = REPO / 'data_benchmark' / c.parent / 'c_lowk.csv'
    return p if p.exists() else None


def load_benchmark(entry: dict, suspects: dict) -> dict[tuple[int, int], dict]:
    """Per pair: absolute r, the reference variants of c, and how many
    points were dropped as OCR-suspect.

    `published` is the table as it stands.  `c22_mfphi` overwrites the (2,2)
    pair beyond sigma22 with -beta*phi, following extended_rc() in
    c22_outer_diagnosis.py -- a one-pair patch on the symptom.  `lowk` is
    data_benchmark/dcf_lowk.py's repair of the same defect at its source: all
    three pairs, chat_ij(0) taken from the equation of state instead of from the
    integral of a contaminated tail.  It is resampled onto THIS entry's own r
    so that all three variants are scored at identical points, and continued
    past the repaired file's range by the exact asymptote the repair imposes
    there anyway.  Suspect points are dropped only where they can be
    localised -- i.e. for a directly tabulated DCF.  For a c obtained by
    inverting a corrupted g the damage is spread over every r and dropping
    points would only hide it, so those tables keep every point and carry their
    n_suspect flag instead.
    """
    tab = read_table(REPO / 'data_benchmark' / entry['c_path'])
    ratio = float(entry['sigma_ratio'])
    sm = sig_mix_of(ratio)
    per_pair = entry['r_convention'] == 'perpair'
    rkey = 'R_star' if per_pair else 'r'
    T = float(entry['T_star'])
    E = eps_of(float(entry['eps_ratio']))
    drop_from = entry['suspect_key'] if entry['fidelity'] == 'direct' else ''
    lp = lowk_path(entry)
    low = read_table(lp) if lp is not None else None

    out = {}
    for (i, j) in PAIRS:
        tag = PAIR_TAG[(i, j)]
        r = tab[rkey] * (sm[i, j] if per_pair else 1.0)
        c = tab[f'c{tag}'].copy()
        keep = np.isfinite(c) & (r > 0) & (r <= float(entry[f'r_max_{tag}']) + 1e-9)
        n_drop = 0
        if drop_from:
            flagged = suspects.get((drop_from, f'c{tag}'), set())
            if flagged:
                rstar = np.round(tab[rkey], 4)
                bad = np.isin(rstar, list(flagged))
                n_drop = int((bad & keep).sum())
                keep &= ~bad
        r, c = r[keep], c[keep]
        variants = {'published': c}
        if entry['potential'] in ('lj', 'ljhs'):
            c2 = c.copy()
            if (i, j) == (1, 1):
                m = r > sm[1, 1]
                c2[m] = -phi(r[m], sm[1, 1], E['22']) / T
            variants['c22_mfphi'] = c2
        if low is not None and r.size:
            rl, cl = low['r'], low[f'c{tag}']
            v = np.interp(r, rl, cl)
            beyond = r > rl[-1]
            if beyond.any():
                v[beyond] = -phi(r[beyond], sm[i, j], E[tag]) / T
            variants['lowk'] = v
        out[(i, j)] = {'r': r, 'c': variants, 'n_drop': n_drop}
    return out


# --------------------------------------------------------------------------
# model side
# --------------------------------------------------------------------------
C_MAX = 1.0e6          # |c| this large is a runaway branch, not a fluid


def _c_is_physical(c) -> bool:
    """A converged c must at least be finite and bounded.

    The DIIS residual test is a statement about the ITERATION, not about the
    solution: on the small-species-rich dense states it will happily report
    convergence with max|c| = 4e37.  The physical inner-core c reaches a few
    hundred at eta ~ 0.5, so 1e6 rejects only genuine runaways.
    """
    if c is None:
        return False
    return bool(np.all(np.isfinite(c)) and np.max(np.abs(c)) < C_MAX)


def solve_oz(ga, closure: str, sigs, eps, x, rho, T, n_tails, fast_init: bool = False):
    """Converge OZ; (r, c, iters) or (None, None, None).

    The default path is the **coupling ramp**: solve at the hard-sphere limit
    and continue the Yukawa coupling up to 1 in five steps.  That is not merely
    a robustness trick — it is what SELECTS THE SOLUTION.  MSA at high packing
    with a small-species-rich composition has more than one root, and the ramp
    returns the one continuously connected to the hard-sphere fluid, which is
    the physical branch by construction.

    `fast_init=True` starts DIIS from a poly-model initial guess instead.  It is
    4x cheaper (20 iterations against 80) and it is OFF BY DEFAULT, because it
    does not select a branch:

      * at 26 of 138 state points it converges onto a runaway with
        max|c| ~ 1e37 while the DIIS residual test reports success — the
        residual is a statement about the iteration, not the solution, which is
        what `_c_is_physical` is for;
      * at 3 further MD points (eta >= 0.4, x1 = 0.7-0.9) it lands on a second,
        finite root that the ramp never reaches — e.g. md_sweep_r3.0_eta0.4_x0.7
        gives chat22(0) = -26.1 against the ramp's -661.6.

    Both failures are silent without the gate.  Every published number in
    numerical_notes/results/benchmark/ is from the ramp.
    """
    from oz_numerical import OZSolverMSA
    from sc_checks import FMSA_poly_adapter
    from FMSA_MC_cleaned_2cpp import FMSA_poly_term_species

    s = OZSolverMSA(ga, n_r=N_R)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        if fast_init:
            try:
                poly = FMSA_poly_term_species(sigs, eps, x, rho, T, n_tails=n_tails)
                init_c = s.init_c_from_model(FMSA_poly_adapter(poly))
                s.solve(method='diis', closure=closure, tol=1e-6, max_iter=500,
                        alpha=0.2, init_c=init_c.copy())
            except Exception:
                pass
        if not _c_is_physical(getattr(s, '_c_converged', None)):
            s._c_converged = None
            try:
                s.solve(method='diis', closure=closure, tol=1e-6, max_iter=500,
                        alpha=0.2, n_coupling_steps=5)
            except Exception:
                pass
    if not _c_is_physical(getattr(s, '_c_converged', None)):
        return None, None, None
    return s.r, s._c_converged, s._n_iters


def build_models(entry):
    """-> (R_ij, {model: (r_model, c_model)}, {model: (converged, iters)})."""
    from fmsa_ga_matrix_mix import FMSA_GA_matrix_mix
    from fmsa_double_prop import FMSA_double_prop

    from ljhs_mapping import model_inputs

    ratio = float(entry['sigma_ratio'])
    sigs, eps, mkw = model_inputs(entry.get('potential', 'lj'), ratio,
                                  entry['eps_ratio'], float(entry['T_star']))
    x1 = float(entry['x1'])
    x = np.array([x1, 1.0 - x1])
    rho = float(entry['rho_star'])
    T = float(entry['T_star'])
    n_tails = 2

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        ga = FMSA_GA_matrix_mix(sigs, eps, x, rho, T, n_tails=n_tails,
                                verbose=False, **mkw)
        dp = FMSA_double_prop(sigs, eps, x, rho, T, n_tails=n_tails,
                              verbose=False, **mkw)

    R_ij = np.asarray(ga.R_ij, float)
    r_fine = np.linspace(1e-3, 15.0 * float(R_ij.max()), 12000)

    curves, status = {}, {}
    for name, closure in (('OZ+MSA', 'msa'), ('OZ+HNCB', 'hncb')):
        r, c, nit = solve_oz(ga, closure, sigs, eps, x, rho, T, n_tails)
        curves[name] = (r, c)
        status[name] = (r is not None, nit)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        for name, ref in (('FMSA-DP(fmt)', 'fmt'), ('FMSA-DP(py)', 'py')):
            try:
                curves[name] = (r_fine, dp.get_c_total(r_fine, hs_ref=ref))
                status[name] = (True, 0)
            except Exception:
                curves[name] = (None, None)
                status[name] = (False, None)
    return R_ij, curves, status


# --------------------------------------------------------------------------
# the two paths
# --------------------------------------------------------------------------
def contact_peak(r, c, R):
    """max of c over [R, 1.15 R] -- the observable that separated the closures."""
    m = (r >= R) & (r <= 1.15 * R)
    return float(np.max(c[m])) if m.any() else float('nan')


def interp_within(r_target, r_src, c_src, lo, hi):
    """Interpolate a model curve using only source points inside (lo, hi).

    c_ij(r) jumps at R_ij -- the core and the tail are different functional
    forms.  Interpolating across that jump smears one region's values into the
    other and, at a benchmark point sitting within one grid spacing of R,
    produces a difference of order the jump itself.  Restricting the source to
    the region being scored removes it.

    Cubic, not linear: just outside contact the short-range Yukawa (z2 ~ 14/sigma)
    turns over within a few of the OZ solver's grid spacings, and linear
    interpolation there leaves an error of order 1% of the benchmark rms -- big
    enough to blur the fact that the MSA and FMSA-DP tails are the same function.
    """
    sel = (r_src > lo) & (r_src < hi)
    if sel.sum() < 2:
        return np.full(np.shape(r_target), np.nan)
    rs, cs = r_src[sel], c_src[sel]
    if sel.sum() < 4:
        return np.interp(r_target, rs, cs)
    # A benchmark point can fall between R and the model's first grid point past
    # it -- within one spacing, where -beta*u is steepest.  Clamping there is
    # what a plain interpolation does, and it costs ~1% of the benchmark rms.
    # Extrapolate the cubic across that one spacing instead; refuse anything
    # further out.
    from scipy.interpolate import CubicSpline
    margin = float(np.max(np.diff(rs))) if rs.size > 1 else 0.0
    out = CubicSpline(rs, cs, extrapolate=True)(r_target)
    too_far = (r_target < rs[0] - margin) | (r_target > rs[-1] + margin)
    return np.where(too_far, np.nan, out)


def score_state(entry: dict, suspects: dict) -> tuple[list, list]:
    bench = load_benchmark(entry, suspects)
    R_ij, curves, status = build_models(entry)
    meta = {k: entry[k] for k in ('label', 'source', 'potential', 'sigma_ratio',
                                  'eps_ratio', 'T_star', 'x1', 'rho_star', 'eta',
                                  'fidelity', 'outer_trust')}
    x1 = float(entry['x1'])
    x = np.array([x1, 1.0 - x1])
    rho_tot = float(entry['rho_star'])
    rho_i = x * rho_tot

    rows, scalars = [], []
    refs = [r for r in REFS if r in bench[(0, 0)]['c']]
    for ref in refs:
        chat_tot = {m: np.zeros((2, 2)) for m in MODELS}
        chat_tot['benchmark'] = np.zeros((2, 2))
        for name in MODELS + ['benchmark']:
            if name != 'benchmark' and curves[name][0] is None:
                continue
            for (i, j) in PAIRS:
                tag, R = PAIR_TAG[(i, j)], float(R_ij[i, j])
                rb = bench[(i, j)]['r']
                cb = bench[(i, j)]['c'][ref]
                r_hi = float(rb[-1]) if rb.size else 0.0
                if name == 'benchmark':
                    rm, cm = rb, cb
                else:
                    rm_all, cm_all = curves[name]
                    cm_all = cm_all[i, j]
                    keep = (rm_all > 0) & (rm_all <= r_hi)
                    rm, cm = rm_all[keep], cm_all[keep]
                for region, msk_b, msk_m in (
                        ('inner', rb < R, rm < R),
                        ('outer', (rb >= R) & (rb <= r_hi), (rm >= R) & (rm <= r_hi))):
                    if name != 'benchmark':
                        lo, hi = (0.0, R) if region == 'inner' else (R, np.inf)
                        cm_on_bench = interp_within(rb, rm_all, cm_all, lo, hi)
                    ib = float(4 * np.pi * _trapz(rb[msk_b] ** 2 * cb[msk_b], rb[msk_b])) \
                        if msk_b.sum() > 1 else float('nan')
                    im = float(4 * np.pi * _trapz(rm[msk_m] ** 2 * cm[msk_m], rm[msk_m])) \
                        if msk_m.sum() > 1 else float('nan')
                    if name == 'benchmark':
                        chat_tot['benchmark'][i, j] += 0.0 if np.isnan(ib) else ib
                        chat_tot['benchmark'][j, i] = chat_tot['benchmark'][i, j]
                        continue
                    chat_tot[name][i, j] += 0.0 if np.isnan(im) else im
                    chat_tot[name][j, i] = chat_tot[name][i, j]
                    # a benchmark point the model cannot resolve (further than one
                    # model spacing outside its region) is scored by nobody
                    ok = msk_b & np.isfinite(cm_on_bench)
                    d = cm_on_bench[ok] - cb[ok]
                    n = int(ok.sum())
                    rms_b = float(np.sqrt(np.mean(cb[ok] ** 2))) if n else float('nan')
                    rmse = float(np.sqrt(np.mean(d ** 2))) if n else float('nan')
                    rows.append(dict(meta, ref=ref, model=name, pair=LABELS[(i, j)],
                                     region=region, n_pts=n,
                                     rmse=rmse,
                                     mae=float(np.mean(np.abs(d))) if n else float('nan'),
                                     maxe=float(np.max(np.abs(d))) if n else float('nan'),
                                     nrmse=(rmse / rms_b) if n and rms_b > 0 else float('nan'),
                                     rms_bench=rms_b,
                                     chat0_model=im, chat0_bench=ib,
                                     contact_model=contact_peak(rm, cm, R) if region == 'outer' else '',
                                     contact_bench=contact_peak(rb, cb, R) if region == 'outer' else ''))

        for name in ['benchmark'] + MODELS:
            if name != 'benchmark' and curves[name][0] is None:
                scalars.append(dict({k: meta[k] for k in SCALAR_FIELDS if k in meta},
                                    ref=ref, model=name, converged=0,
                                    iters='', chat0_11='', chat0_12='', chat0_22='',
                                    chi_ratio='', det_spinodal=''))
                continue
            C0 = chat_tot[name]
            denom = 1.0 - rho_tot * float(x @ C0 @ x)
            X = np.sqrt(np.outer(rho_i, rho_i))
            scalars.append(dict({k: meta[k] for k in SCALAR_FIELDS if k in meta},
                                ref=ref, model=name,
                                converged=1 if name == 'benchmark' else int(status[name][0]),
                                iters='' if name == 'benchmark' else status[name][1],
                                chat0_11=C0[0, 0], chat0_12=C0[0, 1], chat0_22=C0[1, 1],
                                chi_ratio=(1.0 / denom) if denom != 0 else float('inf'),
                                det_spinodal=float(np.linalg.det(np.eye(2) - X * C0))))
    return rows, scalars


def _worker(entry_and_suspects):
    entry, suspects = entry_and_suspects
    try:
        return entry['label'], score_state(entry, suspects), None
    except Exception:
        return entry['label'], ([], []), traceback.format_exc(limit=3)


# --------------------------------------------------------------------------
def read_pool(args) -> list[dict]:
    rows = [r for r in csv.DictReader(POOL_INDEX.open()) if r['usable'] == '1']
    if args.label:
        rows = [r for r in rows if r['label'] in args.label]
    if args.source:
        rows = [r for r in rows if r['source'] in args.source]
    if args.limit:
        rows = rows[:args.limit]
    return rows


def validate() -> int:
    """Regression on the one state point where the answer is already recorded.

    numerical_notes/results/single_point/oz_closures_comparison.md sec.3, pair (2,2):
    GCMC inner -229.14 / outer +133.53; MSA inner -223.50; HNCB inner -235.78,
    and the inner-core ranking MSA 2.2% < HNCB 3.4%.  If this drifts, nothing
    downstream is worth reading.
    """
    entry = next(r for r in csv.DictReader(POOL_INDEX.open())
                 if r['label'] == 'tn1331dcf_T1_eps0.50_x0.2500')
    rows, scalars = score_state(entry, load_suspects())
    sel = [r for r in rows if r['ref'] == 'published']
    print('=' * 78)
    print('  REGRESSION vs oz_closures_comparison.md sec.3  '
          f'(sigma=[1,2], rho*={entry["rho_star"]}, T*={entry["T_star"]})')
    print('=' * 78)
    target = {('(2,2)', 'inner', 'benchmark'): -229.14,
              ('(2,2)', 'outer', 'benchmark'): +133.53,
              ('(2,2)', 'inner', 'OZ+MSA'): -223.50,
              ('(2,2)', 'inner', 'OZ+HNCB'): -235.78}
    print(f"  {'pair':>6} {'region':>6} {'model':>13} {'chat0':>10} {'recorded':>10} {'delta':>8}")
    for (i, j) in PAIRS:
        for region in ('inner', 'outer'):
            lbl = LABELS[(i, j)]
            b = [r for r in sel if r['pair'] == lbl and r['region'] == region]
            if not b:
                continue
            ref_val = b[0]['chat0_bench']
            t = target.get((lbl, region, 'benchmark'))
            print(f"  {lbl:>6} {region:>6} {'benchmark':>13} {ref_val:10.2f} "
                  f"{t if t is not None else float('nan'):10.2f} "
                  f"{(ref_val - t) if t is not None else float('nan'):8.2f}")
            for r in b:
                t = target.get((lbl, region, r['model']))
                print(f"  {'':>6} {'':>6} {r['model']:>13} {r['chat0_model']:10.2f} "
                      f"{t if t is not None else float('nan'):10.2f} "
                      f"{(r['chat0_model'] - t) if t is not None else float('nan'):8.2f}")
    print('\n  inner-core |chat0| error, %:')
    print(f"    {'model':>13}  " + '  '.join(f'{LABELS[p]:>8}' for p in PAIRS) + '     mean')
    for m in MODELS:
        es = []
        for p in PAIRS:
            b = [r for r in sel if r['pair'] == LABELS[p] and r['region'] == 'inner'
                 and r['model'] == m]
            if b and b[0]['chat0_bench']:
                es.append(abs(b[0]['chat0_model'] - b[0]['chat0_bench'])
                          / abs(b[0]['chat0_bench']) * 100)
        if es:
            print(f"    {m:>13}  " + '  '.join(f'{e:7.1f}%' for e in es)
                  + f"   {np.mean(es):6.1f}%")
    print('\n  chi_T/chi_id (recorded: published 0.104, c22->-beta*phi 0.058,')
    print('                          MSA 0.058, HNCB 0.056):')
    for s in scalars:
        if s['chi_ratio'] != '':
            print(f"    {s['ref']:>12}  {s['model']:>13}  {float(s['chi_ratio']):8.4f}")
    return 0


ETA_BINS = [0.10, 0.25, 0.35, 0.43, 0.60]
ETA_LABELS = ['0.10-0.25', '0.25-0.35', '0.35-0.43', '0.43-0.60']
CLEAN = ['tn1331_dcf', 'md']          # benchmarks whose c is worth scoring against

# One scoring run feeds three write-ups, each restricted to where its data came
# from: numerical_notes/results/benchmark/{literature,LAMMPS_MD,all_samples}.
# `sources` is what the scope may look at; `clean` is the subset whose c is
# trustworthy enough to rank models with (the RDF-inverted tables are kept for
# section 6, which is about them, and excluded from every ranking -- see the
# write-ups).
SCOPES = {
    'all':        dict(sources=['tn1331_dcf', 'tn1331_rdf', 'md', 'yujin_fig13'],
                       clean=['tn1331_dcf', 'md'],
                       title='ALL SOURCES POOLED'),
    'literature': dict(sources=['tn1331_dcf', 'tn1331_rdf', 'yujin_fig13'],
                       clean=['tn1331_dcf'],
                       title='LITERATURE BENCHMARKS ONLY  (NIST TN 1331, Yu & Jin)'),
    'md':         dict(sources=['md'],
                       clean=['md'],
                       title='OUR OWN LAMMPS MD ONLY'),
}


def report(scope: str = 'all') -> int:
    """Every aggregate table quoted in the benchmark write-ups.

    scope = all        -> benchmark/all_samples/dcf_multistate_comparison.md
            literature -> benchmark/literature/literature_dcf_benchmark.md
            md         -> benchmark/LAMMPS_MD/md_dcf_benchmark.md
    """
    import pandas as pd
    pd.set_option('display.width', 220)

    cfg = SCOPES[scope]
    SRC, CLEAN = cfg['sources'], cfg['clean']

    d = pd.read_csv(METRICS_CSV)
    d['eta'] = d.eta.astype(float)
    d['state'] = d.label.str.split('_', n=1).str[1]
    s = pd.read_csv(SCALARS_CSV)
    s['eta'] = s.eta.astype(float)
    s['chi_ratio'] = pd.to_numeric(s.chi_ratio, errors='coerce')
    d_all, s_all_src = d, s
    d = d[d.source.isin(SRC)]
    s = s[s.source.isin(SRC)]
    print('#' * 78)
    print(f'#  SCOPE: {scope} -- {cfg["title"]}')
    print('#' * 78)

    def blk(title):
        print('\n' + '=' * 78 + f'\n  {title}\n' + '=' * 78)

    def piv(frame, values, aggfunc='median', index='model', columns='pair', nd=4):
        t = frame.pivot_table(index=index, columns=columns, values=values,
                              aggfunc=aggfunc, observed=False)
        if columns == 'pair':
            t['mean'] = t.mean(axis=1)
        print(t.round(nd).to_string())

    blk('0.  POOL')
    for src in SRC:
        q = d[d.source == src]
        if not q.label.nunique():
            continue
        print(f'  {src:<12} {q.label.nunique():>3} state points, '
              f'{q[q.outer_trust == 1].label.nunique():>3} outer-trusted')

    blk('1.  POINTWISE, INNER CORE (r < R_ij) -- NRMSE, median over state points')
    for src, ttl in (('tn1331_dcf', 'TN 1331, directly tabulated DCF'),
                     ('md', 'our own MD, g to half the box'),
                     ('tn1331_rdf', 'TN 1331 RDF, OZ-inverted here -- SEE SECTION 6')):
        if src not in SRC:
            continue
        q = d[(d.ref == 'published') & (d.region == 'inner') & (d.source == src)]
        print(f'\n  -- {ttl}  (n = {q.label.nunique()})')
        piv(q, 'nrmse')

    blk('2.  POINTWISE, INNER CORE -- NRMSE by packing fraction (clean sources)')
    q = d[(d.ref == 'published') & (d.region == 'inner') & d.source.isin(CLEAN)].copy()
    q['ebin'] = pd.cut(q.eta, ETA_BINS, labels=ETA_LABELS)
    piv(q, 'nrmse', columns='ebin')
    print('  n states per bin:',
          q.groupby('ebin', observed=False).label.nunique().to_dict())

    blk('3.  POINTWISE, OUTER (r > R_ij) -- outer-trusted benchmarks only\n'
        '      OZ+MSA and FMSA-DP share the tail -beta*u by construction;\n'
        '      their outer rows are the same number, not two measurements.')
    q = d[(d.ref == 'published') & (d.region == 'outer') & (d.outer_trust == 1)]
    print(f'\n  -- NRMSE (n = {q.label.nunique()})')
    piv(q, 'nrmse')
    print('\n  -- MAE, absolute units of c')
    piv(q, 'mae')
    if 'md' in SRC and len(SRC) > 1:
        print(f"\n  -- NRMSE, our MD only (n = {q[q.source == 'md'].label.nunique()})")
        piv(q[q.source == 'md'], 'nrmse')
    q2 = q.copy()
    q2['cperr'] = (q2.contact_model - q2.contact_bench).abs() / q2.contact_bench.abs() * 100
    print('\n  -- contact peak of c(r), median relative error %')
    piv(q2, 'cperr', nd=1)

    blk('4.  COMPRESSIBILITY ROUTE -- |chat_ij(0)| relative error %, by region\n'
        '      chat(0) restricted to the benchmark r range, not the true k->0 limit')
    for ref in REFS:
        for region in ('inner', 'outer'):
            q = d[(d.ref == ref) & (d.region == region) & d.source.isin(CLEAN)].copy()
            if region == 'outer':
                q = q[q.outer_trust == 1]
            q['err'] = (q.chat0_model - q.chat0_bench).abs() / q.chat0_bench.abs() * 100
            print(f'\n  -- ref = {ref:<10} region = {region:<6} (n = {q.label.nunique()})')
            piv(q, 'err', nd=1)

    blk('5.  COMPRESSIBILITY ROUTE -- chi_T/chi_id, and the dual-reference check')
    bad = sorted(set(s[(s.model == 'benchmark') & (s.chi_ratio < 0)].label))
    if bad:
        print('  benchmarks disqualified first -- their own chat(0) implies a NEGATIVE\n'
              '  isothermal compressibility, which no fluid has, so they cannot score\n'
              '  anything.  Dropped from the aggregates below:')
        for src in SRC:
            labs = [l for l in bad
                    if not s[(s.label == l) & (s.source == src)].empty]
            if not labs:
                continue
            print(f'    {src}  ({len(labs)}):')
            for lab in labs:
                z = s[(s.model == 'benchmark') & (s.label == lab)
                      & (s.ref == 'published')].iloc[0]
                print(f'      {lab:<38} eta={z.eta:.3f}  chat(0) = '
                      f'{z.chat0_11:+.1f} / {z.chat0_12:+.1f} / {z.chat0_22:+.1f}'
                      f'  -> chi = {z.chi_ratio:+.3f}')
        note = {'all': 'Almost all are the RDF-inverted tables (section 6); the rest\n'
                       '  are SPARSE OCR casualties and MD points at eta = 0.50.',
                'literature': 'Almost all are the RDF-inverted tables (section 6);\n'
                              '  the rest are SPARSE OCR casualties.',
                'md': 'All but one are at eta = 0.50.'}[scope]
        print(f'\n  {note}  Wherever they cluster, that is where the g -> c\n'
              '  inversion has stopped being usable at all.\n')
    s_all = s
    s = s[~s.label.isin(bad)]
    q = s[s.source.isin(CLEAN)]
    print('  median over clean sources:')
    print(q.pivot_table(index='model', columns='ref', values='chi_ratio',
                        aggfunc='median').round(4).to_string())
    for src in CLEAN:
        b = s[(s.model == 'benchmark') & (s.source == src)].copy()
        if b.empty:
            continue
        b['ebin'] = pd.cut(b.eta, ETA_BINS, labels=ETA_LABELS)
        t = b.pivot_table(index='ebin', columns='ref', values='chi_ratio',
                          aggfunc='median', observed=False)
        t['published / c22_mfphi'] = t['published'] / t['c22_mfphi']
        print(f'\n  benchmark only, source = {src}, by packing fraction -- the two\n'
              '  reference variants must coincide wherever the c22 artifact is absent:')
        print(t.round(4).to_string())

    if 'tn1331_rdf' in SRC and 'tn1331_dcf' in SRC:
        blk('6.  WHAT THE RDF-INVERTED POOL IS ACTUALLY MEASURING')
        q = d[(d.ref == 'published') & (d.region == 'inner')
              & d.source.isin(['tn1331_dcf', 'tn1331_rdf'])]
        both = (set(q[q.source == 'tn1331_dcf'].state)
                & set(q[q.source == 'tn1331_rdf'].state))
        q = q[q.state.isin(both)]
        t = q.pivot_table(index='model', columns='source', values='nrmse', aggfunc='median')
        t['inflation'] = t['tn1331_rdf'] / t['tn1331_dcf']
        print(f'  the same {len(both)} state points, scored against each version of the\n'
              '  benchmark -- direct Appendix C, and Appendix A inverted here:')
        print(t.round(4).to_string())
        print(f"\n  best-to-worst on the direct DCF : "
              f"{list(t['tn1331_dcf'].sort_values().index)}")
        print(f"  best-to-worst on the inverted   : "
              f"{list(t['tn1331_rdf'].sort_values().index)}")

    if 'md' in SRC:
        blk('7.  SIZE RATIO AND COMPOSITION (our MD, the only source that varies them)')
        q = d[(d.source == 'md') & (d.ref == 'published')]
        for region in ('inner', 'outer'):
            qq = q[q.region == region]
            if region == 'outer':
                qq = qq[qq.outer_trust == 1]
            print(f'\n  -- {region} NRMSE by sigma22/sigma11'
                  + ('  (outer-trusted only)' if region == 'outer' else ''))
            piv(qq, 'nrmse', columns='sigma_ratio')
            print('     n states:',
                  qq.groupby('sigma_ratio').label.nunique().to_dict())
        print('\n  -- inner NRMSE by composition x1')
        piv(q[q.region == 'inner'], 'nrmse', columns='x1')
        print('     n states:',
              q[q.region == 'inner'].groupby('x1').label.nunique().to_dict())
        print('\n  -- inner NRMSE by (sigma_ratio, pair) -- where the size ratio bites')
        for pair in ('(1,1)', '(1,2)', '(2,2)'):
            print(f'     pair {pair}')
            piv(q[(q.region == 'inner') & (q.pair == pair)], 'nrmse',
                columns='sigma_ratio')

    blk('8.  OZ CONVERGENCE  (over the whole pool, disqualifications included)')
    nc = s_all[(s_all.model.isin(['OZ+MSA', 'OZ+HNCB'])) & (s_all.converged == 0)
               & (s_all.ref == 'published')]
    tot = s_all[(s_all.model.isin(['OZ+MSA', 'OZ+HNCB'])) & (s_all.ref == 'published')]
    print(f'  {len(nc)} of {len(tot)} solves failed to converge')
    for _, r in nc.iterrows():
        print(f'    {r.label:<40} {r.model:<9} eta={r.eta}')
    print('\n  median iterations:')
    print(tot[tot.converged == 1].groupby('model').iters.median().to_string())

    if scope == 'all':
        blk('9.  DO THE TWO INDEPENDENT CLEAN BENCHMARKS AGREE?\n'
            '      TN 1331 (1989, 864 particles, published tables) and our own MD\n'
            '      (2025 LAMMPS, N=2048, our inversion) share no code and no data.\n'
            '      If they rank the models differently, neither ranking is safe.')
        q = d[(d.ref == 'published') & d.source.isin(CLEAN)]
        for region in ('inner', 'outer'):
            qq = q[q.region == region]
            if region == 'outer':
                qq = qq[qq.outer_trust == 1]
            t = qq.pivot_table(index='model', columns='source', values='nrmse',
                               aggfunc='median')
            for c in t.columns:
                t[f'rank_{c}'] = t[c].rank()
            print(f'\n  -- {region} NRMSE, median, by source')
            print(t.round(4).to_string())
            print('     n states:', qq.groupby('source').label.nunique().to_dict())

        print('\n  -- matched geometry: sigma ratio 2.0 only, both sources')
        qq = q[(q.region == 'inner') & (q.sigma_ratio.astype(float) == 2.0)]
        t = qq.pivot_table(index='model', columns='source', values='nrmse',
                           aggfunc='median')
        print(t.round(4).to_string())
        print('     n states:', qq.groupby('source').label.nunique().to_dict())

        print('\n  -- matched geometry AND packing: sigma ratio 2.0, eta in [0.35,0.47]')
        qq = q[(q.region == 'inner') & (q.sigma_ratio.astype(float) == 2.0)
               & q.eta.between(0.35, 0.47)]
        t = qq.pivot_table(index='model', columns='source', values='nrmse',
                           aggfunc='median')
        print(t.round(4).to_string())
        print('     n states:', qq.groupby('source').label.nunique().to_dict())

        print('\n  -- pooled ranking (both clean sources at once), inner and outer')
        for region in ('inner', 'outer'):
            qq = q[q.region == region]
            if region == 'outer':
                qq = qq[qq.outer_trust == 1]
            per_state = (qq.groupby(['label', 'model'], as_index=False).nrmse.mean())
            t = per_state.groupby('model').nrmse.median().sort_values()
            print(f'     {region}: ' + ', '.join(f'{m} {v:.3f}' for m, v in t.items()))

    if 'lowk' in set(d.ref):
        blk('10. WHAT THE LONG-WAVELENGTH REPAIR DOES TO THE RANKING\n'
            '      ref = lowk is data_benchmark/dcf_lowk.py: chat_ij(0) taken from\n'
            '      the equation of state, c_ij -> -beta u_ij imposed outside the\n'
            '      measured keep radius.  Same state points, same models, same\n'
            '      scoring -- only the benchmark changed.')
        have = d[d.ref == 'lowk'].label.unique()
        q = d[d.label.isin(have) & d.source.isin(CLEAN)]
        print(f'  {len(set(have) & set(d[d.source.isin(CLEAN)].label))} clean state '
              f'points carry a repaired variant, of '
              f'{d[d.source.isin(CLEAN)].label.nunique()}')

        for region in ('inner', 'outer'):
            print(f'\n  -- {region} NRMSE, median, mean over pairs, by reference')
            qq = q[q.region == region]
            per = qq.groupby(['label', 'model', 'ref'], as_index=False).nrmse.mean()
            t = per.pivot_table(index='model', columns='ref', values='nrmse',
                                aggfunc='median')
            for c in t.columns:
                t[f'rank_{c}'] = t[c].rank()
            print(t.round(4).to_string())
            print('     NOTE the outer rows are scored over every repaired state, not\n'
                  '     only the outer_trusted ones: under ref = lowk the outer region\n'
                  '     is the exact asymptote, so the trust flag -- which grades the\n'
                  '     published tail -- no longer selects anything.')
            if region == 'inner':
                print('\n     the same, per pair')
                piv(qq[qq.ref == 'lowk'], 'nrmse')

        print('\n  -- chi_T/chi_id: the benchmark itself, and how far each model is\n'
              '     from it.  This is the observable the repair was built to fix.')
        sb = s_all_src[s_all_src.label.isin(have) & s_all_src.source.isin(CLEAN)].copy()
        b = sb[sb.model == 'benchmark'][['label', 'ref', 'chi_ratio']]
        b = b.rename(columns={'chi_ratio': 'chi_bench'})
        mm = sb[sb.model != 'benchmark'].merge(b, on=['label', 'ref'])
        mm = mm[mm.chi_bench > 0]
        mm['err'] = (mm.chi_ratio - mm.chi_bench).abs() / mm.chi_bench * 100
        print('\n     benchmark chi_T/chi_id, median by reference:')
        print(b.groupby('ref').chi_bench.median().round(4).to_string())
        print('\n     median |model - benchmark| / benchmark, %:')
        t = mm.pivot_table(index='model', columns='ref', values='err',
                           aggfunc='median')
        print(t.round(1).to_string())
        print('\n     benchmarks with chi_T < 0 (physically impossible), by reference:')
        neg = sb[(sb.model == 'benchmark') & (sb.chi_ratio < 0)]
        print(neg.groupby('ref').label.nunique().reindex(REFS).fillna(0)
              .astype(int).to_string())

        print('\n  -- how often each model is best at a state point (pair-averaged)')
        for region in ('inner', 'outer'):
            qq = q[q.region == region]
            if region == 'outer':
                qq = qq[qq.outer_trust == 1]
            per_state = qq.groupby(['label', 'model'], as_index=False).nrmse.mean()
            win = per_state.loc[per_state.groupby('label').nrmse.idxmin()]
            print(f'     {region}: ' + str(win.model.value_counts().to_dict()))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--label', nargs='*', default=None)
    ap.add_argument('--source', nargs='*', default=None,
                    help='tn1331_dcf / tn1331_rdf / md')
    ap.add_argument('--workers', type=int, default=6,
                    help='keep this modest while an MD sweep is running')
    ap.add_argument('--validate', action='store_true',
                    help='regression against the recorded single-state results')
    ap.add_argument('--report', nargs='?', const='all', default=None,
                    choices=list(SCOPES),
                    help='aggregate tables from an existing metrics CSV, for one '
                         'of the three benchmark write-ups (default: all)')
    args = ap.parse_args()

    if args.validate:
        return validate()
    if args.report:
        return report(args.report)

    pool = read_pool(args)
    suspects = load_suspects()
    print(f'scoring {len(pool)} state points with {args.workers} workers')

    metrics, scalars, failed = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, (e, suspects)): e['label'] for e in pool}
        for n, fut in enumerate(as_completed(futs), 1):
            label, (rows, scal), err = fut.result()
            if err:
                failed.append((label, err))
                print(f'  [{n}/{len(pool)}] {label}: FAILED')
            else:
                metrics.extend(rows)
                scalars.extend(scal)
                print(f'  [{n}/{len(pool)}] {label}: {len(rows)} rows')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=METRIC_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(metrics)
    with SCALARS_CSV.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=SCALAR_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(scalars)
    print(f'\nwrote {METRICS_CSV} ({len(metrics)} rows)')
    print(f'wrote {SCALARS_CSV} ({len(scalars)} rows)')

    nonconv = [s for s in scalars if s['model'] in ('OZ+MSA', 'OZ+HNCB')
               and s['converged'] == 0 and s['ref'] == 'published']
    print(f'\nOZ non-convergence: {len(nonconv)} of '
          f'{2 * len(pool)} solves')
    for s in nonconv:
        print(f'    {s["label"]:<40} {s["model"]:<9} eta={s["eta"]}')
    if failed:
        print(f'\n{len(failed)} state points failed outright:')
        for label, err in failed:
            print(f'  {label}\n{err}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
