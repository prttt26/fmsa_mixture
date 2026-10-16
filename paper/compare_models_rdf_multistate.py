#!/usr/bin/env python3
"""OZ-MSA, OZ-HNCB and FMSA-DP against the benchmark RDF pool.

The companion of compare_models_dcf_multistate.py, on the other correlation
function.  Same pool sources, same three-way scope split, same metric-table
layout -- but scoring g_ij(r) instead of c_ij(r), which changes the study in
three ways that matter:

1. NO INVERSION ANYWHERE ON THE BENCHMARK SIDE.  TN 1331 Appendix A tabulates
   g_ij directly and our MD measures it directly.  The DCF study had to invert
   Appendix A to get a c, and that inversion dominated the error (it inflated
   every model's score 3.5-7.7x and reversed the ranking).  Here both sources
   are `direct`, so the 39 RDF-inverted entries that had to be quarantined in
   the DCF pool come back as first-class benchmarks.
2. IT IS A GENUINE THREE-WAY COMPARISON EVERYWHERE.  In the DCF study OZ-MSA
   and FMSA-DP are the SAME function outside R_ij (both = -beta*u), so the
   outer tables were a two-way comparison written three times.  No closure
   fixes g outside the core, so all three models are independent at every r.
3. THE DISCRIMINATING OBSERVABLE IS THE FIRST PEAK, not an integral.  The
   contact-peak result of the DCF study (mean-field tail misses c_11 at contact
   by 51%, HNCB gets it to 5%) has a direct counterpart here, and it is
   measured on the quantity a simulation reports most reliably.

Models: OZ+MSA and OZ+HNCB from oz_numerical (g = 1 + h at the converged c),
FMSA-DP from fmsa_double_prop.get_g_fourier -- the k-space route of
numerical_notes/results/fmsa_dp_rdf_routes.md.  FMSA-DP's RDF has ONE hard-sphere
reference (Baxter/PY, the one its first order perturbs around), so unlike the
DCF study there is no fmt/py pair.

Metrics, per (state, pair)
--------------------------
pointwise   NRMSE = RMSE / rms(benchmark) over the scored range, and over the
            first shell alone (r <= the benchmark's first minimum).
peak        height and position of the first maximum -- the observable that
            separates the closures.
minimum     depth of the first minimum, which is where a bad tail shows up.
integral    coordination number N_ij = 4 pi rho_j Int_0^{r_min} r^2 g_ij dr.

    python compare_models_rdf_multistate.py --build-pool     # -> rdf_pool/
    python compare_models_rdf_multistate.py --workers 8      # -> metric CSVs
    python compare_models_rdf_multistate.py --report md      # one scope's tables
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
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

BENCH = REPO / 'data_benchmark'
POOL_DIR = BENCH / 'rdf_pool'
POOL_INDEX = POOL_DIR / 'pool_index.csv'
SUSPECT = BENCH / 'tn1331' / 'suspect.csv'
OUT_ROOT = REPO / 'numerical_notes' / 'results' / 'benchmark_rdf'
OUT_DIR = OUT_ROOT / 'all_samples'
METRICS_CSV = OUT_DIR / 'rdf_multistate_metrics.csv'

PAIRS = [(0, 0), (0, 1), (1, 1)]
PAIR_TAG = {(0, 0): '11', (0, 1): '12', (1, 1): '22'}
LABELS = {(0, 0): '(1,1)', (0, 1): '(1,2)', (1, 1): '(2,2)'}
MODELS = ['OZ+MSA', 'OZ+HNCB', 'FMSA-DP']
N_R = 4096
R_SCORE_MAX = 2.5          # in units of sigma_ij, as in the DCF pool

SCOPES = {
    'all':        dict(sources=['tn1331_rdf', 'md'], clean=['tn1331_rdf', 'md'],
                       title='ALL SOURCES POOLED'),
    'literature': dict(sources=['tn1331_rdf'], clean=['tn1331_rdf'],
                       title='LITERATURE ONLY  (NIST TN 1331 Appendix A)'),
    'md':         dict(sources=['md'], clean=['md'],
                       title='OUR OWN LAMMPS MD ONLY'),
}

METRIC_FIELDS = [
    'label', 'source', 'sigma_ratio', 'eps_ratio', 'T_star', 'x1', 'rho_star',
    'eta', 'fidelity', 'quality', 'n_suspect', 'plateau_ok', 'model', 'pair', 'converged',
    'n_pts', 'rmse', 'nrmse', 'nrmse_shell', 'rms_bench',
    'peak_model', 'peak_bench', 'rpeak_model', 'rpeak_bench',
    'min_model', 'min_bench', 'coord_model', 'coord_bench',
]


def sig_mix_of(ratio: float) -> np.ndarray:
    s = np.array([1.0, float(ratio)])
    return 0.5 * (s[:, None] + s[None, :])


def eta_of(rho: float, x1: float, ratio: float) -> float:
    x = np.array([x1, 1.0 - x1])
    return float((np.pi / 6.0) * rho * np.sum(x * np.array([1.0, ratio]) ** 3))


# ==========================================================================
# pool
# ==========================================================================
def _read_csv(path: Path) -> dict:
    rows = [r for r in path.open() if not r.startswith('#')]
    rdr = csv.DictReader(rows)
    cols = {n: [] for n in rdr.fieldnames}
    for row in rdr:
        for n, v in row.items():
            cols[n].append(float(v) if v not in ('', None) else np.nan)
    return {n: np.asarray(v) for n, v in cols.items()}


def _grade(r_by_pair: dict, g_by_pair: dict,
           source_hint: str = 'OCR spike', far: list | None = None) -> dict:
    """Uniform physics gates, applied identically to every source.

    Two things are DISQUALIFYING, because no fluid can produce them:
      core      g must be ~0 well inside the pair diameter (this checks the r
                convention as much as the data);
      positive  g >= 0 everywhere, and g < 20 (a spike that large is OCR damage).

    A third is disqualifying only when it is gross:

    homogeneous  g_ij must come back toward 1.  Oscillation at liquid packing is
                 physics -- an outer-third mean of 0.83 or 1.2 is normal, and
                 `plateau_ok` records the tight [0.7, 1.35] band for callers who
                 want it -- but a mean OUTSIDE [0.5, 1.5] is not oscillation, it
                 is a two-phase state, and no single-phase liquid-state theory
                 can be scored against one.  The grid makes this unmissable: the
                 LJ-HS block has g_11 peaking at 35 with a mean of 3.8 out at
                 4-6 sigma while g_12 is depleted to 0.84 -- the LJ component
                 condensing out of an athermal hard-sphere matrix, since LJ's own
                 T_c is 1.31 and there is nothing in a WCA solvent to keep it
                 dispersed.  48 of 144 LJ-HS points and 20 of 352 LJ grid points
                 are two-phase this way; NO pre-existing entry is affected.
    """
    core, pos, plat = [], [], []
    for (i, j) in PAIRS:
        r, g = r_by_pair[(i, j)], g_by_pair[(i, j)]
        if r.size < 8:
            return dict(usable=0, skip_reason=f'pair {PAIR_TAG[(i,j)]} has {r.size} points',
                        core_max='', g_min='', plateau='')
        sig_ij = r_by_pair['sigma'][(i, j)]
        m = r < 0.8 * sig_ij
        core.append(float(np.max(np.abs(g[m]))) if m.any() else 0.0)
        pos.append(float(np.min(g)))
        m = r > 0.6 * float(r[-1])
        plat.append(float(np.mean(g[m])) if m.any() else np.nan)
    gmax = max(float(np.max(g_by_pair[(i, j)])) for (i, j) in PAIRS)
    bad = []
    if max(core) > 0.15:
        bad.append(f'g={max(core):.2f} inside 0.8 sigma (r convention or bad table)')
    if min(pos) < -0.02:
        bad.append(f'g_min={min(pos):.3f} < 0')
    if gmax > 20.0:
        bad.append(f'g_max={gmax:.1f} > 20 ({source_hint})')
    fin = [p for p in plat if np.isfinite(p)]
    # The two-phase test needs the LONG range, so it runs on the untruncated g
    # (`far`) -- available for our MD, not for tables that stop at 2.5 sigma.
    if far is not None and (max(far) > 1.5 or min(far) < 0.5):
        bad.append('two-phase: g plateau = '
                   + ';'.join(f'{v:.2f}' for v in far) + ', not ~1')
    plateau_ok = int(all(0.7 <= p <= 1.35 for p in fin))
    return dict(usable=0 if bad else 1, skip_reason='; '.join(bad),
                core_max=f'{max(core):.4f}', g_min=f'{min(pos):.4f}',
                g_max=f'{gmax:.3f}', plateau_ok=plateau_ok,
                plateau=';'.join(f'{p:.3f}' for p in plat))


def _load_tn1331(path, meta: dict):
    """`path` is the pool's g_path, so a repaired table can be swapped in."""
    tab = _read_csv(BENCH / path)
    sm = sig_mix_of(2.0)
    r_by, g_by = {'sigma': {}}, {}
    for (i, j) in PAIRS:
        tag = PAIR_TAG[(i, j)]
        r = tab['R_star'] * sm[i, j]
        g = tab[f'g{tag}']
        keep = np.isfinite(g) & (r > 0) & (tab['R_star'] <= R_SCORE_MAX + 1e-9)
        r_by[(i, j)], g_by[(i, j)] = r[keep], g[keep]
        r_by['sigma'][(i, j)] = sm[i, j]
    return r_by, g_by


def _load_md(run: Path):
    p = json.loads((run / 'params.json').read_text())
    tab = _read_csv(run / 'g.csv')
    sm = sig_mix_of(p['ratio'])
    # long-range plateau, measured BEFORE the 2.5-sigma cap, for the two-phase gate
    rr = tab['r']
    fm = rr > 0.6 * float(rr[-1])
    far = [float(np.mean(tab[f'g{PAIR_TAG[q]}'][fm])) for q in PAIRS] if fm.any() else None
    r_by, g_by = {'sigma': {}}, {}
    for (i, j) in PAIRS:
        tag = PAIR_TAG[(i, j)]
        r, g = tab['r'], tab[f'g{tag}']
        keep = np.isfinite(g) & (r > 0) & (r <= R_SCORE_MAX * sm[i, j] + 1e-9)
        r_by[(i, j)], g_by[(i, j)] = r[keep], g[keep]
        r_by['sigma'][(i, j)] = sm[i, j]
    return r_by, g_by, p, far


def build_pool(use_repaired: bool = False) -> int:
    """Assemble every DIRECT g_ij source into one index.

    `use_repaired` swaps in tn1331/rdf_repaired/<stem>.csv where audit_tn1331_rdf.py
    has written one.  It is OFF by default and should stay off for the headline
    numbers: those tables were repaired against our own MD, so using them would
    make the literature source no longer independent of the MD source, which is
    the one thing the cross-source check in the write-up is for.  Turn it on to
    measure what the scan damage costs -- see
    numerical_notes/results/benchmark_rdf/literature/tn1331_md_audit.md.
    """
    rows = []
    repaired_dir = BENCH / 'tn1331' / 'rdf_repaired'
    idx = {r['file']: r for r in csv.DictReader((BENCH / 'tn1331' / 'index.csv').open())
           if r['kind'] == 'rdf'}
    for f, meta in sorted(idx.items()):
        stem = Path(f).stem
        if not meta['rho_star']:
            rows.append(dict(label=f'tn1331rdf_{stem}', source='tn1331_rdf',
                             usable=0, plateau_ok=0,
                             skip_reason='no rho_star in index.csv'))
            continue
        rho, x1 = float(meta['rho_star']), float(meta['x1'])
        r_by, g_by = _load_tn1331(f'tn1331/rdf/{stem}.csv', meta)
        grade = _grade(r_by, g_by)
        rows.append(dict(label=f'tn1331rdf_{stem}', source='tn1331_rdf',
                         g_path=f'tn1331/rdf/{stem}.csv', r_convention='perpair',
                         sigma_ratio=2.0, eps_ratio=float(meta['eps']),
                         T_star=float(meta['tstar']), x1=x1, rho_star=rho,
                         eta=f'{eta_of(rho, x1, 2.0):.4f}', fidelity='direct',
                         quality=meta['quality'], n_suspect=meta['n_suspect'],
                         suspect_key=f'rdf/{stem}.csv', **grade))
    for run in sorted((BENCH / 'md' / 'runs').iterdir()):
        if not (run / 'g.csv').exists():
            continue
        r_by, g_by, p, far = _load_md(run)
        grade = _grade(r_by, g_by, source_hint='clustered, see plateau', far=far)
        rows.append(dict(label=f'md_{run.name}', source='md',
                         potential=('ljhs' if p.get('kind') == 'ljhs' else 'lj'),
                         g_path=f'md/runs/{run.name}/g.csv', r_convention='absolute',
                         sigma_ratio=p['ratio'], eps_ratio=p['eps_ratio'],
                         T_star=p['T'], x1=p['x1'], rho_star=p['rho'],
                         eta=f'{eta_of(p["rho"], p["x1"], p["ratio"]):.4f}',
                         fidelity='direct', quality='OK', n_suspect=0,
                         suspect_key='', **grade))
    fields = ['label', 'source', 'potential', 'g_path', 'r_convention',
              'sigma_ratio', 'eps_ratio',
              'T_star', 'x1', 'rho_star', 'eta', 'fidelity', 'quality', 'n_suspect',
              'suspect_key', 'repaired', 'core_max', 'g_min', 'g_max', 'plateau',
              'plateau_ok', 'usable', 'skip_reason']
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r.setdefault('potential', 'lj')
    with POOL_INDEX.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {POOL_INDEX}  ({len(rows)} entries)')
    for src in ('tn1331_rdf', 'md'):
        q = [r for r in rows if r['source'] == src]
        print(f'  {src:<12} {len(q):>3} entries  '
              f'{sum(int(r.get("usable", 0)) for r in q):>3} usable  '
              f'{sum(int(r.get("plateau_ok", 0)) for r in q):>3} plateau-clean')
    skipped = [r for r in rows if not int(r.get('usable', 0))]
    if skipped:
        print(f'\n  skipped ({len(skipped)}):')
        for r in skipped:
            print(f'    {r["label"]:<38} {r.get("skip_reason", "")}')
    return 0


# ==========================================================================
# scoring
# ==========================================================================
def load_benchmark(entry: dict, suspects: dict):
    if entry['source'] == 'tn1331_rdf':
        r_by, g_by = _load_tn1331(entry['g_path'], entry)
        drop = entry['suspect_key']
        if drop:
            tab = _read_csv(BENCH / entry['g_path'])
            sm = sig_mix_of(2.0)
            for (i, j) in PAIRS:
                flagged = suspects.get((drop, f'g{PAIR_TAG[(i,j)]}'), set())
                if not flagged:
                    continue
                bad = np.isin(np.round(tab['R_star'], 4), list(flagged))
                r_all = tab['R_star'] * sm[i, j]
                keep = np.isin(np.round(r_all, 6), np.round(r_by[(i, j)], 6)) & ~bad
                r_by[(i, j)] = r_all[keep]
                g_by[(i, j)] = tab[f'g{PAIR_TAG[(i,j)]}'][keep]
        return r_by, g_by
    r_by, g_by, _, _ = _load_md(BENCH / entry['g_path'].rsplit('/g.csv')[0])
    return r_by, g_by


def _features(r, g, R):
    """(peak height, peak position, first-minimum depth, r at that minimum)."""
    m = (r >= 0.8 * R) & np.isfinite(g)
    if m.sum() < 5:
        return (np.nan,) * 4
    rr, gg = r[m], g[m]
    q = int(np.argmax(gg))
    peak, rpeak = float(gg[q]), float(rr[q])
    tail = gg[q:]
    if tail.size < 3:
        return peak, rpeak, np.nan, np.nan
    qm = q + int(np.argmin(tail))
    return peak, rpeak, float(gg[qm]), float(rr[qm])


def build_models(entry):
    from fmsa_ga_matrix_mix import FMSA_GA_matrix_mix
    from fmsa_double_prop import FMSA_double_prop
    from oz_numerical import OZSolverMSA
    from compare_models_dcf_multistate import _c_is_physical

    from ljhs_mapping import model_inputs

    ratio = float(entry['sigma_ratio'])
    sigs, eps, mkw = model_inputs(entry.get('potential', 'lj'), ratio,
                                  entry['eps_ratio'], float(entry['T_star']))
    x1 = float(entry['x1'])
    x = np.array([x1, 1.0 - x1])
    rho, T = float(entry['rho_star']), float(entry['T_star'])

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        ga = FMSA_GA_matrix_mix(sigs, eps, x, rho, T, n_tails=2,
                                verbose=False, **mkw)
        dp = FMSA_double_prop(sigs, eps, x, rho, T, n_tails=2,
                              verbose=False, **mkw)

    R_ij = np.asarray(ga.R_ij, float)
    curves, status = {}, {}
    for name, closure in (('OZ+MSA', 'msa'), ('OZ+HNCB', 'hncb')):
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore')
            s = OZSolverMSA(ga, n_r=N_R)
            try:
                s.solve(method='diis', closure=closure, tol=1e-6, max_iter=500,
                        alpha=0.2, n_coupling_steps=5)
            except Exception:
                pass
        h = getattr(s, '_h_converged', None)
        ok = h is not None and _c_is_physical(getattr(s, '_c_converged', None)) \
            and bool(np.all(np.isfinite(h)))
        curves[name] = (s.r, 1.0 + h) if ok else (None, None)
        status[name] = ok
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        try:
            r_dp, g_dp = dp.get_g_fourier(k_max=1500.0, n_k=200000, return_r=True)
            curves['FMSA-DP'] = (r_dp, g_dp)          # (N, N, n_r), as for OZ
            status['FMSA-DP'] = True
        except Exception:
            curves['FMSA-DP'] = (None, None)
            status['FMSA-DP'] = False
    return R_ij, curves, status


def score_state(entry: dict, suspects: dict) -> list:
    r_by, g_by = load_benchmark(entry, suspects)
    R_ij, curves, status = build_models(entry)
    meta = {k: entry[k] for k in ('label', 'source', 'sigma_ratio', 'eps_ratio',
                                  'T_star', 'x1', 'rho_star', 'eta', 'fidelity',
                                  'quality', 'n_suspect', 'plateau_ok')}
    x1 = float(entry['x1'])
    rho_i = np.array([x1, 1.0 - x1]) * float(entry['rho_star'])

    rows = []
    for (i, j) in PAIRS:
        rb, gb = r_by[(i, j)], g_by[(i, j)]
        R = float(R_ij[i, j])
        pk_b, rpk_b, mn_b, rmn_b = _features(rb, gb, R)
        cb = (4.0 * np.pi * rho_i[j] * _trapz(rb[rb <= rmn_b] ** 2
                                              * gb[rb <= rmn_b], rb[rb <= rmn_b])
              if np.isfinite(rmn_b) else np.nan)
        for name in MODELS:
            rm, gm = curves[name]
            if rm is None:
                rows.append(dict(meta, model=name, pair=LABELS[(i, j)], converged=0))
                continue
            gm_ij = np.asarray(gm)[i, j]       # every curve is (N, N, n_r)
            gi = np.interp(rb, rm, gm_ij)
            ok = np.isfinite(gi) & np.isfinite(gb) & (rb >= 0.8 * R)
            d = gi[ok] - gb[ok]
            n = int(ok.sum())
            rms_b = float(np.sqrt(np.mean(gb[ok] ** 2))) if n else np.nan
            rmse = float(np.sqrt(np.mean(d ** 2))) if n else np.nan
            shell = ok & (rb <= (rmn_b if np.isfinite(rmn_b) else np.inf))
            ds = gi[shell] - gb[shell]
            rms_s = float(np.sqrt(np.mean(gb[shell] ** 2))) if shell.any() else np.nan
            pk_m, rpk_m, mn_m, rmn_m = _features(np.asarray(rm), np.asarray(gm_ij), R)
            sel = np.asarray(rm) <= (rmn_b if np.isfinite(rmn_b) else 0.0)
            cm = (4.0 * np.pi * rho_i[j] * _trapz(np.asarray(rm)[sel] ** 2
                                                  * np.asarray(gm_ij)[sel],
                                                  np.asarray(rm)[sel])
                  if sel.sum() > 2 else np.nan)
            rows.append(dict(meta, model=name, pair=LABELS[(i, j)], converged=1,
                             n_pts=n, rmse=rmse,
                             nrmse=(rmse / rms_b) if n and rms_b > 0 else np.nan,
                             nrmse_shell=(float(np.sqrt(np.mean(ds ** 2))) / rms_s)
                             if shell.any() and rms_s > 0 else np.nan,
                             rms_bench=rms_b,
                             peak_model=pk_m, peak_bench=pk_b,
                             rpeak_model=rpk_m, rpeak_bench=rpk_b,
                             min_model=mn_m, min_bench=mn_b,
                             coord_model=cm, coord_bench=cb))
    return rows


def load_suspects() -> dict:
    out: dict = {}
    if not SUSPECT.exists():
        return out
    for row in csv.DictReader(SUSPECT.open()):
        out.setdefault((row['file'], row['column']), set()).add(round(float(row['R_star']), 4))
    return out


def _worker(arg):
    entry, suspects = arg
    try:
        return entry['label'], score_state(entry, suspects), None
    except Exception:
        return entry['label'], [], traceback.format_exc(limit=3)


# ==========================================================================
def report(scope: str = 'all') -> int:
    import pandas as pd
    pd.set_option('display.width', 200)
    cfg = SCOPES[scope]
    d = pd.read_csv(METRICS_CSV)
    d = d[d.source.isin(cfg['sources'])]
    d['eta'] = d.eta.astype(float)
    print('#' * 78)
    print(f'#  SCOPE: {scope} -- {cfg["title"]}')
    print('#' * 78)

    def blk(t):
        print('\n' + '=' * 78 + f'\n  {t}\n' + '=' * 78)

    def piv(frame, values, aggfunc='median', columns='pair', nd=4, index='model'):
        t = frame.pivot_table(index=index, columns=columns, values=values,
                              aggfunc=aggfunc, observed=False)
        if columns == 'pair':
            t['mean'] = t.mean(axis=1)
        print(t.round(nd).to_string())

    blk('0.  POOL')
    for src in cfg['sources']:
        q = d[d.source == src]
        if q.empty:
            continue
        print(f'  {src:<12} {q.label.nunique():>3} state points')
    nc = d[(d.converged == 0)]
    print(f'  model failures: {nc.label.nunique()} state points, '
          f'{sorted(set(nc.model))}')

    dd = d[d.converged == 1]
    blk('1.  POINTWISE  NRMSE over the whole scored range (r >= 0.8 R_ij)')
    for src in cfg['sources']:
        q = dd[dd.source == src]
        if q.empty:
            continue
        print(f'\n  -- {src}  (n = {q.label.nunique()})')
        piv(q, 'nrmse')
    if 'tn1331_rdf' in cfg['sources']:
        q = dd[(dd.source == 'tn1331_rdf') & (dd.n_suspect == 0)]
        print(f'\n  -- tn1331_rdf, OCR-unflagged tables only  (n = {q.label.nunique()})')
        piv(q, 'nrmse')
        q = dd[(dd.source == 'tn1331_rdf') & (dd.n_suspect > 0)]
        print(f'\n  -- tn1331_rdf, tables carrying OCR flags  (n = {q.label.nunique()})')
        piv(q, 'nrmse')
        print('  The gap between those two blocks is the scan damage, not the models:')
        print('  suspect.csv flags only the ENDS of a corrupted run, so a flagged')
        print('  table keeps most of its corrupted values.')

    blk('2.  POINTWISE  NRMSE inside the FIRST SHELL (r <= first minimum)')
    piv(dd, 'nrmse_shell')

    blk('3.  THE FIRST PEAK -- height and position')
    q = dd.copy()
    q['pk_err'] = (q.peak_model - q.peak_bench).abs() / q.peak_bench.abs() * 100
    q['rp_err'] = (q.rpeak_model - q.rpeak_bench).abs()
    print('\n  -- peak height, median relative error %')
    piv(q, 'pk_err', nd=1)
    print('\n  -- peak position, median |Delta r| / sigma_11')
    piv(q, 'rp_err', nd=3)
    print('\n  -- signed peak height error % (is the model high or low?)')
    q['pk_signed'] = (q.peak_model - q.peak_bench) / q.peak_bench.abs() * 100
    piv(q, 'pk_signed', nd=1)

    blk('4.  THE FIRST MINIMUM, and the coordination number')
    q['mn_err'] = (q.min_model - q.min_bench).abs() / q.min_bench.abs() * 100
    q['cd_err'] = (q.coord_model - q.coord_bench).abs() / q.coord_bench.abs() * 100
    print('\n  -- first-minimum depth, median relative error %')
    piv(q, 'mn_err', nd=1)
    print('\n  -- coordination number N_ij, median relative error %')
    piv(q, 'cd_err', nd=1)

    blk('5.  BY PACKING FRACTION -- NRMSE (whole range)')
    q = dd.copy()
    q['ebin'] = pd.cut(q.eta, [0.10, 0.25, 0.35, 0.43, 0.60],
                       labels=['0.10-0.25', '0.25-0.35', '0.35-0.43', '0.43-0.60'])
    piv(q, 'nrmse', columns='ebin')
    print('  n states per bin:', q.groupby('ebin', observed=False).label.nunique().to_dict())

    if 'md' in cfg['sources']:
        blk('6.  SIZE RATIO AND COMPOSITION (our MD)')
        q = dd[dd.source == 'md']
        print('\n  -- NRMSE by sigma22/sigma11')
        piv(q, 'nrmse', columns='sigma_ratio')
        print('     n states:', q.groupby('sigma_ratio').label.nunique().to_dict())
        print('\n  -- peak-height error % by sigma22/sigma11')
        qq = q.copy()
        qq['pk_err'] = (qq.peak_model - qq.peak_bench).abs() / qq.peak_bench.abs() * 100
        piv(qq, 'pk_err', columns='sigma_ratio', nd=1)
        print('\n  -- NRMSE by composition x1')
        piv(q, 'nrmse', columns='x1')

    if scope == 'all':
        blk('7.  DO THE TWO INDEPENDENT BENCHMARKS AGREE?')
        for val, lbl in (('nrmse', 'NRMSE (whole range)'),
                         ('nrmse_shell', 'NRMSE (first shell)')):
            t = dd.pivot_table(index='model', columns='source', values=val,
                               aggfunc='median')
            for c in t.columns:
                t[f'rank_{c}'] = t[c].rank()
            print(f'\n  -- {lbl}')
            print(t.round(4).to_string())
        q = dd.copy()
        q['pk_err'] = (q.peak_model - q.peak_bench).abs() / q.peak_bench.abs() * 100
        print('\n  -- peak-height error %, by source')
        print(q.pivot_table(index='model', columns='source', values='pk_err',
                            aggfunc='median').round(1).to_string())
        print('\n  -- matched geometry: sigma ratio 2.0 only')
        qq = dd[dd.sigma_ratio.astype(float) == 2.0]
        print(qq.pivot_table(index='model', columns='source', values='nrmse',
                             aggfunc='median').round(4).to_string())
        print('     n states:', qq.groupby('source').label.nunique().to_dict())
        print('\n  -- how often each model is best at a state point (pair-averaged)')
        per = dd.groupby(['label', 'model'], as_index=False).nrmse.mean()
        win = per.loc[per.groupby('label').nrmse.idxmin()]
        print('    ', win.model.value_counts().to_dict())
    return 0


def read_pool(args) -> list:
    rows = [r for r in csv.DictReader(POOL_INDEX.open()) if r['usable'] == '1']
    if args.label:
        rows = [r for r in rows if r['label'] in args.label]
    if args.source:
        rows = [r for r in rows if r['source'] in args.source]
    if args.limit:
        rows = rows[:args.limit]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build-pool', action='store_true')
    ap.add_argument('--use-repaired', action='store_true',
                    help='swap in MD-repaired TN 1331 tables (NOT the default; see '
                         'build_pool docstring)')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--label', nargs='*', default=None)
    ap.add_argument('--source', nargs='*', default=None)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--report', nargs='?', const='all', default=None, choices=list(SCOPES))
    args = ap.parse_args()

    if args.build_pool:
        return build_pool(args.use_repaired)
    if args.report:
        return report(args.report)
    if not POOL_INDEX.exists():
        build_pool(args.use_repaired)

    pool = read_pool(args)
    suspects = load_suspects()
    print(f'scoring {len(pool)} state points with {args.workers} workers')
    metrics, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, (e, suspects)): e['label'] for e in pool}
        for n, fut in enumerate(as_completed(futs), 1):
            label, rows, err = fut.result()
            if err:
                failed.append((label, err))
                print(f'  [{n}/{len(pool)}] {label}: FAILED')
            else:
                metrics.extend(rows)
                print(f'  [{n}/{len(pool)}] {label}: {len(rows)} rows')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=METRIC_FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(metrics)
    print(f'\nwrote {METRICS_CSV} ({len(metrics)} rows)')
    if failed:
        print(f'\n{len(failed)} state points failed:')
        for label, err in failed:
            print(f'  {label}\n{err}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
