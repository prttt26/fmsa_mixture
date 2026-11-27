#!/usr/bin/env python3
"""Numbers of the SI subsection "Uncertainty of the core comparison": every interval is a cluster bootstrap.

Clusters are the series of the pool (source, sigma ratio, eps ratio, T*, x1); states of one series are not independent.
Reads numerical_notes/results/benchmark/all_samples/dcf_multistate_metrics.csv, clean sources (tn1331_dcf, md), inner core.
Core NRMSE per state = mean of the three pair NRMSE; chat(0) error = |chat_model - chat_bench| / |chat_bench| per (state, pair) row.
Writes numerical_notes/results/benchmark/all_samples/dcf_core_uncertainty_tables.md.

    python code/run.py code/fmsa_dcf/comparisons/dcf_core_uncertainty_tables.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

B, SEED = 5000, 20261125
rng = np.random.default_rng(SEED)
D = Path('numerical_notes/results/benchmark/all_samples')
M = {'OZ+HNCB': 'HNCB', 'OZ+MSA': 'MSA', 'FMSA-DP(py)': 'PYHS', 'FMSA-DP(fmt)': 'WB'}
ORDER = ['HNCB', 'MSA', 'PYHS', 'WB']

d = pd.read_csv(D / 'dcf_multistate_metrics.csv')
d = d[d.model.isin(M) & d.source.isin(['tn1331_dcf', 'md']) & (d.region == 'inner')].copy()
d['model'] = d.model.map(M)
d['err'] = (d.chat0_model - d.chat0_bench).abs() / d.chat0_bench.abs()
low = d[d.ref == 'lowk']
states = sorted(low.label.unique())
meta = low.drop_duplicates('label').set_index('label')
cl_state = meta.apply(lambda r: f'{r.source}|{r.sigma_ratio}|{r.eps_ratio}|{r.T_star}|{r.x1}', axis=1)


def boot(vals, clusters, fn=np.median):
    vals = np.asarray(vals, float)
    g = pd.Series(vals).groupby(np.asarray(clusters)).apply(lambda x: x.values).tolist()
    n = len(g)
    out = [fn(np.concatenate([g[i] for i in rng.integers(0, n, n)])) for _ in range(B)]
    return np.percentile(out, [2.5, 97.5])


def fmt(x, nd=4):
    return f'{x:+.{nd}f}'


def state_table(ref, states_):
    q = d[(d.ref == ref) & d.label.isin(states_)]
    return q.groupby(['label', 'model']).nrmse.mean().unstack()


def row_table(ref):
    q = d[(d.ref == ref) & d.label.isin(states)]
    return q.pivot_table(index=['label', 'pair'], columns='model', values='err')


def cl_of(index):
    labs = index.get_level_values(0) if isinstance(index, pd.MultiIndex) else index
    return cl_state.loc[labs].values


def summary(W, models=ORDER):
    rows = []
    for m in models:
        v = W[m].dropna()
        lo, hi = boot(v.values, cl_of(v.index))
        q1, q3 = np.percentile(v, [25, 75])
        rows.append(f'| {m} | {len(v)} | {v.median():.4f} | [{lo:.4f}, {hi:.4f}] | {q1:.3f}--{q3:.3f} |')
    return rows


def paired(W, a, b):
    dd = (W[a] - W[b]).dropna()
    lo, hi = boot(dd.values, cl_of(dd.index))
    pos = int((dd > 0).sum())
    return f'| {a} - {b} | {len(dd)} | {fmt(dd.median())} | [{fmt(lo)}, {fmt(hi)}] | {pos}/{len(dd)} |'


out = ['# Uncertainty of the core comparison (cluster bootstrap, %d resamples, seed %d)\n' % (B, SEED)]
T = state_table('lowk', states)
out += ['## Core NRMSE, repaired reference (per state: mean of the three pairs; HNCB on the states where it converged)\n',
        '| model | n | median | 95% interval | IQR |', '|---|---|---|---|---|'] + summary(T)
out += ['\n## Paired differences of the core NRMSE (positive: first model has the larger error), same states\n',
        '| first - second | n | median | 95% interval | first larger |', '|---|---|---|---|---|']
for a, b in [('PYHS', 'WB'), ('MSA', 'WB'), ('MSA', 'PYHS'), ('HNCB', 'WB'), ('HNCB', 'PYHS'), ('HNCB', 'MSA')]:
    out.append(paired(T, a, b))
out += ['\n## Reference treatment, same 519 states: PYHS - WB\n', '| reference | median | 95% interval |', '|---|---|---|']
for ref in ['published', 'lowk']:
    Tr = state_table(ref, states)
    dd = (Tr['PYHS'] - Tr['WB']).dropna()
    lo, hi = boot(dd.values, cl_of(dd.index))
    out.append(f'| {ref} | {fmt(dd.median())} | [{fmt(lo)}, {fmt(hi)}] |')
out += ['\n## By pair (repaired reference), PYHS - WB, per-state NRMSE of that pair\n', '| pair | median | 95% interval |', '|---|---|---|']
for pr in ['(1,1)', '(1,2)', '(2,2)']:
    q = d[(d.ref == 'lowk') & (d.pair == pr) & d.label.isin(states)].groupby(['label', 'model']).nrmse.mean().unstack()
    dd = (q['PYHS'] - q['WB']).dropna()
    lo, hi = boot(dd.values, cl_of(dd.index))
    out.append(f'| {pr} | {fmt(dd.median())} | [{fmt(lo)}, {fmt(hi)}] |')
out += ['\n## By source (repaired reference), PYHS - WB\n', '| source | n states | median | 95% interval |', '|---|---|---|---|']
for src in ['md', 'tn1331_dcf']:
    idx = [s for s in states if meta.loc[s, 'source'] == src]
    dd = (T.loc[idx, 'PYHS'] - T.loc[idx, 'WB']).dropna()
    lo, hi = boot(dd.values, cl_state.loc[dd.index].values)
    out.append(f'| {src} | {len(dd)} | {fmt(dd.median())} | [{fmt(lo)}, {fmt(hi)}] |')
for ref in ['lowk', 'published']:
    R = row_table(ref)
    out += [f'\n## chat(0) relative error over the core, {ref} reference (rows = state x pair)\n',
            '| model | rows | median | 95% interval | IQR |', '|---|---|---|---|---|'] + summary(R)
    out += ['\n| first - second | rows | median | 95% interval | first larger |', '|---|---|---|---|---|']
    for a, b in [('PYHS', 'WB'), ('MSA', 'WB'), ('HNCB', 'WB'), ('HNCB', 'PYHS'), ('HNCB', 'MSA')]:
        out.append(paired(R, a, b))
(D / 'dcf_core_uncertainty_tables.md').write_text('\n'.join(out) + '\n')
print('\n'.join(out))
