#!/usr/bin/env python3
"""The states at which OZ-HNCB does not converge, and the core NRMSE on three bases (SI: comparison on the HNCB-converged states).

Repaired reference, clean sources, inner core; core NRMSE per state = mean of the three pairs.
  own        every method on the states where it is available (HNCB 478, the others 519)
  common     the 478 states where all four are available
  worst      519 states with a failed OZ-HNCB solve counted as the worst result
Writes numerical_notes/results/benchmark/all_samples/hncb_converged_only.md.

    python code/run.py code/fmsa_dcf/comparisons/hncb_converged_only.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

D = Path('numerical_notes/results/benchmark/all_samples')
M = {'OZ+HNCB': 'HNCB', 'OZ+MSA': 'MSA', 'FMSA-DP(py)': 'PYHS', 'FMSA-DP(fmt)': 'WB'}
d = pd.read_csv(D / 'dcf_multistate_metrics.csv')
d = d[d.model.isin(M) & d.source.isin(['tn1331_dcf', 'md']) & (d.region == 'inner') & (d.ref == 'lowk')].copy()
d['model'] = d.model.map(M)
T = d.groupby(['label', 'model']).nrmse.mean().unstack()
meta = d.drop_duplicates('label').set_index('label')
fail = T.index[T.HNCB.isna()]
ok = T.index[T.HNCB.notna()]
out = [f'# States at which OZ-HNCB does not converge: {len(fail)} of {len(T)}\n']
m = meta.loc[T.index, ['source', 'eta', 'sigma_ratio']].copy()
m['fail'] = m.index.isin(fail)
m['ebin'] = pd.cut(m.eta, [0, 0.2, 0.3, 0.4, 0.5, 0.7])
out += ['## by eta (pool state variable, Lennard-Jones diameters)\n', '| eta | states | failed |', '|---|---|---|']
for k, g in m.groupby('ebin', observed=True):
    out.append(f'| {k} | {len(g)} | {int(g.fail.sum())} |')
out += ['\n## by source\n', '| source | states | failed |', '|---|---|---|']
for k, g in m.groupby('source'):
    out.append(f'| {k} | {len(g)} | {int(g.fail.sum())} |')
out += ['\n## median core NRMSE of the other methods on the failed and on the converged states\n', '| states | n | MSA | PYHS | WB |', '|---|---|---|---|---|']
for nm, idx in (('failed', fail), ('converged', ok)):
    out.append(f'| {nm} | {len(idx)} | ' + ' | '.join(f'{T.loc[idx, k].median():.3f}' for k in ['MSA', 'PYHS', 'WB']) + ' |')
out += ['\n## median core NRMSE on three bases\n', '| basis | HNCB | MSA | PYHS | WB |', '|---|---|---|---|---|']
own = {k: T[k].median() for k in M.values()}
com = {k: T.loc[ok, k].median() for k in M.values()}
worst = dict(own); worst['HNCB'] = np.median(T.HNCB.fillna(np.inf).values)
for nm, r in (('own states', own), ('478 common states', com), ('519 states, failure as worst', worst)):
    out.append(f'| {nm} | ' + ' | '.join(f'{r[k]:.4f}' for k in ['HNCB', 'MSA', 'PYHS', 'WB']) + ' |')
(D / 'hncb_converged_only.md').write_text('\n'.join(out) + '\n')
print('\n'.join(out))
