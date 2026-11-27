#!/usr/bin/env python3
"""Tables and the state-dependence figure for the FMSA-vs-exact-MSA truncation error.

Reads numerical_notes/results/truncation/truncation_{pure,binary}{,_crit}.csv (truncation_state_dependence.py)
and truncation_critical_temperature.csv (truncation_critical_temperature.py).

Distance to the critical region is measured two ways:
  a_exact   = 1 - rho sum x_i x_j chat_ij(0) of the EXACT MSA (= 1/S(0); the spinodal is a = 0, ideal gas a = 1)
  T / Tc    with Tc the exact-MSA critical temperature of the same system
Errors are the worst pair at each state:
  chat0     |chat_F(0) - chat_E(0)| / |chat_E(0)|
  res_tot   L2 of (c_E - c_F) over the core / L2 of c_E over the core  (error relative to the TOTAL core DCF)
  res_c1    the same residual / L2 of the first-order correction c1 = c_F - c_HS
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

R = Path('numerical_notes/results/truncation')
tc = pd.read_csv(R / 'truncation_critical_temperature.csv').set_index('system')
PAIRS = {'pure': ['11'], 'binary': ['11', '12', '22']}


def load(system):
    d = pd.concat([pd.read_csv(R / f'truncation_{system}.csv'), pd.read_csv(R / f'truncation_{system}_crit.csv')],
                  ignore_index=True).drop_duplicates(['T', 'rho'])
    d = d[(d.ok == 1) & (d.a_exact > 0)].copy()
    P = PAIRS[system]
    d['chat0'] = d[[f'chat0_rel_{p}' for p in P]].max(axis=1)
    d['res_tot'] = d[[f'res_tot_{p}' for p in P]].max(axis=1)
    d['res_c1'] = d[[f'res_c1_{p}' for p in P]].max(axis=1)
    d['T_over_Tc'] = d['T'] / tc.loc[system, 'Tc_exact']
    d['system'] = system
    return d


def main():
    D = {s: load(s) for s in PAIRS}
    both = pd.concat(D.values(), ignore_index=True)
    both.to_csv(R / 'truncation_states_used.csv', index=False)
    lines = ['| system | T_c exact MSA | T_c FMSA | FMSA / exact - 1 |', '|---|---|---|---|']
    for s in PAIRS:
        r = tc.loc[s]
        lines.append(f'| {s} | {r.Tc_exact:.4f} | {r.Tc_fmsa:.4f} | {100 * (r.Tc_fmsa / r.Tc_exact - 1):+.1f}% |')
    out = ['## Critical temperatures (a_min -> 0)\n'] + lines
    for var, edges, lab in (('a_exact', [0, 0.1, 0.3, 1, 3, 10, 1e9], 'a_exact (=1/S(0))'),
                            ('T_over_Tc', [0, 1.0, 1.1, 1.25, 1.5, 2, 3, 99], 'T/Tc(exact)')):
        out.append(f'\n## Errors by {lab} (median / max over the states in the bin; worst pair at each state)\n')
        out.append(f'| system | {lab} bin | n | chat0 rel err | core resid / total core DCF | core resid / first-order correction |')
        out.append('|---|---|---|---|---|---|')
        for s, d in D.items():
            b = pd.cut(d[var], edges, include_lowest=True)
            for iv, g in d.groupby(b, observed=True):
                if len(g) < 3:
                    continue
                f = lambda c: f'{100 * g[c].median():.2f}% / {100 * g[c].max():.2f}%'
                out.append(f'| {s} | {iv} | {len(g)} | {f("chat0")} | {f("res_tot")} | {f("res_c1")} |')
    # the reference binary state
    ref = D['binary'].iloc[((D['binary']['T'] - 1.0).abs() + (D['binary'].rho - 0.139).abs() * 10).argsort()[:1]]
    out.append('\n## State nearest the SI reference binary (T*=1, rho*=0.139) in the grid\n')
    out.append(ref[['T', 'rho', 'eta', 'a_exact', 'T_over_Tc', 'chat0', 'res_tot', 'res_c1']].round(4).to_string(index=False))
    (R / 'truncation_tables.md').write_text('\n'.join(out) + '\n')
    print('\n'.join(out))

    # figure: two rows (distance measure) x three errors
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    cols = {'pure': '#1f77b4', 'binary': '#d62728'}
    mk = {'pure': 'o', 'binary': 's'}
    labs = [('chat0', r'$\hat c(0)$ rel. error (worst pair)'),
            ('res_tot', 'core residual / total core DCF'),
            ('res_c1', 'core residual / first-order correction')]
    for r_, (xv, xl, logx) in enumerate((('a_exact', r'$a=1/S(0)$ of the exact MSA (0 = spinodal, 1 = ideal gas)', True),
                                          ('T_over_Tc', r'$T/T_c$ (exact MSA)', False))):
        for c_, (yv, yl) in enumerate(labs):
            ax = axes[r_, c_]
            for s, d in D.items():
                ax.scatter(d[xv], 100 * d[yv], s=9, color=cols[s], marker=mk[s], alpha=0.55, label=s, linewidths=0)
            ax.set_yscale('log')
            if logx:
                ax.set_xscale('log')
            else:
                ax.set_xlim(0.9, 3.2)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl + ' (%)')
            if r_ == 0 and c_ == 0:
                ax.legend(frameon=False, title='system')
    # reference binary state
    rs = D['binary'].iloc[((D['binary']['T'] - 1.0).abs() + (D['binary'].rho - 0.139).abs() * 10).argsort()[:1]]
    for r_, xv in enumerate(('a_exact', 'T_over_Tc')):
        for c_, (yv, _) in enumerate(labs):
            axes[r_, c_].scatter(rs[xv], 100 * rs[yv], s=70, facecolors='none', edgecolors='k', linewidths=1.4, zorder=5)
    fig.tight_layout()
    fig.savefig(R / 'truncation_state_dependence.png', dpi=150)
    fig.savefig(R / 'truncation_state_dependence.pdf')
    print('figure ->', R / 'truncation_state_dependence.png')


if __name__ == '__main__':
    main()
