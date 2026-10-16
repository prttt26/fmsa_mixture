#!/usr/bin/env python3
"""Figure 7 panels (b)-(d): the pooled RDF benchmark.

Parallel to Figure 5, with the differences the RDF forces.

  (b) NRMSE against eta          } two measures each, not three regions
  (c) NRMSE against sigma ratio  }
  (d) first-peak height, model against benchmark

⚠ TWO DIFFERENCES FROM FIGURE 5, both properties of the observable.

1. There is no in-core / out-core split.  The RDF metrics carry no region
   column: g(r) is zero inside the core by construction, so there is nothing to
   score there.  What the RDF has instead is a WHOLE-RANGE nrmse and a
   first-shell nrmse_shell, and those are the two linestyles here.

2. There is one FMSA-DP, not two.  FMSA-DP's RDF has a single hard-sphere
   reference (Baxter/PY), because the Wiener-Hopf factorisation needs an
   analytic OZ solution and that exists only for PY -- see Figure 6's panels.md.

Panel (d) is the parity plot Figure 5 could not have on its pointwise side: the
first-peak height IS a scalar per (state, pair), and unlike the DCF's contact
value it lives where the models actually differ.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator

SRC = 'numerical_notes/results/benchmark_rdf/all_samples/rdf_multistate_metrics.csv'
OUT = 'numerical_notes/results/fmsa_dp_rdf_samples.png'
LEG = 'numerical_notes/results/fmsa_dp_rdf_samples_legend.png'
CLEAN = ['md', 'tn1331_rdf']
# MSA family first, HNCB last -- as Figure 5
# ONE palette -- `figure_style.py`; linestyle here carries whole-range against
# first-shell, the same documented exception as the DCF companion.
import figure_style as S                                        # noqa: E402
_MODELS = ('OZ+MSA', 'FMSA-DP', 'OZ+HNCB')
STYLE = {m: S.color(m) for m in _MODELS}
DISP = {'FMSA-DP': 'FMSA-DP(PY)'}
MK = {m: S.marker(m) for m in _MODELS}
LS = {'whole range': '-', 'first shell': '--'}
N_ETA_BINS = 8


def ticks3(ax):
    """At least three labelled y ticks; widen the range if 1-2-5 is not enough."""
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    # %g per VALUE, not ScalarFormatter.  ScalarFormatter derives one
    # precision for the whole axis from its interval, which on a log axis
    # rounded the 0.05 tick to "0.1" -- two ticks reading 0.1, one of them
    # off by 2x.  A duplicated tick label is worse than a missing one.
    f = FuncFormatter(lambda v, _p: f'{v:g}')
    ax.yaxis.set_major_formatter(f)
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    lo, hi = ax.get_ylim()
    for _ in range(4):
        if sum(1 for t in ax.yaxis.get_majorticklocs() if lo <= t <= hi) >= 3:
            break
        lo, hi = lo / 1.6, hi * 1.6
        ax.set_ylim(lo, hi)


def trend(ax, d, xcol):
    for meas, col_name in (('whole range', 'nrmse'), ('first shell', 'nrmse_shell')):
        for m, col in STYLE.items():
            q = d[(d.model == m) & d[col_name].notna()]
            if q.empty:
                continue
            if q[xcol].nunique() <= 6:                   # sigma ratio: categorical
                t = (q.groupby(xcol, observed=True)[col_name].median()
                      .rename('y').reset_index().rename(columns={xcol: 'x'}))
            else:
                b = pd.qcut(q[xcol], N_ETA_BINS, duplicates='drop')
                t = q.groupby(b, observed=True).agg(x=(xcol, 'median'),
                                                    y=(col_name, 'median'))
            ax.plot(t.x, t.y, ls=LS[meas], color=col, lw=1.4,
                    marker=MK[m], ms=3.4, mfc='none', mew=0.9)
    ticks3(ax)


def legend_strip():
    """Both shapes -- see figure_legend.py.

    Three panels here, so the block goes in the empty fourth slot of the 2-up
    grid and the strip is unused; that is a .tex choice, not one this script
    should make.
    """
    from matplotlib.lines import Line2D

    import figure_legend
    h = [Line2D([], [], color=c, marker=MK[m], ms=5, mfc='none', mew=1.0,
                ls='none', label=DISP.get(m, m)) for m, c in STYLE.items()]
    h += [Line2D([], [], color='0.35', ls=LS[k], lw=1.6, label=k) for k in LS]
    figure_legend.emit(h, [x.get_label() for x in h], 'fmsa_dp_rdf_samples',
                       out_dir=os.path.dirname(LEG), handlelength=2.2)


def main() -> None:
    d = pd.read_csv(SRC)
    d = d[d.source.isin(CLEAN)]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    trend(axes[0], d, 'eta')
    axes[0].set_xlabel(r'$\eta$'); axes[0].set_ylabel('NRMSE of $g_{ij}(r)$')
    axes[0].set_title(r'(b) NRMSE vs $\eta$')

    trend(axes[1], d, 'sigma_ratio')
    axes[1].set_xlabel(r'$\sigma_2/\sigma_1$'); axes[1].set_ylabel('NRMSE of $g_{ij}(r)$')
    axes[1].set_title(r'(c) NRMSE vs $\sigma$ ratio')

    ax = axes[2]
    q = d[d.peak_bench.notna()]
    for m, col in STYLE.items():
        s = q[q.model == m]
        ax.plot(s.peak_bench, s.peak_model, MK[m], ms=2.8, mfc='none',
                mec=col, mew=0.7, alpha=0.75)
    lim = [float(q.peak_bench.min()) * 0.95, float(q.peak_bench.max()) * 1.05]
    ax.plot(lim, lim, 'k-', lw=0.9, zorder=0)
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel('benchmark first-peak height')
    ax.set_ylabel('model first-peak height')
    ax.set_title('(d) first-peak height')

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print('  wrote', OUT)
    legend_strip()
    for m in STYLE:
        s = q[q.model == m]
        under = float(((s.peak_model - s.peak_bench) / s.peak_bench).median())
        print(f'  {DISP.get(m, m):16s} median peak bias {under*100:+5.1f}%')


if __name__ == '__main__':
    main()
