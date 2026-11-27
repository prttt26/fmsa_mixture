#!/usr/bin/env python3
"""Figure 5 panels (b)-(f): the pooled DCF benchmark, by what is being measured.

Panel (a) of Figure 5 is the scorecard, `dcf_multistate_overall_d`, written by
dcf_multistate_figures.py.  This script writes the rest.

  (b) chat_ij(0) parity -- FULL region, inner + outer summed, against the
      benchmark.  chat(0) is one scalar per (state, pair), so it plots directly.
  (c) chat(0) relative error against eta          } three region trends each:
  (d) chat(0) relative error against sigma ratio  } in-core, out-core, overall
  (e) pointwise NRMSE against eta                 }
  (f) pointwise NRMSE against sigma ratio         }

⚠ WHY (b) HAS NO POINTWISE TWIN.  c(r) is a function, not a scalar, so the
parity analogue would pool every r of every (state, pair) -- order 1e5 points,
which is a 2-D density plot rather than a scatter, and a different claim.  The
scalar that DOES transfer is the contact value, carried in the metrics as
contact_model/contact_bench; (e)-(f) use the NRMSE instead because that is what
the ranking is built on.

Encoding: colour = model, linestyle = region (solid overall, dashed in-core,
dotted out-core).  The overall curve is NOT the mean of the other two: it is
recomputed from the per-region rmse and point counts, since NRMSE is normalised
by the benchmark's own rms over the range it covers.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, LogFormatterSciNotation

SRC = ('numerical_notes/results/benchmark/all_samples/dcf_multistate_metrics.csv')
OUT = 'numerical_notes/results/fmsa_dp_dcf_samples.png'
REF = 'lowk'                      # the repaired benchmark, as Figure 4 uses
CLEAN = ['md', 'tn1331_dcf']
# MSA family together, HNCB last -- see dcf_multistate_figures.py
# ONE palette -- `figure_style.py`.  Colour and marker come from there; the
# LINESTYLE does not, because in this figure it carries the REGION and not the
# solver class.  That is the documented exception to the paper-wide rule: the key
# says so, and no panel here mixes the two meanings.
import figure_style as S                                        # noqa: E402
_MODELS = ('OZ+MSA', 'FMSA-DP(fmt)', 'FMSA-DP(py)', 'OZ+HNCB')
STYLE = {m: S.color(m) for m in _MODELS}
DISP = {m: S.disp(m) for m in ('FMSA-DP(fmt)', 'FMSA-DP(py)')}
LS = {'overall': '-', 'in-core': '--', 'out-core': ':'}
MK = {m: S.marker(m) for m in _MODELS}
# Redundant encoding: colour AND marker carry the model, linestyle carries the
# region.  Colour alone fails in greyscale and for the ~8% of readers with a
# colour vision deficiency, and four hues at 1.5 pt linewidth are hard to tell
# apart at panel size even in colour.
N_ETA_BINS = 8


def combined(d: pd.DataFrame) -> pd.DataFrame:
    """Full-region chat(0) and a properly recombined pointwise NRMSE.

    chat(0) is additive across the split at R_ij, so the full value is the sum.
    NRMSE is not: it is rmse/rms_bench over each region's own range, so the
    overall value is rebuilt from the squared errors and the point counts.
    """
    key = ['label', 'model', 'pair', 'eta', 'sigma_ratio']
    g = d.groupby(key, as_index=False).agg(
        chat0_model=('chat0_model', 'sum'), chat0_bench=('chat0_bench', 'sum'),
        sse=('rmse', lambda s: float(np.sum(np.square(s)))),
        n=('n_pts', 'sum'),
        ssb=('rms_bench', lambda s: float(np.sum(np.square(s)))))
    g['nrmse'] = np.sqrt(g.sse / 2.0) / np.sqrt(g.ssb / 2.0)
    return g


def trend(ax, d, xcol, ycol, logy=True):
    """Median of `ycol` against `xcol`, one line per (model, region).

    sigma_ratio is CATEGORICAL -- the grid samples exactly 1.5, 2.0, 2.5, 3.0 --
    so it is grouped on its own values.  Quantile-binning it merged neighbouring
    ratios and drew three points where the data has four, which read as a coarse
    trend when it was really a binning artefact.  eta has 65 distinct values and
    is binned properly.
    """
    for reg, sub in d.groupby('region'):
        for m, col in STYLE.items():
            q = sub[sub.model == m]
            if q.empty:
                continue
            if q[xcol].nunique() <= 6:                      # categorical axis
                t = (q.groupby(xcol, observed=True)[ycol].median()
                      .rename('y').reset_index().rename(columns={xcol: 'x'}))
            else:
                b = pd.qcut(q[xcol], N_ETA_BINS, duplicates='drop')
                t = q.groupby(b, observed=True).agg(x=(xcol, 'median'),
                                                    y=(ycol, 'median'))
            ax.plot(t.x, t.y, ls=LS[reg], color=col, lw=1.4,
                    marker=MK[m], ms=3.4, mfc='none', mew=0.9)
    if logy:
        ax.set_yscale('log')
        # At least THREE labelled ticks.  Over a narrow range (say 0.05-0.3) the
        # decade locator labels only 0.1, so 1-2-5 subdivisions are requested and
        # the range is widened if that still is not enough.
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto'))
        # %g per VALUE, not ScalarFormatter.  ScalarFormatter derives one
        # precision for the whole axis from its interval, which on a log axis
        # rounded the 0.05 tick to "0.1" -- two ticks reading 0.1, one of them
        # off by 2x.  A duplicated tick label is worse than a missing one.
        f = FuncFormatter(lambda v, _p: f'{v:g}')
        ax.yaxis.set_major_formatter(f)
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        lo, hi = ax.get_ylim()
        for _ in range(4):
            n = sum(1 for t in ax.yaxis.get_majorticklocs() if lo <= t <= hi)
            if n >= 3:
                break
            lo, hi = lo / 1.6, hi * 1.6
            ax.set_ylim(lo, hi)


def legend_strip(path):
    """Shared key for (b)-(f): model on one row, region on the next.

    The panels carry no legend of their own -- five of them would show the same
    twelve-entry key five times.
    """
    from matplotlib.lines import Line2D

    import figure_legend
    h = [Line2D([], [], color=c, marker=MK[m], ms=5, mfc='none', mew=1.0,
                ls='none', label=DISP.get(m, m)) for m, c in STYLE.items()]
    h += [Line2D([], [], color='0.35', ls=LS[r], lw=1.6, label=r)
          for r in ('overall', 'in-core', 'out-core')]
    # Five panels, so the BLOCK takes the sixth slot of a 2x3 grid.  Two columns
    # -- models left, regions right -- keeps seven entries readable at panel
    # width.  The strip is written too, for a layout with six panels.
    figure_legend.emit(h, [x.get_label() for x in h], 'fmsa_dp_dcf_samples',
                       out_dir=os.path.dirname(path), block_ncol=2,
                       handlelength=2.0)


def main() -> None:
    d = pd.read_csv(SRC)
    d = d[(d.ref == REF) & d.source.isin(CLEAN)]
    full = combined(d)
    full['region'] = 'overall'
    per = d.rename(columns={'region': 'r0'}).copy()
    per['region'] = per.r0.map({'inner': 'in-core', 'outer': 'out-core'})
    per['relerr'] = (per.chat0_model - per.chat0_bench).abs() / per.chat0_bench.abs()
    full['relerr'] = (full.chat0_model - full.chat0_bench).abs() / full.chat0_bench.abs()
    allr = pd.concat([per[['model', 'pair', 'eta', 'sigma_ratio', 'region',
                           'relerr', 'nrmse']],
                      full[['model', 'pair', 'eta', 'sigma_ratio', 'region',
                            'relerr', 'nrmse']]], ignore_index=True)

    # 3.1, down from 4.3 then 3.6: five panels plus the key make three rows at
    # 2-up, and each shortening paid for a specific overflow -- 93 pt for the
    # original aspect, then 64 pt more once panel labels moved above the panels
    # and cost a line per row.
    fig, axes = plt.subplots(1, 5, figsize=(24.0, 3.1))

    # (b) parity
    ax = axes[0]
    for m, col in STYLE.items():
        q = full[full.model == m]
        ax.plot(q.chat0_bench, q.chat0_model, MK[m], ms=2.6, mfc='none',
                mec=col, mew=0.7, alpha=0.75)
    # SYMLOG on both axes.  chat(0) changes sign and spans seven decades: the
    # median |chat(0)| is 28 and the 99th percentile 2.2e3, but 16 of 6240 rows
    # exceed 1e4 -- all at x1 = 0.9, where chat_22(0) = (1 - M_22)/rho_2 divides
    # by a vanishing rho_2.  On linear axes that handful sets the range and the
    # other 6224 points collapse onto the origin.
    lim = float(np.nanmax(np.abs(np.r_[full.chat0_bench, full.chat0_model]))) * 1.3
    ax.plot([-lim, lim], [-lim, lim], 'k-', lw=0.9, zorder=0)
    ax.set_xscale('symlog', linthresh=1.0)
    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    # Every THIRD decade.  Symlog over seven decades either side puts ~15 labels
    # on each axis by default, which is denser than the information in them.
    ticks = [-1e6, -1e3, -1, 0, 1, 1e3, 1e6]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(LogFormatterSciNotation(linthresh=1.0))
    ax.set_xlabel(r'benchmark $\hat c_{ij}(0)$')
    ax.set_ylabel(r'model $\hat c_{ij}(0)$')
    ax.set_title(r'(b) $\hat c(0)$, full region (symlog)')

    for ax, xcol, ycol, ttl in (
            (axes[1], 'eta', 'relerr', r'(c) $\hat c(0)$ error vs $\eta$'),
            (axes[2], 'sigma_ratio', 'relerr', r'(d) $\hat c(0)$ error vs $\sigma$ ratio'),
            (axes[3], 'eta', 'nrmse', r'(e) pointwise NRMSE vs $\eta$'),
            (axes[4], 'sigma_ratio', 'nrmse', r'(f) pointwise NRMSE vs $\sigma$ ratio')):
        trend(ax, allr, xcol, ycol)
        ax.set_xlabel(r'$\eta$' if xcol == 'eta' else r'$\sigma_2/\sigma_1$')
        ax.set_ylabel('relative error' if ycol == 'relerr' else 'NRMSE')
        ax.set_title(ttl)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    legend_strip('numerical_notes/results/fmsa_dp_dcf_samples_legend.png')
    print('  wrote', OUT)
    print(f'  states {full.label.nunique() if "label" in full else "-"}, '
          f'rows {len(full)}')


if __name__ == '__main__':
    main()
