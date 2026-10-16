#!/usr/bin/env python3
"""Figures for the multi-state DCF comparison.

Reads the metric tables written by compare_models_dcf_multistate.py and, for the
curve figure, re-solves two representative state points.

    python dcf_multistate_figures.py --scope all          # every source pooled
    python dcf_multistate_figures.py --scope literature   # TN 1331 / Yu & Jin only
    python dcf_multistate_figures.py --scope md           # our own LAMMPS MD only

The scope picks the sources, the output directory and the file prefix, so one
scoring run feeds all three write-ups under numerical_notes/results/benchmark/:

    all         -> all_samples/dcf_multistate_*.png
    literature  -> literature/lit_dcf_*.png
    md          -> LAMMPS_MD/md_dcf_*.png

Conventions follow the existing DCF figures in this repo: the benchmark is open
black circles, the analytic FMSA-DP curves are solid, the iterative OZ solvers
are dashed, R_ij is a grey dotted vertical.

    <prefix>_overall.png         the scorecard: every measure, all pairs averaged
    <prefix>_pointwise.png       NRMSE against packing fraction
    <prefix>_chat0.png           chat_ij(0) parity, inner and outer
    <prefix>_dual_reference.png  the two benchmark variants against eta
    <prefix>_sigma_ratio.png     error against size ratio (scopes including MD)
    <prefix>_curves.png          c_ij(r) at a clean and an artefacted state
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'data_benchmark'))

import compare_models_dcf_multistate as M            # noqa: E402
from convert import read_table                       # noqa: E402

BENCH = REPO / 'numerical_notes' / 'results' / 'benchmark'
DATA = BENCH / 'all_samples'          # the metric tables live with the pooled write-up

# One scoring run, three write-ups.  Each scope draws only the sources its
# write-up is about, into that write-up's directory, with its own file prefix.
SCOPE_DIR = {'all': BENCH / 'all_samples',
             'literature': BENCH / 'literature',
             'md': BENCH / 'LAMMPS_MD'}
SCOPE_PREFIX = {'all': 'dcf_multistate', 'literature': 'lit_dcf', 'md': 'md_dcf'}

# Which benchmark variant the figures are drawn against.  'published' is the
# benchmark as it stands; 'lowk' is the long-wavelength repair of
# data_benchmark/dcf_lowk.py applied as THE reference rather than as a check.
# ⚠ Under 'lowk' the OUTER panels are largely not a measurement: the repair
# imposes c -> -beta*u beyond the keep radius, which is exactly the tail OZ+MSA
# and FMSA-DP use by construction (median 90 % of the outer window on pair
# (1,2), 100 % on (2,2)).  Inner panels and chat(0) carry no such circularity.
REF = 'published'

# Which view each state gets a row of.  0 full, 1 outer zoom, 2 inner core.
# The inner core (r < R_ij) is where the benchmark's inner NRMSE lives and where
# the models actually differ; the outer row mostly shows OZ+MSA and FMSA-DP
# coinciding, since they share the tail -beta*u by construction.  Default is
# unchanged so the published composite keeps its shape.
ZOOMS = (0, 1)

SCOPE_SRC_NAME = {'tn1331_dcf': 'TN 1331 tabulated DCF',
                  'tn1331_rdf': 'TN 1331 RDF, inverted here',
                  'md': 'our LAMMPS MD',
                  'yujin_fig13': 'Yu & Jin Fig. 13'}

SCOPE = 'all'
OUT = SCOPE_DIR[SCOPE]
PREFIX = SCOPE_PREFIX[SCOPE]
CLEAN = ['tn1331_dcf', 'md']


def set_scope(scope: str) -> None:
    global SCOPE, OUT, PREFIX, CLEAN
    SCOPE, OUT, PREFIX = scope, SCOPE_DIR[scope], SCOPE_PREFIX[scope]
    CLEAN = M.SCOPES[scope]['clean']


def clean_names() -> str:
    return ' + '.join(SCOPE_SRC_NAME[s] for s in CLEAN)


PAIR_ORDER = ['(1,1)', '(1,2)', '(2,2)']

# Order is the MSA family first -- OZ+MSA and both FMSA-DP variants share the
# closure, and FMSA-DP is its first-order truncation -- then OZ+HNCB, which is a
# different closure brought in as an instrument.  Grouping them puts the
# comparison the construction is answerable to side by side, instead of
# separating it with an unrelated method.
# ⭐ ONE palette for the whole paper -- see `figure_style.py`.  This script used
# to own its own, and `#7a4fbf` here was one or two units per channel away from
# the `#7b3fa0` the speed and phase figures give to EXACT MSA: on paper they are
# the same purple naming two different methods.  FMSA-DP's two hard-sphere
# references are now two lightnesses of one orange, which is what they are.
import figure_style as S                                        # noqa: E402
STYLE = S.styles('OZ+MSA', 'FMSA-DP(fmt)', 'FMSA-DP(py)', 'OZ+HNCB')
SRC_MARK = {'tn1331_dcf': 'o', 'md': 's', 'tn1331_rdf': 'x'}

# Curve panels only.  Both FMSA-DP variants are kept: they differ only in the
# hard-sphere reference, but that difference is visible in the core and is part
# of what the panel is for -- see the PY-vs-WBFMT gap of ~0.75 there.  The labels
# name the reference each one actually uses, matching the paper's acronyms.
CURVE_HIDE: set[str] = set()
CURVE_RENAME = {m: S.disp(m) for m in STYLE}
CURVE_LEGEND = None     # which column carries it; None = none, use the strip
disp = lambda m: CURVE_RENAME.get(m, m)   # display name, used by every figure
                        # Figure 4 shares one legend under the (b)-(d) grid,
                        # so no panel carries its own -- a per-panel legend
                        # plus the strip would show the same key twice.

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 11,
                     'legend.fontsize': 9, 'figure.titlesize': 13})


def load():
    d = pd.read_csv(DATA / 'dcf_multistate_metrics.csv')
    d['eta'] = d.eta.astype(float)
    s = pd.read_csv(DATA / 'dcf_multistate_scalars.csv')
    s['eta'] = s.eta.astype(float)
    s['chi_ratio'] = pd.to_numeric(s.chi_ratio, errors='coerce')
    return d, s


KEEP = ['label', 'source', 'sigma_ratio', 'eta', 'outer_trust', 'ref', 'model',
        'region', 'T_star', 'eps_ratio', 'x1']    # all constant per state point


def combined(d, values='nrmse'):
    """Collapse the three pairs into one 'all pairs' number per state point.

    A plain mean over (1,1), (1,2), (2,2) -- the same aggregation as the `mean`
    column of the tables in the write-up, so the figures and the tables say the
    same thing.  Returned with the same columns as `d` and pair = 'all pairs'.
    """
    g = (d.groupby(KEEP, as_index=False)[values].mean())
    g['pair'] = 'all pairs'
    return g


def with_combined(d, values='nrmse'):
    cols = KEEP + ['pair', values]
    return pd.concat([d[cols], combined(d, values)[cols]], ignore_index=True)


# --------------------------------------------------------------------------
def fig_pointwise(d):
    d = with_combined(d[d.source.isin(CLEAN) & (d.ref == REF)])
    cols = PAIR_ORDER + ['all pairs']
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 8.2), sharex=True, sharey='row')
    fig.suptitle('Pointwise DCF error against packing fraction — clean benchmarks '
                 f'({clean_names()}).  Last column is the three pairs '
                 'averaged.\n'
                 'outer row: OZ+MSA and FMSA-DP share the tail $-\\beta u$, so their '
                 'points coincide exactly', y=0.99)
    for col, pair in enumerate(cols):
        for row, region in enumerate(('inner', 'outer')):
            ax = axes[row, col]
            q = d[(d.region == region) & (d.pair == pair)]
            if region == 'outer':
                q = q[q.outer_trust == 1]
            # In the outer region OZ+MSA and both FMSA-DP variants are the same
            # curve; drawing three coincident series just hides two of them.
            shown = (list(STYLE) if region == 'inner'
                     else ['OZ+HNCB', 'FMSA-DP(py)'])
            for model in shown:
                color, ls, _ = STYLE[model]
                name = (model if region == 'inner'
                        else ('OZ+HNCB' if model == 'OZ+HNCB'
                              else r'OZ+MSA $\equiv$ FMSA-DP'))
                qq = q[q.model == model]
                for src, mk in SRC_MARK.items():
                    z = qq[qq.source == src]
                    if not len(z):
                        continue
                    ax.scatter(z.eta, z.nrmse, s=26, c=color, marker=mk,
                               alpha=0.75, linewidths=0.6,
                               edgecolors='none' if mk != 'x' else color,
                               label=name if (src == CLEAN[0]
                                              and col == 0) else None)
            if pair == 'all pairs':          # median trend, to read the ranking off
                for model in shown:          # binned as in the write-up's tables
                    color, _, _ = STYLE[model]
                    z = q[q.model == model]
                    g = z.groupby(pd.cut(z.eta, M.ETA_BINS), observed=True).agg(
                        eta=('eta', 'median'), nrmse=('nrmse', 'median'))
                    ax.plot(g.eta, g.nrmse, color=color, lw=2.4, zorder=6,
                            solid_capstyle='round')
            ax.set_yscale('log')
            if row == 0:
                ax.set_title('all pairs (mean)' if pair == 'all pairs'
                             else f'pair {pair}',
                             fontweight='bold' if pair == 'all pairs' else 'normal')
            if row == 1:
                ax.set_xlabel(r'$\eta$')
            if col == 0:
                ax.set_ylabel(f'NRMSE  [{ "inner core  $r<R_{ij}$" if row == 0 else "outer  $r>R_{ij}$"}]')
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + [f'{x} (outer)' for x in l1], loc='lower center',
               ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.005))
    marks = {'tn1331_dcf': 'circles: TN 1331 tabulated DCF', 'md': 'squares: our MD'}
    fig.text(0.5, 0.028, '     '.join(marks[s] for s in CLEAN) + '     '
             'only benchmarks that pass the tail-scale gate '
             r'($s_{ij}=c_{ij}/(-\beta\varphi_{ij})\in[0.5,2]$) are scored in the '
             'outer row', ha='center', fontsize=9, color='#444')
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    p = OUT / f'{PREFIX}_pointwise.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


def fig_chat0(d):
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.6))
    fig.suptitle(r'Compressibility route: $\hat c_{ij}(0)$ over the benchmark $r$ '
                 'range, model against benchmark', y=0.98)
    for col, pair in enumerate(PAIR_ORDER):
        for row, region in enumerate(('inner', 'outer')):
            ax = axes[row, col]
            ref = 'published' if row == 0 else ('lowk' if REF == 'lowk' else 'c22_mfphi')
            q = d[(d.ref == ref) & (d.region == region) & (d.pair == pair)
                  & d.source.isin(CLEAN)]
            if region == 'outer':
                q = q[q.outer_trust == 1]
            lo = hi = None
            for model, (color, ls, mk) in STYLE.items():
                z = q[q.model == model]
                if not len(z):
                    continue
                ax.scatter(z.chat0_bench, z.chat0_model, s=26, c=color, marker=mk,
                           alpha=0.75, linewidths=0,
                           label=disp(model) if (row == 0 and col == 0) else None)
                v = np.concatenate([z.chat0_bench.values, z.chat0_model.values])
                lo = np.nanmin(v) if lo is None else min(lo, np.nanmin(v))
                hi = np.nanmax(v) if hi is None else max(hi, np.nanmax(v))
            if lo is not None:
                pad = 0.05 * (hi - lo)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='k',
                        lw=0.8, ls='-', zorder=0)
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f'pair {pair} — {region}'
                         + ('' if row == 0 else '  (ref: $c_{22}$ outer $\\to-\\beta\\varphi$)'))
            ax.set_xlabel(r'benchmark $\hat c_{ij}(0)$')
            if col == 0:
                ax.set_ylabel(r'model $\hat c_{ij}(0)$')
    h, lb = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lb, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    p = OUT / f'{PREFIX}_chat0.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


def _smooth_trend(x, y, n_out: int = 200, frac: float = 0.30, log: bool = True):
    """A smooth trend through scattered (x, y), spanning the data's own range.

    Local linear regression with tricube weights (a LOWESS, done by hand so the
    figure does not acquire a statsmodels dependency).  `frac` is the bandwidth
    as a fraction of the x range.  Fitted in log y when the quantity spans a
    decade, which chi_T/chi_id does, so the curve is smooth in the eye's metric
    rather than dominated by the largest values.

    Returns a dense (x, y) pair -- no nodes, no markers, and it starts and ends
    at the smallest and largest x that were actually measured.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    if log:
        ok &= (np.asarray(y) > 0)
    x, y = np.asarray(x, float)[ok], np.asarray(y, float)[ok]
    if x.size < 4 or x.max() <= x.min():
        return np.array([]), np.array([])
    yy = np.log(y) if log else y
    xs = np.linspace(x.min(), x.max(), n_out)
    half = max(frac * (x.max() - x.min()), 1e-9)
    out = np.full(n_out, np.nan)
    for q, x0 in enumerate(xs):
        w = np.clip(1.0 - (np.abs(x - x0) / half) ** 3, 0.0, None) ** 3
        if (w > 0).sum() < 3:
            continue
        X = np.column_stack([np.ones_like(x), x - x0])
        A = X.T @ (w[:, None] * X)
        b = X.T @ (w * yy)
        try:
            out[q] = np.linalg.solve(A, b)[0]
        except np.linalg.LinAlgError:
            pass
    return xs, (np.exp(out) if log else out)


def fig_dual_reference(s, src=None):
    src = src or CLEAN[0]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    fig.suptitle(r'The two benchmark variants, and where they stop agreeing', y=1.0)
    # a benchmark whose own chat(0) gives a negative compressibility is not a
    # benchmark -- all three such tables are SPARSE OCR casualties
    bad = set(s[(s.model == 'benchmark') & (s.chi_ratio < 0)].label)
    s = s[~s.label.isin(bad)]
    b = s[(s.model == 'benchmark') & (s.source == src)]
    pub = b[b.ref == REF].set_index('label')
    mf = b[b.ref == ('lowk' if REF == 'lowk' else 'c22_mfphi')].set_index('label')
    common = pub.index.intersection(mf.index)

    ax = axes[0]
    ax.scatter(pub.loc[common, 'eta'], pub.loc[common, 'chi_ratio'], s=34,
               facecolors='none', edgecolors='k', label='benchmark, as published')
    ax.scatter(mf.loc[common, 'eta'], mf.loc[common, 'chi_ratio'], s=30, c='#1baf7a',
               marker='D', label=r'benchmark, $c_{22}$ outer $\to-\beta\varphi$')
    # The models are the busy part of this panel and the eye cannot follow four
    # interleaved scatters, so each gets a smooth local-regression trend across
    # the whole eta range that was measured.  Every raw point stays, faintly:
    # the line is an aid, not a replacement, and the scatter is what carries the
    # state-to-state spread.  The benchmark keeps bare points -- it is the
    # reference, and the question here is where its two variants separate, which
    # a line would smooth over.
    for model, (color, ls, mk) in STYLE.items():
        z = s[(s.model == model) & (s.source == src) & (s.ref == REF)]
        ax.scatter(z.eta, z.chi_ratio, s=20, c=color, marker=mk, alpha=0.28,
                   linewidths=0)
        e, c = _smooth_trend(z.eta.values, z.chi_ratio.values)
        if e.size:
            ax.plot(e, c, color=color, ls=ls, lw=2.2, zorder=6, label=disp(model),
                    solid_capstyle='round')
        else:                      # too few points to trend: keep it in the legend
            ax.scatter([], [], s=22, c=color, marker=mk, label=disp(model))
    ax.axvline(0.25, color='gray', lw=0.8, ls=':')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\chi_T/\chi_{id}=[1-\rho\sum x_ix_j\hat c_{ij}(0)]^{-1}$')
    ax.set_title(f'compressibility, {SCOPE_SRC_NAME[src]} state points')
    ax.legend(fontsize=8, loc='upper right')

    ax = axes[1]
    ratio = (pub.loc[common, 'chi_ratio'] / mf.loc[common, 'chi_ratio'])
    ax.scatter(pub.loc[common, 'eta'], ratio, s=38, c='#c23b22')
    ax.axhline(1.0, color='k', lw=0.9)
    ax.axvline(0.25, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\chi_T^{\rm published}\ /\ \chi_T^{c_{22}\to-\beta\varphi}$')
    ax.set_title('the correction does nothing below $\\eta\\approx0.25$\n'
                 'and reaches a factor 1.8 by $\\eta\\approx0.46$')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = OUT / f'{PREFIX}_dual_reference.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


def fig_sigma_ratio(d):
    q = with_combined(d[(d.source == 'md') & (d.ref == REF)])
    q = q[q.region == 'inner']
    cols = PAIR_ORDER + ['all pairs']
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.6), sharey=True)
    fig.suptitle('Size-ratio dependence — our own MD, the only source that varies '
                 r'$\sigma_{22}/\sigma_{11}$.  Inner core; lines are medians, faint '
                 'points the individual state points.  Last panel is the three pairs '
                 'averaged.'
                 '\nThe degradation lives almost entirely in the unlike pair, so the '
                 'pair-averaged view understates it — and OZ+HNCB is the only model '
                 'that holds it.', y=1.04)
    for col, pair in enumerate(cols):
        ax = axes[col]
        for model, (color, ls, mk) in STYLE.items():
            z = q[(q.model == model) & (q.pair == pair)]
            g = z.groupby('sigma_ratio').nrmse.median()
            ax.plot(g.index, g.values, color=color, ls=ls, marker=mk, ms=6, lw=1.8,
                    label=disp(model) if col == 0 else None)
            ax.scatter(z.sigma_ratio, z.nrmse, s=12, c=color, alpha=0.28, linewidths=0)
        ax.set_xlabel(r'$\sigma_{22}/\sigma_{11}$')
        ax.set_title('all pairs (mean)' if pair == 'all pairs' else f'pair {pair}',
                     fontweight='bold' if pair == 'all pairs' else 'normal')
        ax.set_yscale('log')
    axes[0].set_ylabel('NRMSE, inner core')
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    p = OUT / f'{PREFIX}_sigma_ratio.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


def _scorecard(ax, clean, legend=True):
    """Panel (d) of the overall composite, and Fig. 9(a) of the paper.

    Factored out so it can ALSO be drawn on its own: cropped from the 2x2
    composite at `--pub-cols 2` it comes out 2.1 in wide against a 3.18 in
    slot, because the bar axes under-fill their quarter of the grid, and
    ``includegraphics`` would then magnify its 9 pt text to 13.7 pt beside a
    panel drawn at the right size.  `--scorecard` renders it as a one-panel
    figure so the print size is set here and not by LaTeX.
    """
    rows = []
    for region in ('inner', 'outer'):
        q = clean[(clean.ref == REF) & (clean.region == region)]
        if region == 'outer':
            q = q[q.outer_trust == 1]
        c = combined(q)
        for model in STYLE:
            rows.append((f'pointwise NRMSE\n({region})', model,
                         float(c[c.model == model].nrmse.median())))
        ce = combined(q.assign(err=(q.chat0_model - q.chat0_bench).abs()
                               / q.chat0_bench.abs()), values='err')
        for model in STYLE:
            rows.append((f'$\\hat c(0)$ rel. error\n({region})', model,
                         float(ce[ce.model == model].err.median())))
    q = clean[(clean.ref == REF) & (clean.region == 'outer')
              & (clean.outer_trust == 1)]
    cp = combined(q.assign(err=(q.contact_model - q.contact_bench).abs()
                           / q.contact_bench.abs()), values='err')
    for model in STYLE:
        rows.append(('contact peak\nrel. error', model,
                     float(cp[cp.model == model].err.median())))

    df = pd.DataFrame(rows, columns=['metric', 'model', 'value'])
    metrics = list(dict.fromkeys(df.metric))
    y = np.arange(len(metrics))
    h = 0.19
    for n, model in enumerate(STYLE):
        color = STYLE[model][0]
        v = [float(df[(df.metric == m) & (df.model == model)].value.iloc[0])
             for m in metrics]
        # No `bar_label`.  Four models x five measures is twenty numbers at 6.5 pt
        # printed inside a half-page panel, and they compete with the thing a bar
        # chart is for -- comparing LENGTHS.  The values that carry an argument are
        # quoted in the text and in Table VI; the panel ranks.
        ax.barh(y + (n - 1.5) * h, v, height=h, color=color, label=disp(model),
                edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.invert_yaxis()
    # linear, deliberately: on a log axis the bar LENGTH is not the value, which
    # is exactly the comparison a bar chart is for.
    ax.set_xlim(0, 1.04 * df.value.max())   # headroom was for the bar labels
    # '(all pairs)' moved to the caption: at 2-up width the full sentence is
    # wider than the axes and the tight-bbox crop ate 'better'.
    ax.set_xlabel('median relative error\n(shorter is better)')
    ax.set_title('scorecard: every measure is a dimensionless\nrelative error, '
                 'so one axis is honest')
    if legend:
        ax.legend(fontsize=8, loc='best')  # 'best' keeps it off the longest bars


def fig_overall(d):
    """The three pairs collapsed into one number — how the models rank overall.

    Every quantity on the scorecard is a dimensionless relative error, so they
    share one axis honestly.
    """
    clean = d[d.source.isin(CLEAN)]
    # 7.8 not 9.6: panel (d) of this composite is the paper's scorecard, and
    # stacked with its RDF twin at full width the pair overran \textheight.
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 7.8))
    fig.suptitle('Overall model performance — the three pairs averaged', y=0.98)

    # --- (0,0) and (0,1): overall NRMSE against eta, per region ---------------
    for ax, region, ttl in ((axes[0, 0], 'inner', 'inner core  $r<R_{ij}$'),
                            (axes[0, 1], 'outer', 'outer  $r>R_{ij}$')):
        q = combined(clean[(clean.ref == REF) & (clean.region == region)])
        if region == 'outer':
            q = q[q.outer_trust == 1]
        shown = list(STYLE) if region == 'inner' else ['OZ+HNCB', 'FMSA-DP(py)']
        for model in shown:
            color, _, mk = STYLE[model]
            name = (model if region == 'inner'
                    else ('OZ+HNCB' if model == 'OZ+HNCB'
                          else r'OZ+MSA $\equiv$ FMSA-DP'))
            z = q[q.model == model]
            # Trend lines are drawn PER SOURCE.  The low-eta bin is TN 1331 only
            # and the middle bins are almost all MD, so a single line through all
            # of them would show a source change and look like an eta trend.
            for src, (lsrc, msrc) in (('tn1331_dcf', ('--', 'o')),
                                      ('md', ('-', 's'))):
                zz = z[z.source == src]
                if not len(zz):
                    continue
                ax.scatter(zz.eta, zz.nrmse, s=24, c=color, marker=msrc,
                           alpha=0.45, linewidths=0)
                g = zz.groupby(pd.cut(zz.eta, M.ETA_BINS), observed=True).agg(
                    eta=('eta', 'median'), nrmse=('nrmse', 'median'), n=('eta', 'size'))
                g = g[g.n >= 2]
                if len(g) > 1:
                    ax.plot(g.eta, g.nrmse, color=color, ls=lsrc, lw=2.2,
                            marker=msrc, ms=6, zorder=6)
            ax.plot([], [], color=color, lw=2.2, marker=mk, ms=6, label=name)
        if 'tn1331_dcf' in CLEAN and len(CLEAN) > 1:
            ax.plot([], [], color='#555', ls='--', marker='o', ms=5, lw=1.6,
                    label='— TN 1331 (dashed)')
            ax.plot([], [], color='#555', ls='-', marker='s', ms=5, lw=1.6,
                    label='— our MD (solid)')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\eta$')
        ax.set_ylabel('NRMSE, all pairs')
        ax.set_title(ttl)
        ax.legend(fontsize=8, ncol=2)

    # --- (1,0): the axis this scope's data actually varies -------------------
    # our MD varies the size ratio and nothing else; TN 1331 is all sigma ratio 2
    # but spans T* and eps22/eps11.  Plotting the axis a source does not vary
    # produces a single point and says nothing.
    ax = axes[1, 0]
    if 'md' in CLEAN:
        xvar, xlab, src_ttl = 'sigma_ratio', r'$\sigma_{22}/\sigma_{11}$', 'our MD'
        base = d[(d.source == 'md') & (d.ref == REF)]
    else:
        xvar, xlab, src_ttl = 'T_star', r'$T^*$', 'TN 1331'
        base = d[d.source.isin(CLEAN) & (d.ref == REF)]
    for region, ls, alpha in (('inner', '-', 1.0), ('outer', ':', 0.75)):
        q = combined(base[base.region == region])
        if region == 'outer':
            q = q[q.outer_trust == 1]
        for model, (color, _, mk) in STYLE.items():
            if region == 'outer' and model in ('OZ+MSA', 'FMSA-DP(fmt)'):
                continue                      # coincident with FMSA-DP(py)
            g = q[q.model == model].groupby(xvar).nrmse.median()
            lbl = (model if region == 'inner' else
                   ('OZ+HNCB, outer' if model == 'OZ+HNCB'
                    else r'OZ+MSA $\equiv$ FMSA-DP, outer'))
            ax.plot(g.index, g.values, color=color, ls=ls, marker=mk, ms=6,
                    lw=2.0, alpha=alpha, label=lbl)
    ax.set_xlabel(xlab)
    ax.set_ylabel('NRMSE, all pairs')
    ax.set_yscale('log')
    ax.set_title(f'against {xlab} ({src_ttl})\nsolid = inner core, dotted = outer')
    ax.legend(fontsize=7.5, ncol=2, loc='upper left')

    _scorecard(axes[1, 1], clean)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / f'{PREFIX}_overall.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


def fig_scorecard(d):
    """The scorecard ALONE, sized for one slot of a 2-up row.

    Fig. 9 stacks the DCF and RDF scorecards; side by side each one gets
    0.49\textwidth = 3.18 in, and a panel has to be DRAWN near that width for its
    9 pt text to print at 9 pt.  Cropping panel (d) out of the 2x2 composite gives
    2.1 in instead -- the bar axes under-fill their quarter of the grid -- so this
    renders it on its own.  Same function, same data, same palette: only the
    canvas differs.
    """
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    _scorecard(ax, d[d.source.isin(CLEAN)], legend=False)
    ax.set_title('')
    fig.tight_layout(pad=0.3)
    p = OUT / f'{PREFIX}_scorecard.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')

    # ONE key for BOTH scorecards, emitted here because this is the side that
    # carries all four models -- the RDF pool has no WBFMT leg.  Neither panel
    # keeps a legend of its own: at 2-up width the box covered the two shortest
    # rows, and two boxes would have shown the same four names twice.
    import figure_legend
    from matplotlib.patches import Patch
    h = [Patch(facecolor=STYLE[m][0], edgecolor='none', label=disp(m))
         for m in STYLE]
    figure_legend.emit(h, [x.get_label() for x in h], f'{PREFIX}_scorecard',
                       out_dir=str(OUT), strip_ncol=4, handlelength=1.6)


def fig_legend_strip(label):
    """A standalone horizontal legend, for a figure that shares one across panels.

    The panels become separate PDFs, so a "shared legend at the bottom" cannot be
    an artist inside any one of them -- it has to be its own file that the .tex
    places under the grid.  Entries and styles are taken from the same tables the
    curve panels use, so the strip cannot drift out of step with them.
    """
    pool = {r['label']: r for r in csv.DictReader(M.POOL_INDEX.open())}
    e = pool[label]
    handles = [Line2D([], [], marker='o', ls='none', ms=5, mfc='none',
                      mec=S.DATA['primary'], mew=1.1, label='Literature')]
    for model, (color, ls, _) in STYLE.items():
        if model in CURVE_HIDE:
            continue
        handles.append(Line2D([], [], color=color, ls=ls, lw=1.7,
                              label=CURVE_RENAME.get(model, model)))
    handles.append(Line2D([], [], color='gray', lw=0.9, ls=':',
                          label=r'$R_{ij}$ (contact)'))
    if abs(float(e['sigma_ratio']) - 1.0) > 1e-9:
        handles.append(Line2D([], [], color='#8a6d3b', lw=0.9, ls=(0, (4, 2)),
                              label=r'$|\lambda_{ij}|$ (breakpoint)'))

    # Two rows, not one.  Seven entries in a single row came to 8.40 in against
    # the 7.06 in text width; a strip that has to be scaled down defeats the
    # point of generating panels at printed size.
    ncol = (len(handles) + 1) // 2
    # Both shapes: with three panels the block drops into the empty fourth
    # slot and costs no height, but a fourth panel would want the strip.
    # See figure_legend.py -- that choice belongs to the .tex.
    import figure_legend
    figure_legend.emit(handles, [h.get_label() for h in handles],
                       PREFIX, out_dir=str(OUT), strip_ncol=ncol,
                       block_ncol=1, handlelength=2.0)


def fig_curves(labels):
    pool = {r['label']: r for r in csv.DictReader(M.POOL_INDEX.open())}
    suspects = M.load_suspects()
    fig, axes = plt.subplots(len(ZOOMS) * len(labels), 3,
                             figsize=(15.0, 4.3 * len(ZOOMS) * len(labels)))
    axes = np.atleast_2d(axes)
    fig.suptitle('Representative $c_{ij}(r)$ — a clean low-$\\eta$ state and one '
                 'inside the $c_{22}$ artefact region.\nEvery second row zooms the '
                 'outer region, where the models are indistinguishable except for '
                 'OZ+HNCB and where the benchmark itself is in question.', y=1.0)
    for n, label in enumerate(labels):
        e = pool[label]
        bench = M.load_benchmark(e, suspects)
        R_ij, curves, _ = M.build_models(e)
        for k, zoom in enumerate(ZOOMS):
            row = len(ZOOMS) * n + k
            for col, (i, j) in enumerate(M.PAIRS):
                ax = axes[row, col]
                R = float(R_ij[i, j])
                rb = bench[(i, j)]['r']
                # Under --ref lowk the repaired curve IS the benchmark; the raw
                # one is kept as a faint overlay so the repair stays visible
                # rather than being silently swapped in.
                have = bench[(i, j)]['c']
                key = REF if REF in have else 'published'
                cb = have[key]
                # ⚠ teal + white face -- see the RDF companion.
                ax.plot(rb, cb, 'o', ms=3.6, mfc='white', mec=S.DATA['primary'],
                        mew=0.9, label='Literature', zorder=5)
                # No raw-benchmark overlay.  Under --ref lowk the repaired
                # curve IS the benchmark and is labelled as such; drawing the
                # unrepaired one beside it in faint grey was unreadable at panel
                # size and invited the reader to treat a known artifact as data.
                if key == 'published' and 'c22_mfphi' in have and (i, j) == (1, 1):
                    ax.plot(rb, have['c22_mfphi'], color='#1baf7a',
                            lw=1.6, ls=(0, (4, 2)),
                            label=r'Literature, $c_{22}\to-\beta\varphi$')
                for model, (color, ls, _) in STYLE.items():
                    if model in CURVE_HIDE:
                        continue
                    rm, cm = curves[model]
                    if rm is None:
                        continue
                    m = (rm > 0) & (rm <= rb[-1])
                    ax.plot(rm[m], cm[i, j][m], color=color, ls=ls, lw=1.7,
                            label=CURVE_RENAME.get(model, model))
                ax.axvline(R, color='gray', lw=0.9, ls=':',
                           label=r'$R_{ij}$ (contact)')
                lam = abs(0.5 * (float(e['sigma_ratio']) - 1.0)) if (i, j) == (0, 1) else 0.0
                if lam > 1e-9:
                    ax.axvline(lam, color='#8a6d3b', lw=0.9, ls=(0, (4, 2)),
                               label=r'$|\lambda_{ij}|$ (breakpoint)')
                if zoom == 1:                      # outer
                    sel, xlo, xhi = rb >= R, R, float(rb[-1])
                elif zoom == 2:                    # inner core
                    sel, xlo, xhi = (rb > 0) & (rb <= R), 0.0, R
                else:                              # full
                    sel, xlo, xhi = rb > 0, 0.0, float(rb[-1])
                ax.set_xlim(xlo, xhi)
                lo, hi = float(np.min(cb[sel])), float(np.max(cb[sel]))
                pad = 0.18 * (hi - lo) + 1e-3
                ax.set_ylim(lo - pad, hi + pad)
                ax.axhline(0, color='k', lw=0.5)
                ax.set_xlabel(r'$r/\sigma_{11}$')
                # Every column, not just the first.  figure_split writes one PDF
                # per panel and the .tex places them independently, so a panel
                # that relies on its left neighbour for the y-label is unlabelled
                # the moment the grid is not a single row.
                # ⭐ The PAIR IS IN THE AXIS LABEL, not in a box in the corner.
                # `figure_split --pub` strips titles, so a published panel had no
                # pair identity at all; the fix belongs on the quantity being
                # plotted, which is Fig. 2's convention (r c11 / r c12 / r c22)
                # and costs no extra ink.
                ax.set_ylabel(f'$c_{{{i + 1}{j + 1}}}(r)$')
                ax.set_title(f'pair {M.LABELS[(i, j)]},  $R_{{ij}}$={R:.2f}')
                # One legend per ROW, on the column named by CURVE_LEGEND.
                # figure_split crops each panel separately, so a legend on the
                # composite's first panel only would leave every other panel
                # unreadable -- but one per panel crowds three narrow panels.
                # Set CURVE_LEGEND=None and use the standalone strip written by
                # --legend-strip if the figure carries a shared legend instead.
                if CURVE_LEGEND is not None and col == CURVE_LEGEND:
                    ax.legend(fontsize=6.5, loc='best', framealpha=0.85)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / f'{PREFIX}_curves.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


# Two representative state points per scope: one where the outer region passes
# the tail-scale gate, one inside the c22 artefact region, so the curve figure
# shows both what the models get right and what the benchmark gets wrong.
CURVE_STATES = {
    'all':        ['tn1331dcf_T2_eps1.00_x0.9502',     # eta = 0.210, clean outer
                   'tn1331dcf_T1_eps0.50_x0.2500'],    # eta = 0.455, artefact state
    'literature': ['tn1331dcf_T2_eps1.00_x0.9502',
                   'tn1331dcf_T1_eps0.50_x0.2500'],
    'md':         ['md_sweep_r2.0_eta0.4_x0.5',        # clean outer
                   'md_sweep_r3.0_eta0.5_x0.5'],       # eta = 0.50, where MD fails too
}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scope', default='all', choices=list(SCOPE_DIR),
                    help='which write-up the figures belong to (default: all)')
    ap.add_argument('--legend-strip', action='store_true',
                    help='also write a standalone horizontal legend, for a '
                         'figure that shares one under the panel grid')
    ap.add_argument('--zooms', default='0,1', metavar='LIST',
                    help='rows per state: 0 full, 1 outer zoom, 2 inner core '
                         "(e.g. --zooms 0,2)")
    ap.add_argument('--ref', default='published', choices=['published', 'lowk'],
                    help='benchmark variant the figures are drawn against')
    ap.add_argument('--outdir', default=None,
                    help='write the figures somewhere other than the scope default '
                         '(used for the post-repair re-scoring)')
    ap.add_argument('--data', default=None,
                    help='read the metric CSVs from here instead of all_samples/ '
                         '(used to redraw the frozen pre-repair figures)')
    ap.add_argument('--curve-states', nargs='*', default=None,
                    help='override the two state points of the c_ij(r) figure')
    ap.add_argument('--legend-only', action='store_true',
                    help='write only the shared key, in both shapes, and stop')
    args = ap.parse_args()
    globals()['REF'] = args.ref
    globals()['ZOOMS'] = tuple(int(z) for z in args.zooms.split(','))
    set_scope(args.scope)
    if args.data:
        global DATA
        DATA = Path(args.data)
    if args.outdir:
        global OUT
        OUT = Path(args.outdir)

    if args.legend_only:
        # Pure encoding -- must not cost a full redraw of six figures.
        set_scope(args.scope)
        OUT.mkdir(parents=True, exist_ok=True)
        fig_legend_strip((args.curve_states or CURVE_STATES[SCOPE])[-1])
        return 0

    d, s = load()
    d = d[d.source.isin(M.SCOPES[SCOPE]['sources'])]
    s = s[s.source.isin(M.SCOPES[SCOPE]['sources'])]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'writing {SCOPE} figures:')
    fig_overall(d)
    fig_scorecard(d)          # the same panel, sized for a 2-up slot -- Fig. 9(a)
    fig_pointwise(d)
    fig_chat0(d)
    fig_dual_reference(s)
    if 'md' in M.SCOPES[SCOPE]['sources']:
        fig_sigma_ratio(d)          # only our MD varies the size ratio
    states = args.curve_states or CURVE_STATES[SCOPE]
    fig_curves(states)
    if args.legend_strip:
        fig_legend_strip(states[-1])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
