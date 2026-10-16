#!/usr/bin/env python3
"""Figures for the multi-state RDF comparison.

Reads the metric table written by compare_models_rdf_multistate.py and, for the
curve figure, re-solves two representative state points.

    python rdf_multistate_figures.py --scope all          # every source pooled
    python rdf_multistate_figures.py --scope literature   # NIST TN 1331 App. A
    python rdf_multistate_figures.py --scope md           # our own LAMMPS MD

The scope picks the sources, the output directory and the file prefix, so one
scoring run feeds all three write-ups under numerical_notes/results/benchmark_rdf/:

    all         -> all_samples/rdf_multistate_*.png
    literature  -> literature/lit_rdf_*.png
    md          -> LAMMPS_MD/md_rdf_*.png

Conventions follow the DCF figures: benchmark is open TEAL circles (figure_style
reserves teal for measurements paper-wide), the models
keep the same colours they have there, R_ij is a grey dotted vertical.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import compare_models_rdf_multistate as M               # noqa: E402

BENCH = REPO / 'numerical_notes' / 'results' / 'benchmark_rdf'
DATA = BENCH / 'all_samples'
SCOPE_DIR = {'all': BENCH / 'all_samples', 'literature': BENCH / 'literature',
             'md': BENCH / 'LAMMPS_MD'}
SCOPE_PREFIX = {'all': 'rdf_multistate', 'literature': 'lit_rdf', 'md': 'md_rdf'}
SRC_NAME = {'tn1331_rdf': 'NIST TN 1331 App. A', 'md': 'our LAMMPS MD'}

# MSA family together, HNCB last -- matching dcf_multistate_figures.py.  The
# two scorecards are printed as one figure whose caption asks the reader to
# read them against each other, and they cannot be read against each other
# while the bars are in different orders.
import figure_style as S                                        # noqa: E402
STYLE = S.styles('OZ+MSA', 'FMSA-DP', 'OZ+HNCB')   # one palette -- figure_style.py
PAIR_ORDER = ['(1,1)', '(1,2)', '(2,2)']
# Figure 6 shares one legend under the (b)-(d) grid, as Figure 4 does, so no
# panel carries its own.  None = defer to the strip; an int = that column.
#
# ⚠ There is no FMSA-DP(WBFMT) entry, but the reason is COST, not impossibility.
# An earlier version of this comment said the RDF "cannot" swap reference.  That
# is wrong: g = g_0 + g_1 would assemble perfectly well with a WB-FMT g_0.
#
# What is true is narrower.  The CLOSED-FORM route has one reference by
# construction: get_g0L_laplace is an analytic Laplace expression built on
# det Qhat_0(s), the PY/Baxter factor, and WB-FMT supplies c_HS, not an analytic
# g_0.  To get a WB-FMT g_0 you would solve OZ numerically with that c_HS
# (oz_numerical does exactly this kind of solve) and add the same h_1.
#
# So the DCF's fmt/py pair is free -- swapping the c_HS block is pure
# substitution and both members stay closed-form -- while the RDF's would cost an
# OZ solve and give up the closed form, which is the method's whole point.  Not
# implemented, and worth implementing only if the reference dependence of the RDF
# is itself the question.
CURVE_LEGEND = None

# Display name only -- the key stays 'FMSA-DP' because that is the value in the
# metrics CSV's `model` column and every lookup goes through it.  Naming the
# reference matches Figure 4, which shows FMSA-DP(WBFMT) beside FMSA-DP(PY);
# here there is only the PY one, and saying so is what makes the two figures
# comparable rather than looking like one lost a curve.
RENAME = {'FMSA-DP': 'FMSA-DP(PY)'}
disp = lambda m: RENAME.get(m, m)

SCOPE, OUT, PREFIX, SOURCES = 'all', SCOPE_DIR['all'], 'rdf_multistate', ['tn1331_rdf', 'md']

plt.rcParams.update({'font.size': 10.5, 'axes.titlesize': 11, 'axes.labelsize': 10.5,
                     'legend.fontsize': 9, 'figure.titlesize': 13})


def set_scope(scope: str) -> None:
    global SCOPE, OUT, PREFIX, SOURCES
    SCOPE, OUT, PREFIX = scope, SCOPE_DIR[scope], SCOPE_PREFIX[scope]
    SOURCES = M.SCOPES[scope]['sources']


def load():
    d = pd.read_csv(DATA / 'rdf_multistate_metrics.csv')
    d = d[(d.converged == 1) & d.source.isin(SOURCES)].copy()
    d['eta'] = d.eta.astype(float)
    d['pk_err'] = (d.peak_model - d.peak_bench).abs() / d.peak_bench.abs()
    d['pk_signed'] = (d.peak_model - d.peak_bench) / d.peak_bench.abs()
    d['rp_err'] = (d.rpeak_model - d.rpeak_bench).abs()
    d['cd_err'] = (d.coord_model - d.coord_bench).abs() / d.coord_bench.abs()
    return d


def _bypair(d, col):
    """median over states, then mean over pairs — the tables' aggregation."""
    t = d.pivot_table(index='model', columns='pair', values=col, aggfunc='median')
    return t.mean(axis=1)


# --------------------------------------------------------------------------
def _scorecard(ax, d, legend=True):
    """The RDF scorecard -- panel (a) of the composite, Fig. 9(b) of the paper.

    Factored out for the same reason as its DCF twin: Fig. 9 places the two side
    by side, and a panel has to be DRAWN near its printed width for 9 pt text to
    print at 9 pt.
    """
    metrics = [('nrmse', 'pointwise NRMSE\n(whole range)'),
               ('nrmse_shell', 'pointwise NRMSE\n(first shell)'),
               ('pk_err', 'first-peak height\nrel. error'),
               ('cd_err', 'coordination number\nrel. error')]
    y = np.arange(len(metrics))
    h = 0.24
    for n, model in enumerate(STYLE):
        v = [float(_bypair(d, c).get(model, np.nan)) for c, _ in metrics]
        # No `bar_label` -- see the companion in `dcf_multistate_figures.py`.
        ax.barh(y + (n - 1) * h, v, height=h, color=STYLE[model][0],
                label=disp(model), edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels([lbl for _, lbl in metrics], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('median relative error\n(shorter is better)')
    ax.set_title('scorecard')
    if legend:
        ax.legend(loc='lower right', fontsize=8.5)


def fig_scorecard(d):
    """The RDF scorecard alone, on a 2-up canvas.  Twin of the DCF one."""
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    _scorecard(ax, d, legend=False)   # shared key -- see the DCF companion
    ax.set_title('')
    fig.tight_layout(pad=0.3)
    p = OUT / f'{PREFIX}_scorecard.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


def fig_overall(d):
    # 4.4 not 5.4: panel (a) is the paper's RDF scorecard and stacks with the
    # DCF one at full width; the pair has to fit \textheight.
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.4))
    fig.suptitle('Model performance on the RDF — '
                 f'{" + ".join(SRC_NAME[s] for s in SOURCES)}', y=1.0)

    _scorecard(axes[0], d)

    ax = axes[1]
    for model, (c, ls, mk) in STYLE.items():
        q = d[d.model == model]
        t = q.pivot_table(index='pair', values='pk_signed', aggfunc='median') * 100
        ax.plot(PAIR_ORDER, [t.loc[p, 'pk_signed'] for p in PAIR_ORDER],
                color=c, ls=ls, marker=mk, ms=8, lw=2.0, label=disp(model))
    ax.axhline(0, color='k', lw=0.9)
    ax.set_ylabel('signed first-peak error  [%]')
    ax.set_title('every model UNDERSHOOTS the small-species peak\n'
                 'and the like-large pair is nearly right')
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = OUT / f'{PREFIX}_overall.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


def fig_peak(d):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
    fig.suptitle('First peak of $g_{ij}(r)$ — model against benchmark', y=1.0)
    for col, pair in enumerate(PAIR_ORDER):
        ax = axes[col]
        q = d[d.pair == pair]
        lo, hi = None, None
        for model, (c, ls, mk) in STYLE.items():
            z = q[q.model == model]
            ax.scatter(z.peak_bench, z.peak_model, s=26, c=c, marker=mk, alpha=0.75,
                       linewidths=0, label=disp(model) if col == 0 else None)
            v = np.concatenate([z.peak_bench.values, z.peak_model.values])
            v = v[np.isfinite(v)]
            if v.size:
                lo = np.min(v) if lo is None else min(lo, np.min(v))
                hi = np.max(v) if hi is None else max(hi, np.max(v))
        if lo is not None:
            pad = 0.05 * (hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k-', lw=0.8, zorder=0)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel('benchmark peak height')
        if col == 0:
            ax.set_ylabel('model peak height')
        ax.set_title(f'pair {pair}')
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    p = OUT / f'{PREFIX}_peak.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


def fig_pointwise(d):
    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.6), sharey=True)
    fig.suptitle('RDF error against packing fraction.  Last panel is the three '
                 'pairs averaged per state point.', y=1.02)
    cols = PAIR_ORDER + ['all pairs']
    for col, pair in enumerate(cols):
        ax = axes[col]
        for model, (c, ls, mk) in STYLE.items():
            if pair == 'all pairs':
                z = (d[d.model == model].groupby(['label', 'eta'], as_index=False)
                     .nrmse.mean())
            else:
                z = d[(d.model == model) & (d.pair == pair)]
            ax.scatter(z.eta, z.nrmse, s=22, c=c, marker=mk, alpha=0.55, linewidths=0,
                       label=disp(model) if col == 0 else None)
            g = z.groupby(pd.cut(z.eta, [0.10, 0.25, 0.35, 0.43, 0.60]),
                          observed=True).agg(eta=('eta', 'median'),
                                             nrmse=('nrmse', 'median'))
            ax.plot(g.eta, g.nrmse, color=c, lw=2.2, zorder=6)
        ax.set_yscale('log')
        ax.set_xlabel(r'$\eta$')
        ax.set_title('all pairs (mean)' if pair == 'all pairs' else f'pair {pair}',
                     fontweight='bold' if pair == 'all pairs' else 'normal')
    axes[0].set_ylabel('NRMSE of $g_{ij}(r)$')
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    p = OUT / f'{PREFIX}_pointwise.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


def fig_sigma_ratio(d):
    q = d[d.source == 'md']
    if q.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.suptitle('Size-ratio dependence of the RDF error — our own MD', y=1.0)
    for ax, col, ttl in ((axes[0], 'nrmse', 'NRMSE (whole range)'),
                         (axes[1], 'pk_err', 'first-peak height rel. error')):
        for model, (c, ls, mk) in STYLE.items():
            z = q[q.model == model]
            g = z.groupby('sigma_ratio')[col].median()
            ax.plot(g.index, g.values, color=c, ls=ls, marker=mk, ms=7, lw=2.0,
                    label=disp(model))
        ax.set_xlabel(r'$\sigma_{22}/\sigma_{11}$')
        ax.set_ylabel(ttl)
        ax.set_yscale('log')
        ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / f'{PREFIX}_sigma_ratio.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


def fig_legend_strip():
    """Standalone horizontal legend for a figure that shares one under its grid.

    Mirrors the DCF strip, minus the lambda_ij entry -- see CURVE_LEGEND above
    for why the RDF has no interior breakpoint to mark.
    """
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', ls='none', ms=5, mfc='none',
                      mec=S.DATA['primary'], mew=1.1, label='Literature')]
    for model, (color, ls, _) in STYLE.items():
        handles.append(Line2D([], [], color=color, ls=ls, lw=1.7, label=disp(model)))
    handles.append(Line2D([], [], color='gray', lw=0.9, ls=':',
                          label=r'$R_{ij}$ (contact)'))
    # Both shapes: with three panels the block drops into the empty fourth
    # slot and costs no height, but a fourth panel would want the strip.
    # See figure_legend.py -- that choice belongs to the .tex.
    import figure_legend
    figure_legend.emit(handles, [h.get_label() for h in handles],
                       PREFIX, out_dir=str(OUT),
                       strip_ncol=(len(handles) + 1) // 2,
                       block_ncol=1, handlelength=2.0)


def fig_curves(labels):
    pool = {r['label']: r for r in csv.DictReader(M.POOL_INDEX.open())}
    suspects = M.load_suspects()
    fig, axes = plt.subplots(len(labels), 3, figsize=(15.5, 4.4 * len(labels)))
    axes = np.atleast_2d(axes)
    fig.suptitle('Representative $g_{ij}(r)$ — benchmark against the three models',
                 y=1.0)
    for n, label in enumerate(labels):
        e = pool[label]
        r_by, g_by = M.load_benchmark(e, suspects)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            R_ij, curves, _ = M.build_models(e)
        for col, (i, j) in enumerate(M.PAIRS):
            ax = axes[n, col]
            rb, gb = r_by[(i, j)], g_by[(i, j)]
            # ⚠ teal + white face, like every other measured series in the
            # paper.  These panels kept a hardcoded black through the palette
            # change because they never went through figure_style.
            ax.plot(rb, gb, 'o', ms=3.6, mfc='white', mec=S.DATA['primary'], mew=0.9,
                    label='Literature', zorder=5)
            for model, (c, ls, mk) in STYLE.items():
                rm, gm = curves[model]
                if rm is None:
                    continue
                m = (np.asarray(rm) > 0) & (np.asarray(rm) <= rb[-1])
                ax.plot(np.asarray(rm)[m], np.asarray(gm)[i, j][m], color=c, ls=ls,
                        lw=1.7, label=disp(model))
            ax.axvline(float(R_ij[i, j]), color='gray', lw=0.9, ls=':',
                       label=r'$R_{ij}$ (contact)')
            # ⭐ START NEAR CONTACT, not at r = 0.  g_ij is identically zero
            # below R_ij, so on a 0-to-2.5-sigma axis roughly 40 % of the panel is
            # a flat line at zero carrying no information -- and it compresses the
            # first peak, which is the only thing the models disagree about here.
            # A little margin before R_ij keeps the rise visible as a rise.
            # ⚠ The DCF panels must NOT be cropped this way: c is large and
            # structured inside the core, and that region is the whole story there.
            ax.set_xlim(0.80 * float(R_ij[i, j]), rb[-1])
            # No lambda_ij line here.  It is the DCF's interior breakpoint; the
            # RDF's support is bounded away from the origin (Figure 1), so there
            # is nothing at lambda for it to mark.
            ax.axhline(1.0, color='k', lw=0.5)
            ax.set_xlabel(r'$r/\sigma_{11}$')
            # Every panel: figure_split writes one PDF each and the .tex places
            # them independently, so a panel cannot rely on its left neighbour.
            # ⭐ The PAIR IS IN THE AXIS LABEL, not in a box in the corner.
            # `figure_split --pub` strips titles, so a published panel had no
            # pair identity at all; the fix belongs on the quantity being
            # plotted, which is Fig. 2's convention (r c11 / r c12 / r c22)
            # and costs no extra ink.
            ax.set_ylabel(f'$g_{{{i + 1}{j + 1}}}(r)$')
            ax.set_title(f'pair {M.LABELS[(i, j)]}')
            if CURVE_LEGEND is not None and col == CURVE_LEGEND:
                ax.legend(fontsize=6.5, loc='best', framealpha=0.85)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / f'{PREFIX}_curves.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {p}')


CURVE_STATES = {
    'all':        ['tn1331rdf_T1_eps0.50_x0.7500', 'md_sweep_r3.0_eta0.4_x0.5'],
    'literature': ['tn1331rdf_T1_eps0.50_x0.7500', 'tn1331rdf_T2_eps1.00_x0.9502'],
    'md':         ['md_sweep_r2.0_eta0.4_x0.5', 'md_sweep_r3.0_eta0.4_x0.5'],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scope', default='all', choices=list(SCOPE_DIR))
    ap.add_argument('--outdir', default=None,
                    help='write the figures somewhere other than the scope default '
                         '(used for the post-repair re-scoring)')
    ap.add_argument('--data', default=None,
                    help='read the metric CSVs from here instead of all_samples/ '
                         '(used to redraw the frozen pre-repair figures)')
    ap.add_argument('--legend-strip', action='store_true',
                    help='also write a standalone horizontal legend')
    ap.add_argument('--curve-states', nargs='*', default=None)
    ap.add_argument('--legend-only', action='store_true',
                    help='write only the shared key, in both shapes, and stop')
    args = ap.parse_args()
    set_scope(args.scope)
    if args.data:
        global DATA
        DATA = Path(args.data)
    if args.outdir:
        global OUT
        OUT = Path(args.outdir)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.legend_only:
        # Pure encoding -- must not cost a full redraw of five figures.
        OUT.mkdir(parents=True, exist_ok=True)
        fig_legend_strip()
        return 0

    d = load()
    print(f'writing {SCOPE} figures:')
    fig_overall(d)
    fig_scorecard(d)
    fig_peak(d)
    fig_pointwise(d)
    if 'md' in SOURCES:
        fig_sigma_ratio(d)
    fig_curves(args.curve_states or CURVE_STATES[SCOPE])
    if args.legend_strip:
        fig_legend_strip()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
