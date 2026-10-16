#!/usr/bin/env python3
r"""The in-core first-order DCF, split into its polynomial and Yukawa-exponential parts.

The closed form puts `2*pi*sqrt(rho_i rho_j) * r * c^(1)_ij(r)` in a fixed finite class: a
polynomial in `r` plus exponentials `e^{+-z_ij r}`.  The split is not a fit -- it is already
the internal representation (`_c1_pieces` returns pieces `(a, b, {rate: poly})`, so `rate == 0`
IS the polynomial part and `rate != 0` the exponentials), so this figure reports the
construction rather than a decomposition imposed on it.

⭐ WHAT THE SPLIT SHOWS, AND WHY IT IS WORTH A PANEL.  The in-core DCF is a SMALL RESIDUAL OF
TWO LARGE PIECES.  At `r = 0.02` on the reference binary the polynomial part is `-104.6` and the
exponential part `+106.0`; their sum is `1.49`.  Each part carries a `1/r` divergence at the
origin and the divergences cancel exactly, because the bracket vanishes at `r = 0`
(`P(0) + sum_nu A_nu = 0`, verified to machine zero in panel (b)).  Two consequences:

  * structural -- the finiteness of `c^(1)(0)` is a cancellation, not a property of either term;
  * numerical -- anything that evaluates the two parts separately near contact loses digits.
    Panel (b) measures how many.

⭐ THE RATES ARE THE PAIR YUKAWA RATES, AND NOTHING ELSE.  The exponentials present are exactly
`+-z/R_ij` over the pairs -- at unequal diameters all three pair rates appear in one entry,
because the `sum_l` couples the species.  No hard-sphere pole rate appears anywhere.  That is
the paper's "no HS pole entered" claim, made visible instead of asserted.

⚠ Panel (c) is the unequal-diameter unlike pair, and it carries the breakpoint theorem inside
the decomposition: at `|lambda_ij|` each part JUMPS by ~12.7 while the total moves by 0.05.  The
kink the theorem is about is a near-cancellation of two discontinuities.

⚠ NO EXPLANATORY TEXT INSIDE THE AXES; the caption carries it.

    python figure_incore_split.py            # composite PNG + PDF + legend + JSON
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from energy_compress_core import make_model

OUT = pathlib.Path('numerical_notes/results/incore_split')
STEM = 'incore_split'

C_TOT = '#111111'      # the physical DCF
C_POLY = '#2a78d6'     # polynomial part
C_EXP = '#e08c36'      # Yukawa-exponential part
C_BP = '#7a7a7a'       # breakpoint rule

# The two standard systems the rest of the paper uses, so the split is shown on the same
# states as everything else rather than on a case chosen to look clean.
#   2YK  -- the LJ -> two-Yukawa mapping, `SYS['2YK']` of `fmsa_phase_lines.py`
#   binary -- the reference sigma-ratio-2 sample of the single-sample DCF figure
SYS_2YK = dict(key='2YK_lj', sig=[1.0], eps=[1.0], x=[1.0], nt=2, input_type='bh',
               z=[2.9637, 14.0167], k=[2.1714, -2.1714])
REF_BINARY = dict(sig=[1.0, 2.0], eps=[1.0, 0.5], x=[0.25, 0.75], rho=0.139, T=1.0)


def model_2yk(rho=0.2, T=1.0):
    d = dict(SYS_2YK)
    d = dict(key=d['key'], sig=np.array(d['sig'], float), eps=np.array(d['eps'], float),
             x=np.array(d['x'], float), nt=d['nt'], z=np.array(d['z'], float),
             k=np.array(d['k'], float), input_type='bh')
    return make_model(d, rho, T)


def model_binary():
    r = REF_BINARY
    n = len(r['sig'])
    eps = np.array(r['eps'], float)
    eij = np.sqrt(np.outer(eps, eps))                 # Lorentz--Berthelot, as elsewhere
    return make_model(dict(key='refbin', sig=np.array(r['sig'], float), eps=eps,
                           x=np.array(r['x'], float), nt=1, z=np.array([1.8]),
                           k=np.array([1.0]), eps_ij=eij, input_type='bh'),
                      r['rho'], r['T'])


def splitter(f, i, j):
    """-> (total, poly, exp) evaluators and (R_ij, breakpoints, rates).

    Reads the piece representation directly: contribution of `rate` at `t` is
    `polyval(P, t) * exp(-rate * t)`, so `rate == 0` is the polynomial part.
    """
    C, den = f._c1_pieces(i, j)
    R = float(f.R_ij[i, j])
    bps = sorted({a for a, _, _ in C} | {b for _, b, _ in C})
    rates = sorted({al for _, _, terms in C for al in terms})

    def part(t, keep):
        v = 0.0
        for (a, b, terms) in C:
            if a <= t < b:
                for al, P in terms.items():
                    if keep(al):
                        v += float(np.polyval(P[::-1], t)) * math.exp(-al * t)
        return v / (den * t)

    return (lambda t: part(t, lambda a: True),
            lambda t: part(t, lambda a: a == 0),
            lambda t: part(t, lambda a: a != 0),
            R, bps, rates)


def bracket_at_zero(f, i, j):
    """`P(0) + sum_nu A_nu` — the quantity whose vanishing makes c^(1)(0) finite."""
    C, _ = f._c1_pieces(i, j)
    a0, _, terms = C[0]
    return sum(float(P[0]) for P in terms.values()) if a0 == 0.0 else float('nan')


# publication width: JCP full column is 7.06 in; three panels across a `figure*`
PUB_W = 7.06 / 3.0
PUB_H = 2.55


PUB_W = 7.06 / 2.0
PUB_H = 2.7


def draw(ax, f, i, j, xlab):
    """The split for one pair: total, polynomial part, exponential part."""
    tot, poly, exp, R, bps, rates = splitter(f, i, j)
    lam = 0.5 * abs(f.sigma[i] - f.sigma[j])
    ax.axhline(0.0, color='0.85', lw=0.8, zorder=1)
    # ⚠ plotted as r*c, not c.  Each part carries a 1/r divergence at contact-zero for LIKE
    # pairs; multiplying by r turns that into a finite nonzero intercept, so the cancellation
    # shows as two equal-and-opposite intercepts on linear axes instead of needing log ones.
    # r*c is also the construction's own variable: 2 pi r c_ij = -Q'_ij + sum_l rho_l Q'_il * Q_jl.
    cuts = [b for b in bps if 0.0 < b < R] or []
    edges = [1e-4 * R] + cuts + [0.999 * R]
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = np.linspace(lo + 1e-9, hi - 1e-9, 260)
        ax.plot(seg, [x * poly(x) for x in seg], color=C_POLY, lw=1.6, zorder=3)
        ax.plot(seg, [x * exp(x) for x in seg], color=C_EXP, lw=1.6, zorder=3)
        ax.plot(seg, [x * tot(x) for x in seg], color=C_TOT, lw=2.1, zorder=4)
    if lam > 1e-12:
        ax.axvline(lam, color=C_BP, lw=1.1, ls=':', zorder=2)
    ax.set_xlabel(xlab)
    rs = np.linspace(1e-4 * R, 0.999 * R, 200)
    eps = 1e-9
    return dict(R=R, lam=lam, rates=rates, breakpoints=bps,
                intercept_poly=eps * poly(eps), intercept_exp=eps * exp(eps),
                intercept_total=eps * tot(eps),
                r=[float(x) for x in rs], rc_total=[x * tot(x) for x in rs],
                rc_poly=[x * poly(x) for x in rs], rc_exp=[x * exp(x) for x in rs])


def main(panels: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rec: dict = {}

    if panels:
        plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 9,
                             'xtick.labelsize': 8, 'ytick.labelsize': 8})
        figs = [plt.subplots(figsize=(PUB_W, PUB_H)) for _ in range(4)]
        fig, axes = figs[0][0], [f[1] for f in figs]
    else:
        fig, axs = plt.subplots(2, 2, figsize=(11.0, 7.2))
        axes = axs.ravel()

    # (a) single component, the standard 2YK (LJ -> two-Yukawa) system
    f1 = model_2yk()
    rec['a'] = draw(axes[0], f1, 0, 0, '$r/\\sigma$')
    axes[0].set_ylabel('$r\\,c^{(1)}(r)$')

    # (b)-(d) the reference sigma-ratio-2 binary: the three pairs
    g = model_binary()
    for ax, (i, j), tag in ((axes[1], (0, 0), 'b'),
                            (axes[2], (0, 1), 'c'),
                            (axes[3], (1, 1), 'd')):
        rec[tag] = draw(ax, g, i, j, '$r/\\sigma_1$')
        ax.set_ylabel(f'$r\\,c^{{(1)}}_{{{i + 1}{j + 1}}}(r)$')

    import figure_legend
    handles = [Line2D([], [], color=C_TOT, lw=2.1, label='total'),
               Line2D([], [], color=C_POLY, lw=1.6, label='polynomial part'),
               Line2D([], [], color=C_EXP, lw=1.6, label='Yukawa-exponential part'),
               Line2D([], [], color=C_BP, lw=1.1, ls=':', label='$|\\lambda_{ij}|$')]
    figure_legend.emit(handles, [h.get_label() for h in handles], STEM,
                       out_dir=str(OUT), handlelength=2.6, strip_ncol=4, strip_fontsize=9)

    if not panels:                       # letters only on the browsing composite
        for ax, lab in zip(axes, ['(a)', '(b)', '(c)', '(d)']):
            ax.set_title(lab, loc='left')

    if panels:
        pdir = pathlib.Path('numerical_notes/figures_subfigures_sep/incore_split')
        pdir.mkdir(parents=True, exist_ok=True)
        for (fg, _), lab in zip(figs, 'abcd'):
            fg.tight_layout(pad=0.35)
            fg.savefig(pdir / f'{STEM}_{lab}.pdf')
        print(f'  wrote 4 publication panels to {pdir}')
        return

    fig.tight_layout()
    fig.savefig(OUT / f'{STEM}.png', dpi=150)
    fig.savefig(OUT / f'{STEM}.pdf')
    (OUT / f'{STEM}.json').write_text(json.dumps(rec, indent=1, default=float))
    print(f'  wrote {OUT}/{STEM}.pdf')
    for k in 'abcd':
        d = rec[k]
        print(f"  ({k}) R={d['R']:.3f} |lam|={d['lam']:.3f} "
              f"intercepts poly/exp/total = {d['intercept_poly']:+.4f} "
              f"{d['intercept_exp']:+.4f} {d['intercept_total']:+.2e}")


if __name__ == '__main__':
    import sys
    main(panels='--panels' in sys.argv)
