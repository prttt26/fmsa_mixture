#!/usr/bin/env python3
"""Invert TN 1331's tabulated g_ij(r) through the mixture OZ equation to c_ij(r).

Why
---
Appendix C of NIST TN 1331 does not report a directly sampled direct correlation
function: the report OZ-inverted its own simulated g_ij(r) (Appendix A).  Running
that inversion ourselves does two things:

  1. validates a g -> c pipeline we can then point at any other simulation data
     (our own MD runs at other size ratios, for instance), and
  2. tests the standing suspicion about the c22 outer region.  The project note
     numerical_notes/results/single_point/oz_closures_comparison.md sec.4 found that beyond
     r ~ 1.3 sigma the tabulated c_ij exceed the rigorous c -> -beta*u limit by
     2-7x, worst for the (2,2) pair.  If our inversion of the same g_ij
     reproduces that excess, it is inherent to inverting this g data (finite box,
     truncated range); if it does not, the excess entered in the report's own
     inversion procedure.

Theory
------
Multicomponent Ornstein-Zernike, with rho_l the partial number densities:

    h_ij(r) = c_ij(r) + sum_l rho_l Int c_il(|r-r'|) h_lj(r') dr'

In Fourier space, with H, C the matrices of the 3-D radial transforms and
D = diag(rho_1, rho_2):

    H = C + C D H      =>      C = H (I + D H)^{-1}

The 3-D radial transform of a spherically symmetric f is

    f^(k) = (4 pi / k) Int_0^inf r sin(kr) f(r) dr
    f(r)  = (1 / (2 pi^2 r)) Int_0^inf k sin(kr) f^(k) dk

both of which are sine transforms of r*f(r), so a single discrete sine transform
(DST-I) on a uniform r grid serves for both directions.

Units and grids -- the subtlety that matters
--------------------------------------------
TN 1331 tabulates against the PER-PAIR reduced separation R* = r / sigma_ij, so
each pair's column is on a *different* absolute-r grid.  The OZ equation couples
the pairs at equal absolute r, so every table must first be mapped to absolute
r = R* * sigma_ij and resampled onto one common grid.  Skipping this step is the
single easiest way to get a plausible-looking but wrong answer.

Two directions, and why one of them is the trustworthy one
----------------------------------------------------------
forward (c -> g)
    Well conditioned: c_ij is short ranged, decaying like -beta*u ~ r^-6, so
    setting it to zero past the end of the table costs almost nothing.  Running
    this on TN's own Appendix C reproduces its Appendix A g_ij to about 0.1-1%,
    which is what validates BOTH the extracted tables and this pipeline.  Use it
    as the regression test whenever the extractor or the transforms change.

inverse (g -> c)
    This is the direction we ultimately want for new simulation data, but it
    CANNOT be expected to reproduce Appendix C from Appendix A, for a reason
    that has nothing to do with the code: h_ij = g_ij - 1 is long ranged, and
    the published tables stop at R* = 2.5.  For pair (2,2) that is only
    r = 5 sigma_11, where h is still visibly oscillating.  Huber and Ely
    inverted the h they had in the simulation, not the h they printed: at 864
    particles and rho* = 0.139 the box is about 18 sigma_11 across, so their
    h extended to roughly 9 sigma_11 -- well past what Appendix C's own source
    table shows.  Truncating at 5 sigma_11 and zero-padding therefore loses real
    structure, and the inverse transform amplifies that loss.

    The practical consequence: use this direction on OUR OWN MD output, where
    g_ij can be written out to half the box length, not on the published tables.

Usage
-----
    python oz_invert.py --state T1_eps0.50_x0.2500                  # inverse (default)
    python oz_invert.py --state T1_eps0.50_x0.2500 --direction forward
    python oz_invert.py --state T1_eps0.50_x0.2500 --plot out.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from structure_transforms import (PAIR_TAG, PAIRS, dst_backward, dst_forward,
                                  make_grid)

# Size ratio 2 with Lorentz-Berthelot mixing, in units of sigma_11.
SIGMA = np.array([[1.0, 1.5], [1.5, 2.0]])


def read_table(path: Path) -> dict[str, np.ndarray]:
    with path.open() as fh:
        rdr = csv.DictReader(fh)
        cols = {name: [] for name in rdr.fieldnames or []}
        for row in rdr:
            for name, val in row.items():
                cols[name].append(float(val))
    return {name: np.asarray(vals) for name, vals in cols.items()}


def h_on_grid(rdf: dict[str, np.ndarray], r: np.ndarray) -> np.ndarray:
    """h_ij = g_ij - 1 on the shared absolute-r grid, shape (2, 2, len(r)).

    Each pair is tabulated against R* = r / sigma_ij, so it is converted to
    absolute r before resampling.  Below the table g = 0 (hard core, h = -1);
    above it h is set to 0 -- see the module docstring on why that zero-padding
    is the limiting approximation of the inverse direction.
    """
    h = np.zeros((2, 2, r.size))
    for i, j in PAIRS:
        g = rdf[f'g{PAIR_TAG[(i, j)]}']
        r_abs = rdf['R_star'] * SIGMA[i, j]
        hij = np.where(r < r_abs[0], -1.0, 0.0)
        inside = (r >= r_abs[0]) & (r <= r_abs[-1])
        hij[inside] = np.interp(r[inside], r_abs, g - 1.0)
        h[i, j] = h[j, i] = hij
    return h


def c_on_grid(dcf: dict[str, np.ndarray], r: np.ndarray) -> np.ndarray:
    """c_ij on the shared absolute-r grid, shape (2, 2, len(r)).

    The DCF table starts at R* = 0.025, so the continuation below it is held
    flat at the first tabulated value; c is set to 0 beyond the table, which is
    a mild approximation because c decays like -beta*u.
    """
    c = np.zeros((2, 2, r.size))
    for i, j in PAIRS:
        v = dcf[f'c{PAIR_TAG[(i, j)]}']
        r_abs = dcf['R_star'] * SIGMA[i, j]
        cij = np.zeros(r.size)
        inside = (r >= r_abs[0]) & (r <= r_abs[-1])
        cij[inside] = np.interp(r[inside], r_abs, v)
        cij[r < r_abs[0]] = v[0]
        c[i, j] = c[j, i] = cij
    return c


def oz_solve(f: np.ndarray, rho: np.ndarray, r: np.ndarray, k: np.ndarray,
             direction: str) -> np.ndarray:
    """Apply the matrix OZ relation in k space and transform back.

    direction 'inverse': f = h  ->  C = H (I + D H)^{-1}
    direction 'forward': f = c  ->  H = (I - C D)^{-1} C

    Thin wrapper over structure_transforms; kept so this script's forward/inverse
    regression flow reads the same as before the core was factored out.
    """
    from structure_transforms import ck_from_hk, hk_from_ck

    fk = np.empty((2, 2, k.size))
    for i, j in PAIRS:
        fk[i, j] = fk[j, i] = dst_forward(f[i, j], r, k)

    out_k = ck_from_hk(fk, rho) if direction == 'inverse' else hk_from_ck(fk, rho)

    out = np.empty_like(out_k)
    for i, j in PAIRS:
        out[i, j] = out[j, i] = dst_backward(out_k[i, j], r, k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    ap.add_argument('--state', default='T1_eps0.50_x0.2500',
                    help='state-point stem, as named by extract_tn1331.py')
    ap.add_argument('--data', type=Path, default=here / 'tn1331')
    ap.add_argument('--rho', type=float, default=None,
                    help='total reduced density; default: read from index.csv')
    ap.add_argument('--x1', type=float, default=None,
                    help='mole fraction of species 1; default: read from the state name')
    ap.add_argument('--direction', choices=('inverse', 'forward'), default='inverse',
                    help="'inverse' g->c (what we want for new data); "
                         "'forward' c->g (the pipeline's regression test)")
    ap.add_argument('--r-max', type=float, default=12.0,
                    help='outer radius of the common grid (units of sigma_11)')
    ap.add_argument('--n-grid', type=int, default=4096)
    ap.add_argument('--plot', type=Path, default=None)
    args = ap.parse_args()

    rdf_path = args.data / 'rdf' / f'{args.state}.csv'
    dcf_path = args.data / 'dcf' / f'{args.state}.csv'
    need = rdf_path if args.direction == 'inverse' else dcf_path
    if not need.exists():
        raise SystemExit(f'{args.direction} needs {need}, which does not exist')

    rho_total, x1 = args.rho, args.x1
    index = args.data / 'index.csv'
    if (rho_total is None or x1 is None) and index.exists():
        with index.open() as fh:
            for row in csv.DictReader(fh):
                if row['kind'] == 'rdf' and Path(row['file']).stem == args.state:
                    if rho_total is None and row.get('rho_star'):
                        rho_total = float(row['rho_star'])
                    if x1 is None:
                        x1 = float(row['x1'])
    if rho_total is None or x1 is None:
        raise SystemExit('could not determine rho* / x1; pass --rho and --x1')

    rho = np.array([x1, 1.0 - x1]) * rho_total

    n = args.n_grid
    r, k = make_grid(args.r_max, n)

    print(f'state {args.state}:  rho* = {rho_total:g}, x1 = {x1:g}, '
          f'r_max = {args.r_max:g}, n = {n},  direction = {args.direction}')
    print(f'partial densities: rho_1 = {rho[0]:.5f}, rho_2 = {rho[1]:.5f}')

    if args.direction == 'inverse':
        src = h_on_grid(read_table(rdf_path), r)
        out, sym, ref_path, ref_prefix = oz_solve(src, rho, r, k, 'inverse'), 'c', dcf_path, 'c'
    else:
        src = c_on_grid(read_table(dcf_path), r)
        out, sym, ref_path, ref_prefix = oz_solve(src, rho, r, k, 'forward'), 'g', rdf_path, 'g'
        out = out + 1.0                      # h -> g for comparison

    if not ref_path.exists():
        print(f'\n(no {ref_prefix}-table for this state; nothing to compare against)')
        return 0

    ref = read_table(ref_path)
    print(f'\n{"R*":>6} ' + ' '.join(
        f'{sym}{t}_{w:<4}'.rjust(10) for t in ('11', '12', '22') for w in ('ours', 'TN')))

    resid = {t: [] for t in ('11', '12', '22')}
    for r_star in (0.5, 0.9, 0.95, 1.0, 1.05, 1.1, 1.3, 1.5, 2.0, 2.5):
        idx = np.argmin(np.abs(ref['R_star'] - r_star))
        if abs(ref['R_star'][idx] - r_star) > 1e-6:
            continue
        line = [f'{r_star:6.3f}']
        for i, j in PAIRS:
            tag = PAIR_TAG[(i, j)]
            r_abs = r_star * SIGMA[i, j]
            ours = float(np.interp(r_abs, r, out[i, j])) if r_abs <= r[-1] else np.nan
            theirs = float(ref[f'{ref_prefix}{tag}'][idx])
            if np.isfinite(ours):
                resid[tag].append(abs(ours - theirs))
            line += [f'{ours:10.4f}' if np.isfinite(ours) else f'{"--":>10}',
                     f'{theirs:10.4f}']
        print(' '.join(line))

    print('\nmean |ours - TN| over the sampled points: '
          + ',  '.join(f'{sym}{t} = {np.mean(v):.4f}' if v else f'{sym}{t} = n/a'
                       for t, v in resid.items()))
    if args.direction == 'inverse':
        print('Close agreement is NOT expected here -- the published g stops at\n'
              'R* = 2.5 and h is zero-padded beyond it; see the module docstring.\n'
              'Run --direction forward for the pipeline regression test.')

    if args.plot is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        label_ours = ('OZ inversion of App. A $g_{ij}$' if args.direction == 'inverse'
                      else 'OZ forward from App. C $c_{ij}$')
        ref_label = f'TN 1331 App. {"C" if args.direction == "inverse" else "A"}'
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for ax, (i, j) in zip(axes, PAIRS):
            tag = PAIR_TAG[(i, j)]
            ax.plot(r, out[i, j], label=label_ours)
            ax.plot(ref['R_star'] * SIGMA[i, j], ref[f'{ref_prefix}{tag}'], '--',
                    label=ref_label)
            ax.set_xlabel('$r/\\sigma_{11}$')
            ax.set_ylabel(f'${sym}_{{{tag}}}(r)$')
            ax.set_xlim(0, 2.5 * SIGMA[i, j])
            ax.axhline(0 if sym == 'c' else 1, lw=0.5, color='k')
            ax.set_title(f'pair {tag}')
            ax.legend(fontsize=8)
        fig.suptitle(f'TN 1331 {args.state} -- {args.direction} OZ')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f'wrote {args.plot}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
