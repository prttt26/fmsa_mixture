#!/usr/bin/env python3
"""Convert between the structure quantities of a binary fluid.

Given ONE representation of the correlations -- RDF g_ij(r), DCF c_ij(r), total
correlation h_ij(r), their Fourier transforms, or the Ashcroft-Langreth
structure factor S_ij(k) -- write out ANY of the others. The physics lives in
`structure_transforms.Correlations`; this script is the I/O, unit handling, and
CLI around it.

Inputs
------
--state <stem>        a TN 1331 state point; reads tn1331/{rdf,dcf}/<stem>.csv
                      and (rho*, x1) from tn1331/index.csv.  --from selects g
                      (default) or c.
--input <file.csv>    an arbitrary table (e.g. our own MD).  --from names the
                      quantity; columns are R_star/r + the three pair columns
                      (see --cols).

Units (--r-convention)
    perpair   column 0 is R* = r/sigma_ij, each pair on its own grid; mapped to
              absolute r via --sigma (default: size-ratio-2 Lorentz-Berthelot).
              This is how TN 1331 tables are laid out.
    absolute  column 0 is already an absolute r shared by all pairs (own-MD).

Densities
    Needed for any conversion through the OZ equation (anything <-> c) or to a
    structure factor.  Pure Fourier transforms (g<->hhat, c<->chat) do not need
    them.  Supply --rho (total reduced density) and --x1, or let --state read
    them from index.csv.

Outputs (--to)
    g, h, c            real-space, on the common absolute-r grid
    hk, ck             Fourier space, on the conjugate k grid
    S_AL               Ashcroft-Langreth partial structure factor S_ij(k)
    S_BT               Bhatia-Thornton S_NN, S_CC, S_NC
    all                every quantity the inputs allow
Each is written to converted/<label>/<quantity>.csv, alongside a manifest.json.

A caveat: converting a *truncated* real-space input to c or to k space is only
as good as the input's range.  h_ij is long ranged; zero-padding it past the
table loses structure the inverse transform then amplifies (at rho*=0.139 the
amplification onto chat22(0) is ~450x -- see c22_outer_diagnosis.py).  Feed such
conversions g_ij extended to about half the box length.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from structure_transforms import (PAIR_TAG, PAIRS, Correlations, make_grid)

SIGMA_RATIO2 = np.array([[1.0, 1.5], [1.5, 2.0]])   # size ratio 2, LB mixing

# What column prefix each real/Fourier quantity uses in a CSV.
REAL_PREFIX = {'g': 'g', 'h': 'h', 'c': 'c'}
K_PREFIX = {'hk': 'hk', 'ck': 'ck'}
ALL_TARGETS = ['g', 'h', 'c', 'hk', 'ck', 'S_AL', 'S_BT']


def read_table(path: Path) -> dict[str, np.ndarray]:
    with path.open() as fh:
        rdr = csv.DictReader(ln for ln in fh if not ln.lstrip().startswith('#'))
        cols = {name: [] for name in rdr.fieldnames or []}
        for row in rdr:
            for name, val in row.items():
                cols[name].append(float(val) if val not in (None, '') else np.nan)
    return {name: np.asarray(vals) for name, vals in cols.items()}


def field_on_grid(table: dict[str, np.ndarray], prefix: str, r: np.ndarray,
                  sigma: np.ndarray, per_pair: bool, below, above=0.0
                  ) -> np.ndarray:
    """Resample a 3-pair table onto the common absolute-r grid, shape (2,2,N).

    `below`/`above` set the continuation outside the tabulated range (a constant,
    or the string 'first' to hold the first tabulated value).  For per-pair data
    column 0 is R* = r/sigma_ij; for absolute data it is r itself.
    """
    rkey = 'R_star' if per_pair else ('r' if 'r' in table else 'R_star')
    field = np.zeros((2, 2, r.size))
    for i, j in PAIRS:
        col = table[f'{prefix}{PAIR_TAG[(i, j)]}']
        r_src = table[rkey] * (sigma[i, j] if per_pair else 1.0)
        v = np.zeros(r.size)
        inside = (r >= r_src[0]) & (r <= r_src[-1])
        v[inside] = np.interp(r[inside], r_src, col)
        lo = col[0] if below == 'first' else below
        v[r < r_src[0]] = lo
        v[r > r_src[-1]] = above
        field[i, j] = field[j, i] = v
    return field


def build_correlations(args, r, k, rho) -> Correlations:
    """Load the input table and hand it to the right Correlations constructor."""
    table = read_table(args._input_path)
    per_pair = args.r_convention == 'perpair'
    q = args.__dict__['from']

    if q in ('g', 'h'):
        # below the first point: hard core, g = 0 (h = -1); above: h -> 0
        pref = 'g' if q == 'g' else 'h'
        field = field_on_grid(table, pref, r, args.sigma, per_pair,
                              below=(0.0 if q == 'g' else -1.0), above=(1.0 if q == 'g' else 0.0))
        return (Correlations.from_g if q == 'g' else Correlations.from_h)(field, r, k, rho)
    if q == 'c':
        field = field_on_grid(table, 'c', r, args.sigma, per_pair, below='first')
        return Correlations.from_c(field, r, k, rho)
    if q in ('hk', 'ck'):
        # k-space input: interpolate onto the conjugate k grid (single grid)
        field = np.zeros((2, 2, k.size))
        for i, j in PAIRS:
            field[i, j] = field[j, i] = np.interp(k, table['k'], table[f'{q}{PAIR_TAG[(i, j)]}'])
        return (Correlations.from_hk if q == 'hk' else Correlations.from_ck)(field, r, k, rho)
    if q == 'S_AL':
        field = np.zeros((2, 2, k.size))
        for i, j in PAIRS:
            field[i, j] = field[j, i] = np.interp(k, table['k'], table[f'S{PAIR_TAG[(i, j)]}'])
        return Correlations.from_S_AL(field, r, k, rho)
    raise SystemExit(f'unknown --from {q}')


def write_real(path: Path, r: np.ndarray, field: np.ndarray, prefix: str,
               r_hi: float) -> None:
    mask = r <= r_hi
    with path.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['r'] + [f'{prefix}{PAIR_TAG[p]}' for p in PAIRS])
        for n in np.nonzero(mask)[0]:
            w.writerow([f'{r[n]:.4f}'] + [f'{field[i, j, n]:.6f}' for i, j in PAIRS])


def write_k(path: Path, k: np.ndarray, field: np.ndarray, prefix: str,
            k_hi: float) -> None:
    mask = k <= k_hi
    with path.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['k'] + [f'{prefix}{PAIR_TAG[p]}' for p in PAIRS])
        for n in np.nonzero(mask)[0]:
            w.writerow([f'{k[n]:.4f}'] + [f'{field[i, j, n]:.6f}' for i, j in PAIRS])


def write_bt(path: Path, k: np.ndarray, bt: dict, k_hi: float) -> None:
    mask = k <= k_hi
    with path.open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['k', 'S_NN', 'S_CC', 'S_NC'])
        for n in np.nonzero(mask)[0]:
            w.writerow([f'{k[n]:.4f}'] + [f'{bt[key][n]:.6f}' for key in ('S_NN', 'S_CC', 'S_NC')])


def resolve_state_input(args) -> None:
    """For --state: pick the RDF or DCF file and fill rho/x1 from index.csv."""
    data = args.data
    q = args.__dict__['from']
    sub = {'g': 'rdf', 'h': 'rdf', 'c': 'dcf'}.get(q)
    if sub is None:
        raise SystemExit(f'--state supports --from g or c, not {q}')
    args._input_path = data / sub / f'{args.state}.csv'
    if not args._input_path.exists():
        raise SystemExit(f'no {sub} table for state {args.state} ({args._input_path})')
    index = data / 'index.csv'
    if (args.rho is None or args.x1 is None) and index.exists():
        for row in read_index(index):
            if row['kind'] == sub and Path(row['file']).stem == args.state:
                if args.rho is None and row.get('rho_star'):
                    args.rho = float(row['rho_star'])
                if args.x1 is None:
                    args.x1 = float(row['x1'])


def read_index(path: Path):
    with path.open() as fh:
        yield from csv.DictReader(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--state', help='TN 1331 state-point stem (reads tn1331/)')
    src.add_argument('--input', type=Path, help='an arbitrary CSV table')
    ap.add_argument('--from', dest='from', default='g',
                    choices=['g', 'h', 'c', 'hk', 'ck', 'S_AL'],
                    help='quantity the input represents (default g)')
    ap.add_argument('--to', default='all',
                    help="comma list of {g,h,c,hk,ck,S_AL,S_BT} or 'all'")
    ap.add_argument('--data', type=Path, default=here / 'tn1331')
    ap.add_argument('--out', type=Path, default=here / 'converted')
    ap.add_argument('--label', default=None, help='output subdir name')
    ap.add_argument('--rho', type=float, default=None, help='total reduced density')
    ap.add_argument('--x1', type=float, default=None, help='mole fraction of species 1')
    ap.add_argument('--r-convention', choices=['perpair', 'absolute'], default='perpair')
    ap.add_argument('--sigma', type=float, nargs=4, default=None,
                    metavar=('s11', 's12', 's21', 's22'),
                    help='pair sigma matrix for perpair (default size ratio 2)')
    ap.add_argument('--r-max', type=float, default=12.0)
    ap.add_argument('--n-grid', type=int, default=4096)
    ap.add_argument('--r-out', type=float, default=None,
                    help='max r written to real-space CSVs (default: input range)')
    ap.add_argument('--k-out', type=float, default=25.0,
                    help='max k written to Fourier / S(k) CSVs')
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    args.sigma = (np.array(args.sigma).reshape(2, 2) if args.sigma is not None
                  else SIGMA_RATIO2)

    # --- resolve the input source and densities ---
    if args.state is not None:
        resolve_state_input(args)
        args.label = args.label or f'{args.state}_from_{args.__dict__["from"]}'
    else:
        args._input_path = args.input
        args.label = args.label or f'{args.input.stem}_from_{args.__dict__["from"]}'

    rho = None
    if args.rho is not None and args.x1 is not None:
        rho = np.array([args.x1, 1.0 - args.x1]) * args.rho

    targets = ALL_TARGETS if args.to == 'all' else [t.strip() for t in args.to.split(',')]
    needs_rho = any(t in ('c', 'ck', 'S_AL', 'S_BT') for t in targets) or \
        args.__dict__['from'] in ('c', 'ck', 'S_AL')
    if needs_rho and rho is None:
        raise SystemExit('this conversion needs densities: pass --rho and --x1 '
                         '(or use --state so they come from index.csv)')

    r, k = make_grid(args.r_max, args.n_grid)
    corr = build_correlations(args, r, k, rho)

    # r range to write: the input's own tabulated span, unless overridden
    if args.r_out is None:
        src_r = read_table(args._input_path).get('R_star', read_table(args._input_path).get('r'))
        r_hi = float(src_r[-1]) * (args.sigma.max() if args.r_convention == 'perpair' else 1.0)
    else:
        r_hi = args.r_out

    out_dir = args.out / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for t in targets:
        path = out_dir / f'{t}.csv'
        if t in ('g', 'h', 'c'):
            write_real(path, r, getattr(corr, t), t, r_hi)
        elif t in ('hk', 'ck'):
            write_k(path, k, getattr(corr, t), t, args.k_out)
        elif t == 'S_AL':
            write_k(path, k, corr.S_AL, 'S', args.k_out)
        elif t == 'S_BT':
            write_bt(path, k, corr.S_BT, args.k_out)
        else:
            raise SystemExit(f'unknown --to target {t}')
        written.append(f'{t}.csv')

    manifest = {
        'input': str(args._input_path),
        'from': args.__dict__['from'],
        'r_convention': args.r_convention,
        'rho_total': args.rho, 'x1': args.x1,
        'rho_partial': None if rho is None else rho.tolist(),
        'grid': {'r_max': args.r_max, 'n': args.n_grid, 'r_out': r_hi, 'k_out': args.k_out},
        'S_convention': 'Ashcroft-Langreth (symmetric); Bhatia-Thornton S_NN/S_CC/S_NC',
        'outputs': written,
        'caveat': ('conversions to c or to k space are only as good as the input '
                   'range; long-ranged h zero-padded past the table loses '
                   'structure the inverse transform amplifies (see '
                   'c22_outer_diagnosis.py). Feed g extended to ~half the box.'),
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    print(f'{args.label}: wrote {", ".join(written)} + manifest.json -> {out_dir}')
    if rho is not None:
        print(f'  rho_total={args.rho:g}  x1={args.x1:g}  '
              f'rho=({rho[0]:.5f}, {rho[1]:.5f})')

    if args.plot:
        _plot(out_dir, corr, r, k, r_hi, args.k_out, targets)
    return 0


def _plot(out_dir, corr, r, k, r_hi, k_hi, targets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [t for t in ('g', 'c', 'S_AL') if t in targets] or ['g']
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, panels):
        if t == 'S_AL':
            m = k <= k_hi
            for i, j in PAIRS:
                ax.plot(k[m], corr.S_AL[i, j][m], label=f'S{PAIR_TAG[(i, j)]}')
            ax.set_xlabel('$k\\sigma_{11}$'); ax.set_ylabel('$S_{ij}(k)$')
        else:
            m = r <= r_hi
            field = getattr(corr, t)
            for i, j in PAIRS:
                ax.plot(r[m], field[i, j][m], label=f'{t}{PAIR_TAG[(i, j)]}')
            ax.set_xlabel('$r/\\sigma_{11}$'); ax.set_ylabel(f'${t}_{{ij}}(r)$')
        ax.axhline(1 if t in ('g', 'S_AL') else 0, lw=0.5, color='k')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'plot.png', dpi=140)
    print(f'  wrote {out_dir / "plot.png"}')


if __name__ == '__main__':
    raise SystemExit(main())
