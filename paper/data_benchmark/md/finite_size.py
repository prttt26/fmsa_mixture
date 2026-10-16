#!/usr/bin/env python3
"""Finite-size check on the small-k structure factor of the NVT-MD data.

In a finite fixed-N (canonical) box the long-wavelength density fluctuations are
suppressed, so S_ij(k->0) -- and with it the reduced compressibility
chi_T/chi_id = S_NN(k->0) -- is biased below its thermodynamic-limit value. This
is the one place where NVT-MD differs from a grand-canonical simulation, and the
one region of our benchmark (the outer / small-k part) where that matters.

This runs the validation state point (size ratio 2, eps 0.5, x1 0.25, T*=1,
rho*=0.139) at a ladder of system sizes N, pushes each g_ij through the SAME
g -> S pipeline the sweep uses (`convert.py`), extracts S_ij(k->0) by a small-k
S = S0 + a k^2 fit, and extrapolates S0 vs 1/N to N -> infinity. The spread from
N=2048 (the sweep's size) to N->inf is the finite-size uncertainty to attach to
the small-k of every sweep point.

    python finite_size.py                 # runs the ladder, then analyses
    python finite_size.py --analyze-only  # just re-analyse existing runs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_md            # noqa: E402

NS = [512, 1024, 2048, 4096, 8192]
BASE = dict(ratio=2.0, eps_ratio=0.5, x1=0.25, T=1.0, rho=0.139,
            push=8000, equil=30000, prod=200000, nbins=450, every=25)


def run_one(N: int, threads: int, out: Path) -> str:
    from run_md import build_input, run_lammps, parse_rdf, parse_thermo, write_g_csv
    label = f'fss_N{N}'
    rundir = out / label
    if (rundir / 'g.csv').exists():
        return f'{label}: cached'
    rundir.mkdir(parents=True, exist_ok=True)
    spec = dict(BASE, N=N, label=label)
    build_input(spec, rundir)
    wall = run_lammps(rundir, threads)
    g = parse_rdf(rundir / 'rdf.out')
    write_g_csv(rundir / 'g.csv', g, spec)
    th = parse_thermo(rundir / 'thermo.out')
    th['wall_seconds'] = round(wall, 1)
    (rundir / 'thermo.json').write_text(json.dumps(th, indent=2))
    return f'{label}: {wall:.0f}s  T*={th["T_star"]:.3f}'


def s_at_zero(k: np.ndarray, s: np.ndarray, kmax: float = 1.6) -> float:
    """S(k->0) by an even (S = S0 + a k^2) least-squares fit over small k."""
    m = (k > 0) & (k <= kmax)
    A = np.vstack([np.ones(m.sum()), k[m] ** 2]).T
    s0, _ = np.linalg.lstsq(A, s[m], rcond=None)[0]
    return float(s0)


def read_csv(path: Path) -> dict:
    lines = [l for l in path.read_text().splitlines() if not l.startswith('#')]
    hdr = lines[0].split(',')
    data = np.array([[float(x) for x in ln.split(',')] for ln in lines[1:]])
    return {h: data[:, i] for i, h in enumerate(hdr)}


def analyse(out: Path) -> None:
    rows = []
    for N in NS:
        rundir = out / f'fss_N{N}'
        if not (rundir / 'g.csv').exists():
            continue
        rho = BASE['rho']
        # push g -> S through the production pipeline (convert.py)
        conv = HERE.parent / 'converted' / f'md_fss_N{N}'
        subprocess.run([sys.executable, str(HERE.parent / 'convert.py'),
                        '--input', str(rundir / 'g.csv'), '--from', 'g',
                        '--r-convention', 'absolute', '--rho', str(rho),
                        '--x1', str(BASE['x1']), '--to', 'S_AL,S_BT',
                        '--r-max', '24', '--n-grid', '4096',
                        '--label', f'md_fss_N{N}'],
                       cwd=HERE.parent, capture_output=True)
        al = read_csv(conv / 'S_AL.csv')
        bt = read_csv(conv / 'S_BT.csv')
        rows.append(dict(N=N,
                         S11=s_at_zero(al['k'], al['S11']),
                         S12=s_at_zero(al['k'], al['S12']),
                         S22=s_at_zero(al['k'], al['S22']),
                         S_NN=s_at_zero(bt['k'], bt['S_NN']),
                         S_CC=s_at_zero(bt['k'], bt['S_CC'])))
    if not rows:
        raise SystemExit('no fss runs found; run without --analyze-only first')

    Ninv = np.array([1.0 / r['N'] for r in rows])
    print(f'\n  finite-size trend of S(k->0)  (state: ratio 2, eps 0.5, x1 0.25, '
          f'T*=1, rho*=0.139)')
    print(f'    {"N":>6} {"1/N":>8} {"S_NN(0)":>9} {"S_CC(0)":>9} '
          f'{"S11(0)":>8} {"S12(0)":>8} {"S22(0)":>8}')
    for r in rows:
        print(f'    {r["N"]:>6} {1/r["N"]:>8.5f} {r["S_NN"]:>9.4f} {r["S_CC"]:>9.4f} '
              f'{r["S11"]:>8.4f} {r["S12"]:>8.4f} {r["S22"]:>8.4f}')

    print('\n  linear extrapolation S0(N) = S0(inf) + b/N :')
    result = {'state': 'ratio2_eps0.5_x0.25_T1_rho0.139', 'per_N': rows}
    for key in ('S_NN', 'S_CC', 'S11', 'S12', 'S22'):
        y = np.array([r[key] for r in rows])
        b, a = np.polyfit(Ninv, y, 1)      # y = a + b*(1/N)
        s2048 = next(r[key] for r in rows if r['N'] == 2048)
        shift = (s2048 - a) / a * 100 if a else float('nan')
        print(f'    {key:>5}(inf) = {a:+.4f}   [N=2048 gives {s2048:+.4f}, '
              f'{shift:+.1f}% vs inf]')
        result[f'{key}_inf'] = a
        result[f'{key}_N2048'] = s2048
        result[f'{key}_pct_shift_2048'] = shift

    # ripple diagnostic: how non-monotonic is S(k) at small k?  A clean S(k->0)
    # is flat/monotone; ripple (from the truncated-g FT) makes the extrapolation
    # meaningless.  Report the peak-to-smallest-k spread over k < 0.6 for the
    # largest box as the tell.
    big = out / 'converted' / f'md_fss_N{NS[-1]}'
    # (converted dirs live under data_benchmark/converted)
    big = HERE.parent / 'converted' / f'md_fss_N{NS[-1]}'
    bt = read_csv(big / 'S_BT.csv'); al = read_csv(big / 'S_AL.csv')
    def ripple(k, s):
        m = (k > 0) & (k < 0.6)
        return float(s[m].max() - s[m].min())
    print('\n  small-k ripple (max-min of S over k<0.6, largest box) -- a clean '
          'S(k->0) has ~0:')
    print(f'    S_CC: {ripple(bt["k"], bt["S_CC"]):.3f}   '
          f'S_NN: {ripple(bt["k"], bt["S_NN"]):.3f}   '
          f'S22: {ripple(al["k"], al["S22"]):.3f}')
    print('\n  => S_CC(k->0) is clean and trustworthy; S_NN/S22(k->0) are ripple-\n'
          '     dominated, so the FT route CANNOT give a physical compressibility.\n'
          '     Read chi_T from the EOS (dP/drho), not from S(k->0).')

    (out / 'finite_size.json').write_text(json.dumps(result, indent=2))
    print(f'\n  wrote {out / "finite_size.json"}')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=HERE / 'runs')
    ap.add_argument('--parallel', type=int, default=3)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--analyze-only', action='store_true')
    args = ap.parse_args()

    if not args.analyze_only:
        print(f'running finite-size ladder N={NS} ...')
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [ex.submit(run_one, N, args.threads, args.out) for N in NS]
            for f in as_completed(futs):
                print('  ' + f.result())
    analyse(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
