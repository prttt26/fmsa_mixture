#!/usr/bin/env python3
"""Run the gridded MD batch concurrently, then convert each point to c and S(k).

The grid varies **size ratio, energy ratio, packing fraction and temperature**,
and adds an LJ + hard-sphere family -- see `states.build_grid` for why those axes
and not others.  496 points at the default settings.

Throughput, measured on this machine: LAMMPS at N = 2048 scales 1.97x on 2
threads but only 3.3x on 4 and 5.0x on 8, so **more independent runs at 2 threads
beats fewer at 4 or 8** -- 23 x 2 gives ~135 full points/h against ~110 for
11 x 4.  Do not raise --threads to "go faster".

    python run_grid.py                          # all 496, 23 concurrent x 2
    python run_grid.py --blocks B               # just the LJ-HS family
    python run_grid.py --limit 4 --blocks A1    # a taste
    python run_grid.py --dry-run                # list what would run

Resumable: a point whose g.csv and converted c.csv both exist is skipped, so
re-running after an interruption costs nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_md                          # noqa: E402
from states import build_grid          # noqa: E402


def one_point(label: str, spec: dict, threads: int, out: Path) -> tuple[str, float, str]:
    from run_md import (build_input, run_lammps, parse_rdf, parse_thermo,
                        write_g_csv)
    if (out / label / 'g.csv').exists() and \
            (HERE.parent / 'converted' / f'md_{label}' / 'c.csv').exists():
        return label, 0.0, 'skipped (already done)'
    spec = dict(spec)
    if spec.get('rho') is None:
        spec['rho'] = run_md.rho_of_eta(spec['eta'], spec['x1'], spec['ratio'])
    rundir = out / label
    rundir.mkdir(parents=True, exist_ok=True)
    build_input(spec, rundir)
    try:
        wall = run_lammps(rundir, threads)
    except SystemExit as e:
        return label, 0.0, f'FAILED: {e}'
    g = parse_rdf(rundir / 'rdf.out')
    write_g_csv(rundir / 'g.csv', g, spec)
    th = parse_thermo(rundir / 'thermo.out')
    th['wall_seconds'] = round(wall, 1)
    (rundir / 'thermo.json').write_text(json.dumps(th, indent=2))
    subprocess.run([sys.executable, str(HERE.parent / 'convert.py'),
                    '--input', str(rundir / 'g.csv'), '--from', 'g',
                    '--r-convention', 'absolute', '--rho', str(spec['rho']),
                    '--x1', str(spec['x1']), '--to', 'c,S_AL,S_BT',
                    '--r-max', '24', '--n-grid', '4096', '--label', f'md_{label}'],
                   cwd=HERE.parent, capture_output=True)
    return label, wall, f'U*={th["U_star"]:.3f} P*={th["P_star"]:.3f}'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=HERE / 'runs')
    ap.add_argument('--blocks', default='A1,A2,B', help='which grid blocks')
    ap.add_argument('--parallel', type=int, default=23, help='concurrent runs')
    ap.add_argument('--threads', type=int, default=2, help='threads per run')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    grid = build_grid(args.blocks)
    items = list(grid.items())[:args.limit] if args.limit else list(grid.items())
    if args.dry_run:
        for lbl, spec in items:
            print(f'  {lbl:<44} kind={spec["kind"]:<5} '
                  f'rho={run_md.rho_of_eta(spec["eta"], spec["x1"], spec["ratio"]):.5f}')
        print(f'{len(items)} points')
        return 0
    print(f'grid [{args.blocks}]: {len(items)} points, {args.parallel} concurrent '
          f'x {args.threads} threads', flush=True)
    t0 = time.time()
    done = fail = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(one_point, lbl, spec, args.threads, args.out): lbl
                for lbl, spec in items}
        for fut in as_completed(futs):
            label, wall, msg = fut.result()
            done += 1
            fail += msg.startswith('FAILED')
            el = time.time() - t0
            eta_h = (el / done * (len(items) - done)) / 3600.0
            print(f'  [{done}/{len(items)}] {label}  {wall:.0f}s  {msg}'
                  f'   (elapsed {el/3600:.2f} h, eta {eta_h:.2f} h)', flush=True)
    print(f'grid complete: {done - fail} ok, {fail} failed, '
          f'{(time.time()-t0)/3600:.2f} h', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
