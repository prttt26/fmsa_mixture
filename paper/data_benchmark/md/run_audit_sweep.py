#!/usr/bin/env python3
"""Run every tnaudit_* state point concurrently, then convert each to c and S(k).

One MD run per NIST TN 1331 Appendix A table that carries a density, placed on
exactly that table's state point, so the two g_ij can be differenced value by
value.  Feeds audit_tn1331_rdf.py.

    python run_audit_sweep.py                 # everything not already done
    python run_audit_sweep.py --parallel 4 --threads 8
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
import run_md                                    # noqa: E402
from states import TN1331_AUDIT                  # noqa: E402


def one(label: str, spec: dict, threads: int, out: Path):
    from run_md import build_input, run_lammps, parse_rdf, parse_thermo, write_g_csv
    if (out / label / 'g.csv').exists():
        return label, 0.0, 'skipped (already done)'
    spec = dict(spec)
    spec.setdefault('label', label)
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
    ap.add_argument('--parallel', type=int, default=4)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', type=Path, default=HERE / 'runs')
    args = ap.parse_args()

    todo = {k: v for k, v in TN1331_AUDIT.items()
            if not (args.out / k / 'g.csv').exists()}
    print(f'{len(todo)} of {len(TN1331_AUDIT)} audit points to run, '
          f'{args.parallel} at a time x {args.threads} threads', flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = {ex.submit(one, k, v, args.threads, args.out): k for k, v in todo.items()}
        for n, fut in enumerate(as_completed(futs), 1):
            label, wall, msg = fut.result()
            print(f'  [{n}/{len(todo)}] {label}  {wall:.0f}s  {msg}'
                  f'   (elapsed {(time.time() - t0) / 60:.0f} min)', flush=True)
    print('audit sweep complete', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
