#!/usr/bin/env python3
"""Isothermal compressibility of the validation state from the EOS (dP*/drho*).

The finite-size check (finite_size.py) showed S(k->0) -- hence the compressibility
read off the FT of g -- is ripple-dominated and untrustworthy. The reliable route
is the equation of state: at fixed composition and T,

    chi_T / chi_id = S_NN(k->0) = 1 / (d(beta P)/d rho)_{T,x}
                   = T* / (dP*/drho*)          (reduced units)
                   = 1 / (dP*/drho*)           at T* = 1.

Runs a rho* ladder at the validation composition (size ratio 2, eps 0.5, x1 0.25,
T*=1), fits P*(rho*), differentiates at rho*=0.139, and compares to the theory
band (MSA/PY/HNC/HNCB: 0.056-0.080) and the artefact value from the published
c-hat(0) (0.10).  This is the physically sound compressibility; FT S(k->0) is not.

    python eos.py                 # run the ladder + fit
    python eos.py --analyze-only
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run_md            # noqa: E402

RHOS = [0.100, 0.115, 0.130, 0.139, 0.150, 0.165, 0.180]
RHO0 = 0.139
BASE = dict(ratio=2.0, eps_ratio=0.5, x1=0.25, T=1.0,
            push=8000, equil=30000, prod=200000, nbins=300, every=25)


def run_one(rho, threads, out):
    from run_md import build_input, run_lammps, parse_rdf, parse_thermo, write_g_csv
    label = f'eos_rho{rho:.3f}'
    rundir = out / label
    if (rundir / 'thermo.json').exists():
        return f'{label}: cached'
    rundir.mkdir(parents=True, exist_ok=True)
    spec = dict(BASE, N=2048, rho=rho, label=label)
    build_input(spec, rundir)
    wall = run_lammps(rundir, threads)
    g = parse_rdf(rundir / 'rdf.out')
    write_g_csv(rundir / 'g.csv', g, spec)
    th = parse_thermo(rundir / 'thermo.out')
    th['wall_seconds'] = round(wall, 1)
    (rundir / 'thermo.json').write_text(json.dumps(th, indent=2))
    return f'{label}: {wall:.0f}s  P*={th["P_star"]:.4f}'


def analyse(out):
    rho, P, Perr = [], [], []
    for r in RHOS:
        f = out / f'eos_rho{r:.3f}' / 'thermo.json'
        if not f.exists():
            continue
        th = json.loads(f.read_text())
        rho.append(r); P.append(th['P_star']); Perr.append(th['P_star_sd'])
    rho, P, Perr = np.array(rho), np.array(P), np.array(Perr)
    if len(rho) < 4:
        raise SystemExit('need >=4 eos points; run without --analyze-only first')

    print('\n  EOS  (ratio 2, eps 0.5, x1 0.25, T*=1)')
    print(f'    {"rho*":>7} {"P*":>9} {"+/-":>7}')
    for r, p, e in zip(rho, P, Perr):
        mark = '  <- rho0' if abs(r - RHO0) < 1e-9 else ''
        print(f'    {r:>7.3f} {p:>9.4f} {e:>7.4f}{mark}')

    coef = np.polyfit(rho, P, 3)
    dPdrho = float(np.polyval(np.polyder(coef), RHO0))
    chi = 1.0 / dPdrho * BASE['T']
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(2000):
        Pk = P + rng.normal(0, np.maximum(Perr, 1e-4))
        dc = np.polyder(np.polyfit(rho, Pk, 3))
        samples.append(1.0 / np.polyval(dc, RHO0))
    chi_sd = float(np.std(samples))

    print(f'\n  cubic-fit slope  dP*/drho* at rho*={RHO0} : {dPdrho:.3f}')
    print(f'  chi_T/chi_id = S_NN(0) = 1/(dP*/drho*)     : {chi:.3f} +/- {chi_sd:.3f}')
    print('\n  comparison:')
    print(f'    EOS route (this, physical)       : {chi:.3f} +/- {chi_sd:.3f}')
    print(f'    MSA/PY/HNC/HNCB band (repo)      : 0.056 - 0.080')
    print(f'    TN 1331 published c-hat(0)       : 0.10   (artefact)')
    print(f'    FT S(k->0) of our g (finite_size): unreliable (ripple)')

    (out / 'eos.json').write_text(json.dumps(
        {'rho': rho.tolist(), 'P_star': P.tolist(), 'P_star_sd': Perr.tolist(),
         'rho0': RHO0, 'dPdrho_rho0': dPdrho,
         'chi_T_over_chi_id': chi, 'chi_sd': chi_sd,
         'theory_band': [0.056, 0.080], 'published_chat0': 0.10}, indent=2))
    print(f'\n  wrote {out / "eos.json"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=HERE / 'runs')
    ap.add_argument('--parallel', type=int, default=4)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--analyze-only', action='store_true')
    args = ap.parse_args()
    if not args.analyze_only:
        print(f'running EOS ladder rho*={RHOS} ...')
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [ex.submit(run_one, r, args.threads, args.out) for r in RHOS]
            for f in as_completed(futs):
                print('  ' + f.result())
    analyse(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
