#!/usr/bin/env python3
"""LAMMPS-driven binary Lennard-Jones mixture MD -> partial RDFs g_ij(r).

Produces the benchmark structure data the literature does not tabulate: partial
radial distribution functions for a size-asymmetric binary LJ mixture, out to
~half the box (far past the published tables' 2.5 sigma), so the OZ inversion to
c_ij(r) is not truncation-limited.  Full LJ 12-6, Lorentz-Berthelot mixing,
NVT-MD -- the same model and method as NIST TN 1331 / Ely (1986), so results are
directly comparable.

Pipeline
--------
    run_md.py --state validation           # -> md/runs/<label>/g.csv (+ thermo.json)
    convert.py --input md/runs/<label>/g.csv --from g --r-convention absolute \
               --rho <rho*> --x1 <x1> --to c,S_AL,S_BT

Reduced units (LAMMPS `units lj`): species 1 is the reference, sigma11 = eps11 =
mass = 1, so T* = kT/eps11 and rho* = rho sigma11^3, exactly as in TN 1331 / Ely.

Density is set by PACKING FRACTION eta = (pi/6) rho sum_i x_i sigma_ii^3, not by
rho* sigma11^3, so different size ratios stay a comparable single-phase liquid
(holding rho* sigma11^3 fixed would drive eta -> 0.75 at ratio 3 and crystallise).
The validation point's eta ~ 0.455 is the default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
LMP = Path.home() / '.conda/envs/lammps_NO_python/bin/lmp_pyNO'


# --- state-point physics --------------------------------------------------

WCA_CUT = 2.0 ** (1.0 / 6.0)     # LJ minimum: where a purely repulsive pair stops


def mixing(ratio: float, eps_ratio: float, kind: str = 'lj'):
    """Lorentz-Berthelot sigma/eps matrices (species 1 the reference).

    kind='ljhs' makes species 2 a HARD SPHERE: every pair it takes part in loses
    its attractive tail, which is what eps_2 -> 0 means under Lorentz-Berthelot
    (eps_12 = sqrt(eps_1 * 0) = 0).  MD cannot integrate a discontinuous
    potential, so the hard sphere is the usual WCA stand-in -- LJ truncated at
    its minimum and shifted up -- and the REPULSIVE scale is kept at eps_1 so the
    sphere's effective (Barker-Henderson) diameter is a clean function of T.
    `eps_ratio` is meaningless for this kind and is recorded as 0.
    """
    s12 = 0.5 * (1.0 + ratio)
    sig = ((1.0, s12), (s12, ratio))
    if kind == 'ljhs':
        return sig, ((1.0, 1.0), (1.0, 1.0))
    e12 = math.sqrt(eps_ratio)
    eps = ((1.0, e12), (e12, eps_ratio))
    return sig, eps


def pair_block(sig, eps, kind: str) -> str:
    """The pair_coeff lines plus the pair_modify that goes with them.

    For 'ljhs' the two pairs involving the hard sphere carry their own cutoff at
    the LJ minimum, and `shift yes` turns exactly those into WCA (u_LJ + eps).
    The tail correction is then switched OFF: LAMMPS would compute it from the LJ
    form beyond each cutoff, which for a purely repulsive pair is a tail that is
    not there.  That costs the 1-1 pair its own tail correction -- worth ~1e-5
    eps in the energy at rc = 4.5 sigma and NOTHING in the forces, so g_ij is
    unaffected and only the reported P and U shift slightly.
    """
    L = [f'pair_coeff 1 1 {eps[0][0]:.6f} {sig[0][0]:.4f}']
    if kind == 'ljhs':
        L += [f'pair_coeff 2 2 {eps[1][1]:.6f} {sig[1][1]:.4f} '
              f'{WCA_CUT * sig[1][1]:.4f}',
              f'pair_coeff 1 2 {eps[0][1]:.6f} {sig[0][1]:.4f} '
              f'{WCA_CUT * sig[0][1]:.4f}',
              'pair_modify shift yes']
    else:
        L += [f'pair_coeff 2 2 {eps[1][1]:.6f} {sig[1][1]:.4f}',
              f'pair_coeff 1 2 {eps[0][1]:.6f} {sig[0][1]:.4f}',
              'pair_modify tail yes']
    return '\n'.join(L)


def eta_of(rho: float, x1: float, ratio: float) -> float:
    x2 = 1.0 - x1
    return (math.pi / 6.0) * rho * (x1 * 1.0 + x2 * ratio ** 3)


def rho_of_eta(eta: float, x1: float, ratio: float) -> float:
    x2 = 1.0 - x1
    return eta * 6.0 / math.pi / (x1 * 1.0 + x2 * ratio ** 3)


# --- LAMMPS input ---------------------------------------------------------

INPUT = """\
# binary LJ mixture, size ratio {ratio}, eps ratio {eps_ratio}, x1 ={x1}, T*={T}
# rho*={rho:.5f}  eta={eta:.4f}  N={N}  L={L:.4f}  rc={rc:.3f}
units lj
atom_style atomic
boundary p p p

region box block 0 {L:.6f} 0 {L:.6f} 0 {L:.6f}
create_box 2 box
create_atoms 1 random {N1} {seed1} box
create_atoms 2 random {N2} {seed2} box
mass * 1.0
velocity all create {T} {seedv} loop geom

# --- soft push-off to relax the random overlaps ---
pair_style soft {smax:.3f}
pair_coeff * * 0.0
variable pre equal ramp(0,200)
fix push all adapt 1 pair soft a * * v_pre
fix lim  all nve/limit 0.05
timestep 0.005
thermo 2000
run {push} post no
unfix push
unfix lim

# --- full LJ 12-6, Lorentz-Berthelot, long cutoff + tail correction ---
pair_style lj/cut {rc:.4f}
{pair_block}
neighbor 0.5 bin

# robust warmup: minimize the worst residual overlaps, then bounded-displacement
# dynamics to bleed off overlap energy, before the thermostat runs.  Without this,
# dense small-particle-dominated points explode (E_pair -> 1e19) when LJ turns on.
minimize 1.0e-4 1.0e-6 2000 20000
velocity all create {T} {seedw} loop geom
reset_timestep 0
fix warm all nve/limit 0.05
run 5000 post no
unfix warm

fix nvt all nvt temp {T} {T} {tdamp:.3f}
thermo 5000
run {equil} post no
reset_timestep 0

# --- production: partial g_ij and thermodynamic averages ---
compute rdf all rdf {nbins} 1 1 1 2 2 2 cutoff {rc_rdf:.4f}
fix grdf all ave/time {every} {nrep} {prod} c_rdf[*] file rdf.out mode vector
variable Ustar equal pe
variable Pstar equal press
variable Tstar equal temp
fix tstat all ave/time {every} {trep} {tblock} v_Ustar v_Pstar v_Tstar &
    file thermo.out
run {prod} post no
print "MD_DONE natoms=$(count(all))"
"""


def build_input(p: dict, rundir: Path) -> None:
    kind = p.get('kind', 'lj')
    sig, eps = mixing(p['ratio'], p['eps_ratio'], kind)
    x1 = p['x1']
    N = p['N']
    N1 = round(x1 * N)
    N2 = N - N1
    rho = p['rho']
    L = (N / rho) ** (1.0 / 3.0)
    # long cutoff so g_ij reaches well past the published 2.5*sigma_ij; capped
    # below L/2.  tail correction makes the extra range thermodynamically cheap.
    rc = min(4.5 * p['ratio'], 0.49 * L)
    rc_rdf = min(rc, 0.499 * L)
    prod = p['prod']
    text = INPUT.format(
        ratio=p['ratio'], eps_ratio=p['eps_ratio'], x1=x1, T=p['T'],
        rho=rho, eta=eta_of(rho, x1, p['ratio']), N=N, L=L, rc=rc,
        N1=N1, N2=N2, seed1=12345, seed2=54321, seedv=98765, seedw=24680,
        smax=max(sig[1][1], 1.0), push=p['push'], equil=p['equil'],
        s11=sig[0][0], s22=sig[1][1], s12=sig[0][1],
        e11=eps[0][0], e22=eps[1][1], e12=eps[0][1],
        pair_block=pair_block(sig, eps, kind),
        tdamp=100 * 0.005, nbins=p['nbins'], rc_rdf=rc_rdf,
        every=p['every'], nrep=prod // p['every'], prod=prod,
        tblock=max(prod // 20, p['every']),           # ~20 thermo blocks -> error bars
        trep=max(prod // 20, p['every']) // p['every'])
    (rundir / 'in.lammps').write_text(text)
    (rundir / 'params.json').write_text(json.dumps(
        {**p, 'N1': N1, 'N2': N2, 'L': L, 'rc': rc, 'rc_rdf': rc_rdf,
         'sigma': sig, 'eps': eps, 'kind': kind,
         'eta': eta_of(rho, x1, p['ratio'])}, indent=2))


# --- run and parse --------------------------------------------------------

def run_lammps(rundir: Path, threads: int) -> float:
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    cmd = [str(LMP), '-in', 'in.lammps', '-log', 'log.lammps']
    if threads > 1:
        cmd += ['-sf', 'omp', '-pk', 'omp', str(threads)]
    t0 = time.time()
    res = subprocess.run(cmd, cwd=rundir, env=env, capture_output=True, text=True)
    (rundir / 'stdout.txt').write_text(res.stdout)
    if 'MD_DONE' not in res.stdout:
        raise SystemExit(f'LAMMPS did not finish; see {rundir}/stdout.txt\n'
                         + res.stdout[-1500:] + res.stderr[-500:])
    return time.time() - t0


def parse_rdf(path: Path) -> np.ndarray:
    """Last time-averaged block of rdf.out -> array [r, g11, g12, g22]."""
    rows = []
    for ln in path.read_text().splitlines():
        if ln.startswith('#'):
            continue
        f = ln.split()
        if len(f) == 2:          # "timestep nbins" -> a new block starts
            rows = []
            continue
        if len(f) == 8:
            rows.append([float(f[1]), float(f[2]), float(f[4]), float(f[6])])
    return np.array(rows)


def parse_thermo(path: Path) -> dict:
    vals = []
    for ln in path.read_text().splitlines():
        if ln.startswith('#'):
            continue
        f = ln.split()
        if len(f) == 4:          # timestep, U*, P*, T*
            vals.append([float(f[1]), float(f[2]), float(f[3])])
    a = np.array(vals)
    return {'U_star': a[:, 0].mean(), 'U_star_sd': a[:, 0].std(),
            'P_star': a[:, 1].mean(), 'P_star_sd': a[:, 1].std(),
            'T_star': a[:, 2].mean(), 'n_blocks': len(a)}


def write_g_csv(path: Path, g: np.ndarray, p: dict) -> None:
    with path.open('w', newline='') as fh:
        fh.write(f'# LAMMPS NVT-MD binary LJ mixture, size ratio {p["ratio"]}, '
                 f'eps ratio {p["eps_ratio"]}\n')
        fh.write(f'# x1={p["x1"]} T*={p["T"]} rho*={p["rho"]:.5f} '
                 f'eta={eta_of(p["rho"], p["x1"], p["ratio"]):.4f} N={p["N"]}\n')
        fh.write('# r is absolute (units sigma11); species 1 = small reference\n')
        fh.write('r,g11,g12,g22\n')
        for r, g11, g12, g22 in g:
            fh.write(f'{r:.4f},{g11:.5f},{g12:.5f},{g22:.5f}\n')


# --- driver ---------------------------------------------------------------

def resolve_state(name: str, overrides: dict) -> dict:
    from states import STATES
    if name not in STATES:
        raise SystemExit(f'unknown --state {name}; choices: {list(STATES)}')
    p = dict(STATES[name])
    p.update({k: v for k, v in overrides.items() if v is not None})
    if 'rho' not in p or p.get('rho') is None:
        p['rho'] = rho_of_eta(p['eta'], p['x1'], p['ratio'])
    p['label'] = p.get('label', name)
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--state', required=True, help='name in states.STATES')
    ap.add_argument('--out', type=Path, default=HERE / 'runs')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--N', type=int, default=None)
    ap.add_argument('--prod', type=int, default=None, help='production steps')
    ap.add_argument('--equil', type=int, default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(HERE))
    p = resolve_state(args.state, {'N': args.N, 'prod': args.prod, 'equil': args.equil})
    rundir = args.out / p['label']
    rundir.mkdir(parents=True, exist_ok=True)

    if not LMP.exists():
        raise SystemExit(f'LAMMPS binary not found: {LMP}')
    build_input(p, rundir)
    print(f'[{p["label"]}] ratio={p["ratio"]} eps={p["eps_ratio"]} x1={p["x1"]} '
          f'T*={p["T"]} rho*={p["rho"]:.4f} eta={eta_of(p["rho"], p["x1"], p["ratio"]):.3f} '
          f'N={p["N"]}  running LAMMPS ({args.threads} threads)...')
    wall = run_lammps(rundir, args.threads)

    g = parse_rdf(rundir / 'rdf.out')
    write_g_csv(rundir / 'g.csv', g, p)
    thermo = parse_thermo(rundir / 'thermo.out')
    thermo['wall_seconds'] = round(wall, 1)
    (rundir / 'thermo.json').write_text(json.dumps(thermo, indent=2))

    print(f'[{p["label"]}] done in {wall:.0f}s  ->  {rundir}/g.csv  '
          f'(r to {g[-1, 0]:.1f} sigma11, {len(g)} bins)')
    print(f'  thermo: U*={thermo["U_star"]:.3f}+/-{thermo["U_star_sd"]:.3f}  '
          f'P*={thermo["P_star"]:.3f}+/-{thermo["P_star_sd"]:.3f}  '
          f'T*={thermo["T_star"]:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
