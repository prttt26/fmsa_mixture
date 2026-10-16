"""State-point registry for run_md.py.

Each entry is a binary-LJ-mixture NVT-MD state point. `ratio` = sigma22/sigma11
(species 2 large), `eps_ratio` = eps22/eps11, `x1` = mole fraction of the small
species (the reference). Density is given either directly as `rho` (rho* sigma11^3)
or as packing fraction `eta`; run_md derives rho from eta when rho is absent.

Sampling defaults (N, prod, equil, push, nbins, every) are chosen for smooth
g_ij at modest cost; override on the CLI (`--N`, `--prod`, `--equil`).
"""

# shared sampling defaults
_S = dict(N=2048, push=8000, equil=30000, prod=200000, nbins=450, every=25)

STATES = {
    # --- validation: reproduce the NIST TN 1331 target (size ratio 2) ---
    # rho* = 0.139 exactly (eta ~ 0.455); compared against tn1331 App. A g_ij and
    # the Ely thermo row (P* ~ 0.51, U* ~ -2.728).
    'validation': dict(ratio=2.0, eps_ratio=0.5, x1=0.25, T=1.0, rho=0.139, **_S),

    # --- the two coexisting phases of the T* = 0.55 direct-coexistence run ---
    # Read off the density plateaus of runs_vle/vle_r2.0_e0.5_x0.25_T0.550/full
    # (see run_vle_slab.py).  NOT the same composition: the mixture is asymmetric,
    # so the small strongly-bound species partitions into the liquid -- x1 = 0.040
    # in the vapour against 0.087 in the liquid, from an overall 0.25.
    # ⚠ x1 is small in BOTH, so species-1 statistics set N: at x1 = 0.040 a 2048-atom
    # box holds 82 small particles and g_11 is unusable.  N is raised accordingly.
    'vle_vap_T0.55': dict(ratio=2.0, eps_ratio=0.5, x1=0.0404, T=0.55, rho=0.0109,
                          **{**_S, 'N': 8000, 'prod': 400000, 'equil': 60000}),
    'vle_liq_T0.55': dict(ratio=2.0, eps_ratio=0.5, x1=0.0866, T=0.55, rho=0.0802,
                          **{**_S, 'N': 4000, 'prod': 400000, 'equil': 60000}),

    # --- literature gaps: same eps/x1/T as the target, other size ratios, at the
    # target's packing fraction eta ~ 0.455 so they are a comparable liquid ---
    'ratio1.5': dict(ratio=1.5, eps_ratio=0.5, x1=0.25, T=1.0, eta=0.455, **_S),
    'ratio2.5': dict(ratio=2.5, eps_ratio=0.5, x1=0.25, T=1.0, eta=0.455, **_S),
    'ratio3.0': dict(ratio=3.0, eps_ratio=0.5, x1=0.25, T=1.0, eta=0.455, **_S),
}

# --- direct quality audit of the NIST TN 1331 Appendix A tables ------------
# Six of its state points re-simulated at EXACTLY their (T*, rho*, x1,
# eps22/eps11) with size ratio 2, so that |TN - MD| measures the TABLE's error
# instead of a model's.  The validation run already showed this works: at
# T1_eps0.50_x0.2500 it localised eight corrupted g12 points, every one of them
# low by exactly 1.000 (the scan's lost-integer failure mode), where
# suspect.csv had flagged only two of them and two clean points besides.
#
# Chosen to span the damage: three heavily flagged tables, one flagged but
# graded OK, two unflagged controls.  Every flagged column in this set is g12,
# which is the best-sampled pair (N1*N2 pairs), so the audit is not
# statistics-limited where it matters.  Longer production than the sweep,
# because here the data quality IS the measurement.
_A = dict(_S, prod=400000)

TN1331_AUDIT = {
    # heavily flagged
    'tnaudit_T1_eps0.50_x0.0995': dict(ratio=2.0, eps_ratio=0.5, x1=0.0995,
                                       T=1.0, rho=0.121, **_A),   # 27 flags, all g12
    'tnaudit_T1_eps0.50_x0.9005': dict(ratio=2.0, eps_ratio=0.5, x1=0.9005,
                                       T=1.0, rho=0.583, **_A),   # 14 flags, dense
    'tnaudit_T2_eps1.50_x0.7500': dict(ratio=2.0, eps_ratio=1.5, x1=0.75,
                                       T=2.0, rho=0.244, **_A),   # 10 flags, g11+g12
    # flagged but graded OK
    'tnaudit_T1_eps1.00_x0.7500': dict(ratio=2.0, eps_ratio=1.0, x1=0.75,
                                       T=1.0, rho=0.319, **_A),   # 6 flags
    # unflagged controls -- do they really come out clean?
    'tnaudit_T3_eps1.00_x0.2500': dict(ratio=2.0, eps_ratio=1.0, x1=0.25,
                                       T=3.0, rho=0.094, **_A),
    'tnaudit_T3_eps2.00_x0.5000': dict(ratio=2.0, eps_ratio=2.0, x1=0.5,
                                       T=3.0, rho=0.143, **_A),

    # --- full sweep: every remaining Appendix A table that carries a density
    'tnaudit_T1_eps0.50_x0.5000': dict(ratio=2.0, eps_ratio=0.5, x1=0.5, T=1.0, rho=0.19, **_A),
    'tnaudit_T1_eps0.50_x0.7500': dict(ratio=2.0, eps_ratio=0.5, x1=0.75, T=1.0, rho=0.303, **_A),
    'tnaudit_T1_eps1.00_x0.0995': dict(ratio=2.0, eps_ratio=1.0, x1=0.0995, T=1.0, rho=0.127, **_A),
    'tnaudit_T1_eps1.00_x0.1001': dict(ratio=2.0, eps_ratio=1.0, x1=0.1001, T=1.0, rho=0.147, **_A),
    'tnaudit_T1_eps1.00_x0.2500': dict(ratio=2.0, eps_ratio=1.0, x1=0.25, T=1.0, rho=0.2, **_A),
    'tnaudit_T1_eps1.00_x0.9005': dict(ratio=2.0, eps_ratio=1.0, x1=0.9005, T=1.0, rho=0.494, **_A),
    'tnaudit_T1_eps1.00_x0.9502': dict(ratio=2.0, eps_ratio=1.0, x1=0.9502, T=1.0, rho=0.598, **_A),
    'tnaudit_T1_eps1.50_x0.0498': dict(ratio=2.0, eps_ratio=1.5, x1=0.0498, T=1.0, rho=0.128, **_A),
    'tnaudit_T1_eps1.50_x0.0995': dict(ratio=2.0, eps_ratio=1.5, x1=0.0995, T=1.0, rho=0.131, **_A),
    'tnaudit_T2_eps0.50_x0.0498': dict(ratio=2.0, eps_ratio=0.5, x1=0.0498, T=2.0, rho=0.091, **_A),
    'tnaudit_T2_eps0.50_x0.2500': dict(ratio=2.0, eps_ratio=0.5, x1=0.25, T=2.0, rho=0.107, **_A),
    'tnaudit_T2_eps0.50_x0.9502': dict(ratio=2.0, eps_ratio=0.5, x1=0.9502, T=2.0, rho=0.274, **_A),
    'tnaudit_T2_eps1.00_x0.0995': dict(ratio=2.0, eps_ratio=1.0, x1=0.0995, T=2.0, rho=0.098, **_A),
    'tnaudit_T2_eps1.00_x0.1001': dict(ratio=2.0, eps_ratio=1.0, x1=0.1001, T=2.0, rho=0.102, **_A),
    'tnaudit_T2_eps1.00_x0.2500': dict(ratio=2.0, eps_ratio=1.0, x1=0.25, T=2.0, rho=0.117, **_A),
    'tnaudit_T2_eps1.00_x0.9502': dict(ratio=2.0, eps_ratio=1.0, x1=0.9502, T=2.0, rho=0.298, **_A),
    'tnaudit_T2_eps1.50_x0.5000': dict(ratio=2.0, eps_ratio=1.5, x1=0.5, T=2.0, rho=0.165, **_A),
    'tnaudit_T2_eps1.50_x0.9001': dict(ratio=2.0, eps_ratio=1.5, x1=0.9001, T=2.0, rho=0.306, **_A),
    'tnaudit_T2_eps1.50_x0.9502': dict(ratio=2.0, eps_ratio=1.5, x1=0.9502, T=2.0, rho=0.322, **_A),
    'tnaudit_T2_eps2.00_x0.0498': dict(ratio=2.0, eps_ratio=2.0, x1=0.0498, T=2.0, rho=0.111, **_A),
    'tnaudit_T2_eps2.00_x0.0999': dict(ratio=2.0, eps_ratio=2.0, x1=0.0999, T=2.0, rho=0.116, **_A),
    'tnaudit_T2_eps2.00_x0.1001': dict(ratio=2.0, eps_ratio=2.0, x1=0.1001, T=2.0, rho=0.116, **_A),
    'tnaudit_T2_eps2.00_x0.2500': dict(ratio=2.0, eps_ratio=2.0, x1=0.25, T=2.0, rho=0.134, **_A),
    'tnaudit_T3_eps0.50_x0.0498': dict(ratio=2.0, eps_ratio=0.5, x1=0.0498, T=3.0, rho=0.076, **_A),
    'tnaudit_T3_eps0.50_x0.0999': dict(ratio=2.0, eps_ratio=0.5, x1=0.0999, T=3.0, rho=0.078, **_A),
    'tnaudit_T3_eps0.50_x0.2500': dict(ratio=2.0, eps_ratio=0.5, x1=0.25, T=3.0, rho=0.105, **_A),
    'tnaudit_T3_eps0.50_x0.7500': dict(ratio=2.0, eps_ratio=0.5, x1=0.75, T=3.0, rho=0.131, **_A),
    'tnaudit_T3_eps1.00_x0.5000': dict(ratio=2.0, eps_ratio=1.0, x1=0.5, T=3.0, rho=0.117, **_A),
    'tnaudit_T3_eps1.00_x0.7500': dict(ratio=2.0, eps_ratio=1.0, x1=0.75, T=3.0, rho=0.143, **_A),
    'tnaudit_T3_eps1.00_x0.9001': dict(ratio=2.0, eps_ratio=1.0, x1=0.9001, T=3.0, rho=0.159, **_A),
    'tnaudit_T3_eps1.00_x0.9502': dict(ratio=2.0, eps_ratio=1.0, x1=0.9502, T=3.0, rho=0.163, **_A),
    'tnaudit_T3_eps1.50_x0.9001': dict(ratio=2.0, eps_ratio=1.5, x1=0.9001, T=3.0, rho=0.167, **_A),
    'tnaudit_T3_eps1.50_x0.9502': dict(ratio=2.0, eps_ratio=1.5, x1=0.9502, T=3.0, rho=0.167, **_A),
    'tnaudit_T3_eps2.00_x0.0995': dict(ratio=2.0, eps_ratio=2.0, x1=0.0995, T=3.0, rho=0.096, **_A),
    'tnaudit_T3_eps2.00_x0.2500': dict(ratio=2.0, eps_ratio=2.0, x1=0.25, T=3.0, rho=0.11, **_A),
    'tnaudit_T3_eps2.00_x0.7500': dict(ratio=2.0, eps_ratio=2.0, x1=0.75, T=3.0, rho=0.174, **_A),
    'tnaudit_T3_eps3.00_x0.0498': dict(ratio=2.0, eps_ratio=3.0, x1=0.0498, T=3.0, rho=0.107, **_A),
    'tnaudit_T3_eps3.00_x0.0995': dict(ratio=2.0, eps_ratio=3.0, x1=0.0995, T=3.0, rho=0.126, **_A),
    'tnaudit_T3_eps3.00_x0.1001': dict(ratio=2.0, eps_ratio=3.0, x1=0.1001, T=3.0, rho=0.105, **_A),
    'tnaudit_T3_eps3.00_x0.7500': dict(ratio=2.0, eps_ratio=3.0, x1=0.75, T=3.0, rho=0.213, **_A),
    'tnaudit_T3_eps3.00_x0.9005': dict(ratio=2.0, eps_ratio=3.0, x1=0.9005, T=3.0, rho=0.178, **_A),
}
STATES.update(TN1331_AUDIT)

# Broad EQUIDISTANT sweep (ratio x packing x composition): 4 x 3 x 5 = 60 points.
# ~1.5-2 h at 3 concurrent x 16 threads.  eps ratio fixed at 0.5, T* = 1.
SWEEP_RATIOS = [1.5, 2.0, 2.5, 3.0]            # step 0.5
SWEEP_ETA = [0.30, 0.40, 0.50]                 # step 0.1
SWEEP_X1 = [0.10, 0.30, 0.50, 0.70, 0.90]      # step 0.2


def build_sweep():
    """Expand the sweep grid into STATES-style entries (call from a sweep runner)."""
    out = {}
    for ratio in SWEEP_RATIOS:
        for eta in SWEEP_ETA:
            for x1 in SWEEP_X1:
                label = f'sweep_r{ratio}_eta{eta}_x{x1}'
                out[label] = dict(ratio=ratio, eps_ratio=0.5, x1=x1, T=1.0,
                                  eta=eta, label=label, **_S)
    return out


# ---------------------------------------------------------------------------
# GRIDDED BATCH (2026-07-29): sigma ratio x energy ratio x packing x TEMPERATURE
# ---------------------------------------------------------------------------
# Everything before this point holds eps_ratio = 0.5 and T* = 1 (the `sweep`
# block) or sits on TN 1331's own state points (`tnaudit`).  Neither varies the
# energy ratio or the temperature as a controlled axis, and the two open
# questions in the benchmark need exactly those: the peak deficit is a LOW-T
# effect (-20 % at T*=1 against -4 % at T*=3) and the energy ratio has never
# been swept in our own MD at all.
#
# A1 is a clean 4-way factorial so main effects AND interactions are estimable;
# A2 buys composition coverage at more than one (eps, T) without a 5-way
# explosion; B is the LJ + hard-sphere family (species 2 loses its tail).
#
# Cost, measured on this machine: 2 threads is the throughput optimum (1 thread
# 274 s, 2 threads 139 s = 1.97x, 4 threads 3.3x, 8 threads 5.0x on a 54k-step
# probe), so 23 concurrent x 2 threads ~ 135 full points/h.

GRID_A1 = dict(ratio=[1.5, 2.0, 2.5, 3.0], eps_ratio=[0.5, 1.0, 2.0, 3.0],
               eta=[0.20, 0.30, 0.40, 0.50], T=[1.0, 1.5, 2.0, 3.0], x1=[0.50])
GRID_A2 = dict(ratio=[1.5, 2.0, 3.0], eps_ratio=[0.5, 2.0], eta=[0.30, 0.45],
               T=[1.0, 2.0], x1=[0.10, 0.25, 0.75, 0.90])
GRID_B  = dict(ratio=[1.5, 2.0, 2.5, 3.0], eps_ratio=[0.0],
               eta=[0.20, 0.30, 0.40, 0.50], T=[1.0, 2.0, 3.0],
               x1=[0.25, 0.50, 0.75])

# Block C -- the (sigma ratio 2.0, eps ratio 1.5) cell, added 2026-08-14.
#
# GRID_A1 samples eps_ratio in {0.5, 1.0, 2.0, 3.0} and skips 1.5, so the only
# runs that ever landed in this cell were the eight scattered `tnaudit` points
# placed on TN 1331 states.  Eight points cannot support the pressure surface
# dcf_lowk.py fits (~42 terms at degree 4 with three 1/T powers), and the
# consequence is measurable: the cell fails the low-k anchor on 16 of 16 clean
# states, while 16-run cells at other eps ratios fail 0 of 16.
#
# The axes bracket the 23 states that need anchoring here: x1 spans 0.0498 to
# 0.9502 and eta spans 0.118 to 0.513, so the grid must contain that hull in
# (rho_1, rho_2), not merely come close to it.  x1 runs to 0.03 and 0.97
# rather than 0.05 and 0.95 for exactly that reason: at x1 = 0.05 the grid
# lies ON the ray through the x1 = 0.0498 target instead of outside it, and
# a Delaunay hull check put that state out by a hair.
GRID_C  = dict(ratio=[2.0], eps_ratio=[1.5],
               eta=[0.15, 0.25, 0.35, 0.45, 0.52], T=[1.0, 1.5, 2.0, 3.0],
               x1=[0.03, 0.25, 0.50, 0.75, 0.97])


def _expand(g, prefix, kind):
    out = {}
    for ratio in g['ratio']:
        for er in g['eps_ratio']:
            for eta in g['eta']:
                for T in g['T']:
                    for x1 in g['x1']:
                        if kind == 'ljhs':
                            lbl = f'{prefix}_r{ratio}_h{eta}_T{T}_x{x1}'
                        else:
                            lbl = f'{prefix}_r{ratio}_e{er}_h{eta}_T{T}_x{x1}'
                        out[lbl] = dict(ratio=ratio, eps_ratio=er, x1=x1, T=T,
                                        eta=eta, kind=kind, label=lbl, **_S)
    return out


def build_grid(blocks: str = 'A1,A2,B'):
    """The gridded batch.  `blocks` selects which of A1 / A2 / B to expand."""
    want = {b.strip() for b in blocks.split(',')}
    out = {}
    if 'A1' in want:
        out.update(_expand(GRID_A1, 'grid', 'lj'))
    if 'A2' in want:
        out.update(_expand(GRID_A2, 'gridx', 'lj'))
    if 'B' in want:
        out.update(_expand(GRID_B, 'ljhs', 'ljhs'))
    if 'C' in want:
        out.update(_expand(GRID_C, 'gridc', 'lj'))
    return out


def grid_summary():
    for b in ('A1', 'A2', 'B'):
        g = build_grid(b)
        print(f'  block {b}: {len(g):4d} points   e.g. {next(iter(g))}')
    print(f'  TOTAL    : {len(build_grid()):4d} points')
