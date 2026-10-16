# `md/` — LAMMPS-generated binary-LJ-mixture benchmark structure

Generates partial radial distribution functions g_ij(r) for size-asymmetric
binary Lennard-Jones mixtures by molecular dynamics, out to ~half the box — far
past the published tables' 2.5σ — so the OZ inversion to c_ij(r) (via
`../convert.py`) is not truncation-limited. This is the path the literature
survey pointed to: the only on-target / other-size-ratio structure data available
is the data we make ourselves, exactly as Huber & Ely did.

Full LJ 12-6, Lorentz–Berthelot mixing, NVT-MD — same model and method as NIST
TN 1331 / Ely (1986), so results are directly comparable.

## Files

| | |
|---|---|
| `run_md.py` | build LAMMPS input, run `lmp_pyNO`, parse → `runs/<label>/g.csv` + `thermo.json` |
| `states.py` | state-point registry: `validation`, `ratio1.5/2.5/3.0`, and the sweep grid |
| `validate.py` | compare a run to TN 1331 (structure) and Ely (thermo), and close the g→c loop |
| `run_sweep.py` | run the broad grid concurrently, converting each point to c and S(k) |
| `runs/<label>/` | per-run outputs: `in.lammps`, `log.lammps`, `rdf.out`, `g.csv`, `thermo.json`, `params.json` |

Engine: `~/.conda/envs/lammps_NO_python/bin/lmp_pyNO` (LAMMPS 12 Jun 2025), 48 cores.

## Conventions (must match to compare)

- **Reduced units** (`units lj`): species 1 is the reference, σ11 = ε11 = mass = 1.
  So T\* = kT/ε11 and ρ\* = ρσ11³, as in TN 1331 / Ely. **Species 2 is the large
  one** (σ22 = ratio), matching TN 1331 (opposite to Yu & Jin's labelling).
- **Density by packing fraction.** η = (π/6)ρ Σ x_i σ_ii³ is held fixed across
  size ratios (default η ≈ 0.455, the validation point's value); ρ\* is derived
  per ratio. Holding ρ*σ11³ fixed instead would drive η → 0.75 at ratio 3 and
  crystallise. Each run records both η and ρ\* in `params.json`.
- **Long cutoff.** r_c = min(4.5 σ22, 0.49 L) with `pair_modify tail yes`; the
  RDF is collected to the same range. Including more of the (negligible) LJ tail
  is *more* accurate, not less, and lets g_ij reach ~7–13 σ11.
- Overlaps from random insertion are removed by a **soft-potential push-off** ramp
  before the LJ dynamics — needed at this packing with the large species.

## Validation (size ratio 2 → the TN 1331 target)

`python run_md.py --state validation` then `python validate.py` (N=2048, 163 s):

- **Structure** — our g_ij(r) vs TN 1331 Appendix A, mean |Δ|:
  **g22 = 0.004, g12 = 0.011, g11 = 0.036** (g11 is the minority small species,
  worst statistics; the majority g22 matches to <1%).
- **Thermodynamics** — vs Ely 1986 (x=0.25, ε=0.5, T\*=1):
  **U\* = −2.737 ± 0.002 vs −2.728** (0.3%), **P\* = 0.515 ± 0.001 vs 0.510** (1%),
  T\* = 0.999.
- **Loop closure** — OZ-inverting our long g_ij (to 9 σ11) reproduces TN 1331
  Appendix C's inner-core c_ij: **c11 < 0.2%, c12 ~ 2%, c22 ~ 3–20%**. The larger
  c22 spread is the extreme k→0 amplification the c22 investigation quantified
  (M₂₂² ≈ 453), not a pipeline error; our c22 is now derived from a g that is not
  truncated.

All three pass — the generator is trusted.

## Data produced (the literature gaps)

| state | ratio | ρ\* | η | g_ij reach | U\* | P\* |
|---|---|---|---|---|---|---|
| validation | 2.0 | 0.139 | 0.455 | 9.0 σ11 | −2.737 | 0.515 |
| ratio1.5 | 1.5 | 0.312 | 0.455 | 6.7 σ11 | −2.922 | 1.221 |
| ratio2.5 | 2.5 | 0.073 | 0.455 | 11.2 σ11 | −2.618 | 0.262 |
| ratio3.0 | 3.0 | 0.042 | 0.455 | 13.5 σ11 | −2.540 | 0.150 |

Each has `runs/<label>/g.csv` and a converted `../converted/md_<label>/{c,S_AL,S_BT}.csv`.
Contact positions, peaks, and the g→1 plateau are all physical across ratios.

## The broad sweep — RUN and complete (60 points)

`states.build_sweep()` is an **equidistant** grid: ratios {1.5, 2.0, 2.5, 3.0} ×
packings η {0.30, 0.40, 0.50} × compositions x1 {0.1, 0.3, 0.5, 0.7, 0.9} =
**60 points**, ε ratio 0.5, T\*=1. Each point's g_ij is in `runs/sweep_*/g.csv`
and its c_ij / S(k) in `../converted/md_sweep_*/`.

Per point ≈ 2.7 min solo / ≈ 5.5 min at 3 concurrent; the full grid ran in ≈ 1.7 h.

**All 60 are clean single-phase liquids** — an anomaly scan (thermostat T\*, g_ij
peak heights, and an outer-region order parameter = std of g about 1 for
r > 0.6 r_max) flags **none**: T\* = 0.999–1.001 throughout, g11 peaks 2.6–5.3,
order parameter ≤ 0.023 (a liquid; a crystal would show persistent oscillation).
No frozen/crystalline points at these (η ≤ 0.5, T\*=1) conditions.

### One fix along the way
The first pass lost 19 of 60 points: dense, small-particle-dominated states
(x1 = 0.7/0.9) exploded (E_pair → 1e19, lost atoms) when LJ turned on with residual
overlaps the soft push-off had not fully cleared. Fixed by a **minimize +
bounded-displacement (`fix nve/limit`) warmup** after the LJ potential is set and
before the thermostat — it absorbs the overlap energy. `run_sweep.py` skips
already-completed points, so only the 19 were re-run. The warmup changes the
equilibration path, not the equilibrium state.

Run / extend the sweep with:

```bash
python run_sweep.py                     # full 60-point grid, ~1.7 h; resumes if interrupted
python run_sweep.py --limit 12          # a 12-point taste
python run_sweep.py --parallel 6 --threads 8
```

## Small-k / finite-size trustworthiness (`finite_size.py`)

The one quantity where NVT-MD could differ from a grand-canonical run is the
small-k structure factor / compressibility. `finite_size.py` runs the validation
state at N = 512…8192 and pushes each g through the same g→S pipeline, extracting
S_ij(k→0) by an S = S0 + a·k² fit and extrapolating vs 1/N. The verdict is a mix
of ensemble and FT-truncation effects, and it partitions cleanly:

| quantity | small-k behaviour | trust |
|---|---|---|
| real-space g_ij(r) | validated to a few % vs TN 1331 | ✅ |
| inner/mid c_ij(r) | matches TN 1331 App. C inner core | ✅ |
| S_ij(k) for k ≳ 1.5 (main peak on) | smooth, converged | ✅ |
| **S_CC(k→0)** | flat ≈ 0.19, ripple 0.014, ~3 % across N | ✅ |
| S11(k→0) | ≈ 1.24, ~4 % across N | ✅ |
| **S_NN(k→0)** (= χ_T/χ_id) | **non-monotone** (0.21→hump 0.28), ripple 0.070 | ❌ |
| **S22(k→0)** | swings 0.0004–0.09 in one run, ripple 0.079, 68 % across N | ❌ |

So S_NN(k→0) and S22(k→0) — hence the **compressibility read off S(k→0)** — are
dominated by the truncated-g FT ripple, exactly the k→0 sensitivity the c22
investigation flagged (`../../numerical_notes/rootcause/c22_outer_inversion_artifact.md`).
The N-ladder does not rescue them; a finite-box FT simply cannot deliver a
physical S(k→0) here. **Read χ_T from the EOS (dP/dρ), which the sweep's P\*(ρ)
values provide, not from S(k→0).** Everything else — the real-space structure, the
inner/mid DCF, S(k) past its first peak, and S_CC(k→0) — is trustworthy.

### The physical compressibility, from the EOS (`eos.py`)

Since S(k→0) is untrustworthy, `eos.py` gets χ_T the right way: a ρ\* ladder at
x1=0.25, T\*=1 (ρ\* = 0.100…0.180), fit P\*(ρ\*), and
χ_T/χ_id = S_NN(0) = 1/(dP\*/dρ\*) at ρ\*=0.139. The result nails it:

| route | χ_T/χ_id | |
|---|---|---|
| **EOS dP\*/dρ\* (our MD, physical)** | **0.055 ± 0.000** | ✅ in the theory band |
| MSA / PY / HNC / HNCB (repo) | 0.056 – 0.080 | reference |
| TN 1331 published ĉ(0) | 0.10 | ✗ the c22 artefact, 2× high |
| FT S(k→0) of our g | ~0.2 – 0.25 | ✗ ripple, 4× high |

So our MD compressibility (0.055, at the MSA-like lower edge of the band) is
physical, and it independently confirms both that the published ĉ(0)=0.10 is the
inflated artefact and that the FT S(k→0) route cannot be used. P\*(ρ\*) is smooth
and tightly determined (errors ~0.001–0.003); cached in `runs/eos.json`.

Result cached in `runs/finite_size.json`.

## Regenerate / extend

```bash
python run_md.py --state validation --threads 16      # or ratio1.5 / ratio2.5 / ratio3.0
python validate.py                                    # validation report + closure
python ../convert.py --input runs/<label>/g.csv --from g --r-convention absolute \
       --rho <rho*> --x1 0.25 --to c,S_AL,S_BT        # g -> c, S(k)
```

Add a state point by editing `states.py` (`ratio`, `eps_ratio`, `x1`, `T`, and
either `rho` or `eta`); everything downstream follows.


## The gridded batch (2026-07-30)

`run_grid.py` adds 496 points that vary **size ratio, energy ratio, packing and
temperature** together, plus an LJ + hard-sphere family. Everything before it
sits at eps_ratio = 0.5 and T* = 1, so neither of the last two axes had ever been
swept in our own data. Grid definition and the reasoning are in
`states.build_grid`; the results are
`numerical_notes/results/benchmark/LAMMPS_MD/md_grid_dcf.md` and
`numerical_notes/results/benchmark_rdf/LAMMPS_MD/md_grid_rdf.md`.

    python run_grid.py                  # all 496, 23 concurrent x 2 threads
    python run_grid.py --blocks B       # the LJ-HS family alone
    python run_grid.py --dry-run        # list without running
    bash after_grid.sh                  # wait, then rebuild both pools and re-score

**Use 2 threads, not more.** Measured here on a 54k-step probe: 1 thread 274 s,
2 threads 139 s (1.97x), 4 threads 82.6 s (3.3x), 8 threads 54.9 s (5.0x). More
independent runs at 2 threads beats fewer at 4 or 8 -- 23 x 2 gives ~135 full
points/h against ~110 for 11 x 4. The full batch took 9.45 h with 0 failures;
it runs longer than the throughput figure suggests because cost scales as
rc^3 = (4.5 sigma_22)^3, so a ratio-3 point costs 20x a ratio-1.5 one, and this
machine is shared.

**The LJ-HS family** (`kind='ljhs'`) makes species 2 a hard sphere: every pair it
takes part in loses its tail, which is what eps_2 -> 0 means under
Lorentz-Berthelot. MD cannot integrate a discontinuous potential, so the sphere
is the usual WCA stand-in -- LJ cut at its minimum and shifted, via a per-pair
cutoff and `pair_modify shift yes`. The tail correction is off for those runs
(LAMMPS would compute one from an LJ tail that is not there); that costs the 1-1
pair ~1e-5 eps in the energy and nothing in the forces, so g_ij is unaffected and
only the reported P and U shift slightly. **Read
`numerical_notes/results/benchmark_rdf/LAMMPS_MD/md_grid_rdf.md` section 3 before
drawing hard-sphere conclusions from these runs**: the models read the WCA sphere
as a hard one and overshoot its g peak by 27-81 %.
