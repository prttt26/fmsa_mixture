# FMSA for hard-core Yukawa mixtures — Python reference implementation

The first-order mean spherical approximation (FMSA) for an *N*-component fluid of
additive hard cores carrying *M* Yukawa tails, including Lennard-Jones mixtures
mapped onto a hard core plus two Yukawa tails ("2YK"). Everything below is
evaluated without iterating the Ornstein–Zernike equation.

| quantity | how it is obtained |
|---|---|
| direct correlation function `c_ij(r)` | finite closed form (polynomial × exponential pieces) |
| `ĉ_ij(k)`, structure factors `S_ij(k)` | closed form at any `k` |
| radial distribution function `g_ij(r)` | inversion of the closed-form `ĥ(k)` (Fourier or Laplace route); contact values algebraic |
| energy route: `βF_ex/V`, `βP`, `βμ_i` | λ-integration, Laplace space; derivatives by finite difference |
| compressibility route: `ĉ_ij(0)`, `(∂βP/∂ρ)_x`, stability matrix, `βP` and `βF_ex/V` | read off `ĉ(0)`; `P` and `F` by integration along the constant-composition ray |

Method: R. Pan and J. Wu, *The First-Order Mean Spherical Approximation for
Hard-Core Yukawa Mixtures: the Direct Correlation Function in Finite Closed Form*
(in preparation). It completes for the mixture DCF the programme of Tang and Lu
[J. Chem. Phys. 99, 9828 (1993); Mol. Phys. 84, 89 (1995)] and Tang, Tong and Lu
[Fluid Phase Equilib. 134, 21 (1997)].

## Requirements

Python 3 with NumPy and SciPy (`pip install -r requirements.txt`).
Tested with Python 3.12, NumPy 1.26, SciPy 1.17; other versions have not been tried.

## Files

```
fmsa_double_prop.py       the model: DCF, RDF, S(k), hard-sphere references
fmsa_free_energy.py       energy route (free energy, pressure, chemical potentials)
fmsa_compressibility.py   compressibility route
examples/example_binary.py   every quantity at one binary state
tests/test_fmsa_mixture.py   self-checks (identities and two-path cross-checks)
paper/                    the scripts and data behind the paper's figures and numbers
```

`paper/` is self-contained and has its own README; the package above does not
depend on it.

Keep the three modules in one directory and put it on `sys.path`; they import
each other by module name.

## Quick start

```python
import numpy as np
from fmsa_double_prop import FMSA_double_prop
from fmsa_free_energy import FMSAFreeEnergy
import fmsa_compressibility as comp

m = FMSA_double_prop(sig_pure=[1.0, 2.0], eps_pure=[1.0, 0.5],
                     x=[0.25, 0.75], rhostar=0.139, tstar=1.0, verbose=False)

c  = m.get_c_total(np.linspace(0.05, 3.0, 200))   # c_ij(r), shape (N, N, n_r)
g  = m.get_g_r(np.linspace(0.5, 4.0, 200))        # g_ij(r), shape (N, N, n_r)
S  = m.get_S_k(1.0)                               # S_ij(k), shape (N, N)

fe = FMSAFreeEnergy(m)
fe.get_free_energy(), fe.get_pressure(), fe.get_chemical_potentials()

comp.chat0(m), comp.dbetaP_drho(m), comp.ray_pressure(m), comp.ray_free_energy(m)
```

`python examples/example_binary.py` prints all of these for one state (≈ 3 s).

## Input and units

Reduced units throughout: lengths in σ of the reference species, energies in its ε,
`tstar = k_B T/ε`, `rhostar` = total number density.

* `sig_pure`, `eps_pure` — per-species parameters, shape `(N,)`.
* `x` — mole fractions (sum to 1); partial densities are `x * rhostar`.
* `input_type='lj'` (default) — Lennard-Jones input. Hard-core diameters are
  Barker–Henderson diameters at `tstar`; the tail is the two-Yukawa fit to LJ.
  `input_type='bh'` — `sig_pure` *are* the hard-core diameters; use this for a
  genuine hard-core Yukawa model.
* `n_tails` — number of Yukawa tails (2 for the LJ fit, 1 for a single Yukawa);
  `z`, `k` override the decay rates and amplitudes.
* `eps_ij` — optional cross-interaction matrix replacing the Berthelot mean
  `√(ε_i ε_j)`; needed e.g. for symmetric mixtures with `ε_12 ≠ ε_11`.
* `hs_fmt` — hard-sphere functional for the FMT reference: `'wb1'` (White Bear I,
  default), `'wb2'` (White Bear II), `'rosenfeld'`.

## Things worth knowing

**Two hard-sphere references.** `get_c_total(r, hs_ref=...)` adds either the FMT
hard-sphere DCF (`'fmt'`, default) or the Percus–Yevick one (`'py'`) to the
first-order part. PY is the reference FMSA formally perturbs around (the exact MSA
at zero coupling); FMT gives the better hard-sphere thermodynamics.
`fmsa_compressibility` takes the same choice as `hs='fmt' | 'py'`; with `'fmt'` its
hard-sphere block is the exact Hessian of the same functional the energy route
uses, so the two routes share one hard-sphere reference.

**Energy-route `incore` option.** `incore='lj'` (default) adds the LJ shell
between the hard core and σ_LJ — a correction to the *potential*, meaningful for
LJ input. `incore='none'` keeps exactly the potential the DCF sees (hard core
plus Yukawa tail) and is the like-for-like partner of the compressibility route.
`get_free_energy()` is `βF_ex/V` relative to the ideal gas;
`get_excess_free_energy()` is the *mixing* excess per particle relative to the
pure species, a different quantity.

**The two routes do not agree, by construction.** The energy and compressibility
routes give different pressures and free energies even at `incore='none'`. The
gap is first order in the coupling and is a property of the MSA closure, not a
numerical error: the energy route's first-order term is the exact
high-temperature term, while the compressibility route reads the MSA's
first-order DCF.

**The compressibility route is used only along a ray.** The compressibility
relation is a derivative at fixed composition, so pressure and free energy are
obtained uniquely by integrating in total density at fixed `x`. Chemical
potentials from this route would require leaving the ray, where the result
depends on the path for any approximate closure; they are therefore not
provided. Use the energy route for `μ_i`.

**Stability.** `comp.stability_matrix(m)` returns `M_ij = δ_ij/ρ_i − ĉ_ij(0)`.
The mixture is locally stable while all eigenvalues are positive; `det M = 0`
is the spinodal, covering both the density-like and the composition-like
(demixing) channel.

**RDF accuracy.** The Fourier route's accuracy is set by `k_max` (default 1500):
the residual `|g(r < R)|` is of order 1e-3. Values within ~2 % of contact are
smeared by the jump at contact; take contact values from `get_g_contact`.
`route='laplace'` is the pointwise alternative.

**Range.** The benchmark in the paper covers `T* ≲ 3` for LJ input. For LJ input
at much higher `T*`, the Barker–Henderson core shrinks and the short-range
repulsive tail of the 2YK fit becomes large near contact; first-order results
there should not be trusted. Hard-core Yukawa input (`'bh'`) has no such issue.

**Docstrings.** Comments and docstrings in `fmsa_double_prop.py` and
`fmsa_free_energy.py` refer to the authors' development tree (lecture notes,
Lean files, internal result notes). They are left verbatim so the code is
identical to the version the paper's numbers were produced with; the references
can be ignored.

## Checks

```
python tests/test_fmsa_mixture.py      # ≈ 15 s, 10 checks
```

Each check is an identity or a comparison of two independent computations:
`ĉ(0)` from the closed-form transform against real-space quadrature; symmetry and
continuity of `c_ij(r)` at the unlike-pair breakpoint; two identical species
reducing to one; `g ≈ 0` inside the core; the analytic hard-sphere block against
a finite-difference Hessian of the energy route's hard-sphere free energy (three
functionals); the FMT identity in real space; ray integration of the hard-sphere
block returning the hard-sphere free energy; quadrature convergence; and
Gibbs–Duhem for the energy route.
