"""sc_checks.py — self-consistency checks for FMSA mixture variants.

Standalone functions that test whether the inner-core DCF satisfies three
physical constraints:

  1. Origin BC    : c^(1)_ij(r) remains finite as r → 0.
  2. Contact jump : c_inner(R_ij−) ≈ c_outer(R_ij+).
  3. Symmetry     : c^(1)_ij(r) = c^(1)_ji(r).

All functions operate on the duck-typed interface shared by every variant:

    model.N                      int
    model.rho                    ndarray (N,), per-species number densities
    model.R_ij                   ndarray (N,N), contact diameters
    model.K                      ndarray (n_tails, N, N), Yukawa amplitudes
    model.z_poles                ndarray (n_tails, N, N), Yukawa inverse ranges
    model.n_tails                int
    model.get_c1_inner(r,i,j)   float, c^(1)_ij at r < R_ij
    model.get_HS_FMT(r,i,j)     float, c^HS_ij(r)

Compatible with: FMSA_GA_mix_v0_c0sub, FMSA_GA_mix_v1_gonly, FMSA_ga_polycorr,
                 FMSA_GA_mix_v3_contactpoly, FMSA_GA_matrix_mix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:                      # numpy >= 2.0 renamed trapz -> trapezoid
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class OriginBCResult:
    """Origin-BC check result for one pair (i, j).

    r1, r2          radii at which c^(1) was sampled (r2 = r1 / 10)
    c1_at_r1/r2     c^(1)_ij at those radii
    ratio           |c1(r2)| / |c1(r1)|; ≈1 if finite, ≈10 for 1/r divergence
    passed          ratio < tol (default 2.0)
    """
    r1: float
    r2: float
    c1_at_r1: float
    c1_at_r2: float
    ratio: float
    passed: bool


@dataclass
class ContactResult:
    """Contact-jump check result for one pair (i, j).

    c_in        total inner DCF (c_HS + c^(1)) evaluated at r = R_ij − ε·R_ij
    c_out       outer Yukawa DCF evaluated at r = R_ij + ε·R_ij
    abs_jump    |c_in − c_out|
    rel_jump    abs_jump / |c_out|
    passed      rel_jump < tol (default 1e-2)
    """
    c_in: float
    c_out: float
    abs_jump: float
    rel_jump: float
    passed: bool


@dataclass
class SymmetryResult:
    """Symmetry check result for one pair (i, j).

    c_ij, c_ji  c^(1) evaluated at r = r_frac·R_ij from each ordering
    abs_asym    |c_ij − c_ji|
    rel_asym    abs_asym / |c_ij|
    passed      rel_asym < tol (default 1e-8)
    """
    c_ij: float
    c_ji: float
    abs_asym: float
    rel_asym: float
    passed: bool


@dataclass
class PairResult:
    """All three check results for one pair (i, j)."""
    origin_bc: OriginBCResult
    contact: ContactResult
    symmetry: SymmetryResult


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_origin_bc(
    model,
    i: int,
    j: int,
    r_probe: float = 1e-4,
    tol: float = 2.0,
) -> OriginBCResult:
    """Check that c^(1)_ij(r) is finite as r → 0.

    Evaluates c^(1) at r1 = r_probe·R_ij and r2 = r1/10.  If c^(1) has a
    1/r singularity the ratio |c(r2)/c(r1)| ≈ 10; for a finite limit it is ≈1.

    Parameters
    ----------
    r_probe : fractional probe radius relative to R_ij (default 1e-4).
    tol     : ratio threshold for 'passed' (default 2.0).
    """
    R  = float(model.R_ij[i, j])
    r1 = r_probe * R
    r2 = r_probe / 10.0 * R
    c1 = model.get_c1_inner(r1, i, j)
    c2 = model.get_c1_inner(r2, i, j)
    a1 = abs(c1)
    ratio = abs(c2) / a1 if a1 > 1e-10 else 1.0
    return OriginBCResult(
        r1=r1, r2=r2, c1_at_r1=c1, c1_at_r2=c2,
        ratio=ratio, passed=ratio < tol,
    )


def check_contact_jump(
    model,
    i: int,
    j: int,
    eps: float = 1e-5,
    tol: float = 1e-2,
) -> ContactResult:
    """Check c_inner(R_ij−) ≈ c_outer(R_ij+).

    Inner value: c_HS(r_in, i, j) + c^(1)(r_in, i, j)  at r_in = R·(1 − ε).
    Outer value: Σ_t K_t exp(−z_t·ε·R) / (2π √(ρᵢρⱼ) r_out)  at r_out = R·(1 + ε).

    Parameters
    ----------
    eps : fractional separation from R_ij for both probe points (default 1e-5).
    tol : relative-jump threshold for 'passed' (default 1e-2).
    """
    R      = float(model.R_ij[i, j])
    r_in   = R * (1.0 - eps)
    r_out  = R * (1.0 + eps)

    c_in = model.get_HS_FMT(r_in, i, j) + model.get_c1_inner(r_in, i, j)

    rho_ij = math.sqrt(float(model.rho[i]) * float(model.rho[j]))
    c_out  = _outer_dcf(model, i, j, r_out)

    abs_jump = abs(c_in - c_out)
    ref      = max(abs(c_out), 1e-10)
    rel_jump = abs_jump / ref
    return ContactResult(
        c_in=c_in, c_out=c_out,
        abs_jump=abs_jump, rel_jump=rel_jump,
        passed=rel_jump < tol,
    )


def check_symmetry(
    model,
    i: int,
    j: int,
    r_frac: float = 0.5,
    tol: float = 1e-8,
) -> SymmetryResult:
    """Check c^(1)_ij(r) == c^(1)_ji(r).

    Evaluates c^(1) at r = r_frac·R_ij from both orderings.  Should be
    identical whenever K[t,i,j] = K[t,j,i] and R_ij = R_ji.

    Parameters
    ----------
    r_frac : fraction of R_ij at which to evaluate (default 0.5).
    tol    : relative asymmetry threshold for 'passed' (default 1e-8).
    """
    R    = float(model.R_ij[i, j])
    r    = r_frac * R
    c_ij = model.get_c1_inner(r, i, j)
    c_ji = model.get_c1_inner(r, j, i)
    abs_asym = abs(c_ij - c_ji)
    ref      = max(abs(c_ij), 1e-10)
    rel_asym = abs_asym / ref
    return SymmetryResult(
        c_ij=c_ij, c_ji=c_ji,
        abs_asym=abs_asym, rel_asym=rel_asym,
        passed=rel_asym < tol,
    )


# ---------------------------------------------------------------------------
# Aggregate: all pairs
# ---------------------------------------------------------------------------

def check_all_pairs(
    model,
    r_probe:     float = 1e-4,
    eps:         float = 1e-5,
    r_frac:      float = 0.5,
    bc_tol:      float = 2.0,
    cont_tol:    float = 1e-2,
    sym_tol:     float = 1e-8,
    print_table: bool  = True,
) -> dict[tuple[int, int], PairResult]:
    """Run all three checks for every unique pair (i ≤ j).

    Returns a dict keyed by (i, j) tuples.  If print_table=True (default),
    also prints a formatted summary table to stdout.

    Parameters
    ----------
    r_probe   : fractional probe radius for origin-BC check.
    eps       : fractional probe offset for contact-jump check.
    r_frac    : fractional radius for symmetry check.
    bc_tol    : origin-BC ratio threshold (pass if ratio < bc_tol).
    cont_tol  : contact relative-jump threshold (pass if rel_jump < cont_tol).
    sym_tol   : symmetry relative-asymmetry threshold.
    """
    results: dict[tuple[int, int], PairResult] = {}
    for i in range(model.N):
        for j in range(i, model.N):
            obc  = check_origin_bc(model, i, j, r_probe=r_probe, tol=bc_tol)
            cont = check_contact_jump(model, i, j, eps=eps, tol=cont_tol)
            sym  = check_symmetry(model, i, j, r_frac=r_frac, tol=sym_tol)
            results[(i, j)] = PairResult(origin_bc=obc, contact=cont, symmetry=sym)

    if print_table:
        _print_table(model, results)

    return results


# ---------------------------------------------------------------------------
# DCF route ĉ(0) for a mixture model
# ---------------------------------------------------------------------------

def compute_chat0_ij(
    model,
    i: int,
    j: int,
    n_r: int = 600,
) -> float:
    """Compute ĉ_ij(k=0) = 4π ∫₀^∞ r² c_ij(r) dr for pair (i,j).

    Split into three parts to avoid the 1/r singularity:

      (a) HS inner:  4π ∫₀^R r² c_HS(r) dr            (numerical)
      (b) c^(1) inner: 4π ∫₀^R r² c^(1)(r) dr         (numerical via r·c^(1)·r dr)
      (c) outer Yukawa (analytic):
            4π ∫_R^∞ r² · K_t exp(−z(r−R)) / (2π √(ρᵢρⱼ) r) dr
          = Σ_t [2 K_t / √(ρᵢρⱼ)] · (R/z_t + 1/z_t²)

    Parameters
    ----------
    n_r : number of radial quadrature points inside [0, R_ij].
    """
    R      = float(model.R_ij[i, j])
    rho_ij = math.sqrt(float(model.rho[i]) * float(model.rho[j]))

    r_grid = np.linspace(0.0, R, n_r + 1)[1:]   # exclude r = 0

    chs_vals = np.array([model.get_HS_FMT(r, i, j) for r in r_grid])
    c1_vals  = np.array([model.get_c1_inner(r, i, j) for r in r_grid])

    chat_hs  = 4.0 * np.pi * _trapz(r_grid**2 * chs_vals, r_grid)
    chat_1in = 4.0 * np.pi * _trapz(r_grid**2 * c1_vals,  r_grid)

    chat_out = 0.0
    for t in range(model.n_tails):
        z = float(model.z_poles[t, i, j])
        if z > 0.0:
            K_t = float(model.K[t, i, j])
            chat_out += (2.0 * K_t / rho_ij) * (R / z + 1.0 / z**2)

    return chat_hs + chat_1in + chat_out


def compute_chat0_matrix(
    model,
    n_r: int = 600,
) -> np.ndarray:
    """Compute the full ĉ_ij(k=0) matrix, shape (N, N).

    Uses symmetry: ĉ_ji = ĉ_ij.

    Parameters
    ----------
    n_r : quadrature points per pair.
    """
    N = model.N
    chat = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            v = compute_chat0_ij(model, i, j, n_r=n_r)
            chat[i, j] = v
            chat[j, i] = v
    return chat


def compute_spinodal_det(
    model,
    n_r: int = 600,
) -> float:
    """Compute det(I − X̂ Ĉ(0)) for spinodal / compressibility-route check.

    X̂_ij = √(ρᵢ ρⱼ)  (symmetric density matrix).
    At a spinodal the determinant changes sign; at the critical point it has
    a double zero.  Returns the scalar determinant value > 0 for a stable
    homogeneous phase.
    """
    chat = compute_chat0_matrix(model, n_r=n_r)
    rho  = model.rho
    N    = model.N
    X_sq = np.sqrt(np.outer(rho, rho))    # X̂_ij = √(ρᵢ ρⱼ)
    mat  = np.eye(N) - X_sq * chat
    return float(np.linalg.det(mat))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _outer_dcf(model, i: int, j: int, r: float) -> float:
    """Evaluate c_outer(r) = Σ_t K_t exp(−z_t(r−R)) / (2π√(ρᵢρⱼ) r)."""
    R      = float(model.R_ij[i, j])
    rho_ij = math.sqrt(float(model.rho[i]) * float(model.rho[j]))
    val    = 0.0
    for t in range(model.n_tails):
        z = float(model.z_poles[t, i, j])
        if z > 0.0:
            val += float(model.K[t, i, j]) * math.exp(-z * (r - R))
    return val / (2.0 * math.pi * rho_ij * r)


def _print_table(model, results: dict[tuple[int, int], PairResult]) -> None:
    """Print a formatted summary table of check results."""
    name    = type(model).__name__
    eta     = getattr(model, 'eta',     float('nan'))
    rhostar = getattr(model, 'rhostar', float('nan'))

    print(f"\nSelf-consistency checks — {name}  (N={model.N}, η={eta:.4f}, ρ*={rhostar:.4f})")
    print()

    col_w = [8, 18, 6, 15, 7, 14, 6]
    header_parts = [
        "Pair",
        "OriginBC ratio",  "BC",
        "Contact reljump", "Cont",
        "Sym relasym",     "Sym",
    ]
    header = "  " + "  ".join(f"{h:>{w}}" for h, w in zip(header_parts, col_w))
    print(header)
    print("  " + "-" * (len(header) - 2))

    for (i, j), pr in sorted(results.items()):
        obc  = pr.origin_bc
        cont = pr.contact
        sym  = pr.symmetry
        row = [
            f"({i},{j})",
            f"{obc.ratio:.4f}",
            "pass" if obc.passed  else "FAIL",
            f"{cont.rel_jump:.3e}",
            "pass" if cont.passed else "FAIL",
            f"{sym.rel_asym:.3e}",
            "pass" if sym.passed  else "FAIL",
        ]
        print("  " + "  ".join(f"{v:>{w}}" for v, w in zip(row, col_w)))
    print()


# ---------------------------------------------------------------------------
# Multi-variant comparison
# ---------------------------------------------------------------------------

def compare_variants(
    variants: dict[str, object],
    r_probe:  float = 1e-4,
    eps:      float = 1e-5,
    r_frac:   float = 0.5,
    bc_tol:   float = 2.0,
    cont_tol: float = 1e-2,
    sym_tol:  float = 1e-8,
) -> dict[str, dict[tuple[int, int], PairResult]]:
    """Run all checks on multiple FMSA variants and print a side-by-side table.

    Parameters
    ----------
    variants : dict mapping short label → model object.
               All models must share the same N, R_ij, rho.
    Others   : forwarded to the individual check functions.

    Returns
    -------
    dict mapping label → {(i,j): PairResult}
    """
    all_results: dict[str, dict[tuple[int, int], PairResult]] = {}
    for label, model in variants.items():
        all_results[label] = check_all_pairs(
            model,
            r_probe=r_probe, eps=eps, r_frac=r_frac,
            bc_tol=bc_tol, cont_tol=cont_tol, sym_tol=sym_tol,
            print_table=False,
        )

    _print_comparison_table(variants, all_results)
    return all_results


def _print_comparison_table(
    variants: dict[str, object],
    all_results: dict[str, dict[tuple[int, int], PairResult]],
) -> None:
    """Print a pivot table: rows = pairs, columns = variants, two metric blocks."""
    labels = list(variants.keys())
    pairs  = sorted(next(iter(all_results.values())).keys())
    N      = next(iter(variants.values())).N
    m0     = next(iter(variants.values()))
    eta     = getattr(m0, 'eta',     float('nan'))
    rhostar = getattr(m0, 'rhostar', float('nan'))

    print(f"\nVariant comparison — N={N}, η={eta:.4f}, ρ*={rhostar:.4f}")
    print()

    LW = max(14, max(len(lb) for lb in labels))
    MW = 9
    PW = 6

    def _bc_cell(pr: PairResult) -> str:
        r    = pr.origin_bc.ratio
        mark = "ok" if pr.origin_bc.passed else "!!"
        return f"{r:5.2f} {mark}"

    def _ct_cell(pr: PairResult) -> str:
        v    = pr.contact.rel_jump
        mark = "ok" if pr.contact.passed else "!!"
        return f"{v:.1e} {mark}" if v < 1e-3 else f"{v:5.2f} {mark}"

    print("  Origin BC ratio  (pass if ratio < 2.0,  ×10 = 1/r divergence)")
    _print_block(pairs, labels, all_results, PW, LW, _bc_cell)

    print("  Contact rel-jump  (pass if jump < 1 %)")
    _print_block(pairs, labels, all_results, PW, LW, _ct_cell)


def _print_block(pairs, labels, all_results, PW, LW, cell_fn) -> None:
    """Print one metric block (rows = pairs, cols = labels)."""
    header_row = f"  {'Pair':<{PW}}" + "".join(f"  {lb:>{LW}}" for lb in labels)
    print(header_row)
    print("  " + "-" * (len(header_row) - 2))
    for pair in pairs:
        row = f"  {str(pair):<{PW}}"
        for lb in labels:
            row += f"  {cell_fn(all_results[lb][pair]):>{LW}}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Adapters for non-standard FMSA implementations
# ---------------------------------------------------------------------------

class FMSA_chsY_adapter:
    """Adapter for FMSA_chsY_species (plot_chsY.py) → sc_checks interface.

    Differences bridged:
      - Inner DCF method: compute_cij1_inside(r,i,j) → get_c1_inner(r,i,j)
        (applies flip_cross_sign negation for unlike pairs, matching get_c_total)
      - Packing fraction: xi3 → eta
      - Total density:  sum(rho) → rhostar

    All other attributes (R_ij, K, z_poles, n_tails, rho, N, get_HS_FMT) are
    forwarded directly.
    """

    def __init__(self, model) -> None:
        self._m      = model
        self.N       = model.N
        self.rho     = model.rho
        self.R_ij    = model.R_ij
        self.K       = model.K
        self.z_poles = model.z_poles
        self.n_tails = model.n_tails
        self.eta     = float(model.xi3)
        self.rhostar = float(np.sum(model.rho))

    def get_c1_inner(self, r: float, i: int, j: int) -> float:
        c1 = self._m.compute_cij1_inside(r, i, j)
        if self._m.flip_cross_sign and i != j:
            c1 = -c1
        return c1

    def get_HS_FMT(self, r: float, i: int, j: int) -> float:
        return self._m.get_HS_FMT(r, i, j)


class FMSA_poly_adapter:
    """Adapter for FMSA_poly_term_species (FMSA_MC_cleaned_2cpp.py) → sc_checks interface.

    Differences bridged:
      - R_ij:      radii
      - K:         K_matrices  shape (2, N, N)
      - z_poles:   z_matrices  shape (2, N, N)
      - n_tails:   n_tails_active
      - get_c1_inner: composed from get_e_ij(r,i,j) + polynomial full_p_coeffs[i,j]
                      with prefactor 1/(2π√ρᵢρⱼ · r_c) where r_c = max(r, λ_ij)
      - get_HS_FMT: get_c0_fmtHS_ref(r_c, i, j)

    Mirrors the formula in get_c_total_poly exactly.
    """

    def __init__(self, model) -> None:
        self._m      = model
        self.N       = int(len(model.rho))
        self.rho     = model.rho
        self.R_ij    = model.radii
        self.K       = model.K_matrices   # (2, N, N)
        self.z_poles = model.z_matrices   # (2, N, N)
        self.n_tails = int(model.n_tails_active)
        self.eta     = float(model.eta)
        self.rhostar = float(model.rhostar)

    def get_c1_inner(self, r: float, i: int, j: int) -> float:
        m          = self._m
        lambda_ij  = abs(m.radii[i, i] - m.radii[j, j]) / 2.0
        r_c        = max(r, lambda_ij)
        prefactor  = 1.0 / (2.0 * math.pi * math.sqrt(m.rho[i] * m.rho[j]) * r_c)
        e_ij       = m.get_e_ij(r, i, j)
        n_terms    = m.full_p_coeffs.shape[2]
        p_ij       = float(np.sum(m.full_p_coeffs[i, j] * np.power(r, np.arange(n_terms))))
        return prefactor * (e_ij + p_ij)

    def get_HS_FMT(self, r: float, i: int, j: int) -> float:
        m         = self._m
        lambda_ij = abs(m.radii[i, i] - m.radii[j, j]) / 2.0
        r_c       = max(r, lambda_ij)
        return m.get_c0_fmtHS_ref(r_c, i, j)
