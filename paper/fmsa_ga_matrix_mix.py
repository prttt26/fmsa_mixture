"""FMSA_GA_matrix_mix: multi-species Yukawa DCF via G/A matrix decomposition.

The inner-core first-order MSA correction uses the matrix decomposition
Q̂₀ = P̂ + Ê·exp(−z·σ_min) → Ĝ = P̂·Q̂₀⁻¹,  Â = Ê·Q̂₀⁻¹
instead of the scalar (single-component) Baxter coefficients.

Self-contained — imports only numpy. Suitable for direct C++ translation.
No inheritance; all helpers are inlined.

See numerical_notes/theory/multicomp_g_a_derivation.md for the full derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import root as _sp_root

from FMSA_pure import FMSA_pure


@dataclass
class _PoleCoeffs:
    """Yukawa poles and residue amplitudes for a single species pair."""

    poles: np.ndarray
    coeffs: np.ndarray


class FMSA_GA_matrix_mix:
    """Multi-species Yukawa DCF using the G/A matrix decomposition of Q̂₀.

    Implements the corrected first-order Mean Spherical Approximation (FMSA)
    for N-component Lennard-Jones mixtures modelled with one or two Yukawa tails.
    The inner-core DCF is obtained by decomposing the Baxter Q̂₀ matrix as

        Q̂₀(z) = P̂(z) + Ê(z)·exp(−z·σ_min)

    and defining Ĝ = P̂·Q̂₀⁻¹,  Â = Ê·Q̂₀⁻¹.  The corrected inner-core formula

        c^(1)_ij(r) = Σ_t K_t[i,j] · [(1 − Ĝ²_ij) exp(−z_t(r−R_ij))
                                        − Â²_ij      exp(+z_t(r−R_ij))]
                      / (2π √(ρ_i ρ_j) r),   r < R_ij

    replaces the incorrect (1+A)² coefficient in the FMSA_chsY Term I, which
    fails for pairs where min(σ_i, σ_j) = σ_min (Terms II–IV vanish identically
    for those pairs).  The algebraic identity Ĝ + Â·exp(−z·σ_min) = I is proved
    formally in Lean (MatrixIdentity.lean, Task M.1).

    The total DCF is c_ij(r) = c^HS_ij(r) + c^(1)_ij(r) inside the core and
    the strict MSA tail −β u_ij(r) outside.  The hard-sphere reference uses the
    White Bear FMT (KR normalisation: n₂ = π Σρ Rᵢ²).  The constructor mirrors the setup of
    FMSA_poly_term_species: it computes Barker-Henderson diameters, packing-
    fraction weighted densities, Yukawa K/z matrices, and precomputes Ĝ/Â for
    all distinct pole values.
    """

    def __init__(
        self,
        sig_pure: np.ndarray,
        eps_pure: np.ndarray,
        x: np.ndarray,
        rhostar: float,
        tstar: float,
        n_tails: int = 2,
        z=None,
        k=None,
        input_type: str = 'lj',
        d_pure=None,
        verbose: bool = True,
    ) -> None:
        """Initialise system parameters and precompute all Path B FMSA quantities.

        Parameters
        ----------
        sig_pure   : pure-species LJ diameters (input_type='lj') or BH diameters
                     (input_type='bh'), shape (N,)
        eps_pure   : pure-species LJ well-depths, shape (N,)
        x          : mole fractions, shape (N,); must sum to 1
        rhostar    : total reduced number density ρ* = ρ σ_ref³
        tstar      : reduced temperature T* = kT / ε_ref
        n_tails    : number of active Yukawa tails (1 = HSY, 2 = 2YK LJ fit; default 2)
        z          : Yukawa range factor(s).
                     n_tails=1: float or [float] → λ,  z_ij = λ/R_ij.
                                Default: 1.791 (standard HSY mapping).
                     n_tails=2: float or [z1_fac, z2_fac] → z_t,ij = z_t/σ_ij.
                                Single float broadcasts to both tails.
                                Default: [2.9637, 14.0167] (LJ two-Yukawa fit).
                     None uses the default for the chosen n_tails.
        k          : Yukawa amplitude factor(s).
                     n_tails=1: float A → k_bare_ij = A·ε_mix_ij·R_ij.
                                Default: 1.0 (unit amplitude scale).
                     n_tails=2: float k0 (shared) or [k1_fac, k2_fac] (per-tail).
                                Shared float: k_bare_t = (±k0)·σ_ij·ε_ij·exp(z_t(σ−R)),
                                sign +/− for t = 0/1.  Per-tail: caller carries sign.
                                Default: k0_fac = 2.1714 (LJ two-Yukawa fit).
                     None uses the default for the chosen n_tails.
        input_type : 'lj' (default) — sig_pure/eps_pure are LJ parameters;
                                       Barker-Henderson diameter conversion is applied.
                     'bh'           — sig_pure/eps_pure are already BH pure-species
                                       diameters / well-depths; conversion is skipped.

        After construction the following attributes are available:

          sig_mix, eps_mix : Lorentz-Berthelot mixing matrices, shape (N, N)
          d_bh             : Barker-Henderson effective diameter matrix, shape (N, N)
          sigma            : BH pure-species diameters (diagonal of d_bh), shape (N,)
          R_ij             : contact-distance matrix (arithmetic mean of σ), shape (N, N)
          rho              : number densities x * rhostar, shape (N,)
          xi               : weighted density moments ξₙ = Σ_k ρ_k σ_k^n, n=0..3,
                             shape (4,)  (no π/6 prefactor; same as FMSA_poly_term_species)
          eta, vac         : packing fraction η = (π/6)·ξ₃ and free-volume fraction 1 − η
          Q0               : hard-sphere Baxter Q matrix at z = 0, shape (N, N)
          Qpp              : HS Q double-prime vector, shape (N,)
          z_poles          : Yukawa decay constants z_t,ij, shape (n_tails, N, N)
          epsilon          : Yukawa well-depth matrix k_bare/R_ij, shape (n_tails, N, N)
          k_bare           : bare Yukawa amplitudes before density prefactor,
                             shape (n_tails, N, N)
          K                : full Yukawa amplitude K_t,ij = 2π√(ρᵢρⱼ)·β·k_bare_t,ij,
                             shape (n_tails, N, N)
          _G_mat           : G matrix elements at every pole, shape (n_tails, N, N, N, N);
                             _G_mat[t, m, n] is the (N×N) Ĝ matrix at z_poles[t, m, n]
          _A_mat           : analogous (N×N) Â matrices, shape (n_tails, N, N, N, N)
          _Qinv_mat        : Q̂₀⁻¹ matrices at every pole, shape (n_tails, N, N, N, N)

          A_prop           : propagator matrices, shape (n_tails, N, N, N, N);
                             A_prop[t, m, n, i, j] is element (i,j) of the propagator
                             at pole z_poles[t, m, n]
          b_grow           : list[list[_PoleCoeffs]], N×N; b_grow[i][j] holds the
                             pole/residue expansion of the growing exponential for pair (i,j)
        """
        sig_pure = np.asarray(sig_pure, dtype=float)
        eps_pure = np.asarray(eps_pure, dtype=float)
        x        = np.asarray(x,        dtype=float)
        N = len(sig_pure)

        self.sig_pure = sig_pure
        self.eps_pure = eps_pure
        self.x        = x
        self.rhostar  = rhostar
        self.tstar    = tstar
        self.beta     = beta = 1.0 / tstar
        self.N        = N
        self.n_tails  = n_tails

        # Mixing rules (Lorentz-Berthelot)
        sig_mix = 0.5 * (sig_pure[:, None] + sig_pure[None, :])
        eps_mix = np.sqrt(eps_pure[:, None] * eps_pure[None, :])
        self.sig_mix = sig_mix
        self.eps_mix = eps_mix

        # BH diameters and contact-distance matrix
        if d_pure is not None:
            # Hard cores given explicitly while sig_pure/eps_pure stay LJ, so the
            # TAIL is still parameterised off sigma_LJ.  input_type='bh' cannot do
            # this: there sig_mix is the diameter matrix, so z = z_fac/d and the
            # amplitude loses its exp(z(sigma-d)) shift -- z comes out 3.4 % high
            # and k_bare 12.2 % low for an LJ fluid.  That reading is right for a
            # hard-core Yukawa model (which is what 'bh' is for) and wrong for an
            # LJ whose core simply is not the BH correlation's -- the LJ + hard
            # sphere mixture, where the sphere is a WCA stand-in 2.7-4.4 % LARGER
            # than sigma and therefore unreachable by any BH formula.
            dd = np.asarray(d_pure, dtype=float)
            self.d_bh = 0.5 * (dd[:, None] + dd[None, :])
        elif input_type == 'lj':
            self.d_bh = self._bh_diameters(sig_mix, eps_mix, beta)
        else:  # 'bh': sig_pure already contains BH pure-species diameters
            self.d_bh = sig_mix.copy()
        self.sigma  = sigma = np.diag(self.d_bh)
        self.R_ij   = 0.5 * (sigma[:, None] + sigma[None, :])
        self.diag_r = sigma.copy()   # pure-species BH diameters; alias for FMSAFreeEnergy

        # Derived geometry
        self.lambda_ij = 0.5 * (sigma[None, :] - sigma[:, None])
        self.sigma_min = float(np.min(sigma))

        # Densities: always rho = x * rhostar
        self.rho = rho = x * rhostar

        # Packing-fraction moments xi[n] = Σ_k ρ_k σ_k^n,  n = 0..3  (no π/6 prefactor,
        # matching FMSA_poly_term_species and the Baxter factorization equations)
        self.xi  = np.array([float(np.sum(rho * sigma**n)) for n in range(4)])
        self.eta = (np.pi / 6.0) * self.xi[3]   # packing fraction η = (π/6) Σ ρ σ³
        self.vac = 1.0 - self.eta

        # HS Q-factors
        self._build_Q0_Qpp()

        # Normalise z/k to 1-D ndarrays of length n_tails
        def _to_arr(v, n: int) -> np.ndarray | None:
            if v is None:
                return None
            a = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
            return np.full(n, a[0]) if a.size == 1 else a

        # Yukawa range: z_poles[t,i,j] = z_arr[t] / sig_mix[i,j]  (or / R_ij for HSY)
        if n_tails == 1:
            z_arr = _to_arr(z, 1) if z is not None else np.array([1.791])
            self.z_poles = z_arr[:, None, None] / self.R_ij[None, :, :]
        else:
            z_arr = _to_arr(z, n_tails) if z is not None else np.array([2.9637, 14.0167])
            self.z_poles = z_arr[:, None, None] / sig_mix[None, :, :]

        # Yukawa bare amplitudes k_bare[t,i,j]
        k_arr  = _to_arr(k, n_tails)
        k_bare = np.zeros((n_tails, N, N), dtype=float)
        k0_fac = 2.1714

        if n_tails == 1:
            k_amp = float(k_arr[0]) if k_arr is not None else 1.0
            k_bare[0] = k_amp * eps_mix * self.R_ij
        else:
            for t in range(n_tails):
                if k_arr is not None:
                    k_bare[t] = k_arr[t] * sig_mix * eps_mix * np.exp(
                        self.z_poles[t] * (sig_mix - self.R_ij)
                    )
                else:
                    sign = 1.0 if t == 0 else -1.0
                    k_bare[t] = sign * k0_fac * sig_mix * eps_mix * np.exp(
                        self.z_poles[t] * (sig_mix - self.R_ij)
                    )

        self.k_bare  = k_bare
        self.epsilon = k_bare / self.R_ij

        # K[t,i,j] = 2π √(ρᵢρⱼ) β k_bare[t,i,j]
        rho_sq = np.sqrt(rho[:, None] * rho[None, :])
        self.K = 2.0 * np.pi * beta * rho_sq[None, :, :] * k_bare

        # FULL-MATRICES EXPRESSION (current): dense 5-D ndarrays indexed by
        # (tail, m-pole, n-pole, species-i, species-j).  _G_mat[t, m, n] and
        # _A_mat[t, m, n] are the (N×N) Ĝ/Â matrices at z_poles[t, m, n].
        # Entries sharing the same z value are stored redundantly — negligible
        # for N ≤ 5 (our project max) and n_tails ≤ 2.
        # C++ mapping: double G_mat[n_tails][N][N][N][N] with direct indexing.
        #
        # UNIQUE-Z_POLES EXPRESSION (alternative for large N or many tails):
        # Deduplicate z-values first to save memory:
        #   z_unique[k]   shape (n_unique,)     — sorted unique positive poles
        #   z_idx[t,m,n]  shape (n_tails,N,N)   — integer index into z_unique
        #   _G_mat[k,i,j] shape (n_unique,N,N)  — G matrix for the k-th unique z
        # Access: _G_mat[self.z_idx[t,m,n], i, j].
        # C++: G_mat[z_idx[t][m][n]][i][j].  Saves memory when n_unique << n_tails·N²,
        # but adds one integer indirection per access.
        self._precompute_G_A()   # fills _G_mat, _A_mat, _Qinv_mat

        self.A_prop = self._compute_A_prop()  # shape (n_tails, N, N, N, N)
        self.b_grow = self._compute_b_grow()  # list[list[_PoleCoeffs]], N×N

        # Store original constructor inputs so _make_at_rho can rebuild at new density
        self._z_input    = z
        self._k_input    = k
        self._input_type = input_type
        self._d_pure = None if d_pure is None else np.asarray(d_pure, dtype=float)

        # Self-consistency corrections: polynomial P_ij(r) added at the rc1r level
        # enforces origin BCs only: P(0)=−C₀, P'(0)=slope. No constraints at r=R.
        # For unlike pairs: constant shift only (P(r)=−C₀) to remove the 1/r divergence.
        self._build_pure_refs()
        self._polycorr: dict[tuple[int, int], np.ndarray] = {}
        self._precompute_polycorr()

    # ------------------------------------------------------------------
    # Barker-Henderson diameter conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _bh_diameters(
        sig_mix: np.ndarray,
        eps_mix: np.ndarray,
        beta: float,
        cbh: np.ndarray | None = None,
    ) -> np.ndarray:
        """Barker-Henderson effective diameters from LJ mixing-rule matrices.

        d_bh[i,j] = 2^{1/6} · sig_mix[i,j] / (1 + sqrt(f))^{1/6}
        where f = 1 + cbh[0]/t + cbh[1]/t² + cbh[2]/t⁴,  t = beta·eps_mix[i,j].

        Parameters
        ----------
        sig_mix : LJ diameter mixing matrix, shape (N, N)
        eps_mix : LJ well-depth mixing matrix, shape (N, N)
        beta    : inverse temperature
        cbh     : 3-element coefficient array; default [1.1287, −0.05536, 7.278e-4]

        Returns
        -------
        d_bh : effective diameter matrix, shape (N, N)
        """
        if cbh is None:
            # WARNING -- these coefficients are BIASED, and knowingly kept.
            # The form is d/sigma = 2^(1/6)/(1+sqrt(1+c0 T*+c1 T*^2+c2 T*^4))^(1/6), which
            # inverts to the statement beta*u(d) = c0 + c1 T* + c2 T*^3.  At the TRUE
            # Barker-Henderson diameter that quantity tends to 1 as T*->0 (near contact
            # u ~ -24 eps (r-sigma)/sigma, so d/sigma -> 1 - T*/24) and falls monotonically
            # from there, so 1 is its supremum -- c0 = 1.1287 is above it and cannot come
            # from fitting this quantity over any range.  Measured against the integral:
            # -0.32% at T* = 0.5, -0.90% at 2.12, -3.72% at 10, always low.  It is not a
            # different DEFINITION either (integrating to r_min, to infinity, or over the
            # WCA split is 3-200% off).  Inherited from the original MATLAB (FMSA_MC.m,
            # "empirical fit coefficients", no citation).
            # LEFT ALONE ON PURPOSE: this project studies hard-core Yukawa systems, where
            # input_type='bh' is used and this LJ mapping never runs.  The C++ port
            # (cpp/FMSA_model) does run it, and uses Cotterman, Schwarz & Prausnitz,
            # AIChE J 32, 1787 (1986), d/sigma = (1+0.2977 T*)/(1+0.33163 T*+0.0010477 T*^2),
            # which is within +-0.12% of the integral over T* in [0.5, 10].
            cbh = np.array([1.1287, -0.05536, 0.0007278])
        t   = 1.0 / (beta * eps_mix)
        fac = 1.0 + cbh[0]*t + cbh[1]*t**2 + cbh[2]*t**4
        return 2.0**(1.0/6.0) * sig_mix / (1.0 + np.sqrt(fac))**(1.0/6.0)

    # ------------------------------------------------------------------
    # HS OZ factors
    # ------------------------------------------------------------------

    def _build_Q0_Qpp(self) -> None:
        """Compute HS Baxter Q matrix Q₀[i,j] and double-prime vector Qpp[j].

        Q₀[i,j]  = (2π/vac) · (R_ij + π·ξ₂·σ_i·σ_j / (4·vac))
        Qpp[j]   = (2π/vac) · (1    + π·ξ₂·σ_j     / (2·vac))

        where ξ₂ = xi[2] = Σ ρ σ² (no π/6 prefactor).

        Results stored in self.Q0 (shape N×N) and self.Qpp (shape N).
        """
        sigma, xi, vac = self.sigma, self.xi, self.vac
        self.Q0 = (2.0 * np.pi / vac) * (
            self.R_ij
            + (np.pi * xi[2] / (4.0 * vac)) * sigma[:, None] * sigma[None, :]
        )
        self.Qpp = (2.0 * np.pi / vac) * (
            1.0 + (np.pi * xi[2] / (2.0 * vac)) * sigma
        )

    # ------------------------------------------------------------------
    # Q̂₀ decomposition and Ĝ/Â matrices
    # ------------------------------------------------------------------

    def _build_Qhat(self, z: float) -> np.ndarray:
        """Evaluate the full Baxter Q̂₀(z) matrix at Laplace variable z.

        Q̂₀_ij(z) = δ_ij − √(ρᵢρⱼ) · exp(−z·λ_ij)
                     · [Q₀_ij · φ₁(σ_i, z) + Qpp_j · φ₂(σ_i, z)]

        where φ₁(σ, z) = (1 − zσ − e^{−zσ}) / z²  and
              φ₂(σ, z) = (1 − zσ + (zσ)²/2 − e^{−zσ}) / z³.

        Parameters
        ----------
        z : Laplace / Yukawa pole value

        Returns
        -------
        Qhat : shape (N, N)
        """
        sigma, rho = self.sigma, self.rho
        lam, Q0, Qpp = self.lambda_ij, self.Q0, self.Qpp
        p1 = (1.0 - z*sigma - np.exp(-z*sigma)) / z**2
        zs = z * sigma
        p2 = (1.0 - zs + 0.5*zs**2 - np.exp(-zs)) / z**3
        sqrt_rr = np.sqrt(rho[:, None] * rho[None, :])
        inner   = p1[:, None] * Q0 + p2[:, None] * Qpp[None, :]
        Qhat    = -sqrt_rr * np.exp(-z * lam) * inner
        Qhat   += np.eye(self.N)
        return Qhat

    def _decompose_Q0(self, z: float) -> tuple[np.ndarray, np.ndarray]:
        """Split Q̂₀(z) into rational part P̂(z) and exponential part Ê(z).

        The decomposition Q̂₀(z) = P̂(z) + Ê(z)·exp(−z·σ_min) is derived by
        expanding φ₁/φ₂ into their rational + exp(−zσ_i) parts and using the
        key identity  exp(−z·λ_ij)·exp(−z·σ_i) = exp(−z·R_ij)
                                                  = exp(−z·σ_min)·exp(−z·(R_ij−σ_min)).

        P̂_ij  = δ_ij − √(ρᵢρⱼ)·exp(−z·λ_ij)·[(1−zσ_i)/z²·Q₀_ij
                                                + (1−zσ_i+(zσ_i)²/2)/z³·Qpp_j]
        Ê_ij  = √(ρᵢρⱼ)·exp(−z·(R_ij−σ_min))·[Q₀_ij/z² + Qpp_j/z³]

        See §5 of numerical_notes/theory/multicomp_g_a_derivation.md for the full derivation.

        Parameters
        ----------
        z : Yukawa pole value

        Returns
        -------
        P_hat, E_hat : each shape (N, N)
        """
        sigma, rho = self.sigma, self.rho
        lam, Q0, Qpp = self.lambda_ij, self.Q0, self.Qpp
        iz2 = 1.0 / z**2
        iz3 = 1.0 / z**3
        phi1_rat = (1.0 - z * sigma) * iz2
        phi2_rat = (1.0 - z*sigma + 0.5*(z*sigma)**2) * iz3
        sqrt_rr  = np.sqrt(rho[:, None] * rho[None, :])
        inner    = phi1_rat[:, None] * Q0 + phi2_rat[:, None] * Qpp[None, :]
        P_hat    = -sqrt_rr * np.exp(-z * lam) * inner
        P_hat   += np.eye(self.N)
        E_hat    = sqrt_rr * np.exp(-z * (self.R_ij - self.sigma_min)) * (
            Q0 * iz2 + Qpp[None, :] * iz3
        )
        return P_hat, E_hat

    def _adjugate_Qhat(self, s: complex) -> np.ndarray:
        """Classical adjugate of Q̂₀(s): adj = cofactor matrix transposed.

        Safe at det(Q̂₀(s))=0 — uses no inversion.  For N=2 the 2×2 formula
        is hardcoded; for N>2 minors are computed entry-by-entry.
        """
        M = self._build_Qhat(s)
        N = self.N
        if N == 1:
            return np.array([[1.0 + 0j]])
        if N == 2:
            return np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]], dtype=complex)
        adj = np.zeros((N, N), dtype=complex)
        for ii in range(N):
            for jj in range(N):
                minor = np.delete(np.delete(M, ii, 0), jj, 1)
                adj[jj, ii] = ((-1) ** (ii + jj)) * np.linalg.det(minor)
        return adj

    def _det_prime_Qhat(self, s: complex, h: float = 1e-7) -> complex:
        """d/ds det(Q̂₀(s)) via central finite difference with real step h."""
        return (
            np.linalg.det(self._build_Qhat(s + h))
            - np.linalg.det(self._build_Qhat(s - h))
        ) / (2.0 * h)

    def _find_hs_poles_raw(self, n_pairs: int = 3) -> list:
        """Find n_pairs HS poles in the upper half-plane via Newton polish.

        Initial guesses: imaginary parts near (2m−1)π/σ_min (locations of N=1 poles
        for a hard-sphere system of diameter σ_min), real parts swept over [−3, −0.05].
        Each candidate is polished with scipy.optimize.root on [Re(det), Im(det)]=0.
        Only poles with Im(s)>0.1 and residual<1e-8 are kept; duplicates suppressed.
        """
        sigma_min = float(self.sigma_min)
        results: list = []

        def f2d(xy: np.ndarray) -> list:
            s = complex(xy[0], xy[1])
            d = np.linalg.det(self._build_Qhat(s))
            return [float(d.real), float(d.imag)]

        alpha_guesses = np.linspace(-3.0, -0.05, 8)
        for m in range(1, n_pairs + 4):
            beta0 = (2 * m - 1) * math.pi / sigma_min
            for alpha0 in alpha_guesses:
                sol = _sp_root(f2d, [alpha0, beta0], method="hybr", tol=1e-12)
                if not sol.success:
                    continue
                if max(abs(sol.fun[0]), abs(sol.fun[1])) > 1e-8:
                    continue
                sk = complex(sol.x[0], sol.x[1])
                if sk.imag < 0.1:
                    continue
                if any(abs(sk - prev) < 0.05 for prev in results):
                    continue
                results.append(sk)
                if len(results) >= n_pairs:
                    return results
        return results

    def _ensure_hs_poles(self, n_pairs: int = 3) -> None:
        """Lazy-init HS poles and pre-cache adjugate + det′ at each pole."""
        if hasattr(self, "_hs_poles") and len(self._hs_poles) >= n_pairs:
            return
        self._hs_poles = self._find_hs_poles_raw(n_pairs)
        self._hs_adj = [self._adjugate_Qhat(sk) for sk in self._hs_poles]
        self._hs_det_prime = [self._det_prime_Qhat(sk) for sk in self._hs_poles]

    def _precompute_G_A(self) -> None:
        """Fill _G_mat, _A_mat, _Qinv_mat for every (tail, m, n) pole slot.

        Each slot (t, m, n) stores the (N×N) matrices Ĝ, Â, and Q̂₀⁻¹ evaluated
        at z = z_poles[t, m, n].  Slots with z ≤ 0 are left as zeros.

        Entries that happen to share the same z value are recomputed independently
        (full-matrices expression — see __init__ comment).  For N ≤ 5 the extra
        matrix inversions are negligible; switch to the unique-z_poles expression
        if N grows large.
        """
        N, nt = self.N, self.n_tails
        self._G_mat    = np.zeros((nt, N, N, N, N), dtype=float)
        self._A_mat    = np.zeros((nt, N, N, N, N), dtype=float)
        self._Qinv_mat = np.zeros((nt, N, N, N, N), dtype=float)
        for t in range(nt):
            for m in range(N):
                for n in range(N):
                    z = float(self.z_poles[t, m, n])
                    if z <= 0.0:
                        continue
                    P_hat, E_hat = self._decompose_Q0(z)
                    Qhat         = self._build_Qhat(z)
                    Q0_inv       = np.linalg.inv(Qhat)
                    self._G_mat[t, m, n]    = P_hat @ Q0_inv
                    self._A_mat[t, m, n]    = E_hat @ Q0_inv
                    self._Qinv_mat[t, m, n] = Q0_inv

    def _get_G_A(self, t: int, m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (Ĝ, Â) at pole slot (t, m, n) from precomputed ndarrays.

        Ĝ = _G_mat[t, m, n],  Â = _A_mat[t, m, n],  both shape (N, N).

        The identity Ĝ + Â·exp(−z·σ_min) = I holds by construction
        (proved in Lean, Task M.1; verified numerically by verify_identity).
        """
        return self._G_mat[t, m, n], self._A_mat[t, m, n]

    def _compute_A_prop(self) -> np.ndarray:
        """Build the propagator array A_prop[t, m, n, i, j] from _Qinv_mat.

        For each pole slot (t, m, n) at z = z_poles[t, m, n]:
          A_prop[t, m, n, i, i] = Qinv[i, i] − 1
          A_prop[t, m, n, i, j] = Qinv[i, j] · exp(+z · λ_ij)   (i ≠ j)

        Returns
        -------
        A_prop : shape (n_tails, N, N, N, N)
        """
        N, nt  = self.N, self.n_tails
        lam    = self.lambda_ij
        A_prop = np.zeros((nt, N, N, N, N), dtype=float)
        for t in range(nt):
            for m in range(N):
                for n in range(N):
                    z = float(self.z_poles[t, m, n])
                    if z <= 0.0:
                        continue
                    Qinv = self._Qinv_mat[t, m, n]   # shape (N, N)
                    for i in range(N):
                        for j in range(N):
                            if i == j:
                                A_prop[t, m, n, i, i] = Qinv[i, i] - 1.0
                            else:
                                A_prop[t, m, n, i, j] = Qinv[i, j] * np.exp(z * lam[i, j])
        return A_prop

    def _compute_b_grow(self) -> list:
        """Build the pole/residue expansion b_grow[i][j] for the growing exponential.

        For each pair (i, j), the growing exponential coefficient at every pole
        is the sum of four term classes (identical structure to chsY b_decay / b_grow):
          1. Direct:  K[t,i,j]                                            at z_t[i,j]
          2. Second:  Σ_n K[t,i,n] · A_prop[t,i,n,j,n]                  at z_t[i,n]
          3. Third:   Σ_m K[t,m,j] · A_prop[t,m,j,i,m]                  at z_t[m,j]
          4. Fourth:  Σ_{m,n} K[t,m,n]·A_prop[t,m,n,i,m]·A_prop[t,m,n,j,n]  at z_t[m,n]

        Contributions from different (t,m,n) slots that happen to share the same
        z value are accumulated into the same coefficient (pole deduplication).

        Returns
        -------
        b_grow : N×N list of lists; b_grow[i][j] is a _PoleCoeffs instance.
        """
        N       = self.N
        K       = self.K
        z_poles = self.z_poles
        A_prop  = self.A_prop

        b_grow: list = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                # Accumulate into a plain list of (z, coeff) pairs; merge by z later.
                zc: list[tuple[float, float]] = []

                for t in range(self.n_tails):
                    def _add(z: float, c: float) -> None:
                        if z > 0.0:
                            zc.append((z, c))

                    _add(z_poles[t, i, j], K[t, i, j])                           # term 1
                    for nn in range(N):
                        _add(z_poles[t, i, nn],
                             K[t, i, nn] * A_prop[t, i, nn, j, nn])              # term 2
                    for mm in range(N):
                        _add(z_poles[t, mm, j],
                             K[t, mm, j] * A_prop[t, mm, j, i, mm])              # term 3
                    for mm in range(N):
                        for nn in range(N):
                            _add(z_poles[t, mm, nn],
                                 K[t, mm, nn]
                                 * A_prop[t, mm, nn, i, mm]
                                 * A_prop[t, mm, nn, j, nn])                      # term 4

                # Merge duplicate z values into arrays
                pole_map: dict[float, float] = {}
                for z, c in zc:
                    pole_map[z] = pole_map.get(z, 0.0) + c
                b_grow[i][j] = _PoleCoeffs(
                    poles  = np.array(list(pole_map.keys()),   dtype=float),
                    coeffs = np.array(list(pole_map.values()), dtype=float),
                )
        return b_grow

    # ------------------------------------------------------------------
    # Self-consistency corrections: polynomial P_ij(r) at the rc1r level
    # ------------------------------------------------------------------

    def _build_pure_refs(self) -> None:
        """Build per-species FMSA_pure objects to supply the c^(1)(0) target value.

        Each pure reference uses:
          • the BH diameter and η from this mixture (correct effective hard-sphere size)
          • the species density ρᵢ (not the total ρ)
          • for n_tails=1 (HSY): K and z from the (i,i) Yukawa entry so that
            the target matches the single-component HSY formula at the same K/z.
        """
        self._pure_refs: list[FMSA_pure] = []
        for i in range(self.N):
            # Use the original LJ sigma (sig_pure[i]), not the BH diameter, so that
            # FMSA_pure computes z1/z2 = z_fac / sigma_lj correctly.  The 'bh' path
            # gave z_fac / d_bh, inflating the inverse-range parameters.
            # With input_type='bh' sig_pure ALREADY holds diameters, so FMSA_pure
            # must not run BH on them a second time -- doing so shrank them 3.3 %
            # and 5.1 % and moved the c^(1)(0) target 28 %.  This is now the
            # DEFAULT.  It moves nothing that was published: the only 'bh' caller
            # in the benchmarks is compare_models_hcty.py, whose scored quantity
            # is the converged h from the OZ solver, and that is bit-identical
            # either way (max|diff| = 0) because it never reads _pure_refs.
            # A species with epsilon = 0 has no choice -- BH divides by epsilon.
            pure_bh = (self._input_type == 'bh' or self._d_pure is not None
                       or float(self.eps_pure[i]) == 0.0)
            # with d_pure the hard core is NOT sig_pure[i]; hand over the real one
            sig_i = (float(self._d_pure[i]) if self._d_pure is not None
                     else float(self.sig_pure[i]))
            p = FMSA_pure(
                sigma   = sig_i,
                epsilon = float(self.eps_pure[i]),
                rhostar = float(self.rho[i]),
                tstar   = float(1.0 / self.beta),
                input_type = 'bh' if pure_bh else 'lj',
            )
            if self.n_tails == 1:
                p.K_matrices = np.array([float(self.K[0, i, i]), 0.0])
                p.z_matrices = np.array([float(self.z_poles[0, i, i]),
                                         float(self.z_poles[0, i, i])])
            self._pure_refs.append(p)

    def _target_c1_at_zero(self, i: int, j: int) -> float:
        """Target c^(1)_ij(r→0) from FMSA_pure (like pair) or arithmetic average (unlike).

        The like-pair target is the N=1 FMSA result evaluated just inside r=0
        via the L'Hôpital formula  lim_{r→0} rc1r(r)/r = c^(1)(0).
        """
        if i != j:
            return 0.5 * (self._target_c1_at_zero(i, i) + self._target_c1_at_zero(j, j))
        eps_r = 1e-5 * self.R_ij[i, i]
        return self._pure_refs[i].get_rc1r(eps_r) / eps_r

    def _rc1r_at_zero(self, i: int, j: int) -> float:
        """Analytic limit of the GA rc1r loop as r→0 (the C₀ residue).

        At r=0: dr = −R_ij, exp_core = exp(z·R_ij).

        Like pairs (i==j): all polynomial terms in the correction vanish except
        the constant 24η·S·L, giving the full correction/D² formula.

        Unlike pairs (i≠j): the scalar D at z_{ij}, R_{ij} can be near-zero,
        inflating the 24η·S·L/D² term to O(10⁶).  Use the pure-exponential
        formula (no scalar D² denominator) that matches get_c1_inner for unlike
        pairs: C₀ = K·[(1−G²_{ij})·e^{zR} − A²_{ij}·e^{−zR}].
        """
        R_ij = self.R_ij[i, j]
        eta  = self.eta
        val  = 0.0
        for t in range(self.n_tails):
            z = float(self.z_poles[t, i, j])
            if z <= 0.0:
                continue
            K_t  = float(self.K[t, i, j])
            g_ij = float(self._G_mat[t, i, j, i, j])
            a_ij = float(self._A_mat[t, i, j, i, j])
            eR   = math.exp(z * R_ij)
            if i != j:
                # Pure-exponential formula: avoids D²≈0 blow-up for unlike pairs.
                val += K_t * ((1.0 - g_ij**2) * eR - a_ij**2 / eR)
            else:
                vac  = self.vac
                S  = vac**2*z**3 + 6*eta*vac*z**2 + 18*eta**2*z - 12*eta*(1+2*eta)
                L  = (1+eta/2)*z + (1+2*eta)
                D  = S + 12*eta*L*math.exp(-z*R_ij)
                D2 = D**2
                gD = g_ij*D; aD = a_ij*D
                val += K_t * (eR - (gD**2*eR + aD**2/eR + 24*eta*S*L) / D2)
        return val

    def _d_rc1r_at_zero(self, i: int, j: int) -> float:
        """Analytic d(rc1r_GA)/dr at r=0.

        Like pairs (i==j): the linear polynomial term −24η·Ar·r and the
        exponential derivative both contribute.

        Unlike pairs (i≠j): pure-exponential formula derivative only —
        d/dr [K·((1−G²)·e^{−z·dr} − A²·e^{+z·dr})] at dr=−R_ij gives
        K·(−z·(1−G²_{ij})·e^{zR} − z·A²_{ij}·e^{−zR}).
        """
        R_ij = self.R_ij[i, j]
        eta  = self.eta
        val  = 0.0
        for t in range(self.n_tails):
            z = float(self.z_poles[t, i, j])
            if z <= 0.0:
                continue
            K_t  = float(self.K[t, i, j])
            g_ij = float(self._G_mat[t, i, j, i, j])
            a_ij = float(self._A_mat[t, i, j, i, j])
            eR   = math.exp(z * R_ij)
            emR  = math.exp(-z * R_ij)
            if i != j:
                # Derivative of the pure-exponential unlike-pair formula at r=0.
                val += K_t * (-z * ((1.0 - g_ij**2) * eR + a_ij**2 * emR))
            else:
                vac  = self.vac
                S  = vac**2*z**3 + 6*eta*vac*z**2 + 18*eta**2*z - 12*eta*(1+2*eta)
                L  = (1+eta/2)*z + (1+2*eta)
                D  = S + 12*eta*L*math.exp(-z*R_ij)
                D2 = D**2
                Ar = (1+2*eta)**2*z**4 + vac*(1+2*eta)*z**5
                gD = g_ij*D; aD = a_ij*D
                val += K_t * ((-z*eR) - (gD**2*(-z)*eR + aD**2*z*emR - 24*eta*Ar) / D2)
        return val

    def _rc1r_at_zero_B(self, i: int, j: int) -> float:
        """Route B limit of rc1r as r→0: two-exp base + _compute_poly_B(0) per tail.

        I_decay(0) = 0 always (empty integration interval), so only I_grow(0)
        contributes the Route B correction at the origin.
        """
        val = self._rc1r_at_zero(i, j)
        if i != j:
            for t in range(self.n_tails):
                if float(self.z_poles[t, i, j]) > 0.0:
                    val += self._compute_poly_B(0.0, i, j, t)
        return val

    def _d_rc1r_at_zero_B(self, i: int, j: int) -> float:
        """Route B d(rc1r)/dr at r=0: two-exp base + Σ_t (−z_t)·PolyB_t(0).

        Analytic: d/dr[K·I_grow(r)]|_{r=0} = −z·K·I_grow(0) = −z·_compute_poly_B(0,t).
        This holds for both lam=0 (equal-size) and lam>0 (unequal-size) unlike pairs.
        """
        val = self._d_rc1r_at_zero(i, j)
        if i != j:
            for t in range(self.n_tails):
                z = float(self.z_poles[t, i, j])
                if z > 0.0:
                    val += -z * self._compute_poly_B(0.0, i, j, t)
        return val

    def _rc1r_at_zero_A(self, i: int, j: int) -> float:
        """Route A limit of rc1r as r→0: two-exp base + poly_A(0) per tail.

        poly_A(0) = K · (−24η·S·L) / D_ij²  (linear+quartic+quadratic all vanish).
        Falls back to Route B per tail if D_ij ≈ 0.
        """
        val = self._rc1r_at_zero(i, j)
        if i != j:
            eta = self.eta
            vac = self.vac
            for t in range(self.n_tails):
                z = float(self.z_poles[t, i, j])
                if z <= 0.0:
                    continue
                K    = float(self.K[t, i, j])
                R_ij = float(self.R_ij[i, j])
                S    = vac**2*z**3 + 6*eta*vac*z**2 + 18*eta**2*z - 12*eta*(1 + 2*eta)
                L    = (1 + eta/2)*z + (1 + 2*eta)
                D_ij = S + 12*eta*L*math.exp(-z*R_ij)
                if abs(D_ij) < 1e-8:
                    val += self._compute_poly_B(0.0, i, j, t)
                else:
                    val += K * (-24*eta*S*L) / D_ij**2
        return val

    def _d_rc1r_at_zero_A(self, i: int, j: int) -> float:
        """Route A d(rc1r)/dr at r=0: two-exp derivative + d(poly_A)/dr|_{r=0} per tail.

        d(poly_A)/dr|_{r=0} = K · 24η·Ar / D_ij²  (only the linear r-term survives).
        Falls back to Route B per tail if D_ij ≈ 0.
        """
        val = self._d_rc1r_at_zero(i, j)
        if i != j:
            eta = self.eta
            vac = self.vac
            for t in range(self.n_tails):
                z = float(self.z_poles[t, i, j])
                if z <= 0.0:
                    continue
                K    = float(self.K[t, i, j])
                R_ij = float(self.R_ij[i, j])
                S    = vac**2*z**3 + 6*eta*vac*z**2 + 18*eta**2*z - 12*eta*(1 + 2*eta)
                L    = (1 + eta/2)*z + (1 + 2*eta)
                D_ij = S + 12*eta*L*math.exp(-z*R_ij)
                if abs(D_ij) < 1e-8:
                    val += -z * self._compute_poly_B(0.0, i, j, t)
                else:
                    Ar = (1 + 2*eta)**2*z**4 + vac*(1 + 2*eta)*z**5
                    val += K * 24*eta*Ar / D_ij**2
        return val

    def _solve_polycorr(self, i: int, j: int) -> np.ndarray:
        """Compute P_ij = [p₀, p₁, p₂, p₃, p₄] — Baxter-origin BCs only.

        Two constraints from Baxter theory at r = 0:
          (1) P(0)  = p₀ = −C₀           cancel 1/r singularity
          (2) P'(0) = p₁ = 2π√ρ·T_ij − D_rc1r(0)

        p₂=p₃=p₄=0 by default; OZSolverMSA (oz_numerical.py) fills p₂..p₄ for unlike
        pairs from the converged OZ+MSA solution (Group Z.2, todo_numerical.md).

        No constraints at r = R.  Contact discontinuity is physical — FMSA_pure
        imposes no P(R) or P'(R) condition either; those terms arise naturally
        from the Baxter polynomial in the exact single-component solution.
        """
        prefac   = 2.0 * math.pi * math.sqrt(self.rho[i] * self.rho[j])
        C0       = self._rc1r_at_zero(i, j)
        p0       = -C0
        D_rc1r_0 = self._d_rc1r_at_zero(i, j)
        T_ij     = self._target_c1_at_zero(i, j)
        p1       = prefac * T_ij - D_rc1r_0
        return np.array([p0, p1, 0.0, 0.0, 0.0])

    def _precompute_polycorr(self) -> None:
        """Cache polynomial coefficients for every pair (i,j)."""
        for i in range(self.N):
            for j in range(i, self.N):
                coeffs = self._solve_polycorr(i, j)
                self._polycorr[(i, j)] = coeffs
                self._polycorr[(j, i)] = coeffs

    def _solve_polycorr_B(self, i: int, j: int) -> np.ndarray:
        """Like _solve_polycorr but uses Route B BC values (includes PolyB correction at r=0)."""
        prefac   = 2.0 * math.pi * math.sqrt(self.rho[i] * self.rho[j])
        C0       = self._rc1r_at_zero_B(i, j)
        p0       = -C0
        D_rc1r_0 = self._d_rc1r_at_zero_B(i, j)
        T_ij     = self._target_c1_at_zero(i, j)
        p1       = prefac * T_ij - D_rc1r_0
        return np.array([p0, p1, 0.0, 0.0, 0.0])

    def _precompute_polycorr_B(self) -> None:
        """Cache Route B polynomial coefficients for every pair (i,j).  Idempotent."""
        if hasattr(self, '_polycorr_B'):
            return
        self._polycorr_B: dict[tuple[int, int], np.ndarray] = {}
        for i in range(self.N):
            for j in range(i, self.N):
                coeffs = self._solve_polycorr_B(i, j)
                self._polycorr_B[(i, j)] = coeffs
                self._polycorr_B[(j, i)] = coeffs

    def _solve_polycorr_A(self, i: int, j: int) -> np.ndarray:
        """Like _solve_polycorr but uses Route A BC values (includes poly_A at r=0)."""
        prefac = 2.0 * math.pi * math.sqrt(self.rho[i] * self.rho[j])
        C0 = self._rc1r_at_zero_A(i, j)
        p0 = -C0
        D_rc1r_0 = self._d_rc1r_at_zero_A(i, j)
        T_ij = self._target_c1_at_zero(i, j)
        p1 = prefac * T_ij - D_rc1r_0
        return np.array([p0, p1, 0.0, 0.0, 0.0])

    def _precompute_polycorr_A(self) -> None:
        """Cache Route A polynomial coefficients for every pair (i,j).  Idempotent."""
        if hasattr(self, "_polycorr_A"):
            return
        self._polycorr_A: dict[tuple[int, int], np.ndarray] = {}
        for i in range(self.N):
            for j in range(i, self.N):
                coeffs = self._solve_polycorr_A(i, j)
                self._polycorr_A[(i, j)] = coeffs
                self._polycorr_A[(j, i)] = coeffs

    def _hs_baxter_poly_coeffs(self, i: int, j: int) -> np.ndarray:
        """Z.1 diagnostic: extract 5 HS Baxter q^HS_{ij}(r) coefficients on [0, sigma_min].

        Returns [E, D, C, B, A] such that q^HS_ij(r) = A*r^4 + B*r^3 + C*r^2 + D*r + E.
        Uses Taylor expansion of P̂^HS_{ij}(z) = Q̂^HS_{ij}(z) - Ê^HS_{ij}*exp(-z*sigma_min)
        at z≈0, extracting the 5 Laplace moments M_n = int_0^{sigma_min} r^n * q^HS_ij(r) dr
        via 5-point finite difference, then solving the 5x5 moment-inversion system (B.8).

        NOTE: This is the HS Baxter polynomial q^HS, NOT the Yukawa polycorr P_ij.
        Extending this method to P_ij is analytically blocked by the 1/r factor in
        c^(1)_ij(r), whose Laplace transform involves E₁ (incomplete exponential integral)
        and is therefore non-rational — the z→0 Taylor expansion does not recover polynomial
        moments for Yukawa quantities.  See Group Z theoretical note in todo_numerical.md.
        Use OZSolverMSA (oz_numerical.py, Z.2) for the full 5-coefficient Yukawa polycorr.
        """
        h = 1e-4
        sm = self.sigma_min

        def P_hat_ij(z: float) -> float:
            Qhat = self._build_Qhat(z)
            _, E_hat = self._decompose_Q0(z)
            return float(Qhat[i, j] - E_hat[i, j] * np.exp(-z * sm))

        z0 = 5.0 * h
        fvals = [P_hat_ij(z0 + n * h) for n in range(-2, 3)]
        derivs = np.zeros(5)
        derivs[0] = fvals[2]
        derivs[1] = (fvals[3] - fvals[1]) / (2 * h)
        derivs[2] = (fvals[4] - 2*fvals[2] + fvals[0]) / h**2
        derivs[3] = (fvals[4] - 2*fvals[3] + 2*fvals[1] - fvals[0]) / (2 * h**3)
        derivs[4] = (fvals[4] - 4*fvals[3] + 6*fvals[2] - 4*fvals[1] + fvals[0]) / h**4
        M = np.array([(-1)**n * derivs[n] for n in range(5)])
        A_mat = np.array([[sm**(n+k+1) / (n+k+1) for k in range(5)] for n in range(5)])
        return np.linalg.solve(A_mat, M)   # [E, D, C, B, A]

    # ------------------------------------------------------------------
    # Antiderivative helpers for Term IV (closed-form double convolution)
    # ------------------------------------------------------------------

    @staticmethod
    def _antideriv_expquad(
        z: float, iz: float, iz2: float, iz3: float,
        qP0: float, qP1: float, qP2: float, u: float,
    ) -> float:
        """Antiderivative of exp(z·u)·(qP0 + qP1·u + qP2·u²) evaluated at u."""
        eu = np.exp(z * u)
        return eu * (
            qP0 * iz
            + qP1 * (u * iz - iz2)
            + qP2 * (u**2 * iz - 2.0 * u * iz2 + 2.0 * iz3)
        )

    @staticmethod
    def _antideriv_poly4(
        r0: float, r1: float, r2: float, r3: float, r4: float, u: float,
    ) -> float:
        """Antiderivative of r0 + r1·u + r2·u² + r3·u³ + r4·u⁴ evaluated at u."""
        return u * (r0 + u * (r1 / 2.0 + u * (r2 / 3.0 + u * (r3 / 4.0 + u * r4 / 5.0))))

    def _compute_poly_B(self, r: float, i: int, j: int, t: int) -> float:
        """Route B (Baxter-Direct): analytic q₀_ij convolution correction for unlike pair.

        Returns K_t · (I_decay(r, t) + I_grow(r, t)) for a single Yukawa pole t.
        I_decay = 0 for r < λ_ij; I_grow is nonzero for all r < R_ij.
        """
        if i == j:
            return 0.0
        rr    = math.sqrt(self.rho[i] * self.rho[j])
        Q0_ij = float(self.Q0[i, j])
        Qpp_j = float(self.Qpp[j])
        R     = float(self.R_ij[i, j])
        lam   = max(0.0, float(self.lambda_ij[i, j]))
        z     = float(self.z_poles[t, i, j])
        if z <= 0.0:
            return 0.0
        K     = float(self.K[t, i, j])
        iz    = 1.0 / z;  iz2 = iz * iz;  iz3 = iz2 * iz

        # Flat polynomial q₀_ij(u) = c0 + c1·u + c2·u² on [lam, R]; q₀_ij(R)=0.
        c2 = -rr * Qpp_j / 2.0
        c1 = -rr * Q0_ij + rr * Qpp_j * R
        c0 =  rr * Q0_ij * R - rr * Qpp_j * R * R / 2.0

        # Decaying: I_decay = exp(−z·r)·[F_pos(min(r,R)) − F_pos(lam)]
        if r < lam:
            I_decay = 0.0
        else:
            u_hi = min(r, R)
            F_hi = self._antideriv_expquad(z, iz, iz2, iz3, c0, c1, c2, u_hi)
            F_lo = self._antideriv_expquad(z, iz, iz2, iz3, c0, c1, c2, lam)
            I_decay = math.exp(-z * r) * (F_hi - F_lo)

        # Growing: I_grow = exp(z·(R−r))·∫₀^{min(R−r,σᵢ)} q̃(v)·exp(−z·v) dv
        # q̃(v) = q₀_ij(R−v) = rr·(Q0_ij·v − Qpp_j·v²/2)
        sig_i  = R - lam
        u_hi_g = min(max(R - r, 0.0), sig_i)
        c1g    = rr * Q0_ij
        c2g    = -rr * Qpp_j / 2.0
        G_hi   = self._antideriv_expquad(-z, -iz, iz2, -iz3, 0.0, c1g, c2g, u_hi_g)
        G_lo   = self._antideriv_expquad(-z, -iz, iz2, -iz3, 0.0, c1g, c2g, 0.0)
        I_grow = math.exp(z * (R - r)) * (G_hi - G_lo)

        return K * (I_decay + I_grow)

    def _compute_poly_A(self, r: float, i: int, j: int, t: int) -> float:
        """Route A polynomial correction: N=1 MSA polynomial with unlike-pair R_ij.

        Applies the exact first-order MSA polynomial from FMSA_pure (N=1 formula)
        with the unlike-pair contact distance R_ij replacing the like-pair σ.  The
        denominator D_ij = S(z) + 12η·L(z)·exp(−z·R_ij) uses the total packing
        fraction η and the pair-specific R_ij; D_ij generalises the N=1 Baxter
        denominator D = S + 12η·L·exp(−z·σ) to unlike pairs.

        Formula (mirrors FMSA_pure.get_rc1r, polynomial part):
            poly_A(r) = K · (12η²·Ar·r⁴ − 12η·Br·r² + 24η·Ar·r − 24η·S·L) / D_ij²

        where  S(z)  = vac²z³ + 6η·vac·z² + 18η²z − 12η(1+2η)
               L(z)  = (1+η/2)z + (1+2η)
               Ar(z) = (1+2η)²z⁴ + vac(1+2η)z⁵
               Br(z) = S·L·z² − vac²(1+η/2)z⁶

        Reduces to the exact FMSA_pure polynomial for N=1 like pairs (since R_ii = σᵢ
        and the like-pair branch uses _pure_refs directly, not this function).

        Falls back to Route B if |D_ij| < 1e-8 (numerical near-zero).
        """
        if i == j:
            return 0.0
        z = float(self.z_poles[t, i, j])
        if z <= 0.0:
            return 0.0
        K     = float(self.K[t, i, j])
        R_ij  = float(self.R_ij[i, j])
        eta   = self.eta
        vac   = self.vac
        S     = vac**2*z**3 + 6*eta*vac*z**2 + 18*eta**2*z - 12*eta*(1 + 2*eta)
        L     = (1 + eta/2)*z + (1 + 2*eta)
        D_ij  = S + 12*eta*L*math.exp(-z*R_ij)
        if abs(D_ij) < 1e-8:
            return self._compute_poly_B(r, i, j, t)
        Ar    = (1 + 2*eta)**2*z**4 + vac*(1 + 2*eta)*z**5
        Br    = S*L*z**2 - vac**2*(1 + eta/2)*z**6
        poly_num = (12*eta**2*Ar*r**4 - 12*eta*Br*r**2
                    + 24*eta*Ar*r - 24*eta*S*L)
        return K * poly_num / D_ij**2

    # ------------------------------------------------------------------
    # Cross-species mediated terms (Terms II + III + IV from Eq. 41)
    # ------------------------------------------------------------------

    def _compute_mediated(self, r: float, i: int, j: int) -> float:
        """Compute the mediated correction c^(1)_mediated_ij(r) for r < R_ij.

        Adds Terms II, III, and IV from the multi-species inner-core formula (Eq. 41):

        II  = Σ_a √(ρ_a ρ_i) · Σ_k Rst_k · [Q0[a,i]·I1 + ½Qpp[i]·I2]
        III = Σ_b √(ρ_b ρ_j) · Σ_k Rst_k · [Q0[b,j]·I1 + ½Qpp[j]·I2]
        IV  = −Σ_{a,b} ∫∫ q_ai(u−R_ab−w)·b_grow_ab(w)·q_bj(r−u) dw du  (analytic)

        Return value is in units of c^(1) (not r·c^(1)):
            c^(1)_mediated = (II + III + IV) / (2π·√(ρ_i·ρ_j)·r)

        These terms are zero for N=1 or for like pairs of the smallest species
        (proved: Lean Tasks 1.3, B.4 — lstar = 0 whenever r < R[a,j] for all a).

        Parameters
        ----------
        r    : radial distance; must satisfy 0 < r < R_ij[i,j]
        i, j : 0-based species indices

        Returns
        -------
        float : mediated first-order DCF contribution
        """
        N         = self.N
        rho       = self.rho
        sigma     = self.sigma
        R         = self.R_ij
        lambda_ij = self.lambda_ij
        Q0        = self.Q0
        Qpp       = self.Qpp

        # --- Term II: Σ_a convolution of b_grow_{a,j} against off-diagonal q_{a,i} ---
        II = 0.0
        for a in range(N):
            if r < R[a, j]:
                continue
            bg    = self.b_grow[a][j]
            l     = r - R[a, j]
            alpha = l + lambda_ij[i, a] - sigma[a]
            lstar = max(0.0, min(l, alpha))
            if lstar <= 0.0:
                continue
            term = 0.0
            for z, Rst in zip(bg.poles, bg.coeffs):
                exp_zl = np.exp(z * lstar)
                I1 = (
                    (alpha - lstar) * exp_zl / z
                    + (exp_zl - 1.0) / z**2
                    - alpha / z
                )
                I2 = (
                    (alpha - lstar)**2 * exp_zl / z
                    + 2.0 * (alpha - lstar) * exp_zl / z**2
                    + 2.0 * exp_zl / z**3
                    - alpha**2 / z - 2.0 * alpha / z**2 - 2.0 / z**3
                )
                term += Rst * (Q0[a, i] * I1 + 0.5 * Qpp[i] * I2)
            II += np.sqrt(rho[a] * rho[i]) * term

        # --- Term III: Σ_b convolution of b_grow_{i,b} against off-diagonal q_{b,j} ---
        III = 0.0
        for b in range(N):
            if r < R[i, b]:
                continue
            bg    = self.b_grow[i][b]
            l     = r - R[i, b]
            alpha = l + lambda_ij[j, b] - sigma[b]
            lstar = max(0.0, min(l, alpha))
            if lstar <= 0.0:
                continue
            term = 0.0
            for z, Rst in zip(bg.poles, bg.coeffs):
                exp_zl = np.exp(z * lstar)
                I1 = (
                    (alpha - lstar) * exp_zl / z
                    + (exp_zl - 1.0) / z**2
                    - alpha / z
                )
                I2 = (
                    (alpha - lstar)**2 * exp_zl / z
                    + 2.0 * (alpha - lstar) * exp_zl / z**2
                    + 2.0 * exp_zl / z**3
                    - alpha**2 / z - 2.0 * alpha / z**2 - 2.0 / z**3
                )
                term += Rst * (Q0[b, j] * I1 + 0.5 * Qpp[j] * I2)
            III += np.sqrt(rho[b] * rho[j]) * term

        # --- Term IV: analytic double convolution −Σ_{a,b} ∫∫ q_ai·b_ab·q_bj ---
        IV_raw = 0.0
        for a in range(N):
            for b in range(N):
                bg_ab = self.b_grow[a][b]
                Rab   = R[a, b]

                Delta_ai = lambda_ij[i, a] - sigma[a]
                Alm      = max(Delta_ai, 0.0)         # alpha − lstar (constant)
                c_exp    = Rab + max(0.0, -Delta_ai)  # u where lstar first > 0
                d_exp    = Rab - Delta_ai              # = c_exp − Alm

                u_lo_bj  = r + lambda_ij[b, j] - sigma[b]
                u_hi_bj  = r + lambda_ij[b, j]
                u_lo_eff = max(c_exp, u_lo_bj)
                u_hi_eff = min(r, u_hi_bj)
                if u_lo_eff >= u_hi_eff:
                    continue

                C_bj = np.sqrt(rho[b] * rho[j])
                qP2  = -C_bj * 0.5 * Qpp[j]
                qP1  = -C_bj * (-Q0[b, j] - Qpp[j] * u_lo_bj)
                qP0  = -C_bj * (Q0[b, j] * u_lo_bj + 0.5 * Qpp[j] * u_lo_bj**2)

                Q0ai = Q0[a, i]
                Qi   = 0.5 * Qpp[i]
                Cai  = np.sqrt(rho[a] * rho[i])

                for z, Rst in zip(bg_ab.poles, bg_ab.coeffs):
                    iz  = 1.0 / z
                    iz2 = iz * iz
                    iz3 = iz2 * iz
                    enc = np.exp(-z * c_exp)

                    E1 = (Alm * iz + iz2) * enc
                    E2 = (Alm**2 * iz + 2.0 * Alm * iz2 + 2.0 * iz3) * enc

                    G1 = d_exp * iz - iz2
                    G2 = 2.0 * d_exp * iz - 2.0 * iz2
                    H2 = -d_exp**2 * iz + 2.0 * d_exp * iz2 - 2.0 * iz3

                    A_k = Rst * (Q0ai * E1 + Qi * E2)
                    p2  = Rst * Qi * (-iz)
                    p1  = Rst * (Q0ai * (-iz) + Qi * G2)
                    p0  = Rst * (Q0ai * G1 + Qi * H2)

                    r4 = p2 * qP2
                    r3 = p2 * qP1 + p1 * qP2
                    r2 = p2 * qP0 + p1 * qP1 + p0 * qP2
                    r1 = p1 * qP0 + p0 * qP1
                    r0 = p0 * qP0

                    exp_int = A_k * (
                        self._antideriv_expquad(z, iz, iz2, iz3, qP0, qP1, qP2, u_hi_eff)
                        - self._antideriv_expquad(z, iz, iz2, iz3, qP0, qP1, qP2, u_lo_eff)
                    )
                    poly_int = (
                        self._antideriv_poly4(r0, r1, r2, r3, r4, u_hi_eff)
                        - self._antideriv_poly4(r0, r1, r2, r3, r4, u_lo_eff)
                    )
                    IV_raw += Cai * (exp_int + poly_int)

        IV = -IV_raw
        prefactor = 1.0 / (2.0 * np.pi * np.sqrt(rho[i] * rho[j]))
        return prefactor * (II + III + IV) / r

    # ------------------------------------------------------------------
    # Hard-sphere FMT reference (White Bear functional, KR normalisation)
    # ------------------------------------------------------------------

    @staticmethod
    def _wb_phi_derivs(x: float) -> tuple[float, float, float]:
        """φ(n₃) = [n₃ + v²ln v]/(36π n₃² v²), v = 1−n₃, with φ′ and φ″ (analytic).

        This is the White Bear φ₃ factor of
        Φ₃ = (n₂³ − 3n₂ n_v2²)·[n₃ + (1−n₃)²ln(1−n₃)]/(36π n₃²(1−n₃)²).
        """
        v = 1.0 - x
        lv = math.log(v)
        N  = x + v**2 * lv
        N1 = 1.0 - 2.0 * v * lv - v
        N2 = 2.0 * lv + 3.0
        D  = 36.0 * np.pi * x**2 * v**2
        D1 = 72.0 * np.pi * x * v * (v - x)
        D2 = 72.0 * np.pi * (v**2 - 4.0 * x * v + x**2)
        phi  = N / D
        phi1 = N1 / D - N * D1 / D**2
        phi2 = N2 / D - 2.0 * N1 * D1 / D**2 - N * D2 / D**2 + 2.0 * N * D1**2 / D**3
        return phi, phi1, phi2

    def get_HS_FMT(self, r: float, i: int, j: int) -> float:
        """White Bear FMT hard-sphere mixture pair DCF c^HS_ij(r), closed form.

        Built directly from the literature definitions (Roth, Evans, Lang & Kahl
        2002; weights and Φ as stated e.g. in arXiv:2508.14642):

            ω₀ = δ(R−r)/(4πR²),  ω₁ = δ(R−r)/(4πR),  ω₂ = δ(R−r),
            ω₃ = Θ(R−r),         ω_v2 = r̂·δ(R−r),    ω_v1 = ω_v2/(4πR)
            n₀ = Σρᵢ,  n₁ = Σρᵢ Rᵢ,  n₂ = Σρᵢ·4πRᵢ²,  n₃ = Σρᵢ(4π/3)Rᵢ³ = η
            Φ = −n₀ln(1−n₃) + (n₁n₂ − n_v1·n_v2)/(1−n₃)
                + (n₂³ − 3n₂ n_v2·n_v2)·φ(n₃)
            c_ij(r) = −Σ_αβ (∂²Φ/∂n_α∂n_β)·(ω_α^i ⊛ ω_β^j)(r)

        The vector weights satisfy ŵ_v2(k) = −i·k·ŵ₃(k), so their convolution is
        −∇²(ω₃^i ⊛ ω₃^j) — no separate geometry is needed.  The scalar weight
        convolutions are elementary and piecewise, with the inner break at
        |Rᵢ−Rⱼ| = |λ_ij| (the smaller sphere entirely inside the larger).

        Verified (2026-07-17): −ĉ_ij(0) = ∂²Φ/∂ρᵢ∂ρⱼ for every mixture pair
        (~1e-5, quadrature noise); one-component 1 − ρĉ(0) reproduces
        Carnahan–Starling (1+4η+4η²−4η³+η⁴)/(1−η)⁴ to 0.0000% at every η;
        c(r) → −1 as ρ → 0.

        [Corrected 2026-07-17.  The previous χ₀..χ₃/χ₂₂ formula (inherited
        verbatim from FMSA_GA_matrix_mix, FMSA_pure and FMSA_MC_cleaned_2cpp)
        used n₂ = π Σρσ²/4 — a factor 4 below the literature n₂ = Σρ·4πR² —
        with χ-coefficients that did not compensate: it undershot the
        Carnahan–Starling compressibility by 8.8% at η=0.10 and 51% at η=0.42.]
        """
        Ri = float(self.sigma[i]) / 2.0
        Rj = float(self.sigma[j]) / 2.0
        rho, R = self.rho, self.sigma / 2.0

        n0 = float(np.sum(rho))
        n1 = float(np.sum(rho * R))
        n2 = float(np.sum(rho * 4.0 * np.pi * R**2))
        n3 = float(np.sum(rho * (4.0 * np.pi / 3.0) * R**3))     # = η
        v  = 1.0 - n3
        ph, ph1, ph2 = self._wb_phi_derivs(n3)

        # Phi_ab := ∂²Φ/∂n_a∂n_b, the Hessian of Φ at the bulk
        # (vector densities vanish there); subscripts are weight indices.
        Phi03 = 1.0 / v
        Phi12 = 1.0 / v
        Phi13 = n2 / v**2
        Phi23 = n1 / v**2 + 3.0 * n2**2 * ph1
        Phi22 = 6.0 * n2 * ph
        Phi33 = n0 / v**2 + 2.0 * n1 * n2 / v**3 + n2**3 * ph2
        Phi_v1v2 = -1.0 / v
        Phi_v2v2 = -6.0 * n2 * ph

        lam = abs(Ri - Rj)
        S   = Ri + Rj
        if r >= S:
            return 0.0
        if r < lam:
            # smaller sphere entirely inside the larger one
            A    = 0.0
            Bij  = 4.0 * np.pi * Ri**2 if Ri < Rj else 0.0
            Bji  = 4.0 * np.pi * Rj**2 if Rj < Ri else 0.0
            V    = (4.0 * np.pi / 3.0) * min(Ri, Rj)**3
            lapV = 0.0
        else:
            Dm   = (Ri - Rj)**2
            A    = 2.0 * np.pi * Ri * Rj / r                       # ω₂ⁱ ⊛ ω₂ʲ
            Bij  = np.pi * Ri * (Rj**2 - (r - Ri)**2) / r          # ω₂ⁱ ⊛ ω₃ʲ
            Bji  = np.pi * Rj * (Ri**2 - (r - Rj)**2) / r          # ω₃ⁱ ⊛ ω₂ʲ
            V    = np.pi * (S - r)**2 * (r**2 + 2.0*r*S - 3.0*Dm) / (12.0 * r)
            # ∇²V = (1/r)·d²(rV)/dr² with rV = (π/12)·p(r),
            # p = r⁴ − 3(S²+Dm)r² + (2S³+6S·Dm)r − 3S²Dm  ⟹ p″ = 12r² − 6(S²+Dm)
            lapV = (np.pi / 12.0) * (12.0 * r**2 - 6.0 * (S**2 + Dm)) / r

        scal = (Phi03 * (Bij / (4.0*np.pi*Ri**2) + Bji / (4.0*np.pi*Rj**2))
                + Phi12 * (A / (4.0*np.pi*Ri) + A / (4.0*np.pi*Rj))
                + Phi13 * (Bij / (4.0*np.pi*Ri) + Bji / (4.0*np.pi*Rj))
                + Phi23 * (Bij + Bji) + Phi22 * A + Phi33 * V)
        vec = lapV * (Phi_v1v2 * (1.0/(4.0*np.pi*Ri) + 1.0/(4.0*np.pi*Rj)) + Phi_v2v2)
        return float(-scal + vec)

    # ------------------------------------------------------------------
    # First-order MSA correction (Path B)
    # ------------------------------------------------------------------

    def get_c1_inner(self, r: float, i: int, j: int) -> float:
        """Path B first-order MSA correction c^(1)_ij(r) for r < R_ij.

        Assembles two contributions:

        **Term I (direct, corrected)** — per Yukawa tail t:

        *Like pairs (i==j)*:
            r·c^(1)_direct = K_t · [(1 − G²_ij)·exp(−z·dr) − A²_ij·exp(+z·dr) + poly/D²]
        where D = S + 12η·L·exp(−z·R_ij) is the scalar Baxter denominator.
        For N=1: G[0,0]=g, A[0,0]=a (Lean Task M.2) → identical to FMSA_pure.

        *Unlike pairs (i≠j)* — pure-exponential formula (no scalar D²):
            r·c^(1)_direct = K_t · [(1 − G²_ij)·exp(−z·dr) − A²_ij·exp(+z·dr)]
        The scalar D at z_{ij}, R_{ij} can be near-zero for unlike pairs, inflating the
        24η·S·L/D² polynomial term to O(10⁶) and making C₀ huge.  Omitting the D²
        polynomial avoids this; origin BCs are enforced by the linear polynomial correction.

        **Terms II + III + IV (mediated, cross-species)** — from _compute_mediated:

        Non-zero only for unlike pairs in N>1 where r ≥ R[a,j] for some smaller
        species a.  Zero for N=1 and for like pairs of the smallest species
        (Lean Tasks 1.3, B.4).

        Parameters
        ----------
        r : separation distance (must satisfy r < R_ij[i,j])
        i : species index (row)
        j : species index (column)

        Returns
        -------
        c1 : scalar first-order DCF correction at r
        """
        R_ij   = self.R_ij[i, j]
        eta    = self.eta
        vac    = self.vac
        prefac = 1.0 / (2.0 * np.pi * np.sqrt(self.rho[i] * self.rho[j]) * r)
        rc1r   = 0.0
        for t in range(self.n_tails):
            z = float(self.z_poles[t, i, j])
            if z <= 0.0:
                continue
            K_t = float(self.K[t, i, j])

            # Matrix elements at this tail's pole — direct ndarray access, no dict lookup.
            # _G_mat[t, i, j] is the (N×N) Ĝ matrix at z_poles[t, i, j]; take element (i,j).
            g_ij = float(self._G_mat[t, i, j, i, j])
            a_ij = float(self._A_mat[t, i, j, i, j])

            dr       = r - R_ij
            exp_core = np.exp(-z * dr)

            if i != j:
                # Unlike pairs: pure-exponential formula.  The scalar D at z_{ij},R_{ij}
                # can be near-zero (D≈0), inflating the 24η·S·L/D² polynomial term to
                # O(10⁶).  Avoid by skipping the D² denominator.
                rc1r += K_t * ((1.0 - g_ij**2) * exp_core - a_ij**2 / exp_core)
            else:
                # Like pairs: full correction with scalar Baxter D².
                # For N=1: g_ij·D = S and a_ij·D = 12ηLe^{-zR} → FMSA_pure exactly.
                S  = vac**2*z**3 + 6*eta*vac*z**2 + 18*eta**2*z - 12*eta*(1 + 2*eta)
                L  = (1 + eta/2)*z + (1 + 2*eta)
                D  = S + 12*eta*L*np.exp(-z*R_ij)
                D2 = D**2
                Ar = (1 + 2*eta)**2 * z**4 + vac*(1 + 2*eta)*z**5
                Br = S*L*z**2 - vac**2*(1 + eta/2)*z**6
                gD = g_ij * D
                aD = a_ij * D
                correction = (
                      gD**2      * exp_core
                    + aD**2      / exp_core
                    - 12*eta**2  * Ar * r**4
                    + 12*eta     * Br * r**2
                    - 24*eta     * Ar * r
                    + 24*eta     * S  * L
                )
                rc1r += K_t * (exp_core - correction / D2)

        # Polynomial correction P_ij(r): p0+p1·r always from origin BCs;
        # p2..p4 filled by OZSolverMSA for unlike pairs (Group Z.2), else 0.
        p0, p1, p2, p3, p4 = self._polycorr[(i, j)]
        rc1r += p0 + p1 * r + p2 * r**2 + p3 * r**3 + p4 * r**4

        # Terms II + III + IV: cross-species convolution corrections.
        # Zero for N=1 (Lean Tasks 1.3, B.4); non-zero for unlike pairs in N>1.
        return rc1r * prefac + self._compute_mediated(r, i, j)

    def get_c1_inner_exp(self, r: float, i: int, j: int) -> float:
        """Analytic-exponential FMSA inner-core DCF c^(1)_ij(r), r < R_ij.

        Uses the exact exponential structure throughout — no polynomial approximation
        of the growing tail:

        *Like pairs (i==j)*:
            Delegates to FMSA_pure._pure_refs[i].get_rc1r(r) / r.
            This is the exact Q₀-Baxter first-order result (growing + decaying
            exponentials from the Baxter factorisation, plus polynomial-×-exponential
            terms from the HS reference).  Matches FMSA_pure to machine precision
            for N=1.

        *Unlike pairs (i≠j)*:
            rc₁r = Σ_t K_t·[(1−G²_t)·exp(z_t·(R_ij−r)) − A²_t·exp(−z_t·(R_ij−r))]
                   + p₀ + p₁·r
            where:
              (1−G²_t)·exp(z_t·(R_ij−r)) — growing exponential, analytic from G_mat
              A²_t·exp(−z_t·(R_ij−r))    — decaying exponential, analytic from A_mat
              p₀ = −_rc1r_at_zero(i,j)   — exact 1/r-singularity cancellation
              p₁ from _solve_polycorr      — exact BC match at c₁(0) target

            Both p₀ and p₁ are already stored in _polycorr and use the pure-exponential
            formula; no higher-order polynomial terms (p₂=p₃=p₄=0 exactly).

        *Mediated (N≥3)*:
            Adds _compute_mediated(r, i, j) — analytic cross-pair growing-tail
            contributions (Terms II+III+IV); zero for N≤2 by geometry.
        """
        if i == j:
            return self._pure_refs[i].get_rc1r(r) / r

        # Unlike pairs: two-exponential GA formula + p₀ + p₁
        R_ij   = self.R_ij[i, j]
        prefac = 1.0 / (2.0 * math.pi * math.sqrt(self.rho[i] * self.rho[j]) * r)
        rc1r   = 0.0
        for t in range(self.n_tails):
            z = float(self.z_poles[t, i, j])
            if z <= 0.0:
                continue
            K_t  = float(self.K[t, i, j])
            g_ij = float(self._G_mat[t, i, j, i, j])
            a_ij = float(self._A_mat[t, i, j, i, j])
            dr       = r - R_ij               # < 0 inside core
            exp_core = math.exp(-z * dr)      # = exp(z*(R_ij−r)), grows as r→0
            rc1r += K_t * ((1.0 - g_ij**2) * exp_core - a_ij**2 / exp_core)

        # p₀ and p₁ only — exact BC enforcement, no polynomial approximation
        p0, p1 = self._polycorr[(i, j)][0], self._polycorr[(i, j)][1]
        rc1r += p0 + p1 * r

        return rc1r * prefac + self._compute_mediated(r, i, j)

    def get_c_total_exp(self, r_grid: np.ndarray) -> np.ndarray:
        """Total DCF on a radial grid using the analytic-exponential inner formula.

        Identical to get_c_total except the inner region (r < R_ij) uses
        get_c1_inner_exp instead of get_c1_inner.  The outer region (r ≥ R_ij)
        is unchanged — the MSA tail closure is the same for all variants.

        Returns
        -------
        c_total : shape (N, N, len(r_grid))
        """
        N       = self.N
        r_grid  = np.asarray(r_grid, dtype=float)
        c_total = np.zeros((N, N, len(r_grid)), dtype=float)

        for i in range(N):
            for j in range(N):
                rho_ij = math.sqrt(self.rho[i] * self.rho[j])
                R_ij   = self.R_ij[i, j]
                for ki, r in enumerate(r_grid):
                    if r <= 0.0:
                        continue
                    if r < R_ij:
                        c_total[i, j, ki] = (
                            self.get_HS_FMT(r, i, j) + self.get_c1_inner_exp(r, i, j)
                        )
                    else:
                        c_outer = 0.0
                        for t in range(self.n_tails):
                            z = float(self.z_poles[t, i, j])
                            if z > 0.0:
                                c_outer += float(self.K[t, i, j]) * math.exp(-z * (r - R_ij))
                        c_total[i, j, ki] = c_outer / (2.0 * math.pi * rho_ij * r)

        return c_total

    def get_c1_inner_baxter(self, r: float, i: int, j: int) -> float:
        """Route B (Baxter-Direct) inner c₁(r) for r < R_ij.

        Like pairs: delegates to FMSA_pure (exact Q₀-Baxter result).
        Unlike pairs: two-exponential GA base + Baxter-Direct PolyB correction
        + mediated terms + updated p₀_B, p₁_B BCs.
        """
        if i == j:
            return self._pure_refs[i].get_rc1r(r) / r

        self._precompute_polycorr_B()

        R_ij   = self.R_ij[i, j]
        prefac = 1.0 / (2.0 * math.pi * math.sqrt(self.rho[i] * self.rho[j]) * r)
        rc1r   = 0.0

        for t in range(self.n_tails):
            z = float(self.z_poles[t, i, j])
            if z <= 0.0:
                continue
            K_t  = float(self.K[t, i, j])
            g_ij = float(self._G_mat[t, i, j, i, j])
            a_ij = float(self._A_mat[t, i, j, i, j])
            dr       = r - R_ij
            exp_core = math.exp(-z * dr)
            rc1r += K_t * ((1.0 - g_ij**2) * exp_core - a_ij**2 / exp_core)
            rc1r += self._compute_poly_B(r, i, j, t)

        p0, p1 = self._polycorr_B[(i, j)][0], self._polycorr_B[(i, j)][1]
        rc1r += p0 + p1 * r

        return rc1r * prefac + self._compute_mediated(r, i, j)

    def get_c_total_baxter(self, r_grid: np.ndarray) -> np.ndarray:
        """Total DCF on a radial grid using the Baxter-Direct (Route B) inner formula.

        Identical to get_c_total_exp except the inner region (r < R_ij) uses
        get_c1_inner_baxter.  The outer region (r ≥ R_ij) is unchanged.

        Returns
        -------
        c_total : shape (N, N, len(r_grid))
        """
        self._precompute_polycorr_B()
        N       = self.N
        r_grid  = np.asarray(r_grid, dtype=float)
        c_total = np.zeros((N, N, len(r_grid)), dtype=float)

        for i in range(N):
            for j in range(N):
                rho_ij = math.sqrt(self.rho[i] * self.rho[j])
                R_ij   = self.R_ij[i, j]
                for ki, r in enumerate(r_grid):
                    if r <= 0.0:
                        continue
                    if r < R_ij:
                        c_total[i, j, ki] = (
                            self.get_HS_FMT(r, i, j) + self.get_c1_inner_baxter(r, i, j)
                        )
                    else:
                        c_outer = 0.0
                        for t in range(self.n_tails):
                            z = float(self.z_poles[t, i, j])
                            if z > 0.0:
                                c_outer += float(self.K[t, i, j]) * math.exp(-z * (r - R_ij))
                        c_total[i, j, ki] = c_outer / (2.0 * math.pi * rho_ij * r)

        return c_total

    def get_c1_inner_route_a(self, r: float, i: int, j: int) -> float:
        """Route A (Complex-Pole Residue) inner c₁(r) for r < R_ij.

        Like pairs: delegates to FMSA_pure (exact Q₀-Baxter result).
        Unlike pairs: two-exponential GA base + HS-pole residue poly_A correction
        + updated p₀_A, p₁_A BCs + mediated terms.
        """
        if i == j:
            return self._pure_refs[i].get_rc1r(r) / r

        self._precompute_polycorr_A()

        R_ij = self.R_ij[i, j]
        prefac = 1.0 / (2.0 * math.pi * math.sqrt(self.rho[i] * self.rho[j]) * r)
        rc1r = 0.0

        for t in range(self.n_tails):
            z = float(self.z_poles[t, i, j])
            if z <= 0.0:
                continue
            K_t = float(self.K[t, i, j])
            g_ij = float(self._G_mat[t, i, j, i, j])
            a_ij = float(self._A_mat[t, i, j, i, j])
            exp_core = math.exp(-z * (r - R_ij))
            rc1r += K_t * ((1.0 - g_ij**2) * exp_core - a_ij**2 / exp_core)
            rc1r += self._compute_poly_A(r, i, j, t)

        p0 = self._polycorr_A[(i, j)][0]
        p1 = self._polycorr_A[(i, j)][1]
        rc1r += p0 + p1 * r

        return rc1r * prefac + self._compute_mediated(r, i, j)

    def get_c_total_route_a(self, r_grid: np.ndarray) -> np.ndarray:
        """Total DCF on a radial grid using the Route A (Complex-Pole Residue) inner formula.

        Identical to get_c_total_baxter except the inner region (r < R_ij) uses
        get_c1_inner_route_a.  The outer region (r ≥ R_ij) is unchanged.

        Returns
        -------
        c_total : shape (N, N, len(r_grid))
        """
        self._precompute_polycorr_A()
        N = self.N
        r_grid = np.asarray(r_grid, dtype=float)
        c_total = np.zeros((N, N, len(r_grid)), dtype=float)

        for i in range(N):
            for j in range(N):
                rho_ij = math.sqrt(self.rho[i] * self.rho[j])
                R_ij = self.R_ij[i, j]
                for ki, r in enumerate(r_grid):
                    if r <= 0.0:
                        continue
                    if r < R_ij:
                        c_total[i, j, ki] = (
                            self.get_HS_FMT(r, i, j) + self.get_c1_inner_route_a(r, i, j)
                        )
                    else:
                        c_outer = 0.0
                        for t in range(self.n_tails):
                            z = float(self.z_poles[t, i, j])
                            if z > 0.0:
                                c_outer += float(self.K[t, i, j]) * math.exp(-z * (r - R_ij))
                        c_total[i, j, ki] = c_outer / (2.0 * math.pi * rho_ij * r)

        return c_total

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_c_total(self, r_grid: np.ndarray) -> np.ndarray:
        """Evaluate the total DCF c_ij(r) on a radial grid for all species pairs.

        Applies two closures depending on position relative to R_ij:

          r < R_ij : c_ij(r) = c^HS_ij(r) + c^(1)_ij(r)
                     (Rosenfeld FMT hard-sphere reference + Path B correction)
          r ≥ R_ij : c_ij(r) = −β Σ_t u_t,ij(r)
                     (strict MSA tail: sum of attractive Yukawa potentials)

        Parameters
        ----------
        r_grid : radial grid, shape (M,)

        Returns
        -------
        c_total : shape (N, N, M);  c_total[i, j, k] = c_ij(r_grid[k])
        """
        N       = self.N
        r_grid  = np.asarray(r_grid, dtype=float)
        c_total = np.zeros((N, N, len(r_grid)), dtype=float)

        for i in range(N):
            for j in range(N):
                rho_ij = np.sqrt(self.rho[i] * self.rho[j])
                R_ij   = self.R_ij[i, j]
                for ki, r in enumerate(r_grid):
                    if r <= 0.0:
                        continue
                    if r < R_ij:
                        c_total[i, j, ki] = (
                            self.get_HS_FMT(r, i, j) + self.get_c1_inner(r, i, j)
                        )
                    else:
                        c_outer = 0.0
                        for t in range(self.n_tails):
                            z = float(self.z_poles[t, i, j])
                            if z > 0.0:
                                c_outer += float(self.K[t, i, j]) * np.exp(-z * (r - R_ij))
                        c_total[i, j, ki] = c_outer / (2.0 * np.pi * rho_ij * r)

        return c_total

    def get_c_total_pathb(self, r_grid: np.ndarray) -> np.ndarray:
        """Evaluate total DCF for all unique pairs; mirrors FMSA_poly get_c_total_poly.

        Parameters
        ----------
        r_grid : radial grid, shape (M,)

        Returns
        -------
        c_pairs : shape (n_unique_pairs, M) where pairs are ordered (i,j) for
                  i ≤ j in row-major order: (0,0), (0,1), …, (N-1,N-1).
                  For N=2: index 0 = c₀₀, 1 = c₀₁, 2 = c₁₁.
        """
        N      = self.N
        c_full = self.get_c_total(np.asarray(r_grid, dtype=float))
        n_pairs = N * (N + 1) // 2
        result  = np.zeros((n_pairs, len(r_grid)), dtype=float)
        p = 0
        for i in range(N):
            for j in range(i, N):
                result[p] = c_full[i, j]
                p += 1
        return result

    def verify_identity(self, tol: float = 1e-10) -> dict[float, float]:
        """Check the algebraic identity Ĝ(z) + Â(z)·exp(−z·σ_min) = I.

        Iterates over every (t, m, n) pole slot and evaluates the max element-wise
        absolute deviation from the identity matrix.  Values should be at or near
        machine epsilon (~1e-15) for a well-conditioned system.  Duplicate z values
        across different slots are reported only once.

        Parameters
        ----------
        tol : reserved for a future pass/fail threshold; not used in computation

        Returns
        -------
        errors : dict mapping each unique pole z → max |Ĝ + Â·exp(−z·σ_min) − I|
        """
        errors: dict[float, float] = {}
        for t in range(self.n_tails):
            for m in range(self.N):
                for n in range(self.N):
                    z = float(self.z_poles[t, m, n])
                    if z <= 0.0 or z in errors:
                        continue
                    G   = self._G_mat[t, m, n]
                    A   = self._A_mat[t, m, n]
                    c   = np.exp(-z * self.sigma_min)
                    err = float(np.max(np.abs(G + c * A - np.eye(self.N))))
                    errors[z] = err
        return errors

    # ------------------------------------------------------------------
    # FMSAFreeEnergy interface — attribute aliases and _make_at_rho
    # ------------------------------------------------------------------

    @property
    def radii(self) -> np.ndarray:
        """Contact-distance matrix R_ij — alias expected by FMSAFreeEnergy."""
        return self.R_ij

    @property
    def z_matrices(self) -> np.ndarray:
        """Yukawa decay constants z_poles — alias expected by FMSAFreeEnergy."""
        return self.z_poles

    @property
    def k_bare_matrices(self) -> np.ndarray:
        """Bare Yukawa amplitudes k_bare — alias expected by FMSAFreeEnergy."""
        return self.k_bare

    def _make_at_rho(self, rho_vec: np.ndarray) -> "FMSA_GA_matrix_mix":
        """Return a new instance at partial densities rho_vec (same T, σ, ε, Yukawa params)."""
        rhostar_new = float(rho_vec.sum())
        x_new = rho_vec / rhostar_new
        return type(self)(
            self.sig_pure, self.eps_pure, x_new, rhostar_new, self.tstar,
            n_tails=self.n_tails, z=self._z_input, k=self._k_input,
            input_type=self._input_type, verbose=False,
        )

    # ------------------------------------------------------------------
    # FMSAFreeEnergy interface — HS and Yukawa thermodynamic quantities
    # ------------------------------------------------------------------

    @staticmethod
    def _phi1s(rv: float, sv: float) -> float:
        """φ₁ auxiliary: (1 − sv·rv − exp(−sv·rv)) / sv²."""
        return (1.0 - sv * rv - np.exp(-sv * rv)) / sv**2

    def G0red_laplace_ij(self, i: int, j: int, s: float) -> float:
        """Reduced Laplace kernel of the HS pair RDF at variable s.

        G0red_ij(s) = g̃₀_ij(s) / (2π√(ρᵢρⱼ) exp(−s·R_ij))

        where g̃₀_ij(s) = L[r·g⁰_ij(r)](s) is the analytic Laplace transform
        of the HS RDF (Percus-Yevick / Baxter result):

            G0red_ij(s) = [C₁/s² + C₂(s)/s] / (vac · det Q̂₀(s))

        with  C₁ = 1 + π·ξ₃/(2·vac)
              C₂ = R_ij + π·ξ₂·Rᵢ·Rⱼ/(4·vac)
                   + (π/(2·vac)) Σ_m ρ_m φ₁(σ_m,s) (σ_m/2−σᵢ/2)(σ_m/2−σⱼ/2)
        where ξₙ = Σ_k ρ_k σ_k^n  (no π/6 factor, same as self.xi).
        """
        R_i  = float(self.diag_r[i])
        R_j  = float(self.diag_r[j])
        R_ij = float(self.R_ij[i, j])
        vac  = self.vac
        det_Qs = float(np.linalg.det(self._build_Qhat(s)))
        c1 = 1.0 + np.pi * float(self.xi[3]) / (2.0 * vac)
        c2 = R_ij + np.pi * float(self.xi[2]) * R_i * R_j / (4.0 * vac)
        for m in range(self.N):
            R_m = float(self.diag_r[m])
            c2 += (np.pi / (2.0 * vac)) * float(self.rho[m]) \
                  * self._phi1s(R_m, s) * (R_m - R_i) * (R_m - R_j)
        return (c1 / s**2 + c2 / s) / (vac * det_Qs)

    def get_g0ij_contact(self, i: int, j: int) -> float:
        """BMCSL HS contact value g⁰_ij(R_ij⁺).

            g⁰_ij = 1/vac + 3·ζ₂·f/vac² + 2·ζ₂²·f²/vac³

        where ζ₂ = (π/6)·ξ₂ and f = σᵢσⱼ/(σᵢ+σⱼ).
        """
        di    = float(self.sigma[i])
        dj    = float(self.sigma[j])
        zeta2 = (np.pi / 6.0) * float(self.xi[2])
        vac   = self.vac
        f     = di * dj / (di + dj)
        return 1.0 / vac + 3.0 * zeta2 * f / vac**2 + 2.0 * zeta2**2 * f**2 / vac**3

    def get_g1ij_contact(self, i: int, j: int) -> float:
        """First-order RDF contact value g¹_ij(R_ij⁻).

        Evaluated by undoing the 1/(2π√ρρ r) prefactor of get_c1_inner just
        inside contact: g¹ = c^(1)_ij(R⁻) × 2π√(ρᵢρⱼ)·R_ij.
        """
        R  = float(self.R_ij[i, j])
        r  = R * (1.0 - 1e-8)
        return self.get_c1_inner(r, i, j) \
               * (2.0 * np.pi * np.sqrt(float(self.rho[i]) * float(self.rho[j])) * r)

    def get_G1L_ij_shift(self, s: float, i: int, j: int) -> float:
        """Shifted Laplace of r·g¹_ij — returns 0 (second-order outcore not implemented)."""
        return 0.0
