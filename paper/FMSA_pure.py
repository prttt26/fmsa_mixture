"""FMSA single-component (pure-fluid) system parameters.

Used to check FMSA_poly_term_species works correctly at the pure-substance limit.
"""
from __future__ import annotations

import math

import numpy as np

try:                      # numpy >= 2.0 renamed trapz -> trapezoid
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz
import matplotlib.pyplot as plt
from scipy.special import factorial

n_species = 1
n_tails   = 2


class FMSA_pure:
    """System parameters for a single-component LJ fluid under FMSA.

    Handles the conversion stage of FMSA_poly_term_species.__init__ —
    everything up to (but not including) the analytical solution — simplified
    for a one-component system where mole fractions and mixing rules collapse
    to scalars.

    Attributes
    ----------
    sigma   : float, LJ diameter (or BH diameter when input_type='bh').
    epsilon : float, LJ well depth (used for Yukawa amplitude scaling).
    rhostar : float, reduced number density rho* = rho * sigma^3.
    tstar   : float, reduced temperature T* = kT / epsilon.
    rho     : float, number density (= rhostar for a pure fluid, x = 1).
    beta    : float, inverse temperature 1 / tstar.
    cbh     : ndarray (3,), Barker-Henderson fit coefficients.
    n_tails : int, number of Yukawa tails (2).
    k0_fac  : float, bare Yukawa amplitude prefactor.
    z1_fac  : float, inverse-range scale factor for tail 1.
    z2_fac  : float, inverse-range scale factor for tail 2.
    d_bh    : float, Barker-Henderson effective hard-sphere diameter.
    radius  : float, hard-sphere contact radius (= d_bh for a like pair).
    xi      : ndarray (4,), raw diameter moments xi_n = rho * d_bh^n  (NOT the FMT nα directly).
    eta     : float, packing fraction (pi/6) * xi_3.
    vac     : float, free-volume fraction 1 - eta.
    z1, z2  : float, Yukawa inverse-range constants for tail 1 and tail 2.
    k1_bare : float, bare Yukawa amplitude for tail 1.
    k2_bare : float, bare Yukawa amplitude for tail 2 (negative).
    k1, k2  : float, full amplitudes (2 pi rho beta) * k_bare.
    k_bare_matrices : ndarray (n_tails,), stacked bare amplitudes [k1_bare, k2_bare].
    K_matrices      : ndarray (n_tails,), stacked full amplitudes [k1, k2].
    z_matrices      : ndarray (n_tails,), stacked inverse ranges [z1, z2].
    """

    # ------------------------------------------------------------------
    # BH conversion (standalone, reusable)
    # ------------------------------------------------------------------

    @staticmethod
    def bh_diameter(
        sigma:   float,
        epsilon: float,
        beta:    float,
        cbh: np.ndarray | None = None,
    ) -> float:
        """Barker-Henderson effective diameter for a pure fluid.

        d_bh = 2^{1/6} · σ / (1 + √f)^{1/6}
        where f = 1 + cbh[0]/t + cbh[1]/t² + cbh[2]/t⁴,  t = β·ε.

        Parameters
        ----------
        sigma   : LJ diameter
        epsilon : LJ well depth
        beta    : inverse temperature
        cbh     : 3-element coefficient array; default [1.1287, −0.05536, 7.278e-4]

        Returns
        -------
        float : d_bh
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
        t = 1.0 / (beta * epsilon)
        fac = 1.0 + cbh[0]*t + cbh[1]*t**2 + cbh[2]*t**4
        return float(2.0**(1.0/6.0) * sigma / (1.0 + np.sqrt(fac))**(1.0/6.0))

    def __init__(
        self,
        sigma:      float,
        epsilon:    float,
        rhostar:    float,
        tstar:      float,
        input_type: str = 'lj',
    ) -> None:
        """Initialise system parameters and precompute all conversion quantities.

        Mirrors the pre-analytical-solution block of FMSA_poly_term_species.__init__:
        BH diameter, raw diameter moments (xi), packing fraction, Yukawa poles
        and amplitudes. No mixing rules are needed for a pure fluid.

        Parameters
        ----------
        sigma      : LJ diameter ('lj' mode) or BH diameter ('bh' mode).
        epsilon    : LJ well depth (used for Yukawa amplitude scaling in both modes).
        rhostar    : reduced number density rho* = rho * sigma^3.
        tstar      : reduced temperature T* = kT / epsilon.
        input_type : 'lj' (default) — compute BH diameter from LJ sigma/epsilon.
                     'bh'           — sigma is already the BH diameter; skip conversion.
        """
        self.sigma      = float(sigma)
        self.epsilon    = float(epsilon)
        self.rhostar    = float(rhostar)
        self.tstar      = float(tstar)
        self.input_type = input_type

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
        self.cbh    = cbh    = np.array([1.1287, -0.05536, 0.0007278], dtype=float)
        self.n_tails           = n_tails
        self.k0_fac = k0_fac = 2.1714
        self.z1_fac = z1_fac = 2.9637
        self.z2_fac = z2_fac = 14.0167
        self.beta   = beta   = 1.0 / tstar

        # Pure fluid: x = 1, so rho = rhostar
        self.rho = rho = float(rhostar)

        # ------------------------------------------------------------------
        # Barker-Henderson effective diameter
        # ------------------------------------------------------------------
        if input_type == 'lj':
            self.d_bh = d_bh = self.bh_diameter(sigma, epsilon, beta, cbh)
        else:  # 'bh': sigma is already the BH diameter
            self.d_bh = d_bh = float(sigma)

        # For a pure fluid the contact radius equals the BH diameter (like pair only)
        self.radius = d_bh

        # ------------------------------------------------------------------
        # Raw diameter moments  xi_n = rho * d_bh^n  (n = 0..3); FMT nα derived below
        # ------------------------------------------------------------------
        self.xi  = xi = np.array([rho * d_bh**n for n in range(4)], dtype=float)
        self.eta      = float((np.pi / 6.0) * xi[3])
        self.vac      = float(1.0 - self.eta)

        # ------------------------------------------------------------------
        # Yukawa inverse-range constants  z_t = z_fac / sigma
        # ------------------------------------------------------------------
        self.z1 = z1 = z1_fac / sigma
        self.z2 = z2 = z2_fac / sigma

        # ------------------------------------------------------------------
        # Yukawa amplitudes
        #   k0     = k0_fac * sigma * epsilon
        #   k_bare = ±k0 * exp(z * (sigma - d_bh))   (+tail1, -tail2)
        #   K      = 2π rho beta * k_bare
        # ------------------------------------------------------------------
        k0             = k0_fac * sigma * epsilon
        self.k1_bare   = k1_bare =  k0 * np.exp(z1 * (sigma - d_bh))
        self.k2_bare   = k2_bare = -k0 * np.exp(z2 * (sigma - d_bh))

        param        = 2.0 * np.pi * rho * beta
        self.k1      = k1 = param * k1_bare
        self.k2      = k2 = param * k2_bare

        # Stacked arrays shape (n_tails,) — parallel to multi-species K_matrices
        self.k_bare_matrices = np.array([k1_bare, k2_bare], dtype=float)
        self.K_matrices      = np.array([k1,      k2     ], dtype=float)
        self.z_matrices      = np.array([z1,      z2     ], dtype=float)

    def get_S(self, s: float) -> float:
        """Polynomial numerator of Q(s)·Δ²·s³ (the exp-free part) for a pure fluid.

        The full single-component Baxter Q-function decomposes as:

            Q(s) · Δ² · s³  =  S(s)  +  12η · L(s) · exp(−s · d_bh)

        S(s) is the polynomial remainder after factoring out the exp term; it is
        NOT det Q(s).  The actual Baxter Q-element for a pure fluid is computed by
        get_Q0.  The poles of the PY structure factor satisfy
        S(s) = −12η · L(s) · exp(−s · d_bh).

            S(s) = (1−η)² s³ + 6η(1−η) s² + 18η² s − 12η(1+2η)

        Parameters
        ----------
        s : Laplace variable (scalar, in units of 1/d_bh).

        Returns
        -------
        float : S(s)
        """
        eta = self.eta
        vac = self.vac
        return (
            vac**2 * s**3
            + 6.0 * eta * vac * s**2
            + 18.0 * eta**2 * s
            - 12.0 * eta * (1.0 + 2.0 * eta)
        )
        
    def get_L(self, s: float) -> float:
        """Baxter contact polynomial L(s) for the pure fluid.

        When the Baxter q-function is integrated by parts to form the Laplace
        transform, the upper-limit boundary at r = d_bh produces terms of the
        form L(s)·exp(−s·d_bh).  L(s) therefore encodes the contact boundary
        conditions of the q-function:

            L(s) = (1 + η/2) s + (1 + 2η)

        where the constant term (1+2η) is proportional to q(d_bh) — the value
        of the Baxter q-function at hard-sphere contact — and the slope (1+η/2)
        is proportional to q'(d_bh).  At s = 0 the constant term gives the PY
        compressibility directly:

            χ_T^PY = (1 − η)⁴ / L(0)²  =  (1 − η)⁴ / (1 + 2η)²

        In a multi-component system, L generalises to an N×N matrix of contact
        polynomials, one per species pair.

        Parameters
        ----------
        s : Laplace variable (scalar).

        Returns
        -------
        float : L(s)
        """
        eta = self.eta
        return (1.0 + eta / 2.0) * s + 1.0 + 2.0 * eta
    
    def get_Q0(self, s: float) -> float:
        """Baxter Q-function for the pure fluid (scalar analog of get_Q_matrix).

        In the multi-component formulation, the Baxter factor is an N×N matrix
        Qhat(s) whose determinant and inverse drive the MSA poles and propagator
        A_prop.  For a single-component fluid the matrix collapses to the scalar:

            Q0(s) = [S(s) + 12η · L(s) · exp(−s · d_bh)] / [(1−η)² s³]

        where Δ = 1−η = vac.  This is the full transcendental Q-element — the
        denominator of the PY structure factor.  Its zeros are the MSA poles,
        i.e. roots of S(s) = −12η · L(s) · exp(−s · d_bh) (see get_S).
        
        Parameters
        ----------
        s : Laplace variable (same units as 1/d_bh).

        Returns
        -------
        float : Q0(s)
        """
        vac = self.vac
        S = self.get_S(s)
        L = self.get_L(s)
        # d=1-units form: the argument s is DIMENSIONLESS (s̃ = s_physical·d_bh),
        # so the boundary phase is exp(−s̃), not exp(−s·d_bh).
        # [Corrected 2026-07-16: passing physical s with exp(−s·d_bh) mixed
        #  units and broke get_rc1r at d_bh ≠ 1 — see get_rc1r note.]
        return (S + 12.0 * self.eta * L * np.exp(-s)) / (vac**2 * s**3)
    
    def get_rc1r(self, r: float) -> float:
        """First-order correction r·c1(r) inside the hard core (0 < r < d_bh).

        For a 2-tail Yukawa potential the total is a plain sum over tails — no
        cross-pole mixing occurs for a pure single-component fluid in first-order
        perturbation theory.

            r·c1(r) = Σ_t  K_t · f(z_t, r)

        where the single-tail contribution f(z, r) is (formula for d_bh = 1 units;
        see note below):

            f(z, r) = exp(−z(r−d_bh))
                      − 1 / [(1−η)^4 z^6 Q0(z)^2] · (
                            [S(z)^2 · exp(−z(r−d_bh)) + 144η^2 L(z)^2· exp(z(r−d_bh))] 
                          − r^4 · 12η^2 · [(1+2η)^2 z^4 + (1−η)(1+2η) z^5]
                          + r^2 · 12η   · [S(z)L(z) z^2 − (1−η)^2(1+η/2) z^6]
                          − r   · 24η   · [(1+2η)^2 z^4 + (1−η)(1+2η) z^5]
                          + 24η · S(z) L(z)
                      )

        Parameters
        ----------
        r : radial distance, must satisfy 0 < r < d_bh.

        Returns
        -------
        float : r · c1(r)
        """
        d   = self.d_bh
        eta = self.eta
        vac = self.vac

        # ── General-d evaluation [corrected 2026-07-16] ────────────────────
        # The [chsY] Eq. 42/43 formula above is written in d = 1 reduced units.
        # The theory's unified length unit is σ (= σ_min = 1); the LJ→BH mapping
        # gives d_bh slightly below 1, so the formula must be evaluated at the
        # DIMENSIONLESS variables  z̃ = z_t·d,  r̃ = r/d  (η and K_t invariant:
        # K̃ = d²·K and rc̃1r̃ = d²·rc1r cancel exactly):
        #     rc1r(r) = Σ_t K_t · f_{d=1}(z_t·d, r/d)
        # The previous code passed the physical (per-σ) z_t and unscaled r into
        # the d=1 formula, producing a smooth ~7–10% error in the correction
        # term at LJ state points (verified against the exact OZ coupling
        # derivative in probe_verify_first_order.py; exact at input_type='bh').
        result = 0.0
        for K_t, z_t in zip(self.K_matrices, self.z_matrices):
            zt = z_t * d          # dimensionless pole  z̃ = z·d
            rt = r / d            # dimensionless separation  r̃ = r/d
            Q0z = self.get_Q0(zt)
            Sz  = self.get_S(zt)
            Lz  = self.get_L(zt)

            denom    = vac**4 * zt**6 * Q0z**2
            exp_core = np.exp(-zt * (rt - 1.0))

            correction = (
                (Sz**2 ) * exp_core + 144.0 * eta**2 * Lz**2 / exp_core
                - 12.0 * eta**2 * ((1.0 + 2.0*eta)**2 * zt**4
                                   + vac * (1.0 + 2.0*eta) * zt**5) * rt**4
                + 12.0 * eta   * (Sz*Lz*zt**2
                                   - vac**2 * (1.0 + eta/2.0) * zt**6)  * rt**2
                - 24.0 * eta   * ((1.0 + 2.0*eta)**2 * zt**4
                                   + vac * (1.0 + 2.0*eta) * zt**5)     * rt
                + 24.0 * eta   * Sz * Lz
            )

            result += K_t * (exp_core - correction / denom)

        return result

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

    def get_c0_fmtHS_ref(self, r: float) -> float:
        """White Bear FMT hard-sphere DCF c0(r) for the pure fluid, closed form.

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
        Ri = Rj = float(self.d_bh) / 2.0
        rho = np.array([float(self.rho)])
        R   = np.array([Ri])

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

    def get_chat0(self, n_r: int = 500) -> float:
        """Compute ĉ(k=0) = 4π ∫₀^∞ r² c(r) dr for the current state point.

        Splits into three terms to avoid the 1/r singularity in c(r):

          (i)  HS reference inside:       4π ∫₀^d r² c0(r) dr
          (ii) First-order inside:        (2/ρ) ∫₀^d r · [r·c1(r)] dr
          (iii) Outside Yukawa (analytic): (2K_t/ρ) · (d/z_t + 1/z_t²)  summed over tails

        The outside integral is exact:
          4π ∫_d^∞ r² · K·exp(−z(r−d))/(2πρr) dr = (2K/ρ) · (d/z + 1/z²)

        Parameters
        ----------
        n_r : number of radial quadrature points inside [0, d_bh].

        Returns
        -------
        float : ĉ(k=0)
        """
        d   = self.d_bh
        rho = self.rho

        r = np.linspace(0.0, d, n_r + 1)[1:]   # exclude r = 0

        c0_vals  = np.array([self.get_c0_fmtHS_ref(ri) for ri in r])
        rc1_vals = np.array([self.get_rc1r(ri)          for ri in r])

        chat_hs   = 4.0 * np.pi * _trapz(r**2 * c0_vals, r)
        chat_1in  = (2.0 / rho) * _trapz(r * rc1_vals, r)
        chat_1out = sum(
            (2.0 * K / rho) * (d / z + 1.0 / z**2)
            for K, z in zip(self.K_matrices, self.z_matrices)
        )

        return chat_hs + chat_1in + chat_1out

    @staticmethod
    def find_critical_point(
        sigma:      float = 1.0,
        epsilon:    float = 1.0,
        tstar_lo:   float = 1.0,
        tstar_hi:   float = 2.0,
        rho_lo:     float = 0.01,
        rho_hi:     float = 0.70,
        n_rho:      int   = 120,
        tol_tstar:  float = 1e-4,
        n_r:        int   = 500,
    ) -> tuple[float, float]:
        """Estimate the gas-liquid critical point for a pure LJ fluid under FMSA.

        Uses the compressibility spinodal criterion:
          at the spinodal:   1 − ρ·ĉ(0) = 0
          at the critical point: additionally  ∂[1 − ρ·ĉ(0)]/∂ρ = 0

        The algorithm bisects on T* using the condition that
          f_min(T*) ≡ min_ρ [1 − ρ·ĉ(0)] < 0  iff the spinodal exists at T*.
        T*_c is the highest T* for which f_min < 0.  ρ*_c is the argmin of
        1 − ρ·ĉ(0) at T*_c, refined on a finer grid around the coarse argmin.

        Parameters
        ----------
        sigma, epsilon  : LJ diameter and well depth.
        tstar_lo        : lower T* bracket (must have spinodal, i.e. f_min < 0).
        tstar_hi        : upper T* bracket (must have no spinodal, f_min ≥ 0).
        rho_lo, rho_hi  : density search range for the spinodal scan.
        n_rho           : density grid points per T* evaluation.
        tol_tstar       : bisection convergence tolerance on T*.
        n_r             : quadrature points for get_chat0.

        Returns
        -------
        (tstar_c, rhostar_c) : critical temperature and density (reduced units).
        """
        rho_grid = np.linspace(rho_lo, rho_hi, n_rho)

        def spinodal_vals(tstar: float) -> np.ndarray:
            return np.array([
                1.0 - rho * FMSA_pure(sigma, epsilon, rho, tstar).get_chat0(n_r)
                for rho in rho_grid
            ])

        def f_min(tstar: float) -> float:
            return float(spinodal_vals(tstar).min())

        # --- validate bracket ---
        if f_min(tstar_lo) > 0.0:
            raise ValueError(
                f"No spinodal at tstar_lo={tstar_lo:.3f}; lower tstar_lo."
            )
        if f_min(tstar_hi) < 0.0:
            raise ValueError(
                f"Spinodal still present at tstar_hi={tstar_hi:.3f}; raise tstar_hi."
            )

        # --- bisect T* to find Tc ---
        t_lo, t_hi = float(tstar_lo), float(tstar_hi)
        while t_hi - t_lo > tol_tstar:
            t_mid = 0.5 * (t_lo + t_hi)
            if f_min(t_mid) < 0.0:
                t_lo = t_mid
            else:
                t_hi = t_mid

        tstar_c = 0.5 * (t_lo + t_hi)

        # --- refine rhoc: coarse argmin then zoom ---
        vals_c  = spinodal_vals(tstar_c)
        idx_min = int(np.argmin(vals_c))
        rho_lo_r = rho_grid[max(idx_min - 2, 0)]
        rho_hi_r = rho_grid[min(idx_min + 2, n_rho - 1)]

        rho_fine = np.linspace(rho_lo_r, rho_hi_r, 200)
        vals_fine = np.array([
            1.0 - rho * FMSA_pure(sigma, epsilon, rho, tstar_c).get_chat0(n_r)
            for rho in rho_fine
        ])
        rhostar_c = float(rho_fine[np.argmin(vals_fine)])

        return tstar_c, rhostar_c

    def get_c_total(self, r_eval: np.ndarray) -> np.ndarray:
        """Evaluate the total FMSA direct correlation function on a radial grid.

        Total DCF = hard-sphere FMT reference + first-order Yukawa perturbation:

            Inside core  (r < d_bh):
                c(r) = c0_FMT(r)  +  r·c1(r) / (2π·ρ·r)
                where r·c1(r) is returned by get_rc1r.

            Outside core (r ≥ d_bh):
                c(r) = [K1·exp(−z1·(r−d_bh)) + K2·exp(−z2·(r−d_bh))] / (2π·ρ·r)

        Mirrors FMSA_poly_term_species.get_c_total_poly for the single-species case:
        the species-pair loop collapses to a single pair (0,0), λ = 0 so r_c = r.
        This one is the accurate solution for the pure fluid, not a limit of the multi-species code.

        Parameters
        ----------
        r_eval : 1-D array of radial distances.

        Returns
        -------
        c_total : ndarray, shape (len(r_eval),)
        """
        d   = self.d_bh
        rho = self.rho
        pref_common = 2.0 * np.pi * rho

        c_total = np.empty(len(r_eval), dtype=float)
        for idx, r in enumerate(r_eval):
            if r < d:
                c0    = self.get_c0_fmtHS_ref(r)
                rc1r  = self.get_rc1r(r)
                c_total[idx] = c0 + rc1r / (pref_common * r)
            else:
                tail = sum(
                    K_t * np.exp(-z_t * (r - d))
                    for K_t, z_t in zip(self.K_matrices, self.z_matrices)
                )
                c_total[idx] = tail / (pref_common * r)

        return c_total