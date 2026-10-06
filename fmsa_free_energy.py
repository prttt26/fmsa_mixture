"""Free energy calculator for FMSA classes (composition, not inheritance).

Usage
-----
    fe = FMSAFreeEnergy(fmsa_instance)
    f  = fe.get_free_energy()            # LJ shell, frozen contact (default)
    f  = fe.get_free_energy('selfcons')  # self-consistent core, cavity weight
    f  = fe.get_free_energy('both')      # both regions (they are disjoint)

The free energy is split by spatial region rather than perturbation order:

    β·f = β·f_HS  +  β·f_incore  +  β·f_outcore

  β·f_HS      : White-Bear FMT hard-sphere reference (no pair approximation needed)
  β·f_outcore : contribution from r > R_ij  (Yukawa tail, Laplace space)
                  first order  : −2π·β Σ ρᵢρⱼ k̄·G0red(z)
                  second order : −π·β  Σ (ρᵢρⱼ/norm)·k̄·G1L_shift(z)
  β·f_incore  : two DISJOINT regions, historically both called "incore" —

    'lj'        the LJ SHELL between the hard core and σ_ij, i.e. the interval
                with endpoints R_ij and σ_ij.  For an LJ input d_BH < σ always,
                so this is [R_ij, σ_ij] — OUTSIDE the hard core, ~4.5 % wide.
                It corrects the potential (real LJ vs the Yukawa fit + hard
                core), not the closure, and the DCF route has no counterpart to
                it.  `weight='contact'` (DEFAULT) freezes g₀ at its contact
                value — the Tang & Lu recipe, analytic; `weight='cavity'`
                resolves y₀(r) across the shell instead.

    'selfcons'  the GENUINE core [0, R_ij], where g₀ ≡ 0 and the energy route
                therefore sees nothing at all, while the model's own first-order
                DCF c₁ is large and nonzero — which is precisely the region the
                compressibility route does see, through the in-core part of
                ĉ^(1)(0).  Using −c₁/β as the model's own in-core effective
                potential, at BOTH perturbation orders:

                  β·f₁^core = −2π Σ ρᵢρⱼ ∫₀^{R_ij} c₁_ij(r) w_ij(r) r² dr
                  β·f₂^core = +π  Σ ρᵢρⱼ ∫₀^{R_ij} c₁_ij(r)²      r² dr

                `betaf1_incore_selfcons` takes the weight w (four choices, see
                its docstring); `betaf2_incore_selfcons` takes none — the PY
                closure pins the first-order cavity function to y₁ = −c₁ the
                same way it pins y₀, so f₂ carries c₁ twice and nothing about it
                is a choice.  'selfcons' sums them, as 'lj' sums its own two.

The two regions do not overlap, so 'both' is a legitimate combination.  The
DEFAULT stays 'lj' with the frozen contact weight — unchanged, analytic, and bit
-identical to what this module always returned.  Everything else is here to be
COMPARED against it, not to replace it: un-freezing the shell weight is worth
0.1 % in T*_c while costing a de Hoog inversion per quadrature node (2 ms →
82 ms) and making a δ_ρ = 1e-3 second difference of βf_ex unusable, and the
self-consistent core route widens the X.6/X.7 route gap rather than closing it.
See `numerical_notes/results/incore_selfcons.md`.

The fmsa object must expose:
  Attributes:
    rho              shape (N,)         partial densities
    diag_r           shape (N,)         pure-species BH contact diameters
    radii            shape (N, N)       contact-distance matrix R_ij
    sig_mix          shape (N, N)       LJ σ mixing matrix
    eps_mix          shape (N, N)       LJ ε mixing matrix
    k_bare_matrices  shape (n_tails, N, N)  bare Yukawa amplitudes (no 2π√ρρ)
    z_matrices       shape (n_tails, N, N)  Yukawa decay constants z_{t,ij}
    beta             float              1 / T*
    rhostar          float              total number density Σ ρᵢ
    x                shape (N,)         mole fractions
    n_tails          int                number of internal tail slots (inactive = 0 amplitude)
  Methods:
    G0red_laplace_ij(i, j, s)    -> float   reduced Laplace of r·g₀_ij at s
    get_g0ij_contact(i, j)       -> float   g₀_ij(R_ij)
    get_g1ij_contact(i, j)       -> float   g₁_ij(R_ij)
    get_G1L_ij_shift(s, i, j)    -> float   shifted Laplace of r·g₁_ij at s
    _make_at_rho(rho_vec)        -> fmsa    new instance with given densities
  Methods needed ONLY by the routes that leave the frozen contact value —
  currently `FMSA_double_prop` alone; anything else keeps working on
  incore='lj' and gets a NotImplementedError naming the gap if asked for more:
    c1_incore_moment(i, j, n)    -> float   exact ∫₀^{R_ij} rⁿ c^(1)_ij dr
    c1_incore_sq_moment(i, j)    -> float   exact ∫₀^{R_ij} r² [c^(1)_ij]² dr
    get_c1_inner(r, i, j)        -> float   first-order DCF, closed form
    incore_breakpoints(i, j)     -> array   panel edges of c^(1) and c^HS-PY
    get_y0_r(r, i, j)            -> array   HS cavity function y⁰_ij(r)
    get_y1_r(r, i, j)            -> array   g¹_ij(r), exterior only
"""

from __future__ import annotations

import numpy as np


class FMSAFreeEnergy:
    """Free energy computed from any FMSA instance via composition."""

    def __init__(self, fmsa) -> None:
        self._f = fmsa

    # ------------------------------------------------------------------
    # Hard-sphere reference
    # ------------------------------------------------------------------

    def betaf_hs(self) -> float:
        """White-Bear FMT excess free energy density β·F_HS/V.

        Rosenfeld normalisation n₂ = 4π Σ ρ_k R_k² (R_k = diag_r[k]/2).  In the
        bulk the vector densities vanish, so

            β·F_HS/V = −n0 ln(1−n3) + n1 n2 A(n3) + n2³ φ(n3)

        with A and φ from `fmsa_double_prop._fmt_A_phi` — the SAME routine
        `get_HS_FMT` uses for the pair DCF.  One definition, so the free energy
        and the DCF cannot disagree about which White Bear they are; before
        2026-07-31 this was hard-wired to WB1 while `get_HS_FMT` already
        honoured `hs_fmt`.

        For WB1 this is term-for-term the historical closed form
        −n0 ln(1−n3) + n1n2/(1−n3) + (n2³/36π)[ln(1−n3)/n3² + 1/(n3(1−n3)²)],
        since A = 1/(1−n3) and n2³·φ_WB1 expands to exactly that bracket.

        BOTH flavours reduce exactly to ρ(4η−3η²)/(1−η)² (Carnahan-Starling)
        for one component — they differ only for mixtures.

        Models without an `hs_fmt` attribute are treated as 'wb1', so the older
        FMSA classes keep working unchanged.
        """
        from fmsa_double_prop import _fmt_A_phi

        f   = self._f
        R   = f.diag_r / 2.0
        n0  = float(np.sum(f.rho))
        n1  = float(np.sum(f.rho * R))
        n2  = 4.0 * np.pi * float(np.sum(f.rho * R**2))
        n3  = (4.0 / 3.0) * np.pi * float(np.sum(f.rho * R**3))
        (A, _, _), (ph, _, _) = _fmt_A_phi(n3, getattr(f, 'hs_fmt', 'wb1'))
        return -n0 * np.log(1.0 - n3) + n1 * n2 * A + n2**3 * ph

    # ------------------------------------------------------------------
    # Outcore region  r > R_ij  (both perturbation orders, Laplace space)
    # ------------------------------------------------------------------

    def betaf_outcore(self) -> float:
        """Total outer-core free energy density = f1_outer + f2_yk.

        First order  : −2π·β Σ_{ij} ρᵢρⱼ Σ_t k̄_{t,ij} G0red_ij(z_{t,ij})
        Second order : −π·β  Σ_{ij} (ρᵢρⱼ/norm_ij) Σ_t k̄_{t,ij} G1L_shift(z_{t,ij})
        """
        return self._betaf1_outer() + self._betaf2_yk()

    def _betaf1_outer(self) -> float:
        """First-order outcore: −2π·β Σ ρᵢρⱼ k̄ G0red(z)."""
        f   = self._f
        N   = len(f.rho)
        nt  = f.n_tails
        tot = 0.0
        for i in range(N):
            for j in range(N):
                rho_ij = f.rho[i] * f.rho[j]
                for t in range(nt):
                    tot += rho_ij * f.k_bare_matrices[t, i, j] \
                           * f.G0red_laplace_ij(i, j, f.z_matrices[t, i, j])
        return -2.0 * np.pi * f.beta * tot

    def _betaf2_yk(self) -> float:
        """Second-order outcore (Yukawa): −π·β Σ (ρᵢρⱼ/norm) k̄ G1L_shift(z).

        The 1/(2π√(ρᵢρⱼ)) factor cancels the same prefactor baked into G1L_shift,
        so this term scales as ρ² (not ρ³).
        """
        f    = self._f
        N    = len(f.rho)
        nt   = f.n_tails
        term = 0.0
        for i in range(N):
            for j in range(N):
                # ρᵢρⱼ / (2π√(ρᵢρⱼ)) = √(ρᵢρⱼ)/2π.  Written as the product, not
                # the quotient, so a zero-density component gives 0 rather than
                # 0/0 -- the λ-integration path starts at exactly that point.
                param_ij = np.sqrt(f.rho[i] * f.rho[j]) / (2.0 * np.pi)
                for t in range(nt):
                    term += param_ij * f.k_bare_matrices[t, i, j] \
                            * f.get_G1L_ij_shift(f.z_matrices[t, i, j], i, j)
        return -np.pi * f.beta * term

    # ------------------------------------------------------------------
    # Incore region  r < R_ij  — LJ approximation
    # ------------------------------------------------------------------

    def betaf_incore_lj(self, weight: str = 'contact', n_gauss: int = 48,
                        guard: float = 0.015) -> float:
        """LJ shell free energy density = f1_inner + f2_lj, over [σ_ij, R_ij].

        Integrates u_LJ over the shell between the hard core and σ_ij, weighted
        by the reference RDFs.  Note where that shell actually is: for an LJ
        input d_BH < σ always, so R_ij < σ_ij and the interval is [R_ij, σ_ij],
        just OUTSIDE the hard core (4.5 % wide at the reference state), where
        g₀ ≠ 0.  This term corrects the POTENTIAL (real LJ vs Yukawa fit + hard
        core); the DCF route has no counterpart to it, which is why the
        route-comparison scripts run with it switched off.

        weight :
          'contact'  g₀_ij(R_ij) and g¹_ij(R_ij), frozen at contact — the
                     Tang & Lu recipe.  Fully analytic, ~40× faster, and the
                     DEFAULT, so every existing caller keeps its numbers to the
                     last bit.
          'cavity'   y₀_ij(r) and g¹_ij(r) resolved across the shell instead of
                     frozen.  The weight of an in-core/soft-core perturbation
                     term is the cavity function (Barker-Henderson), of which
                     the contact value is the frozen approximation, so this is
                     the same term with one approximation removed.  At the
                     reference state it moves f1_inner by −5.8 % and f2_lj by
                     −26.9 % — g¹ falls by a factor 3.9 across the shell where
                     g₀ falls by 7 %, so freezing the SECOND-order weight was
                     much the worse of the two.  Costs a de Hoog inversion per
                     quadrature node, and its own accuracy is set by `guard`
                     (~2e-3 relative), not by `n_gauss`.
        """
        if weight == 'contact':
            return self._betaf1_inner() + self._betaf2_lj()
        if weight != 'cavity':
            raise ValueError(f"Unknown weight {weight!r}; choose 'contact' or 'cavity'")
        self._require("the LJ shell's r-resolved weight", 'get_y0_r', 'get_y1_r')
        f = self._f
        N = len(f.rho)
        # The second-order shell weight is g¹, and inside the core the
        # first-order condition h¹ ≡ 0 makes its cavity analogue a 0·∞ limit
        # this class does not compute.  For an LJ input d_BH < σ, so the shell
        # is entirely outside the core and the question never arises -- but say
        # so if it ever does, instead of quietly integrating a zero.
        inside = [(i, j) for i in range(N) for j in range(N)
                  if float(f.sig_mix[i, j]) < float(f.radii[i, j])]
        if inside:
            raise NotImplementedError(
                f"pairs {inside} have sigma_ij < R_ij, so the LJ shell reaches "
                "INSIDE the hard core, where the first-order weight g1 has no "
                "cavity continuation in this class (h1 = 0 there by the core "
                "condition). Use weight='contact'.")

        def shell(order: int) -> float:
            tot = 0.0
            for i in range(N):
                for j in range(N):
                    R_ij = float(f.radii[i, j])
                    s_ij = float(f.sig_mix[i, j])
                    wfn  = f.get_y0_r if order == 0 else f.get_y1_r
                    kern = lambda r: (wfn(r, i, j, guard=guard)
                                      * self._u_lj(r, i, j) * r * r)
                    # endpoints in physical order; the shell integral of the
                    # perturbation runs σ_ij → R_ij, so restore that sign after
                    lo, hi = min(s_ij, R_ij), max(s_ij, R_ij)
                    sgn = 1.0 if R_ij > s_ij else -1.0
                    # the weight has a kink where the contact chord meets the
                    # inversion; a panel edge there, or the rule straddles it
                    edge = R_ij * (1.0 + guard)
                    bps = [edge] if lo < edge < hi else []
                    tot += f.rho[i] * f.rho[j] * sgn * self._gauss_panels(
                        kern, lo, hi, bps, n_gauss)
            return tot

        return (2.0 * np.pi * f.beta * shell(0)
                + 1.0 * np.pi * f.beta * shell(1))

    def _u_lj(self, r, i, j):
        """u^LJ_ij(r) = 4ε_ij[(σ_ij/r)¹² − (σ_ij/r)⁶], vectorised in r."""
        f = self._f
        s = f.sig_mix[i, j] / np.asarray(r, dtype=float)
        return 4.0 * f.eps_mix[i, j] * (s**12 - s**6)

    def _betaf1_inner(self) -> float:
        """First-order incore:  −8π·β Σ ρᵢρⱼ εᵢⱼ Rᵢⱼ³ LJ_ij · g₀_ij(R_ij).

        LJ_ij = (1/9)s¹² − (1/3)s⁶ + (2/9)s³,  s = σ_ij / R_ij.

        Derivation (lecture notes §15.4.1–15.4.3, Eqs. 160/168/181/186).
        ⚠ THIS TERM IS A CANCELLATION, NOT A REPLACEMENT.  (166) defines a₁ with
        the integral running from σ outward; (168) re-runs the outer integral from
        R, justified by "g₀ = 0 below R_ij, so the limit may be replaced" — an
        argument valid only for R_ij ≥ σ_ij.  With the BH diameter R < σ it LOWERS
        the limit instead, so the outer term over-extends into (R, σ) where g₀ is
        not zero, and this term subtracts that over-extension back off:

            a₁^inner = 2πβ Σ ρᵢρⱼ g₀_ij(R_ij) ∫_σ^R u^LJ_ij(r) r² dr
                     = 2πβ Σ ρᵢρⱼ g₀_ij(R_ij) · (−4 εᵢⱼ Rᵢⱼ³ LJ_ij)

        using ∫_σ^R u^LJ r² dr = −4εR³·LJ exactly (verified symbolically).  The
        weight is g₀ MULTIPLIED.  ⚠ The minus sign is the INTEGRATION DIRECTION
        (∫_σ^R = −∫_R^σ for R < σ), NOT attraction: on that shell the LJ is
        REPULSIVE, u > 0, and ∫_R^σ u^LJ r² dr = +1.82e−02 at T* = 1.5.
        ⚠ So this term's magnitude is the WHOLE shell's, not a fit error's.  The
        two-Yukawa form reproduces the LJ shell integral to 0.04–0.17 % over
        T* = 0.9–3.0 (cpp/research/cDFT_WDA_weight/data/shell_integrals_2yk_vs_lj.txt);
        dropping this term leaves the over-extension in, which is a different and
        much larger change than correcting that residual.

        [Corrected 2026-07-30.  This returned +8πβ Σ(…)·(g₀ − 1): the wrong sign
        AND only the "contact-correction" half of the weight, with the "uniform"
        (g₀ = 1) half missing entirely.  It came from the notes' Eq. (187)/(190)
        decomposition, which the notes themselves flag as "wrong in sign" and
        then reproduce anyway.  Their Eq. (186) — the assembled result — is the
        correct one and is what this now implements.  See
        pdf/corrections/energy_route.md.

        The check that settles it: a₂ carries π where a₁ carries 2π and weight
        g₁ where a₁ carries g₀, over the SAME integral, so the two incore terms
        must have the same structure.  `_betaf2_lj` was already
        −4πβ Σ(…)·g₁; this is now −8πβ Σ(…)·g₀, and they finally agree.]
        """
        f         = self._f
        N         = len(f.rho)
        param     = 8.0 * np.pi * f.beta
        rad_ratio = f.sig_mix / f.radii
        contact   = 0.0
        for i in range(N):
            for j in range(N):
                R_ij     = f.radii[i, j]
                g0R      = f.get_g0ij_contact(i, j)
                param_ij = f.rho[i] * f.rho[j] * f.eps_mix[i, j] * R_ij**3
                s        = rad_ratio[i, j]
                lj       = s**12 / 9.0 - s**6 / 3.0 + 2.0 * s**3 / 9.0
                contact += param_ij * lj * g0R
        return -param * contact

    def _betaf2_lj(self) -> float:
        """Second-order incore (LJ): −4π·β Σ ρᵢρⱼ εᵢⱼ g₁_ij(R_ij) Rᵢⱼ³ LJ_ij."""
        f    = self._f
        N    = len(f.rho)
        term = 0.0
        for i in range(N):
            for j in range(N):
                R_ij     = f.radii[i, j]
                s        = f.sig_mix[i, j] / R_ij
                param_ij = f.rho[i] * f.rho[j] * f.eps_mix[i, j]
                lj_sum   = R_ij**3 * (s**12 / 9.0 - s**6 / 3.0 + 2.0 * s**3 / 9.0)
                term    += 4.0 * param_ij * f.get_g1ij_contact(i, j) * lj_sum
        return -np.pi * f.beta * term

    # ------------------------------------------------------------------
    # Core region  r < R_ij  — self-consistent (the model's own c₁)
    # ------------------------------------------------------------------

    #: weights accepted by `betaf1_incore_selfcons`
    SELFCONS_WEIGHTS = ('cavity', 'cavity_bmcsl', 'contact', 'unit')

    def _require(self, what: str, *names: str) -> None:
        """Fail with the missing method NAMED, not with an AttributeError."""
        missing = [n for n in names if not hasattr(self._f, n)]
        if missing:
            raise NotImplementedError(
                f"{type(self._f).__name__} does not expose {', '.join(missing)}; "
                f"{what} needs the closed-form in-core c^(1) and cavity function, "
                "which only FMSA_double_prop provides. "
                "Use incore='lj' (the frozen contact weight)."
            )

    def betaf1_incore_selfcons(self, weight: str = 'cavity',
                               n_gauss: int = 48) -> float:
        """FIRST-order self-consistent core free energy density, r ∈ [0, R_ij].

            β·f₁^core = −2π Σ_ij ρᵢρⱼ ∫₀^{R_ij} c₁_ij(r) w_ij(r) r² dr

        The second order is `betaf2_incore_selfcons`, and `incore='selfcons'`
        sums the two — the core route carries both perturbation orders, like the
        shell route it is the counterpart of.

        Rationale.  The first-order term is a₁ = 2πβ Σ ρᵢρⱼ ∫ g₀ u r² dr.  Inside
        the core g₀ ≡ 0, so the energy route collects nothing there — while the
        model's own c₁ is large and nonzero, and enters ĉ^(1)(0) = 4π∫r²c₁dr,
        hence the compressibility route.  That asymmetry is the in-core half of
        the route gap recorded in X.6/X.7.  MSA fixes c₁ = −βu OUTSIDE the core;
        this term takes −c₁/β as the effective interaction INSIDE it as well, so
        that both routes see one in-core interaction.  The minus sign is that
        substitution: βu → −c₁ in 2πβΣρρ∫g₀ u r²dr.

        **That rationale does not survive measurement — this docstring is not a
        recommendation.**  The term does not close the route gap, it widens it.
        On the pure fluid the energy route sits +5.3 % above the DCF route in
        T*_c with no in-core term at all; adding this one takes it to +27.5 %
        ('unit'), +48.2 % ('contact'), +57.7 % ('cavity').  The in-core c^(1) is
        11.5–15.5 % of ĉ^(1)(0) and the compressibility route already carries it
        there, so inserting it a second time as a potential in ∫g·u
        double-counts rather than reconciles.  Use this to quantify the in-core
        asymmetry, not to correct the free energy: `incore='lj'` stays the
        default and `incore='none'` stays the like-for-like route.  Full table:
        `numerical_notes/results/incore_selfcons.md`.

        [The placeholder this replaces carried `+2π…∫c₁ g₀ r²dr`.  Two things
        were wrong with it: the sign (βu = −c₁, not +c₁), and g₀ as the weight —
        g₀ vanishes identically on [0, R_ij], so as written the expression was
        exactly zero.  The weight that belongs in a core integral is the CAVITY
        function y₀, which is finite there and is what the contact-value
        approximation was already freezing.  See `FMSA_double_prop.get_y0_r`.]

        weight :
          'cavity'       y₀_ij(r) = −c^HS-PY_ij(r), the hard-sphere cavity
                         function, exact and closed-form in the core.  DEFAULT.
                         It is the only weight whose r → R_ij limit reproduces
                         the contact-value approximation exactly
                         (y₀(R_ij⁻) = g₀(R_ij⁺)), so it is that approximation
                         un-frozen rather than a different one.
          'cavity_bmcsl' the same shape, rescaled so its contact value is the
                         BMCSL g₀_ij(R_ij) the LJ shell term already uses.  The
                         model is a PY-structure / White-Bear-thermodynamics
                         hybrid, and the two contact values differ by 3–4 %;
                         this option puts that difference on the cavity branch
                         instead of leaving it between the two terms.
          'contact'      y₀ frozen at g₀_ij(R_ij) — the literal contact-value
                         approximation applied over the whole core.  Exact
                         (closed-form moment), and the worst-founded: y₀ varies
                         by ~2.5× across [0, R_ij], which is exactly what the
                         frozen weight cannot see.
          'unit'         w = 1, the combination ĉ^(1)(0) actually contains.
                         Consistent with the compressibility route by
                         construction, but it drops the correlation enhancement
                         entirely (a factor ~2 at the reference state).

        `weight` applies to this first-order term only; the second-order weight
        is not a choice at all — see `betaf2_incore_selfcons`.

        'contact' and 'unit' are evaluated from the closed-form moment
        ∫₀^R r²c₁dr (no quadrature at all); the cavity weights use fixed-order
        Gauss-Legendre on panels cut at the true breakpoints of c₁ and c^HS-PY,
        between which both are a single smooth polynomial × exponential.
        """
        if weight not in self.SELFCONS_WEIGHTS:
            raise ValueError(f"Unknown weight {weight!r}; choose from "
                             f"{self.SELFCONS_WEIGHTS}")
        f = self._f
        N = len(f.rho)

        if weight in ('contact', 'unit'):
            self._require('the self-consistent core route', 'c1_incore_moment')
            tot = 0.0
            for i in range(N):
                for j in range(N):
                    w = 1.0 if weight == 'unit' else f.get_g0ij_contact(i, j)
                    tot += f.rho[i] * f.rho[j] * w * f.c1_incore_moment(i, j, 2)
            return -2.0 * np.pi * tot

        self._require('the self-consistent core route',
                      'get_c1_inner', 'get_y0_r', 'incore_breakpoints')
        tot = 0.0
        for i in range(N):
            for j in range(N):
                R_ij = float(f.radii[i, j])
                scale = 1.0
                if weight == 'cavity_bmcsl':
                    y_R = f.get_y0_r(R_ij * (1.0 - 1e-11), i, j)
                    scale = f.get_g0ij_contact(i, j) / y_R
                integ = self._gauss_panels(
                    lambda r: (np.array([f.get_c1_inner(float(t), i, j) for t in r])
                               * f.get_y0_r(r, i, j) * r * r),
                    0.0, R_ij, f.incore_breakpoints(i, j), n_gauss)
                tot += f.rho[i] * f.rho[j] * scale * integ
        return -2.0 * np.pi * tot

    def betaf2_incore_selfcons(self) -> float:
        """SECOND-order self-consistent core term, r ∈ [0, R_ij].

            β·f₂^core = +π Σ_ij ρᵢρⱼ ∫₀^{R_ij} c₁_ij(r)² r² dr

        a₂ = πβ Σ ρᵢρⱼ ∫ g₁ u r²dr with βu → −c₁ and g₁ → y₁, so the question is
        what y₁ is inside the core.  Naively substituting g₁ itself gives ZERO,
        because the first-order core condition is h¹ ≡ 0 for r < R_ij — the same
        trap that made the original f₁ placeholder identically zero, one order
        up.  The PY closure pins y₁ exactly as it pins y₀: expanding c = f·y to
        first order gives c₁ = f·y₁ + f₁·y₀, and inside a hard core f = −1 while
        f₁ = 0 (the perturbation does not move the core), so

            y₁_ij(r) = −c₁_ij(r),     r < R_ij

        exactly as y₀ = −c^HS-PY.  So f₂ carries c₁ TWICE — once as the effective
        potential, once as the weight — which is why it is a squared moment, why
        the two minus signs cancel to leave it POSITIVE where f₁ is negative
        (`_betaf2_lj` opposes `_betaf1_inner` on the shell the same way), and why
        it takes no `weight` argument: nothing here is a choice.

        Closed form, no quadrature.  It is one power of density above f₁ (ρ⁴ vs
        ρ³, measured slope 3.9993) since it carries c₁ twice, and small next to
        it: −7.0 % (binary) / −8.5 % (pure), where the same ratio on the LJ shell
        is −35 % / −43 %.
        """
        self._require('the self-consistent core route', 'c1_incore_sq_moment')
        f = self._f
        N = len(f.rho)
        tot = 0.0
        for i in range(N):
            for j in range(N):
                tot += f.rho[i] * f.rho[j] * f.c1_incore_sq_moment(i, j)
        return np.pi * tot

    @staticmethod
    def _gauss_panels(fn, a: float, b: float, breakpoints, n: int) -> float:
        """∫_a^b fn(r) dr by n-point Gauss-Legendre on every smooth panel.

        `breakpoints` are the interior kinks.  Straddling one with a single rule
        destroys the convergence rate; cutting the interval at each of them
        leaves a single analytic piece per panel, where Gauss-Legendre converges
        geometrically.  `fn` takes and returns an array of nodes.
        """
        edges = np.concatenate(([a], np.asarray(breakpoints, dtype=float), [b]))
        xg, wg = np.polynomial.legendre.leggauss(n)
        total = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            if hi <= lo:
                continue
            mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
            total += half * float(np.dot(wg, fn(mid + half * xg)))
        return total

    # ------------------------------------------------------------------
    # Total free energy
    # ------------------------------------------------------------------

    def betaf_incore(self, incore: str = 'lj') -> float:
        """The in-core/shell contribution named by an ``incore`` spec string.

        Grammar — ``term[+term...]`` with ``term = name[:weight]``:

            'none'                        no in-core term at all; β·f_HS + β·f_outcore,
                                          the like-for-like route the X.6–X.12
                                          comparisons use
            'lj'                          LJ shell, frozen contact weight (DEFAULT)
            'lj:cavity'                   LJ shell, y₀(r) resolved — a comparison
                                          column, not a replacement
            'selfcons'                    core [0,R_ij], cavity weight
            'selfcons:unit'               …and the other weights of
            'selfcons:contact'               `betaf1_incore_selfcons`
            'selfcons:cavity_bmcsl'
            'lj+selfcons'  ( = 'both')    both regions — legitimate, they are
                                          disjoint: [R_ij,σ_ij] and [0,R_ij]

        Everything downstream (`get_pressure`, `get_chemical_potentials`,
        `get_excess_free_energy`, …) threads this one string through unchanged,
        so a new weight needs no new signature anywhere.
        """
        if incore == 'none':
            return 0.0
        total = 0.0
        for term in str(incore).replace('both', 'lj+selfcons').split('+'):
            name, _, weight = term.strip().partition(':')
            if name == 'lj':
                total += self.betaf_incore_lj(weight or 'contact')
            elif name == 'selfcons':
                total += (self.betaf1_incore_selfcons(weight or 'cavity')
                          + self.betaf2_incore_selfcons())
            else:
                raise ValueError(
                    f"Unknown incore term {term!r} in {incore!r}; terms are "
                    "'lj', 'selfcons' (optionally ':weight'), joined by '+', "
                    "or the whole spec is 'none'")
        return total

    def get_free_energy(self, incore: str = 'lj') -> float:
        """Total FMSA excess free energy density β·f = β·f_HS + β·f_incore + β·f_outcore.

        Parameters
        ----------
        incore : spec string parsed by `betaf_incore`; 'lj' (default) is the
                 unchanged historical behaviour.

        Returns
        -------
        f : β·F/V
        """
        return self.betaf_hs() + self.betaf_incore(incore) + self.betaf_outcore()

    # ------------------------------------------------------------------
    # Thermodynamics
    # ------------------------------------------------------------------

    def get_total_free_energy_density(self, incore: str = 'lj') -> float:
        """β·Φ = Σ ρᵢ(ln ρᵢ − 1) + β·f_ex  (λᵢ = 1 reduced-unit convention).

        Empty components are skipped rather than evaluated: ρ ln ρ → 0 as
        ρ → 0, but as written it is 0·(−∞) = nan, which then propagated
        silently through `get_pressure` and the total chemical potentials.
        Every pure reference state of `get_excess_free_energy` has N−1 empty
        components, so this is reached routinely, not only at a contrived
        state point.
        """
        f   = self._f
        rho = np.asarray(f.rho, dtype=float)
        nz  = rho > 0.0
        betaf_id = float(np.sum(rho[nz] * (np.log(rho[nz]) - 1.0)))
        return betaf_id + self.get_free_energy(incore=incore)

    # ``delta`` is a RELATIVE step throughout: the actual displacement is
    # delta·ρ*, not delta.  An absolute step is correctly scaled at exactly one
    # density and wrong everywhere else — at ρ* = 0.01 a fixed 1e-4 is already
    # 1% of the state, and below ρ* ≈ 1e-4 it steps straight through zero into
    # negative density.  Scaling with ρ* also puts the step near the optimum
    # h ≈ ρ*·(3ε_mach)^{1/3} ≈ 1e-5·ρ*, since β·f_ex varies on the scale of ρ*
    # in every component direction.

    def _step(self, delta: float) -> float:
        """Absolute finite-difference displacement for a relative ``delta``."""
        return delta * self._f.rhostar

    def get_pressure(self, delta: float = 1e-4, incore: str = 'lj') -> float:
        """β·P = ρ* · ∂(β·Φ)/∂ρ* − β·Φ  via central finite difference.

        ``delta`` is relative — the step is delta·ρ*.
        """
        f       = self._f
        h       = self._step(delta)
        sys_p   = FMSAFreeEnergy(f._make_at_rho(f.x * (f.rhostar + h)))
        sys_m   = FMSAFreeEnergy(f._make_at_rho(f.x * (f.rhostar - h)))
        dbetaf  = (sys_p.get_total_free_energy_density(incore)
                   - sys_m.get_total_free_energy_density(incore)) / (2.0 * h)
        return f.rhostar * dbetaf - self.get_total_free_energy_density(incore)

    def get_chemical_potentials(self, delta: float = 1e-4, incore: str = 'lj') -> np.ndarray:
        """β·μᵢ = ∂(β·f_ex)/∂ρᵢ by finite difference (excess part only).

        ``delta`` is relative — the step is h = delta·ρ*.

        Central where it can be, one-sided at the boundary.  A central
        difference on a component with ρᵢ < h steps to a NEGATIVE density,
        which the model cannot represent — `_P_kernel` takes √(ρᵢρ_m) and
        raises `ValueError: math domain error`.  That is not a corner case:
        ρᵢ = 0 is where `get_excess_free_energy` evaluates.  The replacement is
        the second-order forward formula (−3f₀ + 4f₁ − f₂)/(2h), so the
        boundary points keep the O(h²) accuracy of the interior ones.
        """
        f   = self._f
        N   = len(f.rho)
        rho = np.asarray(f.rho, dtype=float)
        h   = self._step(delta)
        mu  = np.zeros(N, dtype=float)

        def fe_at(rho_vec: np.ndarray) -> float:
            return FMSAFreeEnergy(f._make_at_rho(rho_vec)).get_free_energy(incore)

        for i in range(N):
            if rho[i] - h >= 0.0:
                rho_p = rho.copy(); rho_p[i] += h
                rho_m = rho.copy(); rho_m[i] -= h
                mu[i] = (fe_at(rho_p) - fe_at(rho_m)) / (2.0 * h)
            else:
                rho_1 = rho.copy(); rho_1[i] += h
                rho_2 = rho.copy(); rho_2[i] += 2.0 * h
                mu[i] = (-3.0 * fe_at(rho) + 4.0 * fe_at(rho_1)
                         - fe_at(rho_2)) / (2.0 * h)
        return mu

    def get_chemical_potentials_total(self, delta: float = 1e-4, incore: str = 'lj') -> np.ndarray:
        """β·μᵢ = ∂(β·Φ_total)/∂ρᵢ = ln ρᵢ + ∂(β·f_ex)/∂ρᵢ.

        The ideal part is differentiated ANALYTICALLY:
        ∂/∂ρᵢ Σⱼ ρⱼ(ln ρⱼ − 1) = ln ρᵢ. Differencing it numerically, as this
        did, added ~1e-7 of truncation error to a term known in closed form,
        and could not produce the right answer at ρᵢ = 0 — where βμᵢ is
        genuinely −∞, not a large finite number.
        """
        rho = np.asarray(self._f.rho, dtype=float)
        with np.errstate(divide='ignore'):
            return self.get_chemical_potentials(delta, incore) + np.log(rho)

    def get_excess_free_energy(self, incore: str = 'lj') -> float:
        """Excess (mixing) free energy per particle β·a_ex.

            β·a_ex = (β·f_mix − Σᵢ β·f_{i,pure}(ρᵢ)) / ρ*

        Reference state: each species at its partial density ρᵢ = xᵢ ρ*.
        The ideal-gas contribution cancels exactly in the difference.
        """
        f          = self._f
        N          = len(f.rho)
        betaf_mix  = self.get_free_energy(incore)
        betaf_pure = 0.0
        for i in range(N):
            rho_i = f.rho[i]
            if rho_i <= 0.0:
                continue
            rho_vec    = np.zeros(N, dtype=float)
            rho_vec[i] = rho_i
            betaf_pure += FMSAFreeEnergy(f._make_at_rho(rho_vec)).get_free_energy(incore)
        return (betaf_mix - betaf_pure) / f.rhostar
