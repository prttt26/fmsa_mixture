"""oz_solver.py -- numerical Ornstein-Zernike solver for Yukawa mixtures.

Usage:
    from oz_yukawa_mixture import OZYukawaMixture
    from oz_solver import OZSolver
    model = OZYukawaMixture(sig_pure, eps_pure, x, rhostar, tstar)
    solver = OZSolver(model, n_r=4096, r_max_factor=15.0)
    solver.solve(tol=1e-8, max_iter=500, alpha=0.2, method='diis', closure='msa')
    r, c, h = solver.r, solver._c_converged, solver._h_converged

Supported methods: 'picard', 'ng', 'diis', 'fire', 'gmres' (MSA only).
Supported closures: 'msa', 'py', 'hnc', 'hncb' (HNC + hard-sphere bridge, i.e. the
reference-HNC / Lado prescription).

This is the solver of `oz_numerical.OZSolverMSA` (archived) with everything that served the
old first-order FMSA class removed: it no longer writes `_polycorr` back into the model after
converging (a post-processing step that does not touch the converged c(r)), so it works on any
model object that supplies R_ij, N, rho, beta, n_tails, z_poles, K and get_HS_FMT
(`OZYukawaMixture`). The iteration, the closures and the coupling continuation are unchanged:
on the same model it reproduces the archived solver bit for bit (iterates, converged c and h,
iteration counts).
"""

import numpy as np

try:                      # numpy >= 2.0 renamed trapz -> trapezoid
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz
from scipy.fft import dst


class OZSolver:
    """Ornstein-Zernike + closure iteration for multi-species Yukawa mixtures.

    After `solve()`, `r`, `_c_converged`, `_h_converged` and `_n_iters` hold the radial grid,
    the converged direct and total correlation functions and the iteration count.
    """

    def __init__(self, fmsa, n_r: int = 4096, r_max_factor: float = 15.0):
        self.fmsa = fmsa
        N = fmsa.N
        R_max = float(fmsa.R_ij.max())
        self.r_max = r_max_factor * R_max
        self.n_r = n_r
        self.dr = self.r_max / n_r
        self.r = (np.arange(n_r) + 0.5) * self.dr
        self.dk = np.pi / self.r_max
        self.k = (np.arange(n_r) + 0.5) * self.dk
        self._c_converged = None
        self._k_scale = 1.0   # coupling scale for continuation (1.0 = full Yukawa)
        self._n_iters: int | None = None

    # ------------------------------------------------------------------
    # Hankel (3D isotropic Fourier) transforms via DST-I
    # DST-I is its own inverse up to the factor 2(n_r+1), so both
    # _r2k and _k2r use the same kernel; prefactors differ by 4π² / (2π)³.
    # ------------------------------------------------------------------

    def _r2k(self, f_r: np.ndarray) -> np.ndarray:
        """f(r) → f̂(k): 3D isotropic FT via DST-IV for cell-centred grid.

        For cell-centred r_n=(n+½)dr, k_m=(m+½)dk with dk·dr=π/N, the quadrature
        sin(k_m·r_n)=sin(π(n+½)(m+½)/N) matches DST-IV exactly (scipy includes
        a factor-2 in the definition, so the sum is halved):
            f̂(k_m) = 4π/k_m · dr · Σ_n f_n r_n sin(k_m r_n)
                    = 2π·dr/k_m · scipy_DST4(r·f)_m
        """
        raw = dst(self.r * f_r, type=4)
        return 2.0 * np.pi * self.dr / self.k * raw

    def _k2r(self, f_k: np.ndarray) -> np.ndarray:
        """f̂(k) → f(r): inverse 3D isotropic FT via DST-IV for cell-centred grid.

        Inverse of _r2k:  f(r_n) = dk/(4π²r_n) · scipy_DST4(k·f̂)_n
        Round-trip: N·dk·dr/π = 1 exactly since dk·dr = π/N.
        """
        raw = dst(self.k * f_k, type=4)
        return self.dk / (4.0 * np.pi**2 * self.r) * raw

    # ------------------------------------------------------------------
    # Initial DCF guess
    # ------------------------------------------------------------------

    def _init_c(self) -> np.ndarray:
        """shape (N, N, n_r): HS FMT + FMSA c^(1) inside core; MSA Yukawa outside."""
        N = self.fmsa.N
        c = np.zeros((N, N, self.n_r))
        for i in range(N):
            for j in range(i, N):
                R = float(self.fmsa.R_ij[i, j])
                for idx, r in enumerate(self.r):
                    if r < R:
                        c[i, j, idx] = (self.fmsa.get_HS_FMT(r, i, j)
                                        + self.fmsa.get_c1_inner(r, i, j))
                    else:
                        c[i, j, idx] = self._c_msa_outer(r, i, j)
                c[j, i] = c[i, j]
        return c

    # ------------------------------------------------------------------
    # Yukawa potential and MSA outer tail
    def _precompute_c_hs(self) -> np.ndarray:
        """c^HS on the full grid (0 outside hard core) for perturbation iterate mode."""
        N = self.fmsa.N
        c_hs = np.zeros((N, N, self.n_r))
        for i in range(N):
            for j in range(i, N):
                R = float(self.fmsa.R_ij[i, j])
                for idx, r in enumerate(self.r):
                    if r < R:
                        c_hs[i, j, idx] = self.fmsa.get_HS_FMT(r, i, j)
                c_hs[j, i] = c_hs[i, j]
        return c_hs

    # ------------------------------------------------------------------

    def _init_c_hs_only(self) -> np.ndarray:
        """Initial c with HS FMT inner and zero outer (no Yukawa tail).

        Used as the step-0 starting point in coupling continuation so the
        first iteration begins from the pure HS reference, avoiding any
        FMSA artefacts in the cross-pair DCF.
        """
        N = self.fmsa.N
        c = np.zeros((N, N, self.n_r))
        for i in range(N):
            for j in range(i, N):
                R = float(self.fmsa.R_ij[i, j])
                for idx, r in enumerate(self.r):
                    if r < R:
                        c[i, j, idx] = self.fmsa.get_HS_FMT(r, i, j)
                c[j, i] = c[i, j]
        return c

    def init_c_from_model(self, model) -> np.ndarray:
        """Sample any duck-typed FMSA model's c(r) onto the OZ radial grid.

        Parameters
        ----------
        model : object with R_ij, rho, n_tails, z_poles, K, get_HS_FMT, get_c1_inner.
                Works with FMSA_GA_matrix_mix directly or FMSA_poly_adapter-wrapped poly.

        Returns
        -------
        c : shape (N, N, n_r)
        """
        N = self.fmsa.N
        c = np.zeros((N, N, self.n_r))
        for i in range(N):
            for j in range(i, N):
                R   = float(model.R_ij[i, j])
                rij = float(np.sqrt(model.rho[i] * model.rho[j]))
                for idx, r in enumerate(self.r):
                    if r < R:
                        c[i, j, idx] = (model.get_HS_FMT(r, i, j)
                                        + model.get_c1_inner(r, i, j))
                    else:
                        val = 0.0
                        for t in range(model.n_tails):
                            z = float(model.z_poles[t, i, j])
                            K = float(model.K[t, i, j])
                            if z > 0.0:
                                val += K * np.exp(-z * (r - R))
                        c[i, j, idx] = val / (2.0 * np.pi * rij * r)
                c[j, i] = c[i, j]
        return c

    def _u_yukawa(self, r: float, i: int, j: int) -> float:
        """u_ij(r): Yukawa potential, consistent with K = 2π√ρρ·β·k_bare.

        Returns u such that -β·u = K/(2π√ρρ·r)·exp(-z·(r-R)), i.e.
        u = -K/(2π√ρρ·β·r)·exp(-z·(r-R)) = -k_bare/r·exp(-z·(r-R)).
        k_bare = epsilon * R_ij (epsilon = k_bare/R_ij from fmsa attribute).
        """
        val = 0.0
        R = float(self.fmsa.R_ij[i, j])
        beta = self.fmsa.beta
        pf = 1.0 / (2.0 * np.pi * np.sqrt(self.fmsa.rho[i] * self.fmsa.rho[j]))
        for t in range(self.fmsa.n_tails):
            z = float(self.fmsa.z_poles[t, i, j])
            K = float(self.fmsa.K[t, i, j])
            val += -K / (beta * r) * np.exp(-z * (r - R))
        return val * pf

    def _c_msa_outer(self, r: float, i: int, j: int) -> float:
        """MSA outer tail: c_{ij}(r) = K_t/(2π√ρρ·r)·exp(-z·(r-R)) for r > R_ij.

        Uses K directly — same convention as get_c1_inner — to avoid sign/scale
        ambiguity in the epsilon attribute.
        """
        R = float(self.fmsa.R_ij[i, j])
        pf = 1.0 / (2.0 * np.pi * np.sqrt(self.fmsa.rho[i] * self.fmsa.rho[j]) * r)
        val = 0.0
        for t in range(self.fmsa.n_tails):
            z = float(self.fmsa.z_poles[t, i, j])
            K = float(self.fmsa.K[t, i, j])
            val += K * self._k_scale * np.exp(-z * (r - R))
        return val * pf

    # ------------------------------------------------------------------
    # OZ step in k-space
    # ------------------------------------------------------------------

    def _oz_step(self, c: np.ndarray) -> np.ndarray:
        """c (N,N,n_r) → h (N,N,n_r) via OZ matrix equation: H = (I - C·ρ)⁻¹ · C."""
        N = self.fmsa.N
        rho = self.fmsa.rho
        C_hat = np.zeros((N, N, self.n_r))
        for i in range(N):
            for j in range(N):
                C_hat[i, j] = self._r2k(c[i, j])

        H_hat = np.zeros_like(C_hat)
        for ki in range(self.n_r):
            Ck = C_hat[:, :, ki]
            H_hat[:, :, ki] = np.linalg.solve(np.eye(N) - Ck @ np.diag(rho), Ck)

        h = np.zeros_like(c)
        for i in range(N):
            for j in range(N):
                h[i, j] = self._k2r(H_hat[i, j])
        return h

    # ------------------------------------------------------------------
    # Closure update
    # ------------------------------------------------------------------

    def _compute_hs_bridge(self, source: str = 'fmt',
                           tol: float = 1e-8, max_iter: int = 400) -> None:
        """Precompute the hard-sphere bridge function B_HS(r) for the 'hncb' closure.

        Reference-HNC (Lado) prescription: take the bridge from a hard-sphere
        reference at the *same* diameters and packing, and use it to correct HNC
        for the full (HS + Yukawa) system.

        Here the reference is solved on the same grid with the Yukawa coupling
        switched off (`_k_scale = 0`) under the PY closure, which for hard
        spheres gives c_HS(r) = 0 outside the core and h_HS = γ_HS there.  The
        exact bridge relation g = exp(−βu + γ + B) then gives, for r > R_ij
        (where u_HS = 0),

            B_HS(r) = ln[1 + γ_HS(r)] − γ_HS(r),      γ_HS = h_HS − c_HS

        so the reference bridge used here is the **PY hard-sphere bridge**.  This
        is well defined per pair for mixtures and needs no external
        parametrisation; substituting a simulation-accurate HS bridge
        (Verlet–Weis / Malijevsky–Labik) would be the natural upgrade and is NOT
        done here.

        `source` selects the hard-sphere reference:
          'fmt' (default) — take c_HS(r) directly from the verified White Bear
              FMT closed form (exact for the Carnahan–Starling compressibility)
              and get h_HS from ONE OZ inversion.  No iteration: given c, the OZ
              equation H = (I − Cρ)⁻¹C is linear.
          'py'  — solve the HS system self-consistently under the PY closure,
              i.e. use the (much cruder) PY hard-sphere bridge.

        Sets `self._bridge_hs`, shape (N, N, n_r); zero inside the cores (the
        closure is only applied outside).
        """
        N = self.fmsa.N

        if source == 'fmt':
            # c_HS: White Bear FMT inside the core, exactly 0 outside.
            c_hs = np.zeros((N, N, self.n_r))
            for i in range(N):
                for j in range(N):
                    R = float(self.fmsa.R_ij[i, j])
                    sel = self.r < R
                    c_hs[i, j, sel] = [self.fmsa.get_HS_FMT(float(rv), i, j)
                                       for rv in self.r[sel]]
            h_hs = self._oz_step(c_hs)          # linear in c — one shot
            gamma = h_hs - c_hs
            B = np.zeros_like(gamma)
            for i in range(N):
                for j in range(N):
                    outer = self.r >= float(self.fmsa.R_ij[i, j])
                    g = 1.0 + gamma[i, j, outer]
                    B[i, j, outer] = (np.log(np.maximum(g, 1e-12))
                                      - gamma[i, j, outer])
            self._bridge_hs = B
            return

        k_save, c_save, h_save = self._k_scale, self._c_converged, getattr(self, "_h_converged", None)

        self._k_scale = 0.0                       # pure hard spheres
        c0 = self._init_c_hs_only()
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ok = self._run_iteration(c0, tol, max_iter, 0.2, 'diis', 'py', 5,
                                     'perturbation')
        if not ok or self._c_converged is None:
            self._k_scale, self._c_converged, self._h_converged = k_save, c_save, h_save
            raise RuntimeError("HS reference for the 'hncb' bridge did not converge")

        c_hs, h_hs = self._c_converged, self._h_converged
        gamma = h_hs - c_hs
        B = np.zeros_like(gamma)
        for i in range(N):
            for j in range(N):
                outer = self.r >= float(self.fmsa.R_ij[i, j])
                g = 1.0 + gamma[i, j, outer]
                B[i, j, outer] = np.log(np.maximum(g, 1e-12)) - gamma[i, j, outer]
        self._bridge_hs = B

        self._k_scale, self._c_converged, self._h_converged = k_save, c_save, h_save

    def _closure_update(self, c: np.ndarray, h: np.ndarray, closure: str) -> np.ndarray:
        """Apply hard-core + outer closure to produce new c.

        Inside hard core (r < R_ij): standard Picard update for the g=0 condition.
          h = -1 at convergence → c_new = c_old - h - 1  (= -1 - γ where γ = h - c_old).
          Using c^HS directly here instead would anchor the solution to c^HS, preventing the
          Yukawa correction from modifying the inner DCF.

        Outside hard core: MSA → c_new = -βu (fixed closure, independent of h).
                           HNC → c_new = exp(-βu + γ) - 1 - γ  (uses γ = h - c_old).
                           PY  → c_new = (exp(-βu) - 1) * (1 + γ)  (linear in γ; more stable).
                           HNC: arg clipped to [-50, 50], c_new clipped to [-50, 50].
                           Prevents DIIS Gram matrix overflow (resid^2 * n_r must fit float64).
        """
        N = self.fmsa.N
        c_new = np.zeros_like(h)
        beta = self.fmsa.beta
        for i in range(N):
            for j in range(i, N):
                R  = float(self.fmsa.R_ij[i, j])
                ri = self.fmsa.rho[i];  rj = self.fmsa.rho[j]
                pf = 1.0 / (2.0 * np.pi * np.sqrt(ri * rj))

                inner = self.r < R
                outer = ~inner
                r_out = self.r[outer]

                # Inner: Picard update for g=0 condition
                c_new[i, j, inner] = c[i, j, inner] - h[i, j, inner] - 1.0

                # Outer: vectorised over r_out
                tail = np.zeros_like(r_out)
                for t in range(self.fmsa.n_tails):
                    z = float(self.fmsa.z_poles[t, i, j])
                    K = float(self.fmsa.K[t, i, j])
                    tail += K * self._k_scale * np.exp(-z * (r_out - R))

                if closure == 'msa':
                    c_new[i, j, outer] = tail * pf / r_out
                elif closure == 'py':
                    # PY: c_new = (exp(-βu) - 1) * (1 + γ), γ = h - c_old
                    # Grows linearly in γ (bounded near spinodal, unlike HNC exp(γ)).
                    neg_beta_u = tail * pf / r_out
                    gamma = h[i, j, outer] - c[i, j, outer]
                    c_new[i, j, outer] = (np.exp(neg_beta_u) - 1.0) * (1.0 + gamma)
                elif closure == 'hncb':
                    # HNC + hard-sphere bridge (reference-HNC / Lado):
                    #   c_new = exp(-βu + γ + B_HS) - 1 - γ
                    # B_HS is the bridge of the pure-HS reference at the SAME
                    # packing, precomputed in _compute_hs_bridge().  Reduces to
                    # the HS reference exactly when the Yukawa tail is switched
                    # off, and to plain HNC if B_HS ≡ 0.
                    neg_beta_u = tail * pf / r_out
                    gamma = h[i, j, outer] - c[i, j, outer]
                    B = self._bridge_hs[i, j, outer]
                    arg = np.clip(neg_beta_u + gamma + B, -50.0, 50.0)
                    c_new[i, j, outer] = np.clip(np.exp(arg) - 1.0 - gamma,
                                                 -50.0, 50.0)
                else:  # HNC: c_new = exp(-βu + γ) - 1 - γ, γ = h - c_old
                    # -βu(r) = K/(2π√ρρ·r)·exp(-z(r-R)) = tail*pf/r (same as MSA outer)
                    neg_beta_u = tail * pf / r_out
                    gamma = h[i, j, outer] - c[i, j, outer]
                    arg = np.clip(neg_beta_u + gamma, -50.0, 50.0)
                    c_out = np.exp(arg) - 1.0 - gamma
                    # Clip c_new to ±50: physical HNC outer c is O(1), but DIIS
                    # extrapolation can push c to large values in early iterations.
                    # Without this clip the residual grows to ~exp(500)~5e217,
                    # making B[p,q] = dot(r,r) overflow float64 on the next DIIS call.
                    c_new[i, j, outer] = np.clip(c_out, -50.0, 50.0)

                c_new[j, i] = c_new[i, j]
        return c_new

    # ------------------------------------------------------------------
    # Convergence accelerators
    # ------------------------------------------------------------------

    def _ng_step(self, c: np.ndarray, resid: np.ndarray, history: list) -> np.ndarray:
        """Ng (1974) 3-term vector extrapolation."""
        history.append((c.copy(), resid.copy()))
        if len(history) < 3:
            return c + 0.2 * resid
        (c0, r0), (c1, r1), (c2, r2) = history[-3], history[-2], history[-1]
        d1 = r2 - r1
        d2 = r2 - r0
        A = np.array([
            [np.dot(d1.ravel(), d1.ravel()), np.dot(d1.ravel(), d2.ravel())],
            [np.dot(d1.ravel(), d2.ravel()), np.dot(d2.ravel(), d2.ravel())],
        ])
        b = np.array([np.dot(r2.ravel(), d1.ravel()), np.dot(r2.ravel(), d2.ravel())])
        try:
            ab = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return c + 0.2 * resid
        return c2 + r2 - ab[0] * d1 - ab[1] * d2

    def _diis_step(self, c: np.ndarray, resid: np.ndarray,
                   history: list, m: int = 5, beta: float = 1.0) -> np.ndarray:
        """DIIS (Pulay) m-vector extrapolation.

        beta < 1 gives Anderson mixing: new x = Σ c_k * (x_k + beta*r_k) instead of
        x_k + r_k. Damps DIIS overshoot for nonlinear closures (HNC) while keeping
        Gram-matrix acceleration from history. beta=1 recovers standard Pulay DIIS.
        """
        history.append((c.copy(), resid.copy()))
        if len(history) > m:
            history.pop(0)
        n = len(history)
        if n < 2:
            return c + min(beta, 0.2) * resid  # safe Picard fallback (0.2) when history short
        B = np.zeros((n + 1, n + 1))
        for p in range(n):
            for q in range(n):
                B[p, q] = np.dot(history[p][1].ravel(), history[q][1].ravel())
        B[n, :n] = B[:n, n] = -1.0
        rhs = np.zeros(n + 1)
        rhs[n] = -1.0
        try:
            coeffs = np.linalg.solve(B, rhs)[:n]
        except np.linalg.LinAlgError:
            return c + min(beta, 0.2) * resid
        return sum(coeffs[k] * (history[k][0] + beta * history[k][1]) for k in range(n))

    # ------------------------------------------------------------------
    # GMRES linear solver (convergence guarantee for MSA)
    # ------------------------------------------------------------------

    def _solve_gmres(self, tol: float = 1e-8, iterate: str = 'full') -> bool:
        """Solve MSA OZ as a linear system via GMRES (convergence guarantee).

        MSA pins c(r > R_ij) analytically, leaving only c_in = c[r < R_ij] unknown.
        OZ is linear in c, so  A·c_in = b  is a true linear system:
            A·v = OZ(v_full)[core]          inner perturbation → inner h
            b   = −1 − OZ(c_ref)[core]      c_ref = fixed outer (+ HS in perturbation mode)
        GMRES converges in ≤ n_unknowns steps for any non-singular A.
        """
        from scipy.sparse.linalg import gmres, LinearOperator

        N, n_r = self.fmsa.N, self.n_r

        pair_list = [(i, j) for i in range(N) for j in range(i, N)]
        masks = {(i, j): self.r < float(self.fmsa.R_ij[i, j]) for i, j in pair_list}
        sizes = [int(masks[p].sum()) for p in pair_list]
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        total = int(offsets[-1])

        def pack(c_full: np.ndarray) -> np.ndarray:
            return np.concatenate([c_full[i, j, masks[(i, j)]] for i, j in pair_list])

        def unpack_inner(x: np.ndarray) -> np.ndarray:
            c = np.zeros((N, N, n_r))
            for k, (i, j) in enumerate(pair_list):
                c[i, j, masks[(i, j)]] = x[offsets[k]:offsets[k + 1]]
                c[j, i] = c[i, j]
            return c

        # Fixed outer tail: c_msa_outer for r >= R_ij, 0 inside
        c_out = np.zeros((N, N, n_r))
        for i in range(N):
            for j in range(i, N):
                R = float(self.fmsa.R_ij[i, j])
                for idx, r in enumerate(self.r):
                    if r >= R:
                        c_out[i, j, idx] = self._c_msa_outer(r, i, j)
                c_out[j, i] = c_out[i, j]

        # Reference: outer tail only (full) or outer + c^HS (perturbation)
        c_hs = self._precompute_c_hs() if iterate == 'perturbation' else None
        c_ref = (c_hs + c_out) if c_hs is not None else c_out
        h_ref = self._oz_step(c_ref)
        b = np.full(total, -1.0) - pack(h_ref)

        # A·v = OZ(v_full)[core]  — linear operator, no reference contribution
        A = LinearOperator(
            (total, total),
            matvec=lambda v: pack(self._oz_step(unpack_inner(v))),
            dtype=float,
        )

        c_hs_in = pack(c_hs) if c_hs is not None else np.zeros(total)
        c_init = self._init_c()
        if iterate == 'perturbation':
            # Start from zero deviation: r0 = -b_pert = OZ(c_hs+c_out)[core]+1 is clean.
            # Starting from c^(1) makes r0 ≈ 0 (noisy initial Krylov vector).
            x0 = np.zeros(total)
        else:
            x0 = pack(c_init) - c_hs_in

        x_sol, info = gmres(A, b, x0=x0, rtol=tol, atol=tol, restart=min(total, 100))

        if info == 0:
            print(f"GMRES/{iterate} converged (tol={tol:.1e}, n_unknowns={total})")
            c_in_sol = unpack_inner(x_sol + c_hs_in)
            c_conv = c_out.copy()
            for i in range(N):
                for j in range(i, N):
                    mask = masks[(i, j)]
                    c_conv[i, j, mask] = c_in_sol[i, j, mask]
                    c_conv[j, i] = c_conv[i, j]
            self._c_converged = c_conv
            self._h_converged = self._oz_step(c_conv)
            return True
        else:
            print(f"GMRES/{iterate} did NOT converge (info={info})")
            self._c_converged = None
            self._h_converged = None
            return False

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def _run_iteration(self, c_start: np.ndarray, tol: float, max_iter: int,
                       alpha: float, method: str, closure: str,
                       m_diis: int, iterate: str) -> bool:
        """Run one OZ iteration from c_start.  Sets _c_converged / _h_converged.

        Returns True if converged.  One step only, so that coupling continuation can chain
        several of them.
        """
        c_hs = self._precompute_c_hs() if iterate == 'perturbation' else None
        x = (c_start - c_hs) if c_hs is not None else c_start.copy()

        history = []
        v = np.zeros_like(x)
        dt = alpha
        dt_max = 0.5
        f_inc = 1.1
        f_dec = 0.5
        alpha_fire = 0.1

        delta = np.inf
        converged = False
        h = None
        for it in range(max_iter):
            c_tot = (c_hs + x) if c_hs is not None else x
            h = self._oz_step(c_tot)
            c_new = self._closure_update(c_tot, h, closure)
            resid = ((c_new - c_hs) if c_hs is not None else c_new) - x
            delta = float(np.max(np.abs(resid)))

            if method == 'picard':
                x = x + alpha * resid
            elif method == 'ng':
                x = self._ng_step(x, resid, history)
            elif method == 'diis':
                # For HNC/PY use Anderson mixing (beta=alpha < 1) to damp DIIS overshoot.
                # For MSA use standard Pulay DIIS (beta=1.0) unchanged.
                beta_diis = alpha if closure in ('hnc', 'py') else 1.0
                x = self._diis_step(x, resid, history, m=m_diis, beta=beta_diis)
                if closure == 'hnc':
                    # Belt-and-suspenders clip: prevents any residual cascade from
                    # large DIIS coefficients pushing x out of the physical range.
                    np.clip(x, -50.0, 50.0, out=x)
            elif method == 'fire':
                power = float(np.dot(resid.ravel(), v.ravel()))
                if power > 0:
                    v_norm = np.linalg.norm(v)
                    f_norm = np.linalg.norm(resid)
                    v = ((1 - alpha_fire) * v
                         + alpha_fire * resid * (v_norm / (f_norm + 1e-30)))
                    dt = min(dt * f_inc, dt_max)
                else:
                    v = np.zeros_like(x)
                    dt = dt * f_dec
                v = v + dt * resid
                x = x + dt * v
            else:
                raise ValueError(
                    f"Unknown method '{method}'. Choose: picard, ng, diis, fire, gmres.")

            if (it + 1) % 50 == 0:
                print(f"  iter {it+1:4d}  |Δc| = {delta:.3e}")

            if delta < tol:
                print(f"OZ/{method}/{closure}/{iterate} converged at iter {it+1}, "
                      f"|Δc| = {delta:.2e}")
                self._n_iters = it + 1
                converged = True
                break
        else:
            print(f"OZ/{method}/{closure}/{iterate} did NOT converge after {max_iter} iters, "
                  f"|Δc| = {delta:.2e}")
            self._n_iters = max_iter

        if converged:
            self._c_converged = (c_hs + x) if c_hs is not None else x
            self._h_converged = h
        else:
            self._c_converged = None
            self._h_converged = None
        return converged

    def solve(self, tol: float = 1e-8, max_iter: int = 500, alpha: float = 0.2,
              method: str = 'picard', closure: str = 'msa', m_diis: int = 5,
              iterate: str = 'perturbation', gmres_fallback: bool = False,
              init_c: np.ndarray | None = None,
              n_coupling_steps: int = 1) -> None:
        """Run OZ iteration until convergence.

        Parameters
        ----------
        tol              : convergence threshold on max|Δc|
        max_iter         : maximum iterations per step
        alpha            : mixing parameter for Picard; initial dt for FIRE
        method           : 'picard' | 'ng' | 'diis' | 'fire' | 'gmres'
        closure          : 'msa' | 'py' | 'hnc' | 'hncb'
                           'hncb' = HNC + hard-sphere bridge (reference-HNC);
                           the bridge is auto-computed on first use.
        m_diis           : DIIS history length
        iterate          : 'full' | 'perturbation'
        gmres_fallback   : fall back to GMRES when primary method fails (MSA only)
        init_c           : optional external initial c array, shape (N, N, n_r).
                           Use init_c_from_model(model) to build from a reference model.
                           Ignored when n_coupling_steps > 1.
        n_coupling_steps : if > 1, ramp the Yukawa coupling K from 1/n to full strength
                           in n steps (coupling continuation).  Starts from pure-HS c.
                           Robust fallback when a single-step solve diverges.
        """
        if closure == 'hncb' and not hasattr(self, "_bridge_hs"):
            self._compute_hs_bridge()

        if method == 'gmres':
            self._solve_gmres(tol=tol, iterate=iterate)
            return

        if n_coupling_steps > 1:
            # Coupling continuation: ramp K from 1/n to 1.0 in n steps.
            c0 = self._init_c_hs_only()
            total_iters = 0
            for step in range(n_coupling_steps):
                self._k_scale = (step + 1) / n_coupling_steps
                print(f"Continuation step {step+1}/{n_coupling_steps} "
                      f"(K_scale={self._k_scale:.3f})")
                ok = self._run_iteration(c0, tol, max_iter, alpha, method,
                                         closure, m_diis, iterate)
                total_iters += self._n_iters or 0
                if not ok:
                    print(f"  *** Continuation failed at step {step+1}; aborting ***")
                    self._k_scale = 1.0
                    self._n_iters = total_iters
                    return
                c0 = self._c_converged.copy()
            self._k_scale = 1.0
            self._n_iters = total_iters
            return

        # Single-step solve (original behaviour).
        self._k_scale = 1.0
        c0 = init_c if init_c is not None else self._init_c()
        ok = self._run_iteration(c0, tol, max_iter, alpha, method,
                                 closure, m_diis, iterate)
        if not ok and gmres_fallback and closure == 'msa':
            print("Falling back to GMRES (convergence guarantee)...")
            self._solve_gmres(tol=tol, iterate=iterate)

    def compute_chat0_matrix_from_oz(self) -> np.ndarray:
        """ĉᵢⱼ(k=0) = 4π ∫₀^∞ r² cᵢⱼ(r) dr, integrated directly on the OZ radial grid.

        More reliable than calling compute_chat0_matrix(model) after OZ, because it uses
        the actual converged cᵢⱼ(r) rather than reconstructing via get_c1_inner (which
        may not fully reflect the OZ-corrected like-pair DCF unless update_all_pairs=True).

        Returns
        -------
        chat : shape (N, N)  — symmetric matrix of zero-wavevector DCF values.
        """
        if self._c_converged is None:
            raise RuntimeError("OZ not yet converged; call solve() first.")
        N = self.fmsa.N
        chat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                chat[i, j] = 4.0 * np.pi * _trapz(
                    self.r**2 * self._c_converged[i, j], self.r
                )
        return chat

    def extract_K_oz(self) -> np.ndarray:
        """Extract effective Yukawa amplitudes K_t,ij^OZ from the converged outer tail.

        For OZ+MSA the outer is exactly −βu = Σ_t K_t exp(−z_t(r−R))/(2π√ρρ r),
        so K^OZ = K^bare exactly.  For OZ+PY the γ-feedback modifies the near-contact
        outer tail; the extracted K^OZ captures that renormalisation.

        Method: in the window R < r < R + 2/z_max (where both tails are active), fit
            c_ij^OZ(r) · 2π√ρρ · r  ≈  Σ_t K_t^OZ · exp(−z_t(r−R))
        via ordinary least-squares.  For n_tails=1 the system is overdetermined and
        well-conditioned; for n_tails=2 both exponentials contribute in the window so
        their amplitudes are separately identifiable.

        Returns
        -------
        K_oz : ndarray, shape (n_tails, N, N)
            Same layout as fmsa.K; symmetric (K_oz[t,i,j] == K_oz[t,j,i]).
        """
        if self._c_converged is None:
            raise RuntimeError("Call solve() before extract_K_oz().")
        fmsa = self.fmsa
        N, nt = fmsa.N, fmsa.n_tails
        K_oz = fmsa.K.copy()

        for i in range(N):
            for j in range(i, N):
                R = float(fmsa.R_ij[i, j])
                rho_ij = float(np.sqrt(fmsa.rho[i] * fmsa.rho[j]))
                pf = 2.0 * np.pi * rho_ij

                z = np.array([float(fmsa.z_poles[t, i, j]) for t in range(nt)])
                z_pos = z[z > 0]
                if len(z_pos) == 0:
                    continue
                z_max = float(np.max(z_pos))
                # window: near-contact, both tails present
                w = min(R, 2.0 / z_max)
                mask = (self.r > R) & (self.r < R + w)
                if mask.sum() < nt + 1:
                    mask = self.r > R   # fallback: full outer region

                r_fit = self.r[mask]
                # target: c * 2π√ρρ * r = Σ_t K_t^OZ * exp(-z_t*(r-R))
                y = self._c_converged[i, j, mask] * pf * r_fit

                # design matrix, one column per Yukawa tail
                Phi = np.column_stack([np.exp(-z[t] * (r_fit - R)) for t in range(nt)])

                K_fit, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
                for t in range(nt):
                    K_oz[t, i, j] = K_fit[t]
                    K_oz[t, j, i] = K_fit[t]

        return K_oz

    # ------------------------------------------------------------------
    # Diagnostic
    # ------------------------------------------------------------------

    def check_hankel_roundtrip(self, tol: float = 1e-4) -> float:
        """Verify _k2r(_r2k(f)) ≈ f for a Gaussian test function. Returns max error."""
        f = np.exp(-0.5 * (self.r - self.r_max / 4)**2)
        err = float(np.max(np.abs(self._k2r(self._r2k(f)) - f)))
        status = "OK" if err < tol else "FAIL"
        print(f"Hankel round-trip max error: {err:.3e}  [{status}]")
        return err
