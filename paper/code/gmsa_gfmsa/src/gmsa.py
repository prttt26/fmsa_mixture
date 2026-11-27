"""Full GMSA hard-sphere reference calibrated with the complete Exact MSA.

    from gmsa import GMSA
    hs = GMSA.hard_spheres(.49, diameter=[1., .3], x=[.5, .5])
    chat = hs.get_chat_k([0., 1.])
    g = hs.get_g_r([.2, .3, .65, 1., 1.4])

At each density, fit one common dimensional Yukawa rate and a symmetric
contact-normalized amplitude matrix to BMCSL pair contacts and fixed-composition
inverse compressibility. Each fit evaluation solves the full Blum--Hoye MSA
system in msa_exact_mix.py. GFMSA supplies only an initial guess. The DCF, RDF
and Baxter factor returned here are the complete MSA solution of the fitted
closure, without first-order truncation.

This is a hard-sphere reference, not a physical Yukawa fluid. Its fitted tail
carries no energy. from_fmsa(dp) reads only the hard cores and densities; it
does not combine the physical tails with the reference. Linearizing physical
tails about this full reference is a separate approximation and is not done
here. BMCSL calibration does not impose its whole free-energy Hessian on c(0).

Sources: Waisman, Molecular Physics 25, 45 (1973); Blum and Hoye, J. Stat.
Phys. 19, 317 (1978). The contact and amplitude conventions, including the
source corrections, are those audited in msa_exact_mix.py.
"""

from functools import cached_property
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import least_squares, root

from gfmsa import GFMSA
from msa_exact_mix import HCYMix, SolutionMix, _pieces, _qhat_vec, seed_zero_K, solve_bh_mix


class GMSAFitError(RuntimeError):
    """The full-MSA branch or the requested HS calibration did not converge."""


class GMSA:
    """Full additive HS GMSA; sigma and partial densities have shape (N,).

    rho may also be a total density accompanied by mole fractions x. For an
    all-zero partial-density vector x chooses the limiting composition. This
    composition matters for the otherwise undetermined limiting Yukawa rate.
    Fourier/Laplace shapes are (N,N) for scalars, (ns,N,N) for vectors; radial
    shapes are (N,N,nr). All dimensional parameters share the caller's units.
    """

    def __init__(self, sigma, rho, x=None, tol=1e-9, max_nfev=300):
        self.diameters = np.atleast_1d(np.asarray(sigma, dtype=float)).copy()
        if (self.diameters.ndim != 1 or not len(self.diameters)
                or np.any(~np.isfinite(self.diameters)) or np.any(self.diameters <= 0)):
            raise ValueError("sigma must be a nonempty positive finite vector")
        self.N = len(self.diameters)
        r = np.asarray(rho, dtype=float)
        if np.any(~np.isfinite(r)) or np.any(r < 0):
            raise ValueError("densities must be finite and nonnegative")
        fractions = np.full(self.N, 1/self.N) if x is None else np.asarray(x, dtype=float)
        if (fractions.shape != (self.N,) or np.any(~np.isfinite(fractions))
                or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1., rtol=0, atol=1e-12)):
            raise ValueError("x must be nonnegative, sum to one and match sigma")
        if r.ndim == 0:
            self.rho = float(r)*fractions
            self.x = fractions.copy()
        elif r.shape == (self.N,):
            self.rho = r.copy()
            rt = float(r.sum())
            self.x = r/rt if rt > 0 else fractions.copy()
            if x is not None and rt > 0 and not np.allclose(self.x, fractions, rtol=0, atol=1e-12):
                raise ValueError("x is inconsistent with the supplied partial densities")
        else:
            raise ValueError("rho must be a total density or N partial densities")
        self.number_density = float(self.rho.sum())
        self.eta = float(math.pi*np.sum(self.rho*self.diameters**3)/6)
        if not 0 <= self.eta < 1:
            raise ValueError("packing fraction must lie in [0, 1)")
        if not math.isfinite(tol) or tol <= 0 or not isinstance(max_nfev, int) or max_nfev < 1:
            raise ValueError("tol and max_nfev must be positive")
        self._tol, self._max_nfev = float(tol), max_nfev
        self.R_ij = (self.diameters[:, None]+self.diameters[None, :])/2
        seed = GFMSA.hard_spheres(self.eta, self.diameters, x=self.x)
        self._py_dp = seed.physical_dp
        self.target_contacts = np.array([[seed.get_g_contact(i, j)
                                         for j in range(self.N)] for i in range(self.N)])
        self.target_inverse_compressibility = seed.hs_inverse_compressibility
        self.solution, self.fit_diagnostics = self._fit(seed)
        self.hs_rate = float(self.solution.st.z[0])
        self.hs_k_bare = self.solution.st.K[0].copy()

    @classmethod
    def hard_spheres(cls, eta, diameter=1., x=None, **kwargs):
        """Build a pure HS reference at packing fraction eta."""
        eta = float(eta)
        if not math.isfinite(eta) or not 0 <= eta < 1:
            raise ValueError("eta must be finite and in [0, 1)")
        d = np.atleast_1d(np.asarray(diameter, dtype=float))
        if d.ndim != 1 or not len(d) or np.any(~np.isfinite(d)) or np.any(d <= 0):
            raise ValueError("diameters must be a nonempty positive finite vector")
        fractions = np.full(len(d), 1/len(d)) if x is None else np.asarray(x, dtype=float)
        if (fractions.shape != d.shape or np.any(~np.isfinite(fractions))
                or np.any(fractions < 0) or not np.isclose(fractions.sum(), 1., rtol=0, atol=1e-12)):
            raise ValueError("x must match diameters, be nonnegative and sum to one")
        return cls(d, 6*eta/(math.pi*(fractions@d**3)), x=fractions, **kwargs)

    @classmethod
    def from_fmsa(cls, dp, **kwargs):
        """Extract only the HS reference state; no physical tails are added."""
        d = np.asarray(dp.sigma)
        if not np.allclose(dp.R_ij, (d[:, None]+d[None, :])/2, rtol=0, atol=1e-12):
            raise ValueError("GMSA requires additive hard cores")
        return cls(d, dp.rho, x=dp.x, **kwargs)

    def at_density(self, rho):
        """Refit at a total density (fixed x) or new partial-density vector."""
        values = np.asarray(rho)
        return type(self)(self.diameters, rho,
                          x=self.x if values.ndim == 0 or np.all(values == 0) else None,
                          tol=self._tol, max_nfev=self._max_nfev)

    def _fit(self, seed):
        upper = np.triu_indices(self.N)
        initial = seed.hs_k_bare
        scale = np.maximum(abs(initial[upper]), 1e-10*self.R_ij[upper])
        py_contact = self.target_contacts-seed._hs._b_csum_red/(2*math.pi*self.R_ij)
        gscale = np.maximum(abs(self.target_contacts-py_contact)[upper], 1e-10)
        m2, m3 = self.x@self.diameters**2, self.x@self.diameters**3
        gap = (m2**3/m3**2)*self.eta**3*(4-self.eta)/(1-self.eta)**4
        dscale = max(gap, 1e-10)
        initial_u = np.r_[initial[upper]/scale, math.log(seed.hs_rate)]
        calls = 0

        def state(u):
            if not np.all(np.isfinite(u)) or abs(u[-1]-initial_u[-1]) > math.log(100):
                raise GMSAFitError("outer iteration left the supported positive-rate interval")
            K = np.zeros((self.N, self.N))
            K[upper] = u[:-1]*scale
            K[(upper[1], upper[0])] = K[upper]
            return HCYMix(self.diameters, self.rho, [math.exp(u[-1])], K)

        def solve_at(u):
            nonlocal calls
            calls += 1
            st = state(u)
            if not np.any(st.K):
                Dt, Gt = seed_zero_K(st)
                return SolutionMix(st, Dt, Gt, _pieces(st, Dt, Gt), 0.)
            sol = solve_bh_mix(st, tol=min(1e-10, self._tol/10))
            if sol is None or sol.symmetry_defect > max(self._tol, 1e-10) or np.any(sol.A <= 0):
                raise GMSAFitError("complete MSA did not return a regular symmetric HS branch")
            return sol

        def residual(u):
            sol = solve_at(u)
            return np.r_[(sol.contact-self.target_contacts)[upper]/gscale,
                         (sol.dbetaP_drho-self.target_inverse_compressibility)/dscale]

        if self.number_density == 0:
            sol = solve_at(initial_u)
            return sol, dict(converged=True, evaluations=calls, contact_relative_error=0.,
                             compressibility_relative_error=0., msa_residual=0.,
                             message="analytic zero-density PY limit; fitted amplitudes zero")
        try:
            fit = root(residual, initial_u, tol=min(self._tol, 1e-10),
                       options={"maxfev": self._max_nfev})
            sol = solve_at(fit.x)
        except GMSAFitError:
            fit, sol = None, None

        def errors(sol):
            if sol is None:
                return np.inf, np.inf
            ge = float(np.max(abs(sol.contact/self.target_contacts-1)))
            de = abs(sol.dbetaP_drho/self.target_inverse_compressibility-1)
            return ge, de

        if max(errors(sol)) > self._tol:
            # A bounded trust-region retry stays near the already regular
            # GFMSA seed. Acceptance is based on physical residuals, not flags.
            lower = np.r_[np.full(len(scale), -np.inf), initial_u[-1]-math.log(4)]
            upper_bound = np.r_[np.full(len(scale), np.inf), initial_u[-1]+math.log(4)]
            try:
                fit = least_squares(residual, initial_u, bounds=(lower, upper_bound),
                                    xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                    max_nfev=self._max_nfev)
                sol = solve_at(fit.x)
            except GMSAFitError as exc:
                raise GMSAFitError("GMSA HS calibration failed on the complete MSA branch") from exc
        ge, de = errors(sol)
        if max(ge, de) > self._tol:
            raise GMSAFitError(f"GMSA fit missed tolerance: contact={ge:.3g}, compressibility={de:.3g}")
        return sol, dict(converged=True, evaluations=calls, contact_relative_error=ge,
                         compressibility_relative_error=de, msa_residual=sol.residual,
                         message=str(fit.message))

    @staticmethod
    def _vector(values, name, complex_values=False):
        v = np.asarray(values, dtype=complex if complex_values else float)
        if v.ndim > 1 or np.any(~np.isfinite(v)):
            raise ValueError(f"{name} must be a finite scalar or one-dimensional array")
        return np.atleast_1d(v), v.ndim == 0

    def get_qhat(self, s):
        """Complete density-free Baxter kernel Qhat(s), not its PY factor."""
        sv, scalar = self._vector(s, "s", True)
        q = _qhat_vec(self.solution.st, self.solution.p, sv)
        return q[0] if scalar else q

    def get_baxter_factor(self, s):
        """F(s)=I-sqrt(rho) Qhat(s) sqrt(rho), complex s allowed."""
        return np.eye(self.N)-np.sqrt(self.rho[:, None]*self.rho[None, :])*self.get_qhat(s)

    def get_chat_k(self, k):
        """Complete DCF transform, including k=0 and absent-species limits."""
        wave, scalar = self._vector(k, "k")
        q = self.get_qhat(1j*wave)
        qt = np.swapaxes(q.conj(), 1, 2)
        c = (q+qt-(q*self.rho[None, None, :])@qt).real
        return c[0] if scalar else c

    def get_hhat_k(self, k, order="oz"):
        """Complete OZ h; a first-order truncation is not used by GMSA."""
        if order != "oz":
            raise ValueError("GMSA uses order='oz'; use GFMSA for the first-order model")
        c = self.get_chat_k(k)
        return np.linalg.solve(np.eye(self.N)-c*self.rho[None, :], c)

    def get_S_k(self, k, order="oz"):
        return np.eye(self.N)+np.sqrt(self.rho[:, None]*self.rho[None, :])*self.get_hhat_k(k, order)

    def get_g_contact(self, i=0, j=0):
        if not 0 <= i < self.N or not 0 <= j < self.N:
            raise IndexError("species index out of range")
        return float(self.solution.contact[i, j])

    @property
    def hs_inverse_compressibility(self):
        if self.number_density == 0:
            return 1.
        return float(self.solution.dbetaP_drho)

    def get_gL_laplace(self, s):
        """Complete RDF Laplace transform, Blum--Hoye Eq. (33).

        RHS_ij=e^-sR/(2pi)[A_j/s^2+qp_ij/s+z Ct_ij/(s+z)].
        Solve G(s)(I-diag(rho)Qhat(s))=RHS without division by densities.
        """
        sv, scalar = self._vector(s, "s", True)
        if np.any(sv.real <= 0):
            raise ValueError("Laplace arguments must have positive real part")
        p, z = self.solution.p, self.hs_rate
        s3 = sv[:, None, None]
        source = np.exp(-s3*self.R_ij)/(2*math.pi)*(
            p["A"][None, None, :]/s3**2+p["qp"][None]/s3+z*p["Ct"][0][None]/(s3+z))
        factor = np.eye(self.N)-self.rho[None, :, None]*self.get_qhat(sv)
        g = np.swapaxes(np.linalg.solve(np.swapaxes(factor, 1, 2),
                                       np.swapaxes(source, 1, 2)), 1, 2)
        return g[0] if scalar else g

    def get_g_laplace(self, r_grid, M=48, tol=1e-12):
        """de Hoog inversion, exact known core zero and one-sided contact."""
        r, _ = self._vector(r_grid, "r")
        if np.any(r < 0):
            raise ValueError("radial distances must be nonnegative")
        if not isinstance(M, int) or M < 1 or not 0 < tol < 1:
            raise ValueError("M must be a positive integer and tol must lie in (0,1)")
        g = np.zeros((self.N, self.N, len(r)))
        for i in range(self.N):
            for j in range(self.N):
                outside = r > self.R_ij[i, j]
                if np.any(outside):
                    g[i, j, outside] = self._py_dp._invert_rL(
                        self.get_gL_laplace, r[outside], i, j, M, tol)
                g[i, j, r == self.R_ij[i, j]] = self.get_g_contact(i, j)
        return g

    def get_g_fourier(self, r_grid=None, k_max=1500., n_k=200000, return_r=False):
        """Independent full-OZ Fourier/DST RDF inversion."""
        from fmsa_double_prop import FMSA_double_prop
        r, g = FMSA_double_prop.get_g_fourier(
            self, r_grid, k_max=k_max, n_k=n_k, order="oz", return_r=True)
        for i in range(self.N):
            for j in range(self.N):
                g[i, j, r < self.R_ij[i, j]] = 0.
                g[i, j, r == self.R_ij[i, j]] = self.get_g_contact(i, j)
        return (r, g) if return_r else g

    def get_g_r(self, r_grid=None, route="laplace", **kwargs):
        if route == "laplace":
            if r_grid is None:
                raise ValueError("Laplace inversion requires r_grid")
            return self.get_g_laplace(r_grid, **kwargs)
        if route == "fourier":
            return self.get_g_fourier(r_grid, **kwargs)
        raise ValueError("route must be 'laplace' or 'fourier'")

    def _Q(self, i, j, t, derivative=0):
        """One-tail real Baxter kernel and its smooth first/second derivatives."""
        p, z, d = self.solution.p, self.hs_rate, self.diameters
        lam, R = (d[j]-d[i])/2, self.R_ij[i, j]
        if t < lam:
            return 0.
        W, C = p["Wt"][0, i, j], p["Ct"][0, i, j]
        ew = W*math.exp(-z*(t-lam))
        if t < R:
            if derivative == 0:
                return p["qp"][i, j]*(t-R)+p["A"][j]*(t-R)**2/2+ew+C
            if derivative == 1:
                return p["qp"][i, j]+p["A"][j]*(t-R)-z*ew
            return p["A"][j]+z*z*ew
        tail = ew+C*math.exp(-z*(t-R))
        return (-z)**derivative*tail

    def _convolution(self, i, j, r, derivative=1):
        """Baxter convolution with finite-panel quadrature and exact far tail.

        i has the larger diameter. Then the derivative's lower-support delta
        multiplies Q_jl outside its support and vanishes for r>0. For the
        origin derivative the core/tail jump in Q' is included explicitly.
        """
        total, z, d = 0., self.hs_rate, self.diameters
        for l in range(self.N):
            if self.rho[l] == 0:
                continue
            lower = (d[l]-d[j])/2
            tail_start = max(self.R_ij[i, l]-r, self.R_ij[j, l])
            edges = sorted({lower, tail_start,
                            max(lower, self.R_ij[i, l]-r), max(lower, self.R_ij[j, l])})
            value = 0.
            for lo, hi in zip(edges[:-1], edges[1:]):
                if hi > lo:
                    value += quad(lambda t: self._Q(i, l, t+r, derivative)*self._Q(j, l, t),
                                  lo, hi, epsabs=1e-11, epsrel=1e-11, limit=100)[0]
            qi = self._Q(i, l, tail_start+r)
            qj = self._Q(j, l, tail_start)
            value += (-z)**derivative*qi*qj/(2*z)
            if derivative == 2:
                jump = -z*self.solution.p["Ct"][0, i, l]-self.solution.p["qp"][i, l]
                value += jump*self._Q(j, l, self.R_ij[i, l])
            total += self.rho[l]*value
        return total

    @cached_property
    def _origins(self):
        values = np.empty((self.N, self.N))
        for a in range(self.N):
            for b in range(a, self.N):
                i, j = (a, b) if self.diameters[a] >= self.diameters[b] else (b, a)
                value = (-self._Q(i, j, 0., 2)+self._convolution(i, j, 0., 2))/(2*math.pi)
                values[a, b] = values[b, a] = value
        return values

    def get_c_total(self, r_grid):
        """Complete DCF: Baxter convolution in the core, fitted exact outer tail.

        Core evaluation uses adaptive one-dimensional quadrature; the state
        is the complete MSA root. No first-order DCF reconstruction is used.
        """
        r, _ = self._vector(r_grid, "r")
        if np.any(r < 0):
            raise ValueError("radial distances must be nonnegative")
        c = np.empty((self.N, self.N, len(r)))
        for a in range(self.N):
            for b in range(a, self.N):
                i, j = (a, b) if self.diameters[a] >= self.diameters[b] else (b, a)
                for q, rv in enumerate(r):
                    if rv == 0:
                        value = self._origins[a, b]
                    elif rv >= self.R_ij[i, j]:
                        value = self.hs_k_bare[i, j]*math.exp(-self.hs_rate*(rv-self.R_ij[i, j]))/rv
                    else:
                        value = (-self._Q(i, j, rv, 1)+self._convolution(i, j, rv))/(2*math.pi*rv)
                    c[a, b, q] = c[b, a, q] = value
        return c

    def beta_energy_per_particle(self):
        """Hard spheres have zero excess configurational energy."""
        return 0.

    def betaf_excess(self):
        """BMCSL HS excess beta*F/V from the calibrated fixed-composition EOS."""
        e, v = self.eta, 1-self.eta
        m1, m2, m3 = (self.x@self.diameters**n for n in (1, 2, 3))
        a, b = m1*m2/m3, m2**3/m3**2
        return float(self.number_density*((b-1)*math.log1p(-e)+3*a*e/v+b*e/v**2))
