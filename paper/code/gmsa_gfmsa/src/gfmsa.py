"""GFMSA: generalized first-order mean spherical approximation.

Project definition: the PY first-order response is applied to the sum of a
CS/BMCSL-calibrated hard-sphere closure tail and the physical interaction
tails. Both the HS calibration and the response use the same first-order
truncation. GFMSA adds a selectable HS correction without changing the PY
propagator or integrating the HS closure tail as a physical potential.

Usage::

    from gfmsa import GFMSA
    model = GFMSA(existing_dp)           # additive, unequal hard-core diameters
    # Equivalently: model = existing_dp.with_hs_reference("gfmsa")
    c = model.get_c_total(r)             # PY + HS correction + physical c1
    chat = model.get_chat_k([0., 1.])    # includes the zero mode
    g = model.get_g_r(r, route="laplace")

The HS closure tail is density-dependent and temperature-independent. All
responses still use the PY Baxter factor. For mixtures a common dimensional
HS rate is fitted to BMCSL fixed-composition compressibility and the exact
contact matrix in Eq. (52). We retain the size factor that was approximated
in Eq. (53), and solve a scalar quadratic instead of transcribing Eq. (80).
The full zero-mode DCF matrix is not the BMCSL free-energy Hessian: only its
fixed-composition contraction is calibrated. This is a first-order extension
of Tang and Lu, JCP 103, 7463 (1995), not full GMSA. Truncated
DCF and RDF agree through first order; their complete OZ pairing has a
second-order defect. Use get_hhat_k(order="oz") to inspect that resummation.

Physical tails are kept in physical_dp, unchanged. The energy methods below
integrate only those tails, at fixed hard-core diameter. thermodynamics()
provides energy- and compressibility-route free energies and their bulk
derivatives; it does not add empirical in-core potentials.
"""

from functools import cached_property
import math

import numpy as np

from fmsa_double_prop import FMSA_double_prop, _phi12


def tl95_parameters(eta):
    """Return dimensionless (z_HS, K_HS) from original Eqs. (23)--(24).

    c_HS,outer(r) = K_HS exp[-z_HS (r/d - 1)] / (r/d).
    K_HS is a DCF amplitude without beta. No temperature enters this fit.
    """
    eta = float(eta)
    if not math.isfinite(eta) or not 0 <= eta < 1:
        raise ValueError("eta must be finite and in [0, 1)")
    v = 1 - eta
    z = 2 * (1 + 2 * eta) * (3 + math.sqrt(21 - 15 * eta + 3 * eta**2)) / (
        (4 - eta) * v**2
    )
    S = v**2*z**3 + 6*eta*v*z**2 + 18*eta**2*z - 12*eta*(1 + 2*eta)
    L = (1 + eta/2)*z + 1 + 2*eta
    Q = (S + 12*eta*L*math.exp(-z)) / (v**2*z**3)
    return z, eta**2 * Q**2 / (2*v**3)


class GFMSA:
    """Wrap an additive HS mixture DP without changing its physical tails.

    Shapes follow FMSA_double_prop: (N,N,nr) for radial data, (N,N) for
    scalar Fourier/Laplace arguments and (nk,N,N) for vectors. Use
    at_density() to rebuild the fit at a new total or partial density.
    The wrapped state should not be mutated.
    """

    def __init__(self, physical_dp, *, lazy_radial=False):
        if not isinstance(physical_dp, FMSA_double_prop):
            raise TypeError("physical_dp must be a FMSA_double_prop instance")
        self.physical_dp = physical_dp
        self.N = physical_dp.N
        self.diameters = physical_dp.sigma.copy()
        self.rho = physical_dp.rho.copy()
        self.x = physical_dp.x.copy()
        self.number_density = float(np.sum(self.rho))
        self.eta = float(physical_dp.eta)
        if np.any(~np.isfinite(self.diameters)) or np.any(self.diameters <= 0):
            raise ValueError("hard-core diameters must be finite and positive")
        if np.any(~np.isfinite(self.rho)) or np.any(self.rho < 0):
            raise ValueError("partial densities must be finite and nonnegative")
        if np.any(~np.isfinite(self.x)) or np.any(self.x < 0) or not np.isclose(self.x.sum(), 1.):
            raise ValueError("mole fractions must be nonnegative and sum to one")
        tl95_parameters(self.eta)  # validates packing fraction
        if not math.isfinite(physical_dp.beta) or physical_dp.beta <= 0:
            raise ValueError("beta must be finite and positive")
        rates = physical_dp.z_poles
        if np.any(~np.isfinite(rates)) or np.any(rates <= 0):
            raise ValueError("physical Yukawa rates must be finite and positive")
        if np.any(~np.isfinite(physical_dp.k_bare)):
            raise ValueError("physical Yukawa amplitudes must be finite")
        self.R_ij = physical_dp.R_ij.copy()
        self.hs_rate, self.hs_k_bare = self._fit_hs_tail()
        # Preserve the previous single-component dimensionless attributes.
        if self.N == 1:
            self.diameter = float(self.diameters[0])
            self.hs_z = self.hs_rate*self.diameter
            self.hs_amplitude = float(self.hs_k_bare[0, 0]/self.diameter)
        # beta=1 and eps=1 make the companion's k input a DCF amplitude.
        # A separate companion avoids appending a fictitious potential to
        # physical_dp.k_bare or its energy/free-energy tail loops.
        self._hs = FMSA_double_prop(
            self.diameters, np.ones(self.N), self.x,
            self.number_density, 1., n_tails=1,
            z=[1.], k=[0.], input_type="bh", verbose=False,
            lazy_radial=lazy_radial,
        )
        self._hs.z_poles = np.full((1, self.N, self.N), self.hs_rate)
        self._hs.k_bare = self.hs_k_bare[None, :, :].copy()
        self._hs.epsilon = self._hs.k_bare/self.R_ij[None, :, :]
        self._hs.K = 2*math.pi*np.sqrt(self.rho[:, None]*self.rho[None, :])*self._hs.k_bare
        self._hs._build_b_tables()
        if not lazy_radial:
            self._hs._build_closed_form()

    def _zero_M(self):
        """M(0), M'(0) for Q(s)=I-sqrt(rho) M(s) sqrt(rho)."""
        p, d = self.physical_dp, self.diameters[:, None]
        m0 = -d**2*p.Q0/2+d**3*p.Qpp[None, :]/6
        m1 = d**3*p.Q0/6-d**4*p.Qpp[None, :]/24-p.lambda_ij*m0
        return m0, m1

    def _fit_hs_tail(self):
        """Fit common z and undressed amplitudes from exact BMCSL contacts.

        Set j_ij=R_ij*(g_BMCSL-g_PY). Then J=2pi sqrt(rho_i rho_j) j.
        At fixed J, Eq. (67)'s zero mode is C1(z)=a/z+b/z^2.
        Its density contraction gives a scalar quadratic for z. The fit is
        divided by its common density prefactor before solving, so the zero
        density and absent-species limits remain finite.
        """
        e, d, x, rho, R = self.eta, self.diameters, self.x, self.rho, self.R_ij
        v = 1-e
        m0, m1 = self._zero_M()
        q0 = np.eye(self.N)-m0*rho[None, :]
        q1 = -m1*rho[None, :]
        u = (d[:, None]*d[None, :])**2/R
        a = 4*math.pi*(q1@u@q0.T+q0@u@q1.T+q0@(R*u)@q0.T)
        b = 4*math.pi*q0@u@q0.T
        target = math.pi*(x@d**2)*(x@d**3)*(4-e)/(3*v)
        ac, bc = float(x@a@x), float(x@b@x)
        root = math.sqrt(ac*ac+4*target*bc)
        z = ((ac+root)/(2*target) if ac >= 0 else 2*bc/(root-ac))
        if not math.isfinite(z) or z <= 0:
            raise ValueError("HS common-rate calibration has no positive finite root")
        zeta2 = math.pi*np.sum(rho*d**2)/6
        j = zeta2**2*u/(2*v**3)
        p = self.physical_dp
        try:
            with np.errstate(over='raise', invalid='raise'):
                M = p._build_M(z)
                ar = np.linalg.solve(np.eye(self.N)-M*rho[None, :], M)*np.exp(z*p.lambda_ij)
        except FloatingPointError:
            # z|lambda| > 709: the same matrix without the phase (see _build_M_unphased)
            M = p._build_M_unphased(z)
            ar = np.linalg.solve(np.eye(self.N)-M*rho[None, :], M)
        L = np.eye(self.N)+ar*rho[None, :]
        # Undo double propagation of the prescribed contact-residue matrix.
        kij = np.linalg.solve(L, np.linalg.solve(L, j).T).T
        return z, (kij+kij.T)/2

    @classmethod
    def hard_spheres(cls, eta, diameter=1., x=None):
        """Pure additive HS fluid; diameter may be a vector with fractions x."""
        eta = float(eta)
        tl95_parameters(eta)
        d = np.atleast_1d(np.asarray(diameter, dtype=float))
        if d.ndim != 1 or not len(d) or np.any(~np.isfinite(d)) or np.any(d <= 0):
            raise ValueError("diameters must be a nonempty positive finite vector")
        x = np.full(len(d), 1/len(d)) if x is None else np.asarray(x, dtype=float)
        if x.shape != d.shape or np.any(~np.isfinite(x)) or np.any(x < 0) or not np.isclose(x.sum(), 1.):
            raise ValueError("x must match diameters, be nonnegative and sum to one")
        physical = FMSA_double_prop(
            d, np.ones(len(d)), x,
            6*eta/(math.pi*(x@d**3)), 1., n_tails=1,
            z=[1.8], k=[0.], input_type="bh", verbose=False,
        )
        return cls(physical)

    def at_density(self, number_density, *, lazy_radial=False):
        """Refit at a total density (same x), or a vector of partial densities."""
        values = np.asarray(number_density, dtype=float)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("densities must be finite and nonnegative")
        if values.ndim == 0:
            rt, x = float(values), self.x
        elif values.shape == (self.N,):
            rt = float(values.sum())
            x = values/rt if rt > 0 else self.x
        else:
            raise ValueError("provide a total density or N partial densities")
        p = self.physical_dp
        rebuilt = FMSA_double_prop(
            p.sig_pure, p.eps_pure, x, rt, p.tstar,
            n_tails=p.n_tails, z=p._z_input, k=p._k_input,
            eps_ij=p._eps_ij, input_type=p._input_type, d_pure=p._d_pure,
            hs_fmt=p.hs_fmt, verbose=False, lazy_radial=lazy_radial,
        )
        # The dimensional physical potential is fixed along a density ray.
        # Preserve an explicitly installed tail dictionary as well as the
        # ordinary constructor-derived one; density must not remap tails.
        if (not np.array_equal(rebuilt.z_poles, p.z_poles) or
                not np.array_equal(rebuilt.k_bare, p.k_bare)):
            rebuilt.z_poles = p.z_poles.copy()
            rebuilt.k_bare = p.k_bare.copy()
            rebuilt.epsilon = rebuilt.k_bare/rebuilt.R_ij[None, :, :]
            rebuilt.K = (2*math.pi*rebuilt.beta*
                         np.sqrt(rebuilt.rho[:, None]*rebuilt.rho[None, :])*
                         rebuilt.k_bare)
            rebuilt._build_b_tables()
            if not lazy_radial:
                rebuilt._build_closed_form()
        return type(self)(rebuilt, lazy_radial=lazy_radial)

    @staticmethod
    def _vector(values, name, complex_values=False):
        arr = np.asarray(values, dtype=complex if complex_values else float)
        if arr.ndim > 1 or np.any(~np.isfinite(arr)):
            raise ValueError(f"{name} must be a finite scalar or one-dimensional array")
        return np.atleast_1d(arr), arr.ndim == 0

    def _shape(self, values, scalar):
        out = np.asarray(values).reshape(-1, self.N, self.N)
        return out[0] if scalar else out

    def _py_moment(self, n, i=0, j=0):
        # PY is a constant/polynomial (possibly 1/r) on each core piece.
        # Five-point Gauss integrates r^n*c_PY exactly for n=2,4,6.
        nodes, weights = np.polynomial.legendre.leggauss(5)
        p = self.physical_dp
        edges = np.r_[0., p.incore_breakpoints(i, j), self.R_ij[i, j]]
        result = 0.
        for lo, hi in zip(edges[:-1], edges[1:]):
            r = (lo+hi)/2+(hi-lo)*nodes/2
            result += (hi-lo)/2*sum(w*rv**n*p.get_HS_PY(rv, i, j)
                                    for rv, w in zip(r, weights))
        return result

    def _moment(self, dp, n, i=0, j=0):
        """Integral of r**n c1 including the analytic Yukawa tail."""
        d = self.R_ij[i, j]
        outer = 0.
        for rate, amp in zip(dp.z_poles[:, i, j], dp.beta*dp.k_bare[:, i, j]):
            outer += amp * sum(math.comb(n-1, j)*d**(n-1-j)*math.factorial(j)
                               / rate**(j+1) for j in range(n))
        return dp.c1_incore_moment(i, j, n=n) + outer

    @cached_property
    def _moments(self):
        out = {}
        for n in (2, 4, 6):
            val = np.empty((3, self.N, self.N))
            for i in range(self.N):
                for j in range(self.N):
                    val[:, i, j] = (self._py_moment(n, i, j),
                                    self._moment(self.physical_dp, n, i, j),
                                    self._moment(self._hs, n, i, j))
            out[n] = val
        return out

    def _M_vec(self, s):
        """Complex PY M(s), without density square roots or zero divisions."""
        p = self.physical_dp
        s = np.asarray(s, dtype=complex)[:, None]
        d = self.diameters[None, :]
        # series for |s d| < 1/2 (the closed forms cancel like eps/|s d|^3 as s d -> 0)
        p1, p2 = _phi12(s, d)
        return np.exp(-s[:, :, None]*p.lambda_ij)*(p1[:, :, None]*p.Q0
                    + p2[:, :, None]*p.Qpp[None, None, :])

    def _spectrum(self, k):
        """Return PY, physical c1, and HS correction, including stable k=0."""
        wave, scalar = self._vector(k, "k")
        vals = np.zeros((len(wave), 3, self.N, self.N))
        rate_min = min(1/np.max(self.diameters), self.hs_rate,
                       np.min(self.physical_dp.z_poles, initial=np.inf))
        small = abs(wave) < 1e-3*rate_min
        if np.any(small):
            q = wave[small, None, None, None]
            vals[small] = 4*math.pi*(self._moments[2]
                                    - q**2*self._moments[4]/6
                                    + q**4*self._moments[6]/120)
        other = ~small
        if np.any(other):
            w = wave[other]
            M = self._M_vec(1j*w)
            MT = np.swapaxes(M.conj(), 1, 2)
            py = (M+MT-(M*self.rho[None, None, :])@MT).real
            left = np.eye(self.N)-py*self.rho[None, None, :]
            right = np.swapaxes(left, 1, 2)
            vals[other, 0] = py
            for slot, dp in ((1, self.physical_dp), (2, self._hs)):
                h1 = -4*math.pi*self._h1L(dp, 1j*w).imag/w[:, None, None]
                vals[other, slot] = left@h1@right
        return vals, scalar

    def get_chat_k(self, k):
        """Total DCF transform; HS zero-mode contraction matches BMCSL."""
        vals, scalar = self._spectrum(k)
        return self._shape(vals.sum(axis=1), scalar)

    def get_c1hat_k(self, k):
        """Physical-tail DCF response, excluding the fitted HS correction."""
        vals, scalar = self._spectrum(k)
        return self._shape(vals[:, 1], scalar)

    def get_h1hat_k(self, k):
        """Physical-tail RDF response with the unchanged PY propagator.

        This is the derivative when the physical amplitudes are multiplied
        by tau and the fitted HS tail is held fixed. Mixed HS/physical
        response terms lie beyond the GFMSA truncation.
        """
        vals, scalar = self._spectrum(k)
        inv = np.linalg.inv(np.eye(self.N)-vals[:, 0]*self.rho[None, None, :])
        return self._shape(inv@vals[:, 1]@np.swapaxes(inv, 1, 2), scalar)

    def get_hhat_k(self, k, order="first"):
        """First-order RDF transform, or complete OZ resummation of the DCF."""
        vals, scalar = self._spectrum(k)
        F = np.eye(self.N)-vals[:, 0]*self.rho[None, None, :]
        if order == "first":
            inv = np.linalg.inv(F)
            h = inv@vals[:, 0]+inv@(vals[:, 1]+vals[:, 2])@np.swapaxes(inv, 1, 2)
        elif order == "oz":
            c = vals.sum(axis=1)
            h = np.linalg.solve(np.eye(self.N)-c*self.rho[None, None, :], c)
        else:
            raise ValueError("order must be 'first' or 'oz'")
        return self._shape(h, scalar)

    def get_S_k(self, k, order="first"):
        return np.eye(self.N)+np.sqrt(self.rho[:, None]*self.rho[None, :])*self.get_hhat_k(k, order=order)

    @staticmethod
    def _c1_at(dp, r, i=0, j=0):
        if r >= dp.R_ij[i, j]:
            # The MSA closure fixes this branch exactly. Convolution pieces
            # round support edges, so an exact contact may otherwise fall
            # on the interior side of a rounded endpoint and lose its tail.
            z = dp.z_poles[:, i, j]
            return float(np.sum(dp.beta*dp.k_bare[:, i, j]*
                                np.exp(-z*(r-dp.R_ij[i, j])))/r)
        if r > 0:
            return dp.get_c1_inner(r, i, j)
        # Limit of C(r)/r at the origin; avoids subtracting W(r)-W(-r).
        pieces, denom, ops = dp._c1_pieces_ops(i, j)
        return sum(((P[1] if len(P) > 1 else 0.) - rate*P[0])*ops.origin(a, b, rate)
                   for a, b, terms in pieces if a <= 0 < b
                   for rate, P in terms.items()) / denom

    def get_hs_correction(self, r_grid):
        """Closure correction alone, including its nonzero in-core part."""
        r, _ = self._vector(r_grid, "r")
        if np.any(r < 0):
            raise ValueError("radial distances must be nonnegative")
        return np.array([[[self._c1_at(self._hs, rv, i, j) for rv in r]
                          for j in range(self.N)] for i in range(self.N)])

    def get_c1_total(self, r_grid):
        """Physical-tail radial DCF response, finite also at the origin."""
        r, _ = self._vector(r_grid, "r")
        if np.any(r < 0):
            raise ValueError("radial distances must be nonnegative")
        return np.array([[[self._c1_at(self.physical_dp, rv, i, j) for rv in r]
                          for j in range(self.N)] for i in range(self.N)])

    @cached_property
    def _py_origins(self):
        p = self.physical_dp
        result = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                lam = abs(p.lambda_ij[i, j])
                if lam > 0:
                    result[i, j] = p.get_HS_PY(lam/2, i, j)
                else:
                    edges = p.incore_breakpoints(i, j)
                    width = edges[0] if len(edges) else self.R_ij[i, j]
                    # Cubic extrapolation on the first polynomial PY segment.
                    r = width*1e-3*np.arange(1, 5)
                    values = [p.get_HS_PY(rv, i, j) for rv in r]
                    result[i, j] = np.array([4., -6., 4., -1.])@values
        return result

    def get_c_total(self, r_grid, hs_ref="gfmsa"):
        """DCF with fixed GFMSA reference; rejects extra FMT/CS substitution."""
        if hs_ref not in ("gfmsa", "tl95"):
            raise ValueError("GFMSA uses hs_ref='gfmsa'; use physical_dp for PY/FMT")
        r, _ = self._vector(r_grid, "r")
        if np.any(r < 0):
            raise ValueError("radial distances must be nonnegative")
        p = self.physical_dp
        values = np.empty((self.N, self.N, len(r)))
        for i in range(self.N):
            for j in range(self.N):
                for q, rv in enumerate(r):
                    py = (self._py_origins[i, j] if rv == 0 else
                          p.get_HS_PY(rv, i, j) if rv < self.R_ij[i, j] else 0.)
                    values[i, j, q] = py+self._c1_at(p, rv, i, j)
        return values+self.get_hs_correction(r)

    def _h1L(self, dp, s):
        """Density-free Q^-T B Q^-1, valid for absent components too."""
        s = np.asarray(s, dtype=complex)
        M = self._M_vec(s)
        left = np.linalg.inv(np.eye(self.N)-np.swapaxes(M, 1, 2)*self.rho[None, None, :])
        B = np.zeros((len(s), self.N, self.N), dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                for z, amp in zip(dp._b_poles[i][j], dp._b_coeffs_red[i][j]):
                    B[:, i, j] += amp/(2*math.pi*(s+z))
                B[:, i, j] *= np.exp(-s*self.R_ij[i, j])
        return left@B@np.swapaxes(left, 1, 2)

    def get_g0L_laplace(self, s):
        """HS-reference L[r g](s): PY plus the fitted first-order HS tail."""
        sv, scalar = self._vector(s, "s", complex_values=True)
        if np.any(sv.real <= 0):
            raise ValueError("Laplace arguments must have positive real part")
        g0 = self.physical_dp.get_g0L_laplace(sv)
        return self._shape(g0+self._h1L(self._hs, sv), scalar)

    def get_h1L_laplace(self, s):
        """Physical-tail L[r h1](s), with frozen HS calibration."""
        sv, scalar = self._vector(s, "s", complex_values=True)
        if np.any(sv.real <= 0):
            raise ValueError("Laplace arguments must have positive real part")
        return self._shape(self._h1L(self.physical_dp, sv), scalar)

    def get_gL_laplace(self, s):
        """L[r g](s) for PY + HS correction + physical first-order RDF."""
        return self.get_g0L_laplace(s)+self.get_h1L_laplace(s)

    def get_g0ij_contact(self, i=0, j=0):
        """BMCSL contact of the frozen GFMSA HS reference."""
        return self.get_g_contact(i, j, order="zeroth")

    def get_g1ij_contact(self, i=0, j=0):
        """Physical-tail contact response, excluding the HS correction."""
        if not 0 <= i < self.N or not 0 <= j < self.N:
            raise IndexError("species index out of range")
        return self.physical_dp.get_g1ij_contact(i, j)

    def get_g_contact(self, i=0, j=0, order="first"):
        """Exact BMCSL contact plus physical-tail correction; no double count."""
        if not 0 <= i < self.N or not 0 <= j < self.N:
            raise IndexError("species index out of range")
        g0 = self.physical_dp.get_g_contact(i, j, order="zeroth")
        if order == "zeroth":
            return g0
        if order != "first":
            raise ValueError("order must be 'zeroth' or 'first'")
        return g0 + self.physical_dp.get_g1ij_contact(i, j)

    def get_g_laplace(self, r_grid, M=40, tol=1e-12):
        """de Hoog RDF inversion; returns exact core zero and contact limit."""
        r, _ = self._vector(r_grid, "r")
        if np.any(r < 0):
            raise ValueError("radial distances must be nonnegative")
        if not isinstance(M, int) or M < 1 or not 0 < tol < 1:
            raise ValueError("M must be a positive integer and tol must lie in (0, 1)")
        g = np.zeros((self.N, self.N, len(r)))
        for i in range(self.N):
            for j in range(self.N):
                R = self.R_ij[i, j]
                outside = r > R
                if np.any(outside):
                    g[i, j, outside] = self.physical_dp._invert_rL(
                        self.get_gL_laplace, r[outside], i, j, M, tol)
                g[i, j, r == R] = self.get_g_contact(i, j)
        return g

    def get_g_fourier(self, r_grid=None, k_max=1500., n_k=200000, return_r=False):
        """DST RDF inversion using the combined first-order spectrum."""
        result = FMSA_double_prop.get_g_fourier(
            self, r_grid, k_max=k_max, n_k=n_k, return_r=True)
        r, g = result
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

    def beta_energy_per_particle(self):
        """Physical excess beta*U/N; the HS closure tail carries no energy."""
        if self.number_density == 0:
            return 0.
        return -2*math.pi*self.physical_dp.beta*self._energy_kernel()/self.number_density

    def _energy_kernel(self, physical_weight=1.):
        p = self.physical_dp
        total = 0.
        for i in range(self.N):
            for j in range(self.N):
                z = p.z_poles[:, i, j].astype(complex)
                g = (p.get_g0L_laplace(z)+self._h1L(self._hs, z)
                     + physical_weight*self._h1L(p, z))[:, i, j].real
                total += self.rho[i]*self.rho[j]*np.sum(
                    p.k_bare[:, i, j]*np.exp(z.real*self.R_ij[i, j])*g)
        return float(total)

    @property
    def hs_inverse_compressibility(self):
        """BMCSL d(beta P)/d rho at fixed composition (HS reference only)."""
        e, v = self.eta, 1-self.eta
        m1, m2, m3 = (self.x@self.diameters**n for n in (1, 2, 3))
        return float(1/v**2+6*(m1*m2/m3)*e/v**3
                     + (m2**3/m3**2)*e**2*(9-4*e+e*e)/v**4)

    def betaf_hs(self):
        """BMCSL hard-sphere excess free energy density."""
        e, v = self.eta, 1-self.eta
        m1, m2, m3 = (self.x@self.diameters**n for n in (1, 2, 3))
        a, b = m1*m2/m3, m2**3/m3**2
        return float(self.number_density*((b-1)*math.log1p(-e)+3*a*e/v+b*e/v**2))

    def thermodynamics(self, route="energy", **kwargs):
        """Bulk free energy and its derivatives for one explicit route.

        See GFMSAFreeEnergy for the fixed-composition compressibility
        primitive and its distinction from the full species DCF Hessian.
        """
        from gfmsa_free_energy import GFMSAFreeEnergy
        return GFMSAFreeEnergy(self, route=route, **kwargs)

    def betaf_excess(self, route="energy", **kwargs):
        """Excess beta*F/V at fixed hard core and physical tails.

        The default energy route is BMCSL plus coupling-integrated physical
        energy. The HS RDF correction enters the beta-linear term; only the
        physical response enters the beta-quadratic term. 'compressibility'
        instead integrates the actual DCF zero mode along a composition ray.
        The two routes need not agree when physical tails are nonzero.
        The energy route includes the LJ shell term by default (zero for
        hard-core Yukawa input); pass shell='none' to leave it out.
        """
        return self.thermodynamics(route=route, **kwargs).betaf_excess()
