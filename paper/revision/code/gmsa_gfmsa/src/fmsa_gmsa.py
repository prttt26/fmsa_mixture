"""First-order physical-tail response about a complete GMSA HS reference.

    model = physical_dp.with_hs_reference("gmsa")
    # "gfmsa" retains PY + first-order HS correction + first-order physical
    # correction. This module instead differentiates the full BH equations
    # at the calibrated GMSA root, freezing its auxiliary amplitudes/rate.

All derivatives are with respect to a multiplier tau on beta*k_physical,
at fixed density, geometry and rates. The production response uses an
analytic BH Jacobian and analytic observable derivatives, not differences
of nonlinear solutions. get_msa_state(tau) supplies the same closure to the
existing Exact MSA solver for independent comparisons. Its auxiliary tail
is a closure parameter and must not be counted as a physical potential.

No global branch, interval or Lean certification is claimed. The full
reference G-moment extension and RDF derivative are numerical interfaces
pending the open formal bridges recorded in the Lean handoff.
"""

from functools import cached_property
import math

import numpy as np
from scipy.integrate import quad

from fmsa_double_prop import FMSA_double_prop
from gmsa import GMSA
from msa_exact_mix import HCYMix, SolutionMix, _pieces, _qhat_vec, _residuals


def _pieces_direction(st, Dt, Gt, D1, G1):
    """Exact directional derivative of BH's scaled coefficient dictionary."""
    p = _pieces(st, Dt, Gt)
    rho, sig, z = st.rho, st.sigma, st.z
    C1 = np.empty_like(D1)
    W1 = np.empty_like(D1)
    for t, rate in enumerate(z):
        gam1 = -2*math.pi/rate*rho[None, :]*G1[t]*np.exp(-rate*st.SIG)
        edge = np.exp(-rate*st.LAM)
        C1[t] = (gam1*edge)@Dt[t]+(p["gam"][t]*edge)@D1[t]
        W1[t] = 2*math.pi/rate*((G1[t]*rho[None, :])@Dt[t]
                               +(Gt[t]*rho[None, :])@D1[t])
    M1 = np.zeros(st.N)
    N1 = np.zeros(st.N)
    for t, rate in enumerate(z):
        M1 -= np.einsum('l,lj->j', rho,
                        W1[t]+(1+rate*sig)[:, None]*C1[t])/rate
        N1 += np.einsum('l,lj->j', rho,
            (1+rate*st.LAM.T)/rate**2*W1[t]
            + ((1+rate*sig+(rate*sig)**2/2)[:, None]/rate**2
               +st.LAM.T*((1+rate*sig)/rate)[:, None])*C1[t])
    return dict(Ct=C1, Wt=W1, M=M1, Nc=N1,
                A=st.A0*M1-4*st.B0*N1/sig**2,
                qp=st.QP0*M1[None, :]+st.A0[:, None]*N1[None, :])


def _bh_state_direction(st, Dt, Gt, p, D1, G1):
    """Unscaled BH state Jacobian applied to (D1,G1), with K fixed."""
    p1 = _pieces_direction(st, Dt, Gt, D1, G1)
    q = _qhat_vec(st, p, st.z).real
    q1 = _qhat_vec(st, p1, st.z).real
    r1, r2 = np.empty_like(D1), np.empty_like(G1)
    for t, rate in enumerate(st.z):
        ym, zp = np.exp(-rate*st.LAM.T), np.exp(rate*st.LAM.T)
        Y = ym*(np.eye(st.N)-st.rho[:, None]*q[t].T)
        Z = zp*(np.eye(st.N)-st.rho[:, None]*q[t])
        Y1 = -ym*st.rho[:, None]*q1[t].T
        Z1 = -zp*st.rho[:, None]*q1[t]
        r1[t] = D1[t]@Y+Dt[t]@Y1
        source1 = p1["A"][None, :]+rate*p1["qp"]
        for m, zm in enumerate(st.z):
            source1 = source1+rate**2*zm/(rate+zm)*p1["Ct"][m]
        r2[t] = 2*math.pi*(G1[t]@Z+Gt[t]@Z1)-source1/rate**2
    return np.r_[r1.ravel(), r2.ravel()]


class FMSA_GMSA:
    """Frozen full-GMSA reference plus its first physical-tail response.

    Shapes match GFMSA: scalar transforms (N,N), vector transforms (ns,N,N),
    radial data (N,N,nr). ``reference`` may supply a previously fitted GMSA
    at exactly the same hard cores and partial densities. Do not mutate the
    supplied physical_dp or reference after construction.
    """

    def __init__(self, physical_dp, reference=None):
        if not isinstance(physical_dp, FMSA_double_prop):
            raise TypeError("physical_dp must be a FMSA_double_prop instance")
        p = self.physical_dp = physical_dp
        if not math.isfinite(p.beta) or p.beta <= 0:
            raise ValueError("beta must be finite and positive")
        if np.any(~np.isfinite(p.z_poles)) or np.any(p.z_poles <= 0):
            raise ValueError("physical tail rates must be finite and positive")
        if np.any(~np.isfinite(p.k_bare)):
            raise ValueError("physical amplitudes must be finite")
        for arr in (p.z_poles, p.k_bare):
            if not np.allclose(arr, np.swapaxes(arr, 1, 2), rtol=1e-12, atol=0):
                raise ValueError("physical rates and amplitudes must be pair symmetric")
        ref = GMSA.from_fmsa(p) if reference is None else reference
        if not isinstance(ref, GMSA):
            raise TypeError("reference must be a GMSA instance")
        if (ref.N != p.N or not np.array_equal(ref.diameters, p.sigma)
                or not np.allclose(ref.rho, p.rho, rtol=2e-14, atol=0)
                or not np.allclose(ref.R_ij, p.R_ij, rtol=0, atol=1e-12)):
            raise ValueError("reference must have the same additive cores and partial densities")
        self.reference = ref
        self.N = ref.N
        self.diameters, self.rho, self.x = ref.diameters.copy(), ref.rho.copy(), ref.x.copy()
        self.number_density, self.eta = ref.number_density, ref.eta
        self.R_ij = ref.R_ij.copy()
        self.hs_rate, self.hs_k_bare = ref.hs_rate, ref.hs_k_bare.copy()
        self._build_reference_extension()
        self._build_response()

    def _build_reference_extension(self):
        """Add inactive channels with full-reference, rather than PY, Gt."""
        ref, physical = self.reference, self.physical_dp
        rates = [ref.hs_rate]
        directions = [np.zeros((self.N, self.N))]
        # Identical dimensional rates are one channel, even across physical
        # tails or when a physical rate coincides with the HS rate. Pairwise
        # rates become supported matrices in the common-rate BH dictionary.
        for rate in np.unique(physical.z_poles):
            matrix = np.sum(np.where(physical.z_poles == rate,
                                     physical.beta*physical.k_bare, 0.), axis=0)
            if not np.any(matrix):
                continue
            if rate == ref.hs_rate:
                directions[0] += matrix
            else:
                rates.append(float(rate))
                directions.append(matrix)
        self.physical_direction = np.asarray(directions)
        K0 = np.zeros_like(self.physical_direction)
        K0[0] = ref.hs_k_bare
        st = self._state0 = HCYMix(self.diameters.copy(), self.rho.copy(), rates, K0)
        Dt = np.zeros_like(K0)
        Gt = np.empty_like(K0)
        Dt[0], Gt[0] = ref.solution.Dt[0], ref.solution.Gt[0]
        for t, rate in enumerate(st.z[1:], start=1):
            q = ref.get_qhat(rate).real
            Z = np.exp(rate*st.LAM.T)*(np.eye(self.N)-self.rho[:, None]*q)
            src = (ref.solution.p["A"][None, :]+rate*ref.solution.p["qp"])/rate**2
            src = src+ref.hs_rate/(rate+ref.hs_rate)*ref.solution.p["Ct"][0]
            Gt[t] = np.linalg.solve(Z.T, src.T).T/(2*math.pi)
        self.reference_solution = SolutionMix(st, Dt, Gt, _pieces(st, Dt, Gt), 0.)
        u0 = np.r_[Dt.ravel(), Gt.ravel()]
        residual = _residuals(u0, st, np.ones((st.T, 1, 1)), np.ones((st.T, 1, 1)))
        self.reference_extension_residual = float(np.max(abs(residual)))
        if (not np.all(np.isfinite(u0)) or not np.all(np.isfinite(residual))
                or self.reference_extension_residual > max(1e-8, 10*ref._tol)):
            raise ValueError("inactive physical channels do not extend the supplied BH root")

    def _build_response(self):
        st, sol = self._state0, self.reference_solution
        shape = sol.Dt.shape
        block = sol.Dt.size
        dim = 2*block
        J = np.empty((dim, dim))
        for col in range(dim):
            basis = np.zeros(dim)
            basis[col] = 1.
            J[:, col] = _bh_state_direction(st, sol.Dt, sol.Gt, sol.p,
                                            basis[:block].reshape(shape),
                                            basis[block:].reshape(shape))
        forcing = np.r_[(2*math.pi*self.physical_direction/st.z[:, None, None]).ravel(),
                        np.zeros(block)]
        try:
            u1 = np.linalg.solve(J, forcing)
        except np.linalg.LinAlgError as exc:
            raise ValueError("BH state Jacobian is singular at the supplied GMSA reference") from exc
        if np.any(~np.isfinite(u1)):
            raise ValueError("BH state response is not finite")
        self.Dt1, self.Gt1 = u1[:block].reshape(shape), u1[block:].reshape(shape)
        self._p1 = _pieces_direction(st, sol.Dt, sol.Gt, self.Dt1, self.Gt1)
        error = float(np.linalg.norm(J@u1-forcing, ord=np.inf))
        if error > 1e-9*max(1., float(np.linalg.norm(forcing, ord=np.inf))):
            raise ValueError("BH response linear solve missed its residual tolerance")
        self.response_diagnostics = dict(
            dimension=dim, reference_extension_residual=self.reference_extension_residual,
            linear_residual=error, jacobian_condition=float(np.linalg.cond(J)),
            reference_parameters_frozen=True, certified=False)

    def get_msa_state(self, coupling=1.):
        """Exact MSA closure with HS K fixed and physical K multiplied by tau.

        Can be passed directly to msa_exact_mix.solve_bh_mix. Its K includes
        the auxiliary tail: msa_exact_mix.excess_energy on that raw solution
        would count a fictitious potential. This model's energy excludes it.
        """
        if np.ndim(coupling) != 0 or not math.isfinite(float(coupling)):
            raise ValueError("coupling must be a finite real scalar")
        return HCYMix(self.diameters.copy(), self.rho.copy(), self._state0.z.copy(),
                      self._state0.K+float(coupling)*self.physical_direction)

    def at_density(self, rho):
        """Rebuild physical state and refit HS parameters at a new density."""
        new_ref = self.reference.at_density(rho)
        p = self.physical_dp
        new_dp = FMSA_double_prop(
            p.sig_pure, p.eps_pure, new_ref.x, new_ref.number_density, p.tstar,
            n_tails=p.n_tails, z=p._z_input, k=p._k_input,
            eps_ij=p._eps_ij, input_type=p._input_type, d_pure=p._d_pure,
            hs_fmt=p.hs_fmt, verbose=False)
        return type(self)(new_dp, reference=new_ref)

    def get_q1hat(self, s):
        """Derivative of the density-free Baxter kernel at frozen HS K/rate."""
        sv, scalar = GMSA._vector(s, "s", True)
        q = _qhat_vec(self._state0, self._p1, sv)
        return q[0] if scalar else q

    def get_c1hat_k(self, k):
        """Physical DCF derivative, including zero and absent species."""
        wave, scalar = GMSA._vector(k, "k")
        q0 = self.reference.get_qhat(1j*wave)
        q1 = self.get_q1hat(1j*wave)
        qt0, qt1 = np.swapaxes(q0.conj(), 1, 2), np.swapaxes(q1.conj(), 1, 2)
        c1 = (q1+qt1-(q1*self.rho[None, None, :])@qt0
              -(q0*self.rho[None, None, :])@qt1).real
        return c1[0] if scalar else c1

    def get_chat_k(self, k):
        return self.reference.get_chat_k(k)+self.get_c1hat_k(k)

    def get_h1hat_k(self, k):
        """Derivative of full-reference OZ: L_ref c1 R_ref."""
        c0, c1 = self.reference.get_chat_k(k), self.get_c1hat_k(k)
        left = np.linalg.inv(np.eye(self.N)-c0*self.rho[None, :])
        return left@c1@np.swapaxes(left, -1, -2)

    def get_hhat_k(self, k, order="first"):
        if order == "first":
            return self.reference.get_hhat_k(k)+self.get_h1hat_k(k)
        if order == "oz":
            c = self.get_chat_k(k)
            return np.linalg.solve(np.eye(self.N)-c*self.rho[None, :], c)
        raise ValueError("order must be 'first' or 'oz'")

    def get_S_k(self, k, order="first"):
        return np.eye(self.N)+np.sqrt(self.rho[:, None]*self.rho[None, :])*self.get_hhat_k(k, order)

    def get_g0ij_contact(self, i=0, j=0):
        return self.reference.get_g_contact(i, j)

    def get_g1ij_contact(self, i=0, j=0):
        if not 0 <= i < self.N or not 0 <= j < self.N:
            raise IndexError("species index out of range")
        num = self._p1["qp"][i, j]+np.sum(self._state0.z*self._p1["Ct"][:, i, j])
        return float(num/(2*math.pi*self.R_ij[i, j]))

    def get_g_contact(self, i=0, j=0, order="first"):
        if order == "zeroth":
            return self.get_g0ij_contact(i, j)
        if order == "first":
            return self.get_g0ij_contact(i, j)+self.get_g1ij_contact(i, j)
        raise ValueError("order must be 'zeroth' or 'first'")

    def get_g0L_laplace(self, s):
        return self.reference.get_gL_laplace(s)

    def get_h1L_laplace(self, s):
        """Differentiate G F=E, retaining the full G0*P*Q1 renewal term."""
        sv, scalar = GMSA._vector(s, "s", True)
        if np.any(sv.real <= 0):
            raise ValueError("Laplace arguments must have positive real part")
        s3 = sv[:, None, None]
        p1, z = self._p1, self._state0.z
        source1 = p1["A"][None, None, :]/s3**2+p1["qp"][None]/s3
        for t, rate in enumerate(z):
            source1 = source1+rate*p1["Ct"][t][None]/(s3+rate)
        source1 = np.exp(-s3*self.R_ij)*source1/(2*math.pi)
        q0 = self.reference.get_qhat(sv)
        q1 = self.get_q1hat(sv)
        g0 = self.reference.get_gL_laplace(sv)
        factor = np.eye(self.N)-self.rho[None, :, None]*q0
        rhs = source1+(g0*self.rho[None, None, :])@q1
        g1 = np.swapaxes(np.linalg.solve(np.swapaxes(factor, 1, 2),
                                        np.swapaxes(rhs, 1, 2)), 1, 2)
        return g1[0] if scalar else g1

    def get_gL_laplace(self, s):
        return self.get_g0L_laplace(s)+self.get_h1L_laplace(s)

    def get_g_laplace(self, r_grid, M=48, tol=1e-12):
        # The inversion driver uses this object's total transform/contact.
        return GMSA.get_g_laplace(self, r_grid, M=M, tol=tol)

    _vector = staticmethod(GMSA._vector)

    @property
    def _py_dp(self):
        return self.reference._py_dp

    def get_g_fourier(self, r_grid=None, k_max=1500., n_k=200000, return_r=False):
        r, g = FMSA_double_prop.get_g_fourier(
            self, r_grid, k_max=k_max, n_k=n_k, return_r=True)
        for i in range(self.N):
            for j in range(self.N):
                g[i, j, r < self.R_ij[i, j]] = 0.
                g[i, j, r == self.R_ij[i, j]] = self.get_g_contact(i, j)
        return (r, g) if return_r else g

    def get_g_r(self, r_grid=None, route="laplace", **kwargs):
        return GMSA.get_g_r(self, r_grid, route=route, **kwargs)

    def _Q(self, p, i, j, t, derivative=0):
        """Full or differentiated real Baxter kernel for common finite rates."""
        z = self._state0.z
        lam, R = (self.diameters[j]-self.diameters[i])/2, self.R_ij[i, j]
        if t < lam:
            return 0.
        ew = p["Wt"][:, i, j]*np.exp(-z*(t-lam))
        if t < R:
            if derivative == 0:
                return p["qp"][i, j]*(t-R)+p["A"][j]*(t-R)**2/2+sum(ew+p["Ct"][:, i, j])
            if derivative == 1:
                return p["qp"][i, j]+p["A"][j]*(t-R)-sum(z*ew)
            return p["A"][j]+sum(z*z*ew)
        tail = ew+p["Ct"][:, i, j]*np.exp(-z*(t-R))
        return float(np.sum((-z)**derivative*tail))

    def _convolution_direction(self, i, j, r, derivative=1):
        p0, p1, z = self.reference_solution.p, self._p1, self._state0.z
        total = 0.
        for l in range(self.N):
            if self.rho[l] == 0:
                continue
            lower = (self.diameters[l]-self.diameters[j])/2
            tail_start = max(self.R_ij[i, l]-r, self.R_ij[j, l])
            edges = sorted({lower, tail_start, max(lower, self.R_ij[i, l]-r),
                            max(lower, self.R_ij[j, l])})
            def integrand(t):
                return (self._Q(p1, i, l, t+r, derivative)*self._Q(p0, j, l, t)
                        +self._Q(p0, i, l, t+r, derivative)*self._Q(p1, j, l, t))
            value = sum(quad(integrand, lo, hi, epsabs=1e-10, epsrel=1e-10)[0]
                        for lo, hi in zip(edges[:-1], edges[1:]) if hi > lo)
            def tail_amplitudes(p, a, at):
                lam = (self.diameters[l]-self.diameters[a])/2
                return (p["Wt"][:, a, l]*np.exp(-z*(at-lam))
                        +p["Ct"][:, a, l]*np.exp(-z*(at-self.R_ij[a, l])))
            i0, i1 = tail_amplitudes(p0, i, tail_start+r), tail_amplitudes(p1, i, tail_start+r)
            j0, j1 = tail_amplitudes(p0, j, tail_start), tail_amplitudes(p1, j, tail_start)
            value += np.sum(((-z)**derivative)[:, None]*(i1[:, None]*j0[None, :]
                              +i0[:, None]*j1[None, :])/(z[:, None]+z[None, :]))
            if derivative == 2:
                jump0 = -sum(z*p0["Ct"][:, i, l])-p0["qp"][i, l]
                jump1 = -sum(z*p1["Ct"][:, i, l])-p1["qp"][i, l]
                value += (jump1*self._Q(p0, j, l, self.R_ij[i, l])
                          +jump0*self._Q(p1, j, l, self.R_ij[i, l]))
            total += self.rho[l]*value
        return total

    @cached_property
    def _c1_origins(self):
        values = np.empty((self.N, self.N))
        for a in range(self.N):
            for b in range(a, self.N):
                i, j = (a, b) if self.diameters[a] >= self.diameters[b] else (b, a)
                value = (-self._Q(self._p1, i, j, 0., 2)
                         +self._convolution_direction(i, j, 0., 2))/(2*math.pi)
                values[a, b] = values[b, a] = value
        return values

    def get_c1_total(self, r_grid):
        """Radial physical DCF response, including core and origin limits."""
        r, _ = GMSA._vector(r_grid, "r")
        if np.any(r < 0):
            raise ValueError("radial distances must be nonnegative")
        c = np.empty((self.N, self.N, len(r)))
        for a in range(self.N):
            for b in range(a, self.N):
                i, j = (a, b) if self.diameters[a] >= self.diameters[b] else (b, a)
                R = self.R_ij[i, j]
                for n, rv in enumerate(r):
                    if rv == 0:
                        value = self._c1_origins[a, b]
                    elif rv >= R:
                        value = sum(self.physical_direction[:, i, j]
                                    *np.exp(-self._state0.z*(rv-R)))/rv
                    else:
                        value = (-self._Q(self._p1, i, j, rv, 1)
                                 +self._convolution_direction(i, j, rv))/(2*math.pi*rv)
                    c[a, b, n] = c[b, a, n] = value
        return c

    def get_c_total(self, r_grid):
        return self.reference.get_c_total(r_grid)+self.get_c1_total(r_grid)

    @property
    def hs_inverse_compressibility(self):
        return self.reference.hs_inverse_compressibility

    def betaf_hs(self):
        return self.reference.betaf_excess()

    def _energy_kernel(self, physical_weight=1.):
        p, total = self.physical_dp, 0.
        for i in range(self.N):
            for j in range(self.N):
                rates = p.z_poles[:, i, j]
                g = (self.get_g0L_laplace(rates)
                     +physical_weight*self.get_h1L_laplace(rates))[:, i, j].real
                total += self.rho[i]*self.rho[j]*np.sum(
                    p.k_bare[:, i, j]*np.exp(rates*self.R_ij[i, j])*g)
        return float(total)

    @cached_property
    def _lj_shell_terms(self):
        from gfmsa_free_energy import lj_shell_terms
        return lj_shell_terms(self)

    def betaf_shell(self, shell="lj"):
        """LJ shell beta F/V over [R_ij, sigma_ij], at this model's contacts.

        Production's frozen-contact shell term (see gfmsa_free_energy): g0(R)
        is the GMSA contact (BMCSL by calibration) and g1(R) this model's own
        response, not the PY one. Zero for hard-core Yukawa input
        (sigma_ij = R_ij) and for shell='none'.
        """
        if shell not in ("lj", "none"):
            raise ValueError("shell must be 'lj' or 'none'")
        return sum(self._lj_shell_terms) if shell == "lj" else 0.

    def beta_energy_per_particle(self, shell="lj"):
        """Physical beta U/N plus the shell's; fitted HS closure amplitudes carry no energy."""
        if shell not in ("lj", "none"):
            raise ValueError("shell must be 'lj' or 'none'")
        if self.number_density == 0:
            return 0.
        first, second = self._lj_shell_terms if shell == "lj" else (0., 0.)
        return (-2*math.pi*self.physical_dp.beta*self._energy_kernel()
                + first+2*second)/self.number_density

    def betaf_excess(self, shell="lj"):
        """BMCSL HS beta F/V plus the physical coupling integral and the LJ shell.

        Density/geometry and the calibrated HS tail are fixed in the coupling
        integral. The physical RDF response has weight 1/2. Agreement with a
        compressibility route or a full species Hessian is not imposed.
        shell='none' leaves out the LJ shell term (betaf_shell).
        """
        return (self.betaf_hs()-2*math.pi*self.physical_dp.beta*self._energy_kernel(.5)
                + self.betaf_shell(shell))
