"""GFMSA bulk excess free energies, in beta*F_ex/V units.

At fixed composition x, define C(n)=sum_ij x_i*x_j*c_phys1_hat(0;n).
The calibrated HS DCF gives K_HS=d(beta*P_BMCSL)/dn. Consequently

    f_C = f_BMCSL - integral_0^n (n-t)*C(t) dt,
    beta*P_C = beta*P_BMCSL - integral_0^n t*C(t) dt,
    d(beta*P_C)/dn = K_HS - n*C(n).

Vacuum normalization is f_ex(0)=f_ex'(0)=0 on each composition ray.
This constructs a scalar free energy throughout the density cone, but does
NOT identify its species Hessian with the GFMSA DCF matrix. In particular
the first-order RDF's S(0) is not inverted to compute this compressibility.

For the energy route g_lambda=g_HS_GFMSA+lambda*g_phys1, with fixed HS
closure parameters during physical coupling. Coupling integration gives
f_E=f_BMCSL+A+B/2, where A=beta*integral(u*g_HS) and
B=beta*integral(u*g_phys1). Only the true physical u contributes energy.
Both routes refit the auxiliary HS tail when density/composition changes.
Temperature derivatives refer to fixed diameters/rates/bare potentials;
they do not include a temperature-dependent BH-diameter trajectory.

LJ shell (energy route only, shell='lj', its default there). For an LJ input
the BH diameter R_ij < sigma_ij, and A, B integrate the Yukawa fit from R_ij,
i.e. over the repulsive shell [R_ij, sigma_ij] as well. The production energy
route (FMSAFreeEnergy, incore='lj') removes that shell with g frozen at
contact; the same term is added here, taken from that one definition:

    f_shell = -8*pi*beta sum rho_i rho_j eps_ij R_ij^3 LJ_ij g0_ij(R_ij)
              -4*pi*beta sum rho_i rho_j eps_ij R_ij^3 LJ_ij g1_ij(R_ij),
    LJ_ij = s^12/9 - s^6/3 + 2 s^3/9,  s = sigma_ij/R_ij.

g0(R) is the BMCSL contact, which is the GFMSA reference's own contact by
calibration; g1(R) is the physical response, which the HS correction does not
touch. So with the shell, f_E is the production energy route with g_PY
replaced by g_HS_GFMSA in the first-order Yukawa term, and nothing else.
LJ_ij(1) = 0 exactly, so the shell vanishes for hard-core Yukawa input
(input_type='bh', sigma_ij = R_ij). The compressibility route has no
counterpart (as in production): its C(n) keeps the Yukawa fit on the shell.
"""

from functools import cached_property
import math

import numpy as np
from scipy.integrate import quad_vec


def lj_shell_terms(model):
    """LJ shell (first order, second order) beta*F/V, at the model's own contacts.

    The formula is production's (FMSAFreeEnergy._betaf1_inner, _betaf2_lj,
    contact weight); only the contacts are read from ``model``, which needs
    physical_dp, rho, R_ij, get_g0ij_contact and get_g1ij_contact. Order 1 is
    beta-linear and order 2 beta-quadratic at fixed diameters.
    """
    from types import SimpleNamespace
    from fmsa_free_energy import FMSAFreeEnergy
    p = model.physical_dp
    view = SimpleNamespace(rho=model.rho, radii=model.R_ij, sig_mix=p.sig_mix,
                           eps_mix=p.eps_mix, beta=p.beta,
                           get_g0ij_contact=model.get_g0ij_contact,
                           get_g1ij_contact=model.get_g1ij_contact)
    fe = FMSAFreeEnergy(view)
    return float(fe._betaf1_inner()), float(fe._betaf2_lj())


class GFMSAFreeEnergy:
    """Thermodynamics derived from the selected GFMSA scalar free energy.

    route is 'energy' or 'compressibility'. Pressure is beta*P, chemical
    potentials are beta*mu (excess unless explicitly named total), and
    inverse compressibility is d(beta*P)/dn at fixed composition.
    quadrature tolerances apply to the density-ray C integral, in volume
    units. Finite-difference delta is relative to total number density.
    shell is 'lj' (energy route default) or 'none' (compressibility default,
    and the only value that route accepts); see the module docstring.
    """

    def __init__(self, model, route="energy", *, shell=None, epsabs=1e-10, epsrel=1e-9):
        from gfmsa import GFMSA
        if not isinstance(model, GFMSA):
            raise TypeError("model must be a GFMSA instance")
        if route not in ("energy", "compressibility"):
            raise ValueError("route must be 'energy' or 'compressibility'")
        if shell is None:
            shell = "lj" if route == "energy" else "none"
        if shell not in ("lj", "none"):
            raise ValueError("shell must be 'lj' or 'none'")
        if shell == "lj" and route == "compressibility":
            raise ValueError("the LJ shell term exists on the energy route only")
        if not math.isfinite(epsabs) or epsabs <= 0:
            raise ValueError("epsabs must be finite and positive")
        if not math.isfinite(epsrel) or epsrel <= 0:
            raise ValueError("epsrel must be finite and positive")
        self.model, self.route, self.shell = model, route, shell
        self.epsabs, self.epsrel = float(epsabs), float(epsrel)

    def _at_density(self, density):
        return type(self)(self.model.at_density(density, lazy_radial=True), self.route,
                          shell=self.shell, epsabs=self.epsabs, epsrel=self.epsrel)

    @staticmethod
    def _physical_zero_contraction(model):
        """Finite WH zero coefficient, equal to the actual radial DCF moment.

        h1_hat(0)=-4*pi*partial_s H1L(0), H1L=L*B*L.T,
        L=(I-M.T*rho)^-1. Linear OZ undressing gives c1_hat(0).
        No inverse component densities or radial reconstruction is needed.
        """
        p, x, rho = model.physical_dp, model.x, model.rho
        m0, m1 = model._zero_M()
        left = np.linalg.inv(np.eye(model.N)-m0.T*rho[None, :])
        left1 = left@(m1.T*rho[None, :])@left
        b0, b1 = np.zeros((model.N, model.N)), np.zeros((model.N, model.N))
        for i in range(model.N):
            for j in range(model.N):
                z, b = p._b_poles[i][j], p._b_coeffs_red[i][j]/(2*math.pi)
                b0[i, j] = np.sum(b/z)
                b1[i, j] = -np.sum(b*(model.R_ij[i, j]/z+1/z**2))
        h1prime = left1@b0@left.T+left@b1@left.T+left@b0@left1.T
        cpy = m0+m0.T-(m0*rho[None, :])@m0.T
        factor = np.eye(model.N)-cpy*rho[None, :]
        return float(-4*math.pi*x@factor@h1prime@factor.T@x)

    @cached_property
    def _density_integrals(self):
        """n^2 * [integral(1-lambda)*C, integral lambda*C]."""
        m = self.model
        if m.number_density == 0 or not np.any(m.physical_dp.k_bare):
            return np.zeros(2)

        def integrand(lam):
            node = m.at_density(lam*m.number_density, lazy_radial=True)
            c = self._physical_zero_contraction(node)
            return c*np.array([1-lam, lam])

        value, error = quad_vec(integrand, 0., 1., epsabs=self.epsabs,
                                epsrel=self.epsrel, full_output=False)
        if not np.all(np.isfinite(value)) or not math.isfinite(error):
            raise ArithmeticError("non-finite compressibility density integral")
        # quad_vec returns an error estimate even if its refinement limit is
        # exhausted; do not silently use an unconverged free energy.
        target = max(self.epsabs, self.epsrel*np.linalg.norm(value))
        if error > target:
            raise ArithmeticError("compressibility density integral did not converge")
        return m.number_density**2*value

    @cached_property
    def _shell_terms(self):
        """LJ shell (first order, second order) in beta*F/V; zero for shell='none'.

        GFMSA's contacts are the physical DP's (BMCSL g0, physical g1), so
        this is production's shell term to the bit. Order 1 is beta-linear
        and order 2 beta-quadratic at fixed diameters, like A and B/2.
        """
        if self.shell == "none":
            return 0., 0.
        return lj_shell_terms(self.model)

    def betaf_hs(self):
        return self.model.betaf_hs()

    def betaf_shell(self):
        """LJ shell beta*F/V included in betaf_excess (0 for shell='none')."""
        return sum(self._shell_terms)

    def betaf_excess(self):
        """Excess beta*F/V, with no fictitious auxiliary-tail energy."""
        m = self.model
        if self.route == "energy":
            return (m.betaf_hs()-2*math.pi*m.physical_dp.beta*m._energy_kernel(.5)
                    + self.betaf_shell())
        return float(m.betaf_hs()-self._density_integrals[0])

    def get_free_energy(self):
        """Alias for excess beta*F/V (matching the existing FMSA convention)."""
        return self.betaf_excess()

    def get_total_free_energy_density(self):
        """Ideal + excess beta*F/V; thermal wavelengths set to one."""
        rho = self.model.rho
        occupied = rho > 0
        return float(np.sum(rho[occupied]*(np.log(rho[occupied])-1))
                     + self.betaf_excess())

    def _hs_pressure(self):
        m = self.model
        e, v = m.eta, 1-m.eta
        m1, m2, m3 = (m.x@m.diameters**j for j in (1, 2, 3))
        a, b = m1*m2/m3, m2**3/m3**2
        return m.number_density*(1/v+3*a*e/v**2+b*e**2*(3-e)/v**3)

    def _density_step(self, delta, component=None):
        if not math.isfinite(delta) or not 0 < delta < .05:
            raise ValueError("delta must be finite and in (0, .05)")
        m = self.model
        volume = math.pi*m.diameters**3/6
        density_scale = m.number_density or 1/(m.x@volume)
        packing_slope = m.x@volume if component is None else volume[component]
        return min(delta*density_scale, (1-m.eta)/(8*packing_slope))

    def _density_derivative(self, order, delta, component=None):
        """Fourth-order density derivative, refitting HS at every node."""
        if self.model.number_density == 0:
            return 0.
        h = self._density_step(delta, component)
        base = self.model.number_density if component is None else self.model.rho[component]
        if base >= 2*h:
            offsets = np.arange(-2, 3)
            weights = ([1, -8, 0, 8, -1] if order == 1
                       else [-1, 16, -30, 16, -1])
        else:
            offsets = np.arange(5)
            weights = ([-25, 48, -36, 16, -3] if order == 1
                       else [35, -104, 114, -56, 11])
        values = []
        for offset in offsets:
            if offset == 0:
                values.append(self.betaf_excess())
                continue
            density = self.model.number_density+offset*h
            if component is not None:
                density = self.model.rho.copy()
                density[component] += offset*h
            values.append(self._at_density(density).betaf_excess())
        return float(np.dot(weights, values)/(12*h**order))

    def get_pressure(self, delta=1e-4):
        """beta*P = n + n*partial_n f_ex - f_ex, at fixed x.

        The compressibility route uses the exact derivative identity of its
        density primitive; the energy route uses fourth-order differences.
        """
        self._density_step(delta)
        n = self.model.number_density
        if self.route == "compressibility":
            return float(self._hs_pressure()-self._density_integrals[1])
        return float(n+n*self._density_derivative(1, delta)-self.betaf_excess())

    def get_inverse_compressibility(self, delta=3e-4):
        """d(beta*P)/dn = 1+n*partial_n^2 f_ex, at fixed composition.

        Negative curvature is retained (possible instability); no OZ pole
        resummation or inversion of first-order S(0) is used here.
        """
        self._density_step(delta)
        m = self.model
        if self.route == "compressibility":
            return m.hs_inverse_compressibility-m.number_density*self._physical_zero_contraction(m)
        return float(1+m.number_density*self._density_derivative(2, delta))

    def get_chemical_potentials(self, delta=1e-4):
        """Excess beta*mu_i from this scalar f_ex, including refitting.

        At an empty component use a fourth-order forward derivative. These
        species derivatives do not assert a DCF Hessian identification.
        """
        self._density_step(delta)
        return np.array([self._density_derivative(1, delta, i)
                         for i in range(self.model.N)])

    def get_chemical_potentials_total(self, delta=1e-4):
        """Total beta*mu_i; -infinity for an absent ideal component."""
        with np.errstate(divide="ignore"):
            return self.get_chemical_potentials(delta)+np.log(self.model.rho)

    def beta_energy_per_particle(self):
        """Energy DERIVED from the selected free energy at fixed hard core.

        The energy route gives the direct physical-potential/RDF energy plus
        the shell's (order 1 + 2 * order 2). Since the compressibility
        correction is beta-linear, its beta*U_ex/N equals (f_C-f_HS)/n and is
        generally different.
        """
        m = self.model
        if m.number_density == 0:
            return 0.
        if self.route == "energy":
            first, second = self._shell_terms
            return m.beta_energy_per_particle()+(first+2*second)/m.number_density
        return float(-self._density_integrals[0]/m.number_density)

    def diagnostics(self, delta=3e-4):
        """Explicit route gap and independent structural curvature target."""
        m = self.model
        energy = (self if self.route == "energy" else
                  type(self)(m, "energy", epsabs=self.epsabs, epsrel=self.epsrel))
        compress = (self if self.route == "compressibility" else
                    type(self)(m, "compressibility", epsabs=self.epsabs, epsrel=self.epsrel))
        fe, fc = energy.betaf_excess(), compress.betaf_excess()
        target = compress.get_inverse_compressibility()
        selected = target if self.route == "compressibility" else self.get_inverse_compressibility(delta)
        return {"route": self.route, "shell": energy.shell, "betaf_lj_shell": energy.betaf_shell(),
                "betaf_energy": fe, "betaf_compressibility": fc,
                "betaf_energy_minus_compressibility": fe-fc,
                "inverse_compressibility": selected,
                "dcf_inverse_compressibility": target,
                "inverse_compressibility_minus_dcf": selected-target,
                "physical_rdf_beta_energy_per_particle": m.beta_energy_per_particle(),
                "derived_beta_energy_per_particle": self.beta_energy_per_particle(),
                "species_hessian_equals_dcf_claimed": False, "certified": False}
