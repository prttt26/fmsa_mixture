"""Python script of FMSA using step-wise polynomial (with L-transforms)."""

from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
#from math import comb
from scipy.special import factorial,comb
n_terms = 6
n_terms_b_s = 8
n_species = 2
n_tails = 2
filename = "corrpoly"
#all_terms = np.arange(n_terms)
# GCMC data for binary LJ mixture (sig=[1,2], eps=[1,0.5], x=[0.25,0.75], rhostar=0.139, tstar=1.0).
# Columns: r/sigma_ij, c11, c12, c22.
# ⚠ Column 0 is the PER-PAIR REDUCED separation r/σ_ij (all three pairs reach
#   contact at col0 ≈ 1.0).  To get absolute r for pair (i,j), multiply by
#   sig_mix[i,j] = [[1.0, 1.5], [1.5, 2.0]].  Reading col 0 as absolute r is a
#   recurring bug (breaks c12/c22 axes and under-counts ĉ₁₂/ĉ₂₂ by σ_ij³).
#   Correct GCMC references: ĉ₁₁ = +1.990, ĉ₁₂ = −22.07, ĉ₂₂ = −100.8.
# Generated from the full LJ potential; FMSA uses a 2YK approximation to LJ, so there is a
# fundamental ceiling on how well any FMSA variant can match this data.  The tighter benchmark
# for method development is agreement with FMSA_pure (exact single-component MSA), not GCMC.
MC_DATA = np.array(
    [
        [0.025, -8.9462, -7.2522, -27.7580],
        [0.050, -8.8774, -7.1518, -27.4832],
        [0.075, -8.7585, -7.1425, -27.0438],
        [0.100, -8.6289, -7.1534, -26.4750],
        [0.125, -8.4574, -7.1785, -25.8072],
        [0.150, -8.2776, -7.2121, -25.0704],
        [0.175, -8.0708, -7.2528, -24.2858],
        [0.200, -7.8610, -7.2961, -23.4720],
        [0.225, -7.6366, -7.3391, -22.6412],
        [0.250, -7.4130, -7.3776, -21.7995],
        [0.275, -7.1829, -7.4020, -20.9548],
        [0.300, -6.9560, -7.4102, -20.1110],
        [0.325, -6.7274, -7.3944, -19.2702],
        [0.350, -6.5033, -7.3542, -18.4355],
        [0.375, -6.2804, -7.2792, -17.6068],
        [0.400, -6.0626, -7.1781, -16.7880],
        [0.425, -5.8475, -7.0478, -15.9792],
        [0.450, -5.6374, -6.8952, -15.1815],
        [0.475, -5.4311, -6.7168, -14.3948],
        [0.500, -5.2296, -6.5228, -13.6220],
        [0.525, -5.0324, -6.3135, -12.8642],
        [0.550, -4.8398, -6.0934, -12.1214],
        [0.575, -4.6517, -5.8622, -11.3938],
        [0.600, -4.4681, -5.6255, -10.6850],
        [0.625, -4.2893, -5.3843, -9.9933],
        [0.650, -4.1144, -5.1404, -9.3210],
        [0.675, -3.9445, -4.8945, -8.6676],
        [0.700, -3.7787, -4.6491, -8.0355],
        [0.725, -3.6180, -4.4054, -7.4246],
        [0.750, -3.4612, -4.1638, -6.8353],
        [0.775, -3.3096, -3.9258, -6.2678],
        [0.800, -3.1619, -3.6920, -5.7237],
        [0.825, -3.0193, -3.4639, -5.2031],
        [0.850, -2.8808, -3.2409, -4.7049],
        [0.875, -2.7474, -3.0205, -4.2091],
        [0.900, -2.6120, -2.7927, -3.6329],
        [0.925, -2.4369, -2.4299, -2.8514],
        [0.950, -2.0362, -1.9338, -1.8874],
        [0.975, -1.2603, -1.1926, -0.9108],
        [1.000, -0.2598, -0.4286, -0.1226],
        [1.025, 0.9055, 0.1923, 0.4205],
        [1.050, 1.7495, 0.6705, 0.7377],
        [1.075, 2.2873, 0.8846, 0.8955],
        [1.100, 2.3952, 1.0043, 0.9440],
        [1.125, 2.5324, 1.0066, 0.9608],
        [1.150, 2.2829, 0.9527, 0.9424],
        [1.175, 2.1669, 0.8851, 0.9265],
        [1.200, 1.7854, 0.7993, 0.8852],
        [1.225, 1.5460, 0.7239, 0.8551],
        [1.250, 1.3869, 0.6519, 0.8319],
        [1.275, 1.1083, 0.5931, 0.8110],
        [1.300, 1.0119, 0.5382, 0.7891],
        [1.325, 0.8627, 0.4895, 0.7656],
        [1.350, 0.7066, 0.4546, 0.7421],
        [1.375, 0.6374, 0.4145, 0.7199],
        [1.400, 0.5842, 0.3855, 0.6979],
        [1.425, 0.4406, 0.3659, 0.6713],
        [1.450, 0.4155, 0.3376, 0.6509],
        [1.475, 0.3671, 0.3141, 0.6217],
        [1.500, 0.3156, 0.2936, 0.5930],
        [1.525, 0.3191, 0.2712, 0.5713],
        [1.550, 0.3162, 0.2524, 0.5460],
        [1.575, 0.2725, 0.2397, 0.5215],
        [1.600, 0.2647, 0.2255, 0.4986],
        [1.625, 0.2237, 0.2127, 0.4710],
        [1.650, 0.1642, 0.1998, 0.4495],
        [1.675, 0.1963, 0.1881, 0.4267],
        [1.700, 0.1854, 0.1725, 0.4089],
        [1.725, 0.1457, 0.1657, 0.3955],
        [1.750, 0.1498, 0.1545, 0.3783],
        [1.775, 0.1115, 0.1420, 0.3638],
        [1.800, 0.1509, 0.1358, 0.3443],
        [1.825, 0.1660, 0.1261, 0.3312],
        [1.850, 0.1512, 0.1204, 0.3112],
        [1.875, 0.1267, 0.1191, 0.2924],
        [1.900, 0.1288, 0.1119, 0.2740],
        [1.925, 0.1130, 0.1049, 0.2564],
        [1.950, 0.0915, 0.1042, 0.2339],
        [1.975, 0.1190, 0.0980, 0.2196],
        [2.000, 0.1084, 0.0977, 0.2041],
        [2.025, 0.0964, 0.0895, 0.1852],
        [2.050, 0.0878, 0.0853, 0.1728],
        [2.075, 0.0680, 0.0788, 0.1555],
        [2.100, 0.0592, 0.0777, 0.1451],
        [2.125, 0.0491, 0.0679, 0.1325],
        [2.150, 0.0389, 0.0635, 0.1177],
        [2.175, 0.0580, 0.0634, 0.1064],
        [2.200, 0.0744, 0.0596, 0.0969],
        [2.225, 0.0640, 0.0516, 0.0923],
        [2.250, 0.0643, 0.0424, 0.0797],
        [2.275, 0.0448, 0.0420, 0.0721],
        [2.300, 0.0419, 0.0376, 0.0631],
        [2.325, 0.0357, 0.0301, 0.0569],
        [2.350, 0.0372, 0.0299, 0.0494],
        [2.375, 0.0320, 0.0272, 0.0426],
        [2.400, 0.0553, 0.0229, 0.0398],
        [2.425, 0.0371, 0.0234, 0.0325],
        [2.450, 0.0508, 0.0210, 0.0290],
        [2.475, 0.0300, 0.0180, 0.0222],
        [2.500, 0.0300, 0.0121, 0.0194],
    ],
    dtype=float,
)



# ---------------------------------------------------------------------------
# Analytical Fourier-transform helpers for get_c0_fmtHS_ref_fourier
# ---------------------------------------------------------------------------

def _rn_sin_antideriv(n: int, r: float, k: float) -> float:
    """Antiderivative of r^n sin(kr) evaluated at r  (k ≠ 0, n >= 0)."""
    sk = np.sin(k * r)
    ck = np.cos(k * r)
    if n == 0:
        return -ck / k
    if n == 1:
        return sk / k**2 - r * ck / k
    # Recurrence: I_n = n*r^(n-1)*sk/k^2 - r^n*ck/k - n*(n-1)/k^2 * I_{n-2}
    # Step through the chain with the same parity as n (0,2,4,... or 1,3,5,...).
    start = n % 2
    I = -ck / k if start == 0 else sk / k**2 - r * ck / k
    for m in range(start + 2, n + 1, 2):
        I = m * r**(m - 1) * sk / k**2 - r**m * ck / k - m * (m - 1) / k**2 * I
    return I


def _rn_sin_int(n: int, k: float, a: float, b: float) -> float:
    """∫_a^b r^n sin(kr) dr  (k ≠ 0)."""
    return _rn_sin_antideriv(n, b, k) - _rn_sin_antideriv(n, a, k)


def _exp_sin_antideriv(z: float, r: float, k: float) -> float:
    """Antiderivative of exp(zr) sin(kr)."""
    D = z**2 + k**2
    return np.exp(z * r) * (z * np.sin(k * r) - k * np.cos(k * r)) / D


def _exp_sin_int(z: float, k: float, a: float, b: float) -> float:
    """∫_a^b exp(zr) sin(kr) dr  (k ≠ 0)."""
    return _exp_sin_antideriv(z, b, k) - _exp_sin_antideriv(z, a, k)


def _r_exp_sin_antideriv(z: float, r: float, k: float) -> float:
    """Antiderivative of r exp(zr) sin(kr).

    Derived via Im[∫ r exp((z+ik)r) dr] = Im[exp((z+ik)r)(r/(z+ik) - 1/(z+ik)²)].
    """
    D = z**2 + k**2
    return np.exp(z * r) / D**2 * (
        k * (2.0*z - r*D) * np.cos(k * r)
        + (r*z*D - (z**2 - k**2)) * np.sin(k * r)
    )


def _r_exp_sin_int(z: float, k: float, a: float, b: float) -> float:
    """∫_a^b r exp(zr) sin(kr) dr  (k ≠ 0)."""
    return _r_exp_sin_antideriv(z, b, k) - _r_exp_sin_antideriv(z, a, k)


def _r_exp_antideriv(z: float, r: float) -> float:
    """Antiderivative of r exp(zr)  (for k=0 integrals)."""
    return np.exp(z * r) * (r / z - 1.0 / z**2)


def _r2_exp_antideriv(z: float, r: float) -> float:
    """Antiderivative of r² exp(zr)  (for k=0 integrals)."""
    return np.exp(z * r) * (r**2 / z - 2.0*r / z**2 + 2.0 / z**3)


def _stehfest_coeffs(N: int) -> np.ndarray:
    """Stehfest V-coefficients for N-term numerical Laplace inversion (N even).

    V_n = (-1)^(n+M) * sum_{k=ceil(n/2)}^{min(n,M)}
                k^M * (2k)! / [(M-k)! k! (k-1)! (n-k)! (2k-n)!]
    where M = N/2.
    """
    M = N // 2
    V = np.zeros(N, dtype=float)
    for n in range(1, N + 1):
        s = 0.0
        for k in range((n + 1) // 2, min(n, M) + 1):
            num = (k ** M
                    * int(factorial(2 * k, exact=True)))
            den = (int(factorial(M - k, exact=True))
                    * int(factorial(k, exact=True))
                    * int(factorial(k - 1, exact=True))
                    * int(factorial(n - k, exact=True))
                    * int(factorial(2 * k - n, exact=True)))
            s += num / den
        V[n - 1] = (-1) ** (n + M) * s
    return V



def add_series(current: np.ndarray, scalar: float, poly_num: np.ndarray, z: float,n_terms = n_terms) -> np.ndarray:
    """Add scalar * poly_num(x) / (x+z) to current, as a truncated power series.

    poly_num(x) = sum_{n} poly_num[n] * x^n is the numerator polynomial.
    1/(x+z) = sum_{n=0}^{inf} (-1)^n / z^{n+1} * x^n (small-x series).
    Result is convolution of poly_num and 1/(x+z) series, truncated to n_terms.
    """
    inv_denom = (-1.0 / z) ** np.arange(n_terms) / z  # (-1)^n / z^{n+1}: series of 1/(x+z)
    term = np.convolve(poly_num, inv_denom)
    return current + scalar * term[:n_terms]

def add_series2(current: np.ndarray, scalar: float, poly_num: np.ndarray, z: float) -> np.ndarray:
    """Add scalar * poly_num(x) / (z-x) to current, as a truncated power series.

    poly_num(x) = sum_{n} poly_num[n] * x^n is the numerator polynomial.
    1/(z-x) = sum_{n=0}^{inf} 1 / z^{n+1} * x^n (small-x series).
    Result is convolution of poly_num and 1/(z-x) series, truncated to n_terms.
    """
    inv_denom = (1.0 / z) ** np.arange(n_terms) / z  # 1 / z^{n+1}: series of 1/(-x+z)
    term = np.convolve(poly_num, inv_denom)
    return current + scalar * term[:n_terms]

def flip_signs(poly: np.ndarray) -> np.ndarray:
    """ Get poly_num(-x)
    poly_num(x) = sum_{n} poly_num[n] * x^n is the numerator polynomial.
    """
    length = len(poly)
    signs = np.power(-1, np.arange(length), dtype=float)
    return poly * signs

def exp_taylor_series(r: float, n: int) -> np.ndarray:
    """Return the first n Taylor coefficients of exp(r*s): coeff[k] = r^k / k!"""
    k = np.arange(n)
    return np.power(r, k) / factorial(k)
          
    
class FMSA_poly_term_species:
    """FMSA direct-correlation calculator for a multi-species LJ mixture.

    Implements the first-order Mean Spherical Approximation (FMSA) for a
    binary (or multi-species) Lennard-Jones mixture modelled with two Yukawa
    attractive tails.  The constructor mirrors sections 1-4 of FMSA_MC.m:
    it computes Barker-Henderson hard-sphere diameters, packing-fraction
    weighted densities, two-Yukawa K/Z matrices, the MSA A-series, and the
    inner-core polynomial coefficients P_ij for every species pair.
    """
    def __init__(self,
        sig_pure: np.ndarray,
        eps_pure: np.ndarray,
        x: np.ndarray,
        rhostar: float,
        tstar: float,
        n_tails: int = 2,
        z=None,
        k=None,
        verbose: bool = True,
        input_type: str = 'lj') -> None:
        """Initialise system parameters and precompute all FMSA quantities.

        Parameters
        ----------
        sig_pure : pure-species LJ diameters, shape (n_species,)
        eps_pure : pure-species LJ well depths, shape (n_species,)
        x        : mole fractions, shape (n_species,); must sum to 1
        rhostar  : total reduced number density rho* = rho * sigma_ref^3
        tstar    : reduced temperature T* = kT / epsilon_ref
        n_tails  : number of active Yukawa tails (1 = HSY, 2 = 2YK; default 2).
                   Internal arrays always hold 2 slots; the second is zeroed for n_tails=1.
        z        : Yukawa range parameter(s).
                   n_tails=1: float or [float]  → λ, with z_ij = λ/R_ij.
                               Default: 1.791 (standard HSY mapping).
                   n_tails=2: float or [z1_fac, z2_fac]  → z_t_ij = z_t_fac/σ_ij.
                               Single float broadcasts to both tails.
                               Default: [2.9637, 14.0167] (LJ two-Yukawa fit).
                   None uses the default for the chosen n_tails.
        k        : Yukawa amplitude parameter(s).
                   n_tails=1: float or [float]  → A, with k_bare_ij = A·ε_mix_ij·R_ij.
                               Default: 1.0 (unit amplitude scale).
                   n_tails=2: float (shared k0_fac) or [k1_fac, k2_fac] (per-tail).
                               Single float: k_bare_t = ±k·σ_ij·ε_ij·exp(z_t(σ-R)),
                               sign +/- for t=1/2.  Per-tail: caller sets sign.
                               Default: k0_fac = 2.1714 (LJ two-Yukawa fit).
                   None uses the default for the chosen n_tails.
        verbose  : if False, suppress progress prints (useful when constructing
                   many instances in an optimisation loop).  Default: True.
        input_type : 'lj' (default) — sig_pure/eps_pure are LJ parameters;
                                       BH diameter conversion is applied.
                     'bh'           — sig_pure/eps_pure are already BH pure-species
                                       diameters / well-depths; conversion is skipped.

        Stored inputs z and k are preserved as z_input / k_input and threaded
        through _make_at_rho and get_pressure so finite-difference calls
        respect any custom parameterisation.

        After construction the following attributes are available:
          sig_mix, eps_mix   : mixing-rule matrices (n_species × n_species)
          d_bh               : Barker-Henderson effective diameters
          radii              : contact-radius matrix (= BH diameters on diagonal,
                               arithmetic mean off-diagonal)
          xi                 : raw diameter moments ξₙ = Σ_k ρ_k σ_k^n  (FMT nα derived as n₂ = π·ξ₂/4, etc.)
          eta, vac           : packing fraction and free-volume fraction
          K_matrices         : Yukawa amplitude matrices, shape (2, n, n)
          z_matrices         : Yukawa decay constants,   shape (2, n, n)
          k_bare_matrices    : bare amplitudes before density prefactor, shape (2, n, n)
          a_series           : MSA A-matrix power series, shape (n, n, n_terms)
          a_z                : A-matrix at each z, shape (2, n, n, n, n)
                               a_z[tail, m, n, i, j] = A_ij(z_matrices[tail][m, n])
          full_p_coeffs      : inner-core polynomial coefficients,
                               shape (n_species, n_species, n_terms)
        """
        self.sig_pure = sig_pure
        self.eps_pure = eps_pure
        self.x = x
        # The species count is now the INSTANCE's, taken from the arguments,
        # rather than the module-level `n_species` this class used to read at
        # all 108 of its call sites.  A global made the class silently
        # binary-only: constructing it with three species built 3x3 matrices and
        # then indexed them with loops bounded by 2.
        self.n_species = n_species = len(sig_pure)
        if len(eps_pure) != n_species or len(x) != n_species:
            raise ValueError(
                "FMSA_poly_term_species: sig_pure, eps_pure and x must have the "
                "same length; got %d, %d, %d"
                % (n_species, len(eps_pure), len(x)))
        self.rhostar = rhostar
        self.tstar = tstar
        self.rho = rho = x * rhostar
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
        self.cbh = cbh = np.array([1.1287, -0.05536, 0.0007278], dtype=float)
        # Normalise z and k to lists (length = n_tails_active); None → use defaults.
        # z: float or 1-/2-element list/ndarray; k: same.
        self.z_input = z
        self.k_input = k
        n_tails_active = n_tails          # save before overwriting with internal shape
        self.n_tails_active = n_tails_active
        self.n_tails = n_tails = 2        # internal arrays always hold 2 tail slots
        self.k0_fac = k0_fac = 2.1714
        self.z1_fac = z1_fac = 2.9637
        self.z2_fac = z2_fac = 14.0167
        self.beta = beta = 1.0 / tstar

        sig_mix = np.diag(sig_pure)
        eps_mix = np.diag(eps_pure)
        for i in range(self.n_species):
            for j in range(self.n_species):
                if i!=j:
                    sig_mix[i][j]=(sig_pure[i]+sig_pure[j])/2
                    eps_mix[i][j]=np.sqrt(eps_pure[i] * eps_pure[j])
        self.sig_mix = sig_mix
        self.eps_mix = eps_mix
        
        if input_type == 'lj':
            self.d_bh = self.bh_diameters(sig_mix, eps_mix, beta, cbh)
        else:  # 'bh': sig_pure already contains BH pure-species diameters
            self.d_bh = sig_mix.copy()
        d_bh = self.d_bh    
        self.radii = np.diag(np.diag(self.d_bh))
        for i in range(self.n_species):
            for j in range(self.n_species):
                if i!=j:
                    self.radii[i , j] = (self.d_bh[i, i] + self.d_bh[j, j]) / 2.0
        radii = self.radii     
        self.xi = np.zeros(4, dtype=float)
        self.diag_r = np.diag(self.radii)
        self.lambdas=np.zeros((self.n_species,self.n_species))    
        for i in range(self.n_species):
            for j in range(self.n_species):
                self.lambdas[i,j] = np.abs(self.diag_r[i]-self.diag_r[j])/2
        for n in range(4):
            self.xi[n] = np.sum(self.rho * self.diag_r**n)
        xi = self.xi
        self.eta:float = (np.pi / 6.0) * self.xi[3]
        self.vac:float = 1.0 - self.eta
        vac=self.vac
        
        k1 = np.zeros((self.n_species, self.n_species), dtype=float)
        k2 = np.zeros((self.n_species, self.n_species), dtype=float)
        k1_bare_mat = np.zeros((self.n_species, self.n_species), dtype=float)
        k2_bare_mat = np.zeros((self.n_species, self.n_species), dtype=float)

        def _to_list(val, n):
            """Normalise scalar / 1- or 2-element list/ndarray to a Python list of length n."""
            if val is None:
                return None
            arr = [val] if np.isscalar(val) else list(np.asarray(val).ravel())
            if len(arr) == 1 and n == 2:
                arr = arr * 2          # broadcast single value to both tails
            return arr

        if n_tails_active == 1:
            # HSY: single attractive Yukawa.
            # z[0] = λ (range param); default 1.791 → z1_ij = λ/R_ij.
            # k[0] = amplitude scale on ε_mix·R; default 1.0 → k_bare = k·ε_mix·R.
            z_vals = _to_list(z, 1) or [1.791]
            k_vals = _to_list(k, 1) or [1.0]
            lam = z_vals[0]
            k_amp = k_vals[0]
            z1 = lam / self.radii          # shape (n_species, n_species)
            z2 = z1.copy()                 # dummy; K2=0 so z2 never contributes
            for i in range(self.n_species):
                for j in range(self.n_species):
                    k1_bare_mat[i, j] = k_amp * eps_mix[i, j] * self.radii[i, j]
                    param_ij = 2.0 * np.pi * np.sqrt(rho[i] * rho[j]) * beta
                    k1[i, j] = param_ij * k1_bare_mat[i, j]
                    # k2 / k2_bare remain zero → no second tail
        else:
            # 2-tail LJ fit.
            # z = z1_fac or [z1_fac, z2_fac]; default [2.9637, 14.0167] → z_t = z_t_fac/sig.
            # k = k0_fac (shared) or [k1_fac, k2_fac] (per-tail, k2_fac<0 for repulsion).
            #   Single float k: k_bare_t = (±k)·sig·eps·exp(z_t·(sig-R)), sign +/- for t=1/2.
            #   Two-float k:    k_bare_t = k[t]·sig·eps·exp(z_t·(sig-R)), sign carried by k[t].
            z_vals = _to_list(z, 2) or [self.z1_fac, self.z2_fac]
            k_vals = _to_list(k, 2) if k is not None else None
            z1 = np.array(z_vals[0]) / sig_mix
            z2 = np.array(z_vals[1]) / sig_mix
            for i in range(self.n_species):
                for j in range(self.n_species):
                    param_ij = 2.0 * np.pi * np.sqrt(rho[i] * rho[j]) * beta
                    if k_vals is not None:
                        # per-tail or shared scale: sign is caller's responsibility
                        k1_bare = k_vals[0] * sig_mix[i, j] * eps_mix[i, j] * np.exp(z1[i, j] * (sig_mix[i, j] - self.radii[i, j]))
                        k2_bare = k_vals[1] * sig_mix[i, j] * eps_mix[i, j] * np.exp(z2[i, j] * (sig_mix[i, j] - self.radii[i, j]))
                    else:
                        k0 = self.k0_fac * sig_mix[i, j] * eps_mix[i, j]
                        k1_bare =  k0 * np.exp(z1[i, j] * (sig_mix[i, j] - self.radii[i, j]))
                        k2_bare = -k0 * np.exp(z2[i, j] * (sig_mix[i, j] - self.radii[i, j]))
                    k1_bare_mat[i, j] = k1_bare
                    k2_bare_mat[i, j] = k2_bare
                    k1[i, j] = param_ij * k1_bare
                    k2[i, j] = param_ij * k2_bare

        self.k_bare_matrices = np.stack([k1_bare_mat, k2_bare_mat], axis=0)
        self.K_matrices = K_matrices = np.stack([k1, k2],axis=0)
        self.z_matrices = z_matrices = np.stack([z1, z2],axis=0)
        #self.s_atz = np.unique(self.z_matrices.flatten())
        #self.n_s = self.s_atz.size
        if verbose:
            print("Computing Exact Analytical Solutions...")
        self.a_series = self.compute_a_series()
        if verbose:
            print(self.a_series.shape)
        # a_z[tail, m, n, i, j] = A_ij( z_matrices[tail][m, n] )
        #   axes 0-2  (tail, m, n): select the Yukawa decay constant z_{t,mn}
        #   axes 3-4  (i, j)      : row/col of the A-matrix evaluated at that z
        a_z = np.zeros(shape=(n_tails,self.n_species,self.n_species,self.n_species, self.n_species))
        for tail in range(n_tails):
            z_mat = self.z_matrices[tail]
            #tail_cells = [[np.zeros((n_species, n_species), dtype=float) for _ in range(n_species)] for _ in range(n_species)]
            tail_cells = np.zeros(shape=(self.n_species,self.n_species,self.n_species, self.n_species))
            for m in range(self.n_species):
                for n in range(self.n_species):
                    tail_cells[m][n] = self.amat_scalar(z_mat[m, n])
            #a_z.append(tail_cells)
            a_z[tail] = tail_cells
        self.a_z = a_z

        #r_eval = np.linspace(0.01, 5.0, 500)
        pair_idx = [(i, j) for i in range(self.n_species) for j in range(i, self.n_species)]
        #c_total = np.zeros((len(pair_idx), r_eval.size), dtype=float)
        #diams = np.diag(radii)

        full_p_coeffs = np.zeros(shape=(self.n_species,self.n_species,n_terms),dtype = float)
        for p, (i, j) in enumerate(pair_idx):
            full_p_coeffs[i,j] = self.get_p_coeffs(i, j)
        for i in range(self.n_species):
            for j in range(i):
                full_p_coeffs[i, j] = full_p_coeffs[j, i]
            pass
        self.full_p_coeffs = full_p_coeffs
    
    
    # ------------------------------------------------------------------
    # Barker-Henderson diameter conversion (standalone, reusable)
    # ------------------------------------------------------------------

    @staticmethod
    def bh_diameters(
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

    
    @staticmethod
    def phi1s(rv: float, sv: float) -> float:
        """Scalar phi_1 auxiliary function: (1 - sv*rv - exp(-sv*rv)) / sv^2."""
        return (1.0 - sv * rv - np.exp(-sv * rv)) / sv**2

    @staticmethod
    def phi2s(rv: float, sv: float) -> float:
        """Scalar phi_2 auxiliary function: (1 - sv*rv + (sv*rv)^2/2 - exp(-sv*rv)) / sv^3."""
        return (1.0 - sv * rv + (sv * rv) ** 2 / 2.0 - np.exp(-sv * rv)) / sv**3

    
    
    def get_det_Qs(self, s: float) -> float:
        """Return the Q-matrix determinant det Q(s) at Laplace variable s.

        det Q(s) = 1 - (2π/vac)*A*Σ_m ρ_m φ2(R_m,s)
                     - (2π/vac)*Σ_m ρ_m φ1(R_m,s)*(R_m + π*ξ₂*R_m²/(4*vac))
                     - (π²/(2*vac²))*Σ_{m,n} ρ_m ρ_n φ1(R_m,s) φ1(R_n,s) (R_m−R_n)²

        where A = 1 + π*ξ₃/(2*vac).
        """
        rho, radii, vac, xi2, xi3 = self.rho, self.radii, self.vac, self.xi[2], self.xi[3]
        A = 1.0 + np.pi * xi3 / (2.0 * vac)
        s1 = 0.0
        s2 = 0.0
        s3 = 0.0
        for m in range(self.n_species):
            R_m = radii[m, m]
            s1 += rho[m] * self.phi2s(R_m, s) * A
            s2 += rho[m] * self.phi1s(R_m, s) * (R_m + np.pi * xi2 * R_m**2 / (4.0 * vac))
            for n in range(self.n_species):
                R_n = radii[n, n]
                s3 += rho[m] * rho[n] * self.phi1s(R_m, s) * self.phi1s(R_n, s) * (R_m - R_n) ** 2
        return 1.0 - (2.0 * np.pi / vac) * s1 - (2.0 * np.pi / vac) * s2 - (np.pi**2 / (2.0 * vac**2)) * s3

    def amat_scalar(self, s: float) -> np.ndarray:
        """Evaluate the MSA A-matrix A_ij(s) at Laplace variable s via Q-hat inversion.

        Builds Qhat_ij(s) = delta_ij
                            - sqrt(rho_i*rho_j) * exp(-lambda_ij*s)
                              * [Q0_ij*phi1(sigma_i,s) + Qpp_j*phi2(sigma_i,s)]
        then extracts A_ij(s) = (Qhat^{-1}_ij - delta_ij) * exp(lambda_ij*s).

        This is the pointwise counterpart of compute_a_series, guaranteeing that
        off-diagonal elements carry the correct cofactor sign (which a direct
        scalar formula based on det Q misses for n_species > 1).
        """
        rho, radii, vac, xi2 = self.rho, self.radii, self.vac, self.xi[2]

        Qhat = np.zeros((self.n_species, self.n_species), dtype=float)
        for i in range(self.n_species):
            ri = radii[i, i]
            for j in range(self.n_species):
                rj = radii[j, j]
                lambda_ij = (rj - ri) / 2.0
                q_prime = (2.0 * np.pi / vac) * ((ri + rj) / 2.0 + np.pi * ri * rj * xi2 / (4.0 * vac))
                q_dbl   = (2.0 * np.pi / vac) * (1.0 + np.pi * rj * xi2 / (2.0 * vac))
                Qhat[i, j] = -np.sqrt(rho[i] * rho[j]) * np.exp(-lambda_ij * s) * (
                    q_prime * self.phi1s(ri, s) + q_dbl * self.phi2s(ri, s)
                )
                if i == j:
                    Qhat[i, j] += 1.0

        Qhat_inv = np.linalg.inv(Qhat)

        a = np.zeros((self.n_species, self.n_species), dtype=float)
        for i in range(self.n_species):
            ri = radii[i, i]
            for j in range(self.n_species):
                rj = radii[j, j]
                lambda_ij = (rj - ri) / 2.0
                a[i, j] = (Qhat_inv[i, j] - (1.0 if i == j else 0.0)) * np.exp(lambda_ij * s)

        return a

    def compute_a_series(self) -> np.ndarray:
        """Return the Taylor-series coefficients of the MSA A-matrix A_ij(s) around s=0."""
        rho, radii, vac, xi2 = self.rho, self.radii, self.vac, self.xi[2]
        q = np.zeros(shape=(self.n_species, self.n_species, n_terms), dtype=float)
        phi_order = np.arange(n_terms+3)
        phi_factorial = factorial(phi_order)
        for i in range(self.n_species):
            for j in range(self.n_species):
                ri = radii[i, i]
                rj = radii[j, j]
                lambda_ij = (rj - ri) / 2.0
                q_prime = (2.0 * np.pi / vac) * ((ri + rj) / 2.0 + (np.pi * ri * rj * xi2) / (4.0 * vac))
                q_dbl = (2.0 * np.pi / vac) * (1.0 + (np.pi * rj * xi2) / (2.0 * vac))

                phi = np.power(-ri, phi_order) / phi_factorial
                phi1 = (-phi[2:n_terms+2])
                phi2 = phi[3:n_terms+3]
                e_lam = np.power(-lambda_ij, phi_order) / phi_factorial

                term = np.convolve(e_lam, q_prime * phi1 + q_dbl * phi2)
                q_ij = -np.sqrt(rho[i] * rho[j]) * term[:n_terms]
                if i == j:
                    q_ij[0] += 1.0
                q[i][j] = q_ij

        q0_inv = np.linalg.inv(q[:, :, 0])
        q_inv = np.zeros_like(q)
        q_inv[:, :, 0] = q0_inv
        for n in range(1, n_terms):
            coeff_sum = np.zeros((self.n_species, self.n_species), dtype=float)
            for k in range(1, n + 1):
                coeff_sum += np.matmul(q[:, :, k], q_inv[:, :, n - k])
            q_inv[:, :, n] = -np.matmul(q0_inv, coeff_sum)

        a_series = np.zeros(shape=(self.n_species, self.n_species, n_terms))
        term_order = np.arange(n_terms)
        term_factorial = factorial(term_order)
        for i in range(self.n_species):
            for j in range(self.n_species):
                a_e_lam = q_inv[i, j].copy()
                if i == j:
                    a_e_lam[0] -= 1.0
                lambda_ij = (radii[j, j] - radii[i, i]) / 2.0
                e_lam_pos = np.power(lambda_ij, term_order) / term_factorial
                a_s = np.convolve(a_e_lam, e_lam_pos)
                a_series[i][j] = a_s[:n_terms]

        return a_series

    def build_SY_series(self, i: int, j: int, n_terms: int = n_terms):
        """Assemble the S(s) and Y series for pair (i,j)."""
        a_series, a_z, K_matrices, z_matrices = (
            self.a_series, self.a_z, self.K_matrices, self.z_matrices
        )
        s_series = np.zeros(n_terms, dtype=float)
        y_series = np.zeros(n_terms, dtype=float)

        for tail in range(n_tails):
            k = K_matrices[tail]
            z = z_matrices[tail]
            az = a_z[tail]
            s_series = add_series(s_series, k[i, j], np.array([1, 0, 0, 0, 0], dtype=float), z[i, j], n_terms)
            y_series = add_series(y_series, k[i, j], np.array([1, ], dtype=float), z[i, j], n_terms)

            for n in range(self.n_species):
                s_series = add_series(s_series, k[i, n], flip_signs(a_series[j][n]), z[i, n], n_terms)
                y_series = add_series(y_series, k[i, n], np.array([az[i][n][j, n], ], dtype=float), z[i, n], n_terms)

            for m in range(self.n_species):
                s_series = add_series(s_series, k[m, j], flip_signs(a_series[i][m]), z[m, j], n_terms)
                y_series = add_series(y_series, k[m, j], np.array([az[m][j][i, m], ], dtype=float), z[m, j], n_terms)

            for m in range(self.n_species):
                for n in range(self.n_species):
                    a_prod = np.convolve(flip_signs(a_series[i][m]), flip_signs(a_series[j][n]))
                    s_series = add_series(s_series, k[m, n], a_prod[:n_terms], z[m, n], n_terms)
                    y_series = add_series(
                        y_series,
                        k[m, n],
                        np.array([az[m][n][i, m] * az[m][n][j, n], ], dtype=float),
                        z[m, n],
                        n_terms,
                    )
        return (s_series, y_series)

    def get_laplace_residual(self, i: int, j: int) -> np.ndarray:
        """Return the Laplace-space residual series for pair (i,j)."""
        r_ij = self.radii[i, j]
        s_series, y_series = self.build_SY_series(i, j, n_terms)
        e_s = np.convolve(exp_taylor_series(r_ij, n_terms), s_series)
        return e_s[:n_terms] - y_series

    def get_p_coeffs(self, i: int, j: int) -> np.ndarray:
        """Return polynomial coefficients of the inner-core correction P_ij(r)."""
        diff_series = self.get_laplace_residual(i, j)
        p_coeffs = np.zeros(n_terms, dtype=float)
        p_coeffs[0] = -self.get_e_ij(0.0, i, j)
        for i_term in range(1, n_terms):
            p_coeffs[i_term] = diff_series[i_term] / factorial(i_term)
        return p_coeffs

    def get_e_ij(self, r: float, i: int, j: int) -> float:
        """Return the Yukawa amplitude E_ij(r) inside the hard-sphere core."""
        a_z, K_matrices, z_matrices, radii = self.a_z, self.K_matrices, self.z_matrices, self.radii
        e_val = 0.0
        r_ij = radii[i, j]

        # THE LOOP BOUNDS WERE LITERAL 2s HERE (fixed 2026-08-11).  Every
        # sibling method -- build_SY_series above, compute_a_series, amat_scalar
        # -- writes range(n_species) for exactly these three sums; only this one
        # wrote range(2).  It agreed with them solely because n_species was 2,
        # so the class was binary-only in a way nothing could see until someone
        # changed the global.  At N = 2 this edit is a no-op, which is how it
        # was verified: the reference vectors in cpp/FMSA_model/reference_poly/
        # are byte-identical across it.
        for tail in range(self.n_tails):
            k = K_matrices[tail]
            z = z_matrices[tail]
            az = a_z[tail]
            delta_r = r_ij - r

            e_val += k[i, j] * np.exp(-z[i, j] * (delta_r))
            for n in range(self.n_species):
                e_val += k[i, n] * az[i][n][j, n] * np.exp(-z[i, n] * (delta_r))
            for m in range(self.n_species):
                e_val += k[m, j] * az[m][j][i, m] * np.exp(-z[m, j] * (delta_r))
            for m in range(self.n_species):
                for n in range(self.n_species):
                    e_val += k[m, n] * az[m][n][i, m] * az[m][n][j, n] * np.exp(-z[m, n] * (delta_r))

        return e_val

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

    def _wb_rc0_poly(self, i: int, j: int) -> tuple[float, float, float, float]:
        """Coefficients of r·c0^WB_ij(r) = C0 + C1·r + C2·r² + C4·r⁴  on [λ_ij, R_ij].

        Every White Bear weight convolution times r is a polynomial in r with no
        r³ term (r·ω₂⊛ω₂ = const, r·ω₂⊛ω₃ = quadratic, r·ω₃⊛ω₃ = quartic without
        r³, r·∇²(ω₃⊛ω₃) = quadratic), so the analytic Fourier integrals below
        need only these four coefficients.  See `get_c0_fmtHS_ref` for the
        underlying Φ, weights and normalisations.  [Corrected 2026-07-17 together
        with `get_c0_fmtHS_ref`.]
        """
        Ri = float(self.radii[i, i]) / 2.0
        Rj = float(self.radii[j, j]) / 2.0
        rho, R = self.rho, np.asarray(self.diag_r) / 2.0

        n0 = float(np.sum(rho))
        n1 = float(np.sum(rho * R))
        n2 = float(np.sum(rho * 4.0 * np.pi * R**2))
        n3 = float(np.sum(rho * (4.0 * np.pi / 3.0) * R**3))
        v  = 1.0 - n3
        ph, ph1, ph2 = self._wb_phi_derivs(n3)
        Phi03 = 1.0 / v
        Phi12 = 1.0 / v
        Phi13 = n2 / v**2
        Phi23 = n1 / v**2 + 3.0 * n2**2 * ph1
        Phi22 = 6.0 * n2 * ph
        Phi33 = n0 / v**2 + 2.0 * n1 * n2 / v**3 + n2**3 * ph2
        W   = (-1.0 / v) * (1.0/(4.0*np.pi*Ri) + 1.0/(4.0*np.pi*Rj)) + (-6.0 * n2 * ph)

        S  = Ri + Rj
        Dm = (Ri - Rj)**2
        # r·ω₂ⁱ⊛ω₂ʲ
        a0 = 2.0 * np.pi * Ri * Rj
        # r·ω₂ⁱ⊛ω₃ʲ = πRi(Rj²−Ri²) + 2πRi²·r − πRi·r²   (and i↔j)
        b0i, b1i, b2i = np.pi*Ri*(Rj**2 - Ri**2), 2.0*np.pi*Ri**2, -np.pi*Ri
        b0j, b1j, b2j = np.pi*Rj*(Ri**2 - Rj**2), 2.0*np.pi*Rj**2, -np.pi*Rj
        # r·ω₃ⁱ⊛ω₃ʲ = (π/12)(r⁴ − 3(S²+Dm)r² + (2S³+6S·Dm)r − 3S²Dm)
        v0, v1, v2, v4 = (-np.pi*S**2*Dm/4.0, np.pi*(S**3 + 3.0*S*Dm)/6.0,
                          -np.pi*(S**2 + Dm)/4.0, np.pi/12.0)
        # r·∇²(ω₃ⁱ⊛ω₃ʲ) = (π/12)(12r² − 6(S²+Dm))
        l0, l2 = -np.pi*(S**2 + Dm)/2.0, np.pi

        def _scal(x_i, x_j, xa, xv):
            return (Phi03*(x_i/(4.0*np.pi*Ri**2) + x_j/(4.0*np.pi*Rj**2))
                    + Phi12*xa*(1.0/(4.0*np.pi*Ri) + 1.0/(4.0*np.pi*Rj))
                    + Phi13*(x_i/(4.0*np.pi*Ri) + x_j/(4.0*np.pi*Rj))
                    + Phi23*(x_i + x_j) + Phi22*xa + Phi33*xv)

        C0 = -_scal(b0i, b0j, a0,  v0) + l0 * W
        C1 = -_scal(b1i, b1j, 0.0, v1)
        C2 = -_scal(b2i, b2j, 0.0, v2) + l2 * W
        C4 = -Phi33 * v4
        return float(C0), float(C1), float(C2), float(C4)

    def get_c0_fmtHS_ref_fourier(self, k: float, i: int, j: int) -> float:
        """Analytical Fourier integral of the FMT hard-sphere DCF over [0, R_ij].

        Returns k≠0: ∫_0^R r·c0_fmt(r_c)·sin(kr) dr  or  k=0: ∫_0^R r²·c0_fmt(r_c) dr
        where r_c = max(r, lambda_ij) regularises unlike-pair singularities —
        consistent with the White Bear c0, which is genuinely constant on
        [0, λ_ij] (the smaller sphere sits entirely inside the larger one there).
        """
        R = self.radii[i, j]
        lambda_ij = abs(self.radii[i, i] - self.radii[j, j]) / 2.0
        C0, C1, C2, C4 = self._wb_rc0_poly(i, j)

        if k == 0.0:
            def _poly_r(a: float, b: float) -> float:
                return (C0*(b**2-a**2)/2.0 + C1*(b**3-a**3)/3.0
                        + C2*(b**4-a**4)/4.0 + C4*(b**6-a**6)/6.0)
            if lambda_ij > 0.0:
                c0_lam = self.get_c0_fmtHS_ref(lambda_ij, i, j)
                return c0_lam * lambda_ij**3 / 3.0 + _poly_r(lambda_ij, R)
            return _poly_r(0.0, R)
        else:
            def _poly_sin(a: float, b: float) -> float:
                return (C0 * _rn_sin_int(0, k, a, b)
                        + C1 * _rn_sin_int(1, k, a, b)
                        + C2 * _rn_sin_int(2, k, a, b)
                        + C4 * _rn_sin_int(4, k, a, b))
            if lambda_ij > 0.0:
                c0_lam = self.get_c0_fmtHS_ref(lambda_ij, i, j)
                # [0, λ]: c0(λ)·∫_0^λ r·sin(kr) dr
                return c0_lam * _rn_sin_int(1, k, 0.0, lambda_ij) + _poly_sin(lambda_ij, R)
            return _poly_sin(0.0, R)

    def get_c0_fmtHS_ref(self, r: float, i: int, j: int) -> float:
        """White Bear FMT hard-sphere mixture pair DCF c0_ij(r), closed form.

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
        Ri = float(self.radii[i, i]) / 2.0
        Rj = float(self.radii[j, j]) / 2.0
        rho, R = self.rho, np.asarray(self.diag_r) / 2.0

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

    def cavity_function_weight(self, r: float) -> float:
        """Return sum of rho_i for all species i whose hard-sphere radius <= r.

        Acts as a step-function weight: species i contributes rho_i when
        r >= R_i (i.e. r is outside the hard core of species i).
        """
        diag_r=self.diag_r
        direction_r=np.array(r>=diag_r,dtype=float)
        omega = float(np.sum(direction_r*self.rho))
        return omega
    
    
    def calculate_b_tijs(self,tail:int, i:int, j:int, s: float) -> float:
        """Evaluate b_{t,ij}(s) for a single tail and species pair at Laplace variable s.

        b_{t,ij}(s) collects the same four Yukawa-MSA term classes as get_e_ij
        but in Laplace space (denominator s + z rather than the exponential):
          K[i,j]/(s+z[i,j])
          + sum_n K[i,n]*A[t][i,n][n,j]/(s+z[i,n])
          + sum_m K[m,j]*A[t][m,j][i,m]/(s+z[m,j])
          + sum_{m,n} K[m,n]*A[t][m,n][i,m]*A[t][m,n][j,n]/(s+z[m,n])

        Parameters
        ----------
        tail : Yukawa tail index (0 or 1)
        i, j : 0-based species indices
        s    : Laplace variable (scalar)
        """
        sum_b = 0.0
        sum_b += self.K_matrices[tail][i, j]/(s+self.z_matrices[tail][i, j])
        for n in range(self.n_species):
            sum_b += self.K_matrices[tail][i, n]*self.a_z[tail][i, n][n,j]/(s+self.z_matrices[tail][i, n])
        for m in range(self.n_species):
            sum_b += self.K_matrices[tail][m, j]*self.a_z[tail][m, j][i, m]/(s+self.z_matrices[tail][m, j])
        for m in range(self.n_species):
            for n in range(self.n_species):
                sum_b += self.K_matrices[tail][m, n]*self.a_z[tail][m,n][i, m]*self.a_z[tail][m,n][j, n]/(s+self.z_matrices[tail][m, n])
        return sum_b
    
    def get_btij_s0(self)->np.ndarray:
        """Return b_{t,ij}(s=0) for all tails and species pairs.

        Evaluates calculate_b_tijs at s=0 for every combination of tail,
        i, and j.  The result is stored as self.b_tij_s0 and also returned.

        Returns
        -------
        b_tij_s0 : shape (n_tails, n_species, n_species)
        """
        b_tij_s0 = np.zeros(shape=(self.n_tails,self.n_species,self.n_species))
        for tail in range(self.n_tails):
            for i in range(self.n_species):
                for j in range(self.n_species):
                    b_tij_s0[tail,i,j] = self.calculate_b_tijs(tail,i,j,0)
        self.b_tij_s0 = b_tij_s0
        return b_tij_s0
    
    # Use the same order of a_z, A_ij(z_tmn) = a_z[t][m,n][i,j]
    # b_t2ij(z_t1mn) = b_tijs[t1,m,n][i,j,t2]
    def get_b_ztmn_ij(self) -> None:
        #s_samples=self.s_atz
        self.b_tijs = np.zeros(shape=(self.n_tails,self.n_species,self.n_species,self.n_species,self.n_species,self.n_tails,), dtype=float)
        for tail_s in range (self.n_tails):
            for tail_b in range (self.n_tails):
                for m in range(self.n_species):
                    for n in range(self.n_species):
                        for i in range(self.n_species):
                            for j in range(self.n_species):
                                s = self.z_matrices[tail_s][m, n]
                                self.b_tijs[tail_s,m,n][i,j,tail_b] = self.calculate_b_tijs(tail_b, i, j, s)
        
 
    def get_c_total_nopoly(self, r_eval: np.ndarray) -> np.ndarray:
        """Evaluate the FMSA DCF on r_eval without the inner-core polynomial (P_ij = 0).

        Mirrors the main evaluation loop in FMSA_MC.m but omits all P_ij
        polynomial terms (only the constant p_coeffs[0] at r_c is kept).
        Useful as a baseline to isolate the contribution of the polynomial.

        Parameters
        ----------
        r_eval : 1-D array of distances to evaluate

        Returns
        -------
        c_total : shape (n_pairs, len(r_eval)); rows ordered as (0,0),(0,1),(1,1)
        """
        pair_idx = [(i, j) for i in range(self.n_species) for j in range(i, self.n_species)]
        c_total = np.zeros((len(pair_idx), r_eval.size), dtype=float)
        for p, (i, j) in enumerate(pair_idx):
            p_c = 1.0 / (2.0 * np.pi * np.sqrt(self.rho[i] * self.rho[j]))
            #print(f"{i+1}{j+1} pc_value {p_c}")
            #print(f"{i+1}{j+1} rho_value {np.sqrt(self.rho[i] * self.rho[j])}")
            for idx, r in enumerate(r_eval):
                if r < self.radii[i, j]:
                    lambda_ij = abs(self.radii[i, i] - self.radii[j, j]) / 2.0
                    r_c = max(r, lambda_ij)
                    prefactor = 1.0 / (2.0 * np.pi * np.sqrt(self.rho[i] * self.rho[j]) * r_c)
                    c0 = self.get_c0_fmtHS_ref(r_c, i, j)
                    e_ij = self.get_e_ij(r_c, i, j)
                    e_ijr0 = self.get_e_ij(0.0, i, j)
                    p_ij = -e_ijr0
                    #p_ij = 0.0
                    c_total[p, idx] = c0 + prefactor * (e_ij+p_ij)
                else:
                    prefactor_out = 1.0 / (2.0 * np.pi * np.sqrt(self.rho[i] * self.rho[j]) * r)
                    tail_val = self.K_matrices[0][i, j] * np.exp(-self.z_matrices[0][i, j] * (r - self.radii[i, j])) + self.K_matrices[1][i, j] * np.exp(-self.z_matrices[1][i, j] * (r - self.radii[i, j]))
                    c_total[p, idx] = prefactor_out * tail_val
        return c_total
    
    def get_c_total_poly(self, r_eval: np.ndarray) -> np.ndarray:
        """Evaluate the FMSA DCF on r_eval using the full inner-core polynomial.

        This is the primary evaluation method, directly porting the FMSA_MC.m
        main loop.  For r < R_ij:
          c_ij(r) = c0_ij(r_c) + [E_ij(r) + P_ij(r)] / (2*pi*sqrt(rho_i*rho_j)*r_c)
        For r >= R_ij:
          c_ij(r) = [K1*exp(-z1*(r-R_ij)) + K2*exp(-z2*(r-R_ij))] / (2*pi*sqrt(rho_i*rho_j)*r)
        where r_c = max(r, lambda_ij) regularises the unlike-pair singularity.

        Parameters
        ----------
        r_eval : 1-D array of distances to evaluate

        Returns
        -------
        c_total : shape (n_pairs, len(r_eval)); rows ordered as (0,0),(0,1),(1,1)
        """
        pair_idx = [(i, j) for i in range(self.n_species) for j in range(i, self.n_species)]
        c_total = np.zeros((len(pair_idx), r_eval.size), dtype=float)
        for p, (i, j) in enumerate(pair_idx):
            for idx, r in enumerate(r_eval):
                if r < self.radii[i, j]:
                    lambda_ij = abs(self.radii[i, i] - self.radii[j, j]) / 2.0
                    r_c = max(r, lambda_ij)
                    prefactor = 1.0 / (2.0 * np.pi * np.sqrt(self.rho[i] * self.rho[j]) * r_c)
                    c0 = self.get_c0_fmtHS_ref(r_c, i, j)
                    e_ij = self.get_e_ij(r, i, j)
                    #p_ij = np.sum(self.full_p_coeffs[i,j][1:] * np.power(r_c,all_terms[1:]))
                    p_ij = np.sum(self.full_p_coeffs[i,j][0:] * np.power(r,np.arange(n_terms)[0:]))
                    c_total[p, idx] = c0 + prefactor * (e_ij + p_ij)
                else:
                    prefactor_out = 1.0 / (2.0 * np.pi * np.sqrt(self.rho[i] * self.rho[j]) * r)
                    tail_val = self.K_matrices[0][i, j] * np.exp(-self.z_matrices[0][i, j] * (r - self.radii[i, j])) + self.K_matrices[1][i, j] * np.exp(-self.z_matrices[1][i, j] * (r - self.radii[i, j]))
                    c_total[p, idx] = prefactor_out * tail_val
        return c_total

    def get_ck_total_poly(self, k_eval: np.ndarray) -> np.ndarray:
        """Analytical 3D Fourier transform of the FMSA DCF (matching get_c_total_poly).

        ĉ_ij(k) = (4π/k) ∫_0^∞ r c_ij(r) sin(kr) dr

        All three inner-core contributions are integrated in closed form:
          · c0 (FMT HS): via get_c0_fmtHS_ref_fourier — r·c0(r) is a polynomial
          · E_ij (Yukawa): via _exp_sin_int / _r_exp_sin_int
          · P_ij (polynomial correction): via _rn_sin_int

        The outer Yukawa tail (r ≥ R_ij) uses the exact identity:
            ∫_0^∞ e^{-z u} sin(k(u+R)) du = [z sin(kR) + k cos(kR)] / (z²+k²)

        For k = 0 each piece falls back to its closed-form ∫ r² c(r) dr expression.

        Parameters
        ----------
        k_eval : 1-D array of wavenumbers

        Returns
        -------
        ck_total : shape (n_pairs, len(k_eval)); rows ordered as (0,0),(0,1),(1,1)
        """
        pair_idx = [(i, j) for i in range(self.n_species) for j in range(i, self.n_species)]
        ck_total = np.zeros((len(pair_idx), k_eval.size), dtype=float)

        for p, (i, j) in enumerate(pair_idx):
            R_ij = self.radii[i, j]
            sqrt_rho = np.sqrt(self.rho[i] * self.rho[j])
            lam = abs(self.radii[i, i] - self.radii[j, j]) / 2.0
            like = (lam == 0.0)

            # Collect E_ij(r) = Σ_m C_m exp(z_m r)  amplitude/decay pairs.
            # Mirroring get_e_ij: exp(-z*(R-r)) = exp(-z*R)*exp(z*r).
            e_terms: list[tuple[float, float]] = []
            for tail in range(n_tails):
                km = self.K_matrices[tail]
                zm = self.z_matrices[tail]
                az = self.a_z[tail]
                e_terms.append((km[i, j] * np.exp(-zm[i, j] * R_ij), zm[i, j]))
                for nn in range(self.n_species):
                    e_terms.append((km[i, nn] * az[i][nn][j, nn] * np.exp(-zm[i, nn] * R_ij), zm[i, nn]))
                for mm in range(self.n_species):
                    e_terms.append((km[mm, j] * az[mm][j][i, mm] * np.exp(-zm[mm, j] * R_ij), zm[mm, j]))
                for mm in range(self.n_species):
                    for nn in range(self.n_species):
                        e_terms.append((
                            km[mm, nn] * az[mm][nn][i, mm] * az[mm][nn][j, nn]
                            * np.exp(-zm[mm, nn] * R_ij),
                            zm[mm, nn],
                        ))

            p_coeffs = self.full_p_coeffs[i, j]  # shape (n_terms,)
            pref = 2.0 * np.pi * sqrt_rho          # common denominator

            for ki, k in enumerate(k_eval):
                # ---- FMT c0 ----
                ck_c0 = self.get_c0_fmtHS_ref_fourier(k, i, j)

                # ---- E_ij ----
                ck_e = 0.0
                if k == 0.0:
                    if not like:
                        for Cm, zm_val in e_terms:
                            # [0, λ]: 1/(pref*λ) · ∫_0^λ r² exp(z r) dr
                            ck_e += Cm / (pref * lam) * (
                                _r2_exp_antideriv(zm_val, lam) - _r2_exp_antideriv(zm_val, 0.0)
                            )
                            # [λ, R]: 1/pref · ∫_λ^R r exp(z r) dr
                            ck_e += Cm / pref * (
                                _r_exp_antideriv(zm_val, R_ij) - _r_exp_antideriv(zm_val, lam)
                            )
                    else:
                        for Cm, zm_val in e_terms:
                            ck_e += Cm / pref * (
                                _r_exp_antideriv(zm_val, R_ij) - _r_exp_antideriv(zm_val, 0.0)
                            )
                else:
                    if not like:
                        for Cm, zm_val in e_terms:
                            # [0, λ]: 1/(pref*λ) · ∫_0^λ r exp(zr) sin(kr) dr
                            ck_e += Cm / (pref * lam) * _r_exp_sin_int(zm_val, k, 0.0, lam)
                            # [λ, R]: 1/pref · ∫_λ^R exp(zr) sin(kr) dr
                            ck_e += Cm / pref * _exp_sin_int(zm_val, k, lam, R_ij)
                    else:
                        for Cm, zm_val in e_terms:
                            ck_e += Cm / pref * _exp_sin_int(zm_val, k, 0.0, R_ij)

                # ---- P_ij ----
                ck_p = 0.0
                if k == 0.0:
                    if not like:
                        for n_pow, pn in enumerate(p_coeffs):
                            # [0, λ]: 1/(pref*λ) · ∫_0^λ r^{n+2} dr = λ^{n+3}/(n+3)
                            ck_p += pn / (pref * lam) * lam**(n_pow + 3) / (n_pow + 3)
                            # [λ, R]: 1/pref · ∫_λ^R r^{n+1} dr
                            ck_p += pn / pref * (R_ij**(n_pow + 2) - lam**(n_pow + 2)) / (n_pow + 2)
                    else:
                        for n_pow, pn in enumerate(p_coeffs):
                            ck_p += pn / pref * R_ij**(n_pow + 2) / (n_pow + 2)
                else:
                    if not like:
                        for n_pow, pn in enumerate(p_coeffs):
                            # [0, λ]: 1/(pref*λ) · ∫_0^λ r^{n+1} sin(kr) dr
                            ck_p += pn / (pref * lam) * _rn_sin_int(n_pow + 1, k, 0.0, lam)
                            # [λ, R]: 1/pref · ∫_λ^R r^n sin(kr) dr
                            ck_p += pn / pref * _rn_sin_int(n_pow, k, lam, R_ij)
                    else:
                        for n_pow, pn in enumerate(p_coeffs):
                            ck_p += pn / pref * _rn_sin_int(n_pow, k, 0.0, R_ij)

                # ---- Outer Yukawa tail (r ≥ R_ij), analytical ----
                ck_tail = 0.0
                for t in range(self.K_matrices.shape[0]):
                    Kt = self.K_matrices[t][i, j]
                    zt = self.z_matrices[t][i, j]
                    if k == 0.0:
                        ck_tail += Kt / pref * (zt * R_ij + 1.0) / zt**2
                    else:
                        ck_tail += Kt / pref * (
                            zt * np.sin(k * R_ij) + k * np.cos(k * R_ij)
                        ) / (zt**2 + k**2)

                if k == 0.0:
                    ck_total[p, ki] = 4.0 * np.pi * (ck_c0 + ck_e + ck_p + ck_tail)
                else:
                    ck_total[p, ki] = 4.0 * np.pi / k * (ck_c0 + ck_e + ck_p + ck_tail)

        return ck_total

    def get_all_H1L_laplace(self, s: float) -> np.ndarray:
        """Evaluate the full H1L_ij(s) = G1L_ij(s) matrix for all species pairs.

        H1L and G1L are identical because the first-order perturbation terms of
        the TCF h and the RDF g coincide:

            h(r) = g(r) − 1  =>  h₁(r) = g₁(r)

        so  G1L_ij(s) ≡ L[r · g₁_ij(r)](s) = L[r · h₁_ij(r)](s) ≡ H1L_ij(s).

        Formula:
            H1L_ij(s) = exp(−s R_ij) · Σ_t [(I + A^T) B_t(s) (I + A)]_ij

        where A = amat_scalar(s) and B_t is the b-matrix for Yukawa tail t,
        assembled from four term classes:

            B_t[i,j] = W[i,j]
                      + Σ_n  W[i,n] · az[i,n,n,j]          (term 2)
                      + Σ_m  W[m,j] · az[m,j,i,m]          (term 3)
                      + Σ_mn W[m,n] · az[m,n,i,m]          (term 4)
                               · az[m,n,j,n]
            W[m,n] = K_t[m,n] / (s + z_t[m,n])

        amat_scalar is called once; B_t is built with two O(n_species) loops
        (terms 2-3) and one O(n_species²) loop (term 4), then contracted via
        two matrix multiplications.

        Returns
        -------
        H1L : shape (n_species, n_species)
        """
        A    = self.amat_scalar(s)
        IpAT = np.eye(self.n_species) + A.T
        IpA  = np.eye(self.n_species) + A
        total = np.zeros((self.n_species, self.n_species), dtype=float)

        for tail in range(n_tails):
            W  = self.K_matrices[tail] / (s + self.z_matrices[tail])  # (ns,ns), element-wise
            az = self.a_z[tail]                                        # (ns,ns,ns,ns)

            B = W.copy()                                               # term 1

            # term 2: B[i,j] += Σ_n W[i,n] · az[i,n,n,j]
            #   for fixed n: W[:,n,None] * az[:,n,n,:]  => shape (ns,ns)
            for n in range(self.n_species):
                B += W[:, n, np.newaxis] * az[:, n, n, :]

            # term 3: B[i,j] += Σ_m W[m,j] · az[m,j,i,m]
            #   az[m,:,:,m] has element [j,i]=az[m,j,i,m], so .T gives [i,j]
            #   for fixed m: W[m,None,:] * az[m,:,:,m].T  => shape (ns,ns)
            for m in range(self.n_species):
                B += W[m, np.newaxis, :] * az[m, :, :, m].T

            # term 4: B[i,j] += Σ_{m,n} W[m,n] · az[m,n,i,m] · az[m,n,j,n]
            #   = Σ_{m,n} W[m,n] · outer(az[m,n,:,m], az[m,n,:,n])
            for m in range(self.n_species):
                for n in range(self.n_species):
                    B += W[m, n] * np.outer(az[m, n, :, m], az[m, n, :, n])

            total += IpAT @ B @ IpA

        return np.exp(-s * self.radii) * total

    def get_H1L_laplcase_ij(self, s: float, i: int, j: int) -> float:
        """Return H1L_ij(s) for a single pair; delegates to get_all_H1L_laplace."""
        return self.get_all_H1L_laplace(s)[i, j]
    
    def get_all_G1L_shift(self, s: float) -> np.ndarray:
        G1Ls = self.get_all_H1L_laplace(s)
        G1Ls_shift = np.zeros_like(G1Ls)
        for i in range(self.n_species):
            for j in range(self.n_species):
                R_ij = self.radii[i,j]
                G1L_shift = G1Ls[i,j]*np.exp(s*R_ij)
        return G1Ls_shift
    
    def get_G1L_ij_shift(self, s: float, i: int, j: int) -> float:
        """Shifted first-order Laplace kernel: G1L_ij(s) · exp(s · R_ij).

        Strips the contact-distance exponential factor from H1L_ij(s),
        returning the polynomial part of the Laplace transform:

            G1L_shift_ij(s) = H1L_ij(s) · exp(s · R_ij)
                             = [(I + A(s)ᵀ) B(s) (I + A(s))]_ij

        where B(s) = Σ_t K_t / (s + z_t) (plus A-mixing cross terms).
        Analogous to G0red, which strips exp(-s·R_ij) / (2π√(ρᵢρⱼ)) from H0L.

        Called in betaf2 at s = z_tij (the Yukawa decay constant) to extract
        the residue contribution of each tail to the second-order free energy.

        Parameters
        ----------
        s    : Laplace variable (scalar)
        i, j : 0-based species indices

        Returns
        -------
        float : G1L_ij(s) · exp(s · R_ij)
        """
        G1Lijs = self.get_H1L_laplcase_ij(s, i, j)
        R_ij = self.radii[i, j]
        G1Lij_shift = G1Lijs * np.exp(s * R_ij)
        return G1Lij_shift
    
    def G0red_laplace_ij(self,i:int,j:int, s: float) -> float:
        """Evaluate the reduced kernel of zeroth-order pair correlation G0_ij(s).
        G0red_ij(s)= (g0(s))_ij/ (2*pi*sqrt(rhoi rhoj)exp(-s Rij))
        
        G0red_ij(s) = 1/(vac*detQ(s))*[
            C1/s**2+
            C2/s
            ]
        where C1 = 1+pi*xi3/(2*vac).
        C2 = R_ij+pi*xi2*Ri*Rj/(4*vac)+
            pi/(2*vac)*sum_m rho_m phi1(s)(R_m)*(R_m-R_i)(R_m-R_j)

        Parameters
        ----------
        i, j : 0-based species indices
        s    : Laplace variable (scalar)
        Returns
        -------
        G0red_ij : reduced kernel for incore correlation free energy
        """
        radii,diag_r=self.radii,self.diag_r
        R_i, R_j = diag_r[i],diag_r[j]
        rho_i,rho_j,R_ij,vac = self.rho[i],self.rho[j],self.radii[i, j],self.vac
        param = 2*np.pi*np.sqrt(rho_i*rho_j)*np.exp(-s*R_ij)
        detQs= self.get_det_Qs(s)
        const1 = 1 + np.pi*self.xi[3]/(2*vac)
        const2_term1 = R_ij+np.pi*self.xi[2]*R_i*R_j/(4*vac)
        const2_term2 = 0.0
        for m in range(self.n_species):
            R_m = diag_r[m]
            const2_term2 += (np.pi/(2.0*vac))*self.rho[m]*self.phi1s(R_m,s)*(R_m-R_i)*(R_m-R_j)
        g_0_s = 1/(vac*detQs)*(const1/s**2+const2_term1/s+const2_term2/s)
        return g_0_s
    

    def get_g0ij_r(self, r: float, i: int, j: int, n_stehfest: int = 12) -> float:
        """Return the HS-reference RDF g0_ij(r) via Stehfest numerical Laplace inversion.

        Inverts the analytical Laplace-space result G0red_laplace_ij without
        going through Fourier or k-space.

        The Laplace pair is:
            F_ij(s) = L[r * g0_ij(r)](s)
                    = 2π√(ρ_i ρ_j) exp(-s R_ij) G0red_laplace_ij(i, j, s)

        Stehfest inversion:
            r * g0_ij(r) ≈ (ln2/r) Σ_{n=1}^{N} V_n F_ij(n ln2 / r)

        Parameters
        ----------
        r          : radial distance (scalar)
        i, j       : 0-based species indices
        n_stehfest : number of Stehfest terms (even; 12 recommended)

        Returns
        -------
        g0_ij(r) : 0 for r < R_ij, Laplace-inverted value otherwise
        """
        R_ij = self.radii[i, j]
        if r < R_ij:
            return 0.0
        V = _stehfest_coeffs(n_stehfest)
        ln2 = np.log(2.0)
        pref = 2.0 * np.pi * np.sqrt(self.rho[i] * self.rho[j])
        total = 0.0
        for n in range(1, n_stehfest + 1):
            s_n = n * ln2 / r
            F_s = pref * np.exp(-s_n * R_ij) * self.G0red_laplace_ij(i, j, s_n)
            total += V[n - 1] * F_s
        return (ln2 / r ** 2) * total

    def get_g1ij_r(self, r: float, i: int, j: int, n_stehfest: int = 12) -> float:
        """Return the first-order RDF perturbation g1_ij(r) via Stehfest inversion.

        Since g₁ = h₁ (the first-order perturbation to g and h are identical),
        the Laplace kernel is simply H1L_ij(s) = L[r·g₁_ij(r)](s):

            G1L_ij(s) = H1L_ij(s) = get_all_H1L_laplace(s)[i, j]

        Stehfest inversion:
            r · g₁_ij(r) ≈ (ln2/r) Σ_{n=1}^{N} V_n · G1L_ij(n ln2 / r)

        Parameters
        ----------
        r          : radial distance (scalar)
        i, j       : 0-based species indices
        n_stehfest : number of Stehfest terms (even; 12 recommended)

        Returns
        -------
        g1_ij(r) : 0 for r < R_ij, Laplace-inverted value otherwise
        """
        if r < self.radii[i, j]:
            return 0.0
        V = _stehfest_coeffs(n_stehfest)
        ln2 = np.log(2.0)
        total = 0.0
        for n in range(1, n_stehfest + 1):
            s_n = n * ln2 / r
            total += V[n - 1] * self.get_all_H1L_laplace(s_n)[i, j]
        return (ln2 / r ** 2) * total

    def get_g0ij_contact(self, i: int, j: int) -> float:
        """Return the HS-reference RDF contact value g0_ij(R_ij⁺).

        Uses the BMCSL (Boublík-Mansoori-Carnahan-Starling-Leland) algebraic
        formula for additive hard-sphere mixtures:

            g_ij(R_ij⁺) = 1/vac
                         + 3 ζ₂ d_i d_j / ((d_i+d_j) vac²)
                         + 2 ζ₂² (d_i d_j / (d_i+d_j))² / vac³

        where ζ₂ = (π/6) Σ_k ρ_k d_k²  and  vac = 1 − (π/6) Σ_k ρ_k d_k³.

        Reduces to the Carnahan-Starling value (1−η/2)/(1−η)³ for like pairs
        in a single-component fluid.

        Parameters
        ----------
        i, j : 0-based species indices

        Returns
        -------
        g0_ij at contact (scalar)
        """
        di, dj = self.diag_r[i], self.diag_r[j]
        zeta2 = (np.pi / 6.0) * self.xi[2]   # (π/6) Σ ρ_k d_k²
        vac = self.vac
        f = di * dj / (di + dj)               # = d_i/2 for like pairs
        return 1.0 / vac + 3.0 * zeta2 * f / vac ** 2 + 2.0 * zeta2 ** 2 * f ** 2 / vac ** 3

    def get_g1ij_contact(self, i: int, j: int) -> float:
        """First-order perturbation contact value g₁(R_ij⁺).

        Computes the analytical contact limit of the first-order pair
        distribution function using the FMSA residue formula:

            g₁(R_ij⁺) = [2π√(ρᵢρⱼ) · R_ij]⁻¹ · Σ_t B_t_bare_ij

        where B_t_bare accumulates the same four B-matrix term types as
        get_all_H1L_laplace, but with the Yukawa Laplace factor W_t = K_t/(s+z_t)
        replaced by the bare numerator K_t (i.e. the residue at s = −z_t):

            term 1 (direct):      K_{t,ij}
            term 2 (row mixing):  Σ_n K_{t,in} · A_jn(z_{t,in})
            term 3 (col mixing):  Σ_m K_{t,mj} · A_im(z_{t,mj})
            term 4 (quadratic):   Σ_{m,n} K_{t,mn} · A_im(z_{t,mn}) · A_jn(z_{t,mn})

        A-matrix symmetry (A_ij = A_ji) makes the a_z index ordering in terms
        2–3 equivalent to the vectorised form used in get_all_H1L_laplace.

        Called in betaf2 for the LJ hard-core correction, which integrates
        the pair potential between R_ij and σ_ij weighted by g₁ at contact.

        Parameters
        ----------
        i, j : 0-based species indices

        Returns
        -------
        float : g₁(R_ij⁺), first-order RDF contact value
        """
        K_matrices, a_z = self.K_matrices, self.a_z
        R_ij = self.radii[i, j]
        param = 1 / (2 * np.pi * np.sqrt(self.rho[i] * self.rho[j]) * R_ij)
        total = 0.0
        for tail in range(n_tails):
            total += K_matrices[tail, i, j]
            for n in range(self.n_species):
                total += K_matrices[tail, i, n] * a_z[tail, i, n, j, n]
            for m in range(self.n_species):
                total += K_matrices[tail, m, j] * a_z[tail, m, j, i, m]
            for n in range(self.n_species):
                for m in range(self.n_species):
                    total += K_matrices[tail, m, n] * a_z[tail, m, n, i, m] * a_z[tail, m, n, j, n]
        return param * total

    def _make_at_rho(self, rho_vec: np.ndarray) -> "FMSA_poly_term_species":
        """Return a new instance with partial densities rho_vec (same T, sig, eps)."""
        rhostar_new = float(rho_vec.sum())
        x_new = rho_vec / rhostar_new
        return type(self)(self.sig_pure, self.eps_pure, x_new, rhostar_new, self.tstar,
                         n_tails=self.n_tails_active, z=self.z_input, k=self.k_input,
                         verbose=False)



def main() -> None:
    print("Initializing System Parameters...")
    #%% define system density
    tstar = 1.0
    beta = 1.0 / tstar
    rho_paper = 0.139
    rhostar = rho_paper   # was: rho_paper * sqrt(2); sqrt(2) factor was erroneous (GCMC confirmed at rho*=0.139)
    if self.n_species == 3:
        x = np.array([0.20, 0.70, 0.10], dtype=float)
        sig_pure = np.array([1.0, 2.0, 1.5], dtype=float)
        eps_pure = np.array([1.0, 0.5, 1.0], dtype=float)
    elif self.n_species == 2:
        x = np.array([0.25, 0.75], dtype=float)
        sig_pure = np.array([1.0, 2.0], dtype=float)
        eps_pure = np.array([1.0, 0.5], dtype=float)

    FMSA=FMSA_poly_term_species(sig_pure, eps_pure, x, rhostar, tstar) # type: ignore
    
    full_p_coeffs_save=FMSA.full_p_coeffs.reshape((-1,n_terms))
    np.savetxt(f"params_{self.n_species}species_{n_terms}term.txt",full_p_coeffs_save)
    r_eval = np.linspace(0.01, 5.0, 500)
    pair_idx = [(i, j) for i in range(self.n_species) for j in range(i, self.n_species)]
    c_total = FMSA.get_c_total_poly(r_eval)
    #C1s = FMSA.get_C1s_incore_noclip()
    # diagnostic: poly contributions (c_total-c_nopoly, c_total_2-c_nopoly) vs GCMC correction
    c_nopoly = FMSA.get_c_total_nopoly(r_eval)                
    
    #c_total_2 = FMSA.get_c_total_stepwise_poly_p1s(r_eval)
    print("Loading Monte Carlo Simulation Data...")

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="w")
    line_colors = ["b", "r", "k", "g", "m", "c", "y"]
    marker_styles = ["o", "s", "d", "^", "v", ">", "<"]
    line_styles = ["-", "--", "-.", ":"]
    legend_handles = []
    for p, (i, j) in enumerate(pair_idx):
        color = line_colors[p % len(line_colors)]
        marker = marker_styles[p % len(marker_styles)]
        if i==j and i==1:
            ax.plot(r_eval, c_total[p]/3, color=color, linewidth=2,linestyle = line_styles[1],label=f"1/3 poly c_{{{i + 1}{j + 1}}}(r) ")
        else:
            ax.plot(r_eval, c_total[p], color=color, linewidth=2,linestyle = line_styles[1],label=f"poly c_{{{i + 1}{j + 1}}}(r) ")
        #handle, = ax.plot([], [], color=color, marker=marker, markerfacecolor=color, linewidth=1.5,
        #                  label=f"c_{{{i + 1}{j + 1}}}(r) ")
        #legend_handles.append(handle)

    if self.n_species == 2:
        r_star = MC_DATA[:, 0]
        mc_cols = [MC_DATA[:, 1], MC_DATA[:, 2], MC_DATA[:, 3]]
        for p, (i, j) in enumerate(pair_idx):
            color = line_colors[p % len(line_colors)]
            marker = marker_styles[p % len(marker_styles)]
            r_mc = r_star * FMSA.sig_mix[i, j]
            if i==j and i==1:
                ax.plot(r_mc, mc_cols[p]/3, color=color, marker=marker, markersize=5, markerfacecolor=color, linestyle="none",label=f"1/3 MC c_{{{i + 1}{j + 1}}}(r) ")
            else:
                ax.plot(r_mc, mc_cols[p], color=color, marker=marker, markersize=5, markerfacecolor=color, linestyle="none",label=f"MC c_{{{i + 1}{j + 1}}}(r) ")

    ax.axhline(0.0, color="k", linewidth=1)
    for i, j in pair_idx:
        xpos = FMSA.sig_mix[i, j]
        ax.axvline(xpos, color=(0.8, 0.8, 0.8), linewidth=1)

    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(-15.0, 5.0)
    ax.set_xlabel("Absolute Distance r", fontsize=12, fontweight="bold")
    ax.set_ylabel("Direct Correlation Function c_{ij}(r)", fontsize=12, fontweight="bold")
    ax.set_title("FMSA Analytical Theory ", fontsize=14)

    #ax.legend(handles=legend_handles, loc="best", fontsize=11)
    ax.legend( loc="best", fontsize=11)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{filename}_output_compare_bterms{n_terms_b_s}.jpg")
    print("Done! The comparison plot is ready.")
    plt.show()
    plt.close()
    
    # Compare non-poly contributions
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="w")
    for p, (i, j) in enumerate(pair_idx):
        color = line_colors[p % len(line_colors)]
        marker = marker_styles[p % len(marker_styles)]
        r_ij = FMSA.radii[i, j]
        mask = r_eval <= r_ij
        ax.plot(r_eval[mask], (c_total[p]-c_nopoly[p])[mask], color=color, linewidth=2, linestyle=line_styles[1], label=f"poly c_{{{i + 1}{j + 1}}} ")
        
    if self.n_species == 2:
        r_star = MC_DATA[:, 0]
        mc_cols = [MC_DATA[:, 1], MC_DATA[:, 2], MC_DATA[:, 3]]
        for p, (i, j) in enumerate(pair_idx):
            color = line_colors[p % len(line_colors)]
            marker = marker_styles[p % len(marker_styles)]
            r_ij = FMSA.radii[i, j]
            r_mc = r_star * FMSA.sig_mix[i, j]
            mc_mask = r_mc <= r_ij
            mc_nopoly = np.interp(r_mc[mc_mask], r_eval, c_nopoly[p])
            ax.plot(r_mc[mc_mask], mc_cols[p][mc_mask] - mc_nopoly, color=color, marker=marker, markersize=5, markerfacecolor=color, linestyle="none", label=f"MC c_{{{i + 1}{j + 1}}}")

    ax.axhline(0.0, color="k", linewidth=1)
    for i, j in pair_idx:
        xpos = FMSA.sig_mix[i, j]
        ax.axvline(xpos, color=(0.8, 0.8, 0.8), linewidth=1)

    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(-5.0, 5.0)
    ax.set_xlabel("Absolute Distance r", fontsize=12, fontweight="bold")
    ax.set_ylabel("c_{ij}-c^{HS}_{ij}-P^0_{ij}", fontsize=12, fontweight="bold")
    ax.set_title("Contributions except HS+Manually setted P^0 ", fontsize=14)

    #ax.legend(handles=legend_handles, loc="best", fontsize=11)
    ax.legend( loc="best", fontsize=11)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(f"{filename}_contribs_bterms{n_terms_b_s}.jpg")
    print("Done! The comparison plot is ready.")
    plt.show()

    # 1st derivatives of non-poly contributions
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="w")
    dr = r_eval[1] - r_eval[0]
    for p, (i, j) in enumerate(pair_idx):
        color = line_colors[p % len(line_colors)]
        r_ij = FMSA.radii[i, j]
        mask = r_eval <= r_ij
        r_mid = r_eval[mask][1:-1]
        dpoly = np.diff((c_total[p] - c_nopoly[p])[mask]) / dr
        ax.plot(r_mid, dpoly[:-1], color=color, linewidth=2, linestyle=line_styles[1], label=f"d/dr poly c_{{{i + 1}{j + 1}}} ")
        
    if self.n_species == 2:
        r_star = MC_DATA[:, 0]
        mc_cols = [MC_DATA[:, 1], MC_DATA[:, 2], MC_DATA[:, 3]]
        for p, (i, j) in enumerate(pair_idx):
            color = line_colors[p % len(line_colors)]
            marker = marker_styles[p % len(marker_styles)]
            r_ij = FMSA.radii[i, j]
            r_mc = r_star * FMSA.sig_mix[i, j]
            mc_mask = r_mc <= r_ij
            r_mc_in = r_mc[mc_mask]
            mc_contrib = mc_cols[p][mc_mask] - np.interp(r_mc_in, r_eval, c_nopoly[p])
            dr_mc = np.diff(r_mc_in)
            dmc = np.diff(mc_contrib) / dr_mc
            ax.plot(r_mc_in[:-1], dmc, color=color, marker=marker, markersize=5, markerfacecolor=color, linestyle="none", label=f"d/dr MC c_{{{i + 1}{j + 1}}}")

    ax.axhline(0.0, color="k", linewidth=1)
    for i, j in pair_idx:
        ax.axvline(FMSA.sig_mix[i, j], color=(0.8, 0.8, 0.8), linewidth=1)
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(-10.0, 10.0)
    ax.set_xlabel("Absolute Distance r", fontsize=12, fontweight="bold")
    ax.set_ylabel("d/dr [poly contribution]", fontsize=12, fontweight="bold")
    ax.set_title("1st Derivative of Non-Poly Contributions", fontsize=14)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True)
    fig.tight_layout()
    #plt.savefig(f"{filename}_contribs_deriv_bterms{n_terms_b_s}.jpg")
    plt.show()
    
    
    


if __name__ == "__main__":
    main()
