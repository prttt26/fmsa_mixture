#!/usr/bin/env python3
"""How an LJ + hard-sphere mixture is handed to the theories.

Species 2 is a hard sphere: under Lorentz-Berthelot that means every pair it
takes part in loses its attractive tail, since eps_12 = sqrt(eps_1 * 0) = 0.  The
models already cover this as a LIMIT -- FMSA-DP's tail amplitudes K come out
nonzero only for the 1-1 pair -- but the limit cannot be reached by setting
eps_2 = 0 in the LJ input, because the Barker-Henderson diameter is built from
t = 1/(beta*eps) and eps -> 0 sends the hard core to NaN, not to sigma.  The
diameters have to be supplied directly.

Two diameters, two different integrals
--------------------------------------
LJ species    d = int_0^sigma [1 - exp(-beta u_LJ)] dr, taken from the same
              Cotterman-type correlation every other entry in the study uses, so
              species 1 is treated identically here and everywhere else.  (That
              correlation is 0.6 % off the exact integral; the error is common to
              the whole study, which is the point of keeping it.)

hard sphere   MD cannot integrate a discontinuous potential, so the sphere is the
              usual WCA stand-in (LJ cut at its minimum and shifted up) and its
              diameter is the BH integral of THAT potential -- out to 2^(1/6)
              sigma, not to sigma.  This matters: the WCA sphere is 2.7-4.4 %
              larger than an LJ one of the same sigma (1.0156 vs 0.9730 at
              T* = 1), and since species 2 contributes sigma^3 to the packing,
              using the LJ correlation for it would understate eta by ~10 % at
              size ratio 2.

The cross pair keeps the study's existing convention, R_12 = (d_1 + d_2)/2, i.e.
the additive core built from the two pure-species diameters.  The MD's own 1-2
WCA interaction has its own BH diameter which is not exactly that average -- the
same mapping approximation already present in every LJ entry, not a new one.
"""

from __future__ import annotations

import numpy as np

WCA_CUT = 2.0 ** (1.0 / 6.0)


def d_bh_wca(T: float, sig: float = 1.0, eps: float = 1.0, n: int = 20001) -> float:
    """BH diameter of the WCA potential: integrate to its minimum, not to sigma."""
    rm = WCA_CUT * sig
    r = np.linspace(1e-9, rm, n)
    u = 4.0 * eps * ((sig / r) ** 12 - (sig / r) ** 6) + eps
    return float(np.trapz(1.0 - np.exp(-np.clip(u / T, -500, 500)), r))


def d_bh_lj_correlation(T: float, sig: float = 1.0, eps: float = 1.0) -> float:
    """The correlation the rest of the study uses, so species 1 stays comparable."""
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
    c = np.array([1.1287, -0.05536, 0.0007278])
    t = T / eps
    f = 1.0 + c[0] * t + c[1] * t ** 2 + c[2] * t ** 4
    return float(2.0 ** (1.0 / 6.0) * sig / (1.0 + np.sqrt(f)) ** (1.0 / 6.0))


def model_inputs(potential: str, ratio: float, eps_ratio: float, T: float):
    """-> (sig_pure, eps_pure, kwargs) to splat into the FMSA constructors.

    'lj'   passes the LJ parameters through and lets the class do BH itself.
    'ljhs' passes the LJ SIGMAS as usual -- so the 1-1 tail is parameterised off
           sigma_LJ exactly as every other entry in the study -- with a zero well
           depth for the sphere, which zeroes the tail on both pairs it takes part
           in, and the two hard cores supplied explicitly through `d_pure`.

    Note what is deliberately NOT used: input_type='bh'.  That mode reads
    sig_pure as the diameter matrix, so the Yukawa ranges become z_fac/d and the
    amplitude loses its exp(z(sigma-d)) shift -- z comes out 3.4 % high and
    k_bare 12.2 % low against the 'lj' path at the very same hard cores.  That
    reading is correct for a hard-core Yukawa model, which is what 'bh' exists
    for, and wrong for an LJ species whose core simply does not come from the BH
    correlation.  `d_pure` separates the two roles.
    """
    if potential == 'ljhs':
        d = np.array([d_bh_lj_correlation(T, 1.0, 1.0), d_bh_wca(T, ratio, 1.0)])
        return np.array([1.0, ratio]), np.array([1.0, 0.0]), dict(d_pure=d)
    return np.array([1.0, ratio]), np.array([1.0, float(eps_ratio)]), {}


if __name__ == '__main__':
    print(f"{'T*':>5} {'d_LJ(corr)':>11} {'d_WCA/sig':>10} {'WCA/LJ':>7}")
    for T in (1.0, 1.5, 2.0, 3.0):
        dl, dw = d_bh_lj_correlation(T), d_bh_wca(T)
        print(f'{T:5.1f} {dl:11.5f} {dw:10.5f} {dw/dl:7.4f}')
