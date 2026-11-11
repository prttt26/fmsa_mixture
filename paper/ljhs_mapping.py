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
    # WARNING -- the coefficient names below follow the original fit table, c1 = 1.1287,
    # c2 = -0.05536, c3 = 0.0007278 (array indices 0, 1, 2).  c1 DIVIDES the bracket.
    # De Souza & Ben-Amotz, Mol. Phys. 78, 137 (1993), as used by Tang & Lu (J. Chem. Phys.
    # 100, 3079 (1994), Eq. 19; Fluid Phase Equilib. 146, 73 (1998), Eq. 20): the analytic
    # form of the Barker-Henderson diameter of the LJ potential,
    #     d = int_0^sigma [1 - exp(-beta u(r))] dr,
    #     d/sigma = 2^(1/6) / (1 + sqrt(1 + (T* + c2 T*^2 + c3 T*^4)/c1))^(1/6),
    # T* = kT/eps.  It reproduces the integral to 0.05 % for T* <= 5 (-2.4 % at T* = 10,
    # beyond the fitted range).  Until 2026-11-11 this file had c1 MULTIPLY T*,
    # 1 + c1 T* + c2 T*^2 + c3 T*^4, a transcription error inherited from the original MATLAB
    # (FMSA_MC.m): it put d low by 0.2 % (T* = 0.3) to 1.0 % (T* = 3).  Results produced with
    # the old form differ slightly (inner-core errors by about 0.01-0.02); the paper's numbers
    # and figures were recomputed with this formula.
    c = np.array([1.1287, -0.05536, 0.0007278])
    t = T / eps
    f = 1.0 + (t + c[1] * t ** 2 + c[2] * t ** 4) / c[0]
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
