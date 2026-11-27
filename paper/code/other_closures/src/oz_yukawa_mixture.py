"""oz_yukawa_mixture.py -- the model object that the numerical OZ solver works on.

`OZSolverMSA` (oz_numerical.py) solves the Ornstein--Zernike equation numerically under the
MSA or HNCB closure. It needs an object that supplies, for a Yukawa mixture, the hard cores,
densities and Yukawa tails and the hard-sphere reference c^HS_ij(r) (White Bear FMT), and
nothing about FMSA. This class is that object.

It replaces `FMSA_GA_matrix_mix`, which played this role until 2026-11-11 and also carried an
earlier first-order FMSA implementation that is wrong (the reason its folder is
`archive/fmsa_dcf_wrong/`). That implementation is kept in the archive only; live code does
not import it. Nothing is re-implemented here: the potential (Lorentz--Berthelot mixing, the
Barker--Henderson hard cores, the Yukawa tails) and the hard-sphere reference come from
`FMSA_double_prop`, the maintained implementation, so there is ONE definition of each.
Against the old class the hard cores, densities and tails are bit-identical and the
hard-sphere reference agrees to ~1e-15, so the converged OZ-MSA and OZ-HNCB curves agree to
~1e-14 with identical iteration counts.

The constructor takes the arguments of the old class, so a call site only changes the class
name:

    OZYukawaMixture(sig_pure, eps_pure, x, rhostar, tstar, n_tails=2, z=None, k=None,
                    eps_ij=None, input_type='lj', d_pure=None, hs_fmt='wb1')
"""
from __future__ import annotations

from fmsa_double_prop import FMSA_double_prop

# What the OZ solver and the scripts around it read from the model.
_EXPOSED = ('N', 'rho', 'beta', 'n_tails', 'R_ij', 'z_poles', 'K', 'k_bare', 'epsilon',
            'd_bh', 'sigma', 'sig_mix', 'eps_mix', 'lambda_ij', 'eta', 'xi', 'hs_fmt',
            'tstar', 'rhostar', 'x')


class OZYukawaMixture:
    """Hard-core Yukawa mixture as seen by the numerical OZ solver."""

    def __init__(self, sig_pure, eps_pure, x, rhostar, tstar, n_tails=2, z=None, k=None,
                 eps_ij=None, input_type='lj', d_pure=None, hs_fmt='wb1', **kwargs):
        kwargs.setdefault('verbose', False)
        kwargs.setdefault('lazy_radial', True)       # the radial tables are not needed here
        self._model = FMSA_double_prop(
            sig_pure, eps_pure, x, rhostar, tstar, n_tails=n_tails, z=z, k=k,
            eps_ij=eps_ij, input_type=input_type, d_pure=d_pure, hs_fmt=hs_fmt, **kwargs)

    def __getattr__(self, name):
        # only reached for names not set on the instance
        if name in _EXPOSED:
            return getattr(self._model, name)
        raise AttributeError(name)

    def get_HS_FMT(self, r, i, j):
        """White Bear FMT hard-sphere pair direct correlation function c^HS_ij(r)."""
        return self._model.get_HS_FMT(r, i, j)

    def get_c1_inner(self, r, i, j):
        """First-order (FMSA) in-core correction, used only to seed the solver."""
        return self._model.get_c1_inner(r, i, j)
