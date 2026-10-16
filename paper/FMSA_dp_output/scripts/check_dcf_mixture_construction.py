"""Check mixture DCF transforms by independent real-space quadrature.

Run from the YK_tail_FMSA workspace. The real-space core is integrated piecewise,
and its analytic bare Yukawa tail is added before comparison with the independent
Baxter/OZ spectral assembly. This checks construction, not finite-coupling accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fmsa_double_prop import FMSA_double_prop


def audit(sigma, composition, density, eps, **kwargs):
    model = FMSA_double_prop(np.asarray(sigma), np.diag(eps),
                            np.asarray(composition), density, 1.0,
                            eps_ij=np.asarray(eps), verbose=False, **kwargs)
    ks = np.array([0.2, 0.7, 2.0, 5.0, 11.0])
    errors = []
    targets = []
    symmetry = []
    for k in ks:
        direct = np.empty((model.N, model.N))
        for i in range(model.N):
            for j in range(model.N):
                contact = model.R_ij[i, j]
                knot = abs(model.lambda_ij[i, j])
                value = quad(
                    lambda r: 4*np.pi*np.sqrt(model.rho[i]*model.rho[j])
                    * r*model.get_c1_inner(r, i, j)*np.sin(k*r)/k,
                    0, contact, points=[knot] if 0 < knot < contact else None,
                    epsabs=2e-11, epsrel=2e-11,
                )[0]
                exterior = np.sum(model.K[:, i, j]*np.exp(-1j*k*contact)
                                  / (1j*k+model.z_poles[:, i, j]))
                direct[i, j] = value - 2*exterior.imag/k
        spectral, _ = model._kspace_C1t_F(np.array([k]))
        errors.append(float(np.max(np.abs(direct-spectral[0]))))
        targets.append(float(np.max(np.abs(spectral[0]))))
        symmetry.append(float(np.max(np.abs(direct-direct.T))))
    return dict(N=model.N, n_tails=model.n_tails, input_type=kwargs['input_type'],
                sigma_input=list(sigma), sigma_core=model.sigma.tolist(),
                composition=list(composition), rho=density, tstar=1.0,
                eps_ij=np.asarray(eps).tolist(), eta=float(model.eta),
                z_poles=model.z_poles.tolist(), wavevectors=ks.tolist(),
                absolute_error=max(errors),
                relative_error=max(errors)/max(targets),
                symmetry_error=max(symmetry))


def main():
    cases = [
        audit([1., 2.], [.25, .75], .139,
              np.sqrt(np.outer([1., .5], [1., .5])), input_type='lj', n_tails=2),
        audit([.8, 1.3, 2.1], [.3, .4, .3], .06,
              [[1., .35, .45], [.35, .8, .3], [.45, .3, .6]],
              input_type='bh', n_tails=2, z=[1.2, 2.6], k=[1., -.2]),
        audit([.8, 1.2, 1.7, 2.3], [.2, .3, .3, .2], .04,
              [[1., .3, .2, .4], [.3, .8, .35, .25],
               [.2, .35, .7, .3], [.4, .25, .3, .6]],
              input_type='bh', n_tails=1, z=1.8),
    ]
    for result in cases:
        if result['relative_error'] >= 1e-9:
            raise AssertionError(result)
    print(json.dumps(dict(normalization='density-weighted radial Fourier Ctilde_1',
                          relative_norm='max absolute discrepancy / max spectral magnitude',
                          cases=cases), indent=2))


if __name__ == '__main__':
    main()
