r"""Model-independent compressibility-route thermodynamics on an isopleth.

The model supplies its own zero-wave-number direct correlation matrix or
its own derivative of beta P with respect to total density.  In an
approximate closure different routes need not yield the same free energy.
"""
from __future__ import annotations

from collections.abc import Callable
import math
import numpy as np


def zero_mode_from_even_transform(transform: Callable[[float], object], step: float,
                                  rtol: float = 1e-8, atol: float = 1e-10):
    r"""Estimate a finite zero mode from an even, smooth transform near k=0.

    Evaluate at ``step``, ``step/2`` and ``step/4`` and compare two
    Richardson extrapolations. The caller chooses a step in the transform's
    wave-number units. This is a convergence check, not a rigorous error bound;
    it assumes a regular expansion in even powers and a valid model branch.
    """
    step, rtol, atol = float(step), float(rtol), float(atol)
    if not all(map(math.isfinite, (step, rtol, atol))) or step <= 0 or rtol < 0 or atol < 0:
        raise ValueError('step must be positive and tolerances nonnegative and finite')
    samples = [np.asarray(transform(step / divisor), dtype=float) for divisor in (1, 2, 4)]
    if any(a.shape != samples[0].shape or not np.all(np.isfinite(a)) for a in samples):
        raise ValueError('transform must return finite values of consistent shape')
    coarse = (4 * samples[1] - samples[0]) / 3
    fine = (4 * samples[2] - samples[1]) / 3
    if not np.allclose(coarse, fine, rtol=rtol, atol=atol):
        raise ValueError('zero-mode extrapolation did not converge; check the model and step')
    return float(fine) if fine.ndim == 0 else fine


def pressure_derivative_from_dcf(rho: float, composition, c_hat_zero) -> float:
    r"""Return d(beta P)/d rho = 1 - rho x.T c_hat(0) x at fixed composition.

    ``c_hat_zero`` is the unsymmetrized matrix of three-dimensional Fourier
    transforms, with no density factors absorbed into it.
    """
    rho = float(rho)
    x = np.asarray(composition, dtype=float)
    c = np.asarray(c_hat_zero, dtype=float)
    if not math.isfinite(rho) or rho < 0 or x.ndim != 1 or x.size == 0:
        raise ValueError('rho must be nonnegative and composition a nonempty vector')
    if not np.all(np.isfinite(x)) or np.any(x < 0) or not np.isclose(x.sum(), 1):
        raise ValueError('composition must be finite, nonnegative and sum to one')
    if c.shape != (x.size, x.size) or not np.all(np.isfinite(c)):
        raise ValueError('c_hat_zero must be a finite square matrix matching composition')
    return float(1 - rho * (x @ c @ x))


def free_energy_difference(rho: float, derivative: Callable[[float], float],
                           reference: Callable[[float], float] = lambda _rho: 1.0,
                           n_rho: int = 40) -> float:
    r"""Return beta(f - f_reference) per volume on a fixed-composition ray.

    Both callbacks return d(beta P)/d rho at their supplied total density.
    The callbacks must represent the same composition, temperature and units,
    with the free-energy difference tending to zero sufficiently fast at zero
    density.  The reference defaults to the ideal gas.  The density path must
    stay on a valid model branch; callback exceptions propagate to the caller.
    """
    rho = float(rho)
    if not math.isfinite(rho) or rho < 0:
        raise ValueError('rho must be finite and nonnegative')
    if not isinstance(n_rho, int) or n_rho < 1:
        raise ValueError('n_rho must be a positive integer')
    if rho == 0:
        return 0.0
    nodes, weights = np.polynomial.legendre.leggauss(n_rho)
    points = 0.5 * rho * (nodes + 1)
    total = 0.0
    for point, weight in zip(points, weights):
        delta = float(derivative(float(point))) - float(reference(float(point)))
        if not math.isfinite(delta):
            raise ValueError(f'nonfinite pressure derivative at rho={point}')
        total += weight * delta * (1 / point - 1 / rho)
    return float(0.5 * rho * rho * total)


def free_energy_from_dcf(rho: float, composition, c_hat_zero_at_density: Callable[[float], object],
                         reference: Callable[[float], float] = lambda _rho: 1.0,
                         n_rho: int = 40) -> float:
    r"""Integrate a closure's zero-wave-number DCF on a composition ray.

    The callback must solve or evaluate that closure at the requested total
    density and return the unsymmetrized matrix c_hat(0).  The reference has
    the same meaning as in ``free_energy_difference``.
    """
    x = np.asarray(composition, dtype=float)
    pressure_derivative_from_dcf(0.0, x, np.zeros((x.size, x.size)))
    return free_energy_difference(
        rho, lambda r: pressure_derivative_from_dcf(r, x, c_hat_zero_at_density(r)),
        reference=reference, n_rho=n_rho)


def coupling_free_energy_difference(derivative: Callable[[float], float],
                                    n_coupling: int = 24) -> float:
    r"""Integrate d(beta f)/d lambda from zero to full coupling.

    The callback supplies the model's energy-route integrand per volume at
    fixed temperature, density, and composition.  Its definition must hold
    the model's reference and all other parameters fixed along the path.
    """
    if not isinstance(n_coupling, int) or n_coupling < 1:
        raise ValueError('n_coupling must be a positive integer')
    nodes, weights = np.polynomial.legendre.leggauss(n_coupling)
    total = 0.0
    for node, weight in zip(nodes, weights):
        coupling = 0.5 * (float(node) + 1)
        value = float(derivative(coupling))
        if not math.isfinite(value):
            raise ValueError(f'nonfinite coupling integrand at lambda={coupling}')
        total += float(weight) * value
    return 0.5 * total
