r"""fmsa_compressibility.py -- the compressibility route for an FMSA mixture.

Everything here is read off the direct correlation function at k = 0, which
`FMSA_double_prop` supplies in closed form:

    c_hat_ij(0) = c_hat^HS_ij(0) + c_hat^(1)_ij(0)

  * the first-order part c_hat^(1)(0) (in-core correction plus the MSA tail
    outside the core) comes from the closed-form transform `get_c1hat_k`;
  * the hard-sphere block is chosen by `hs`:
        'fmt'  -d^2(beta F_HS/V)/d rho_i d rho_j for the White-Bear/Rosenfeld
               functional the model was built with (`model.hs_fmt`), evaluated
               ANALYTICALLY from the same `_fmt_A_phi` routine that
               `FMSAFreeEnergy.betaf_hs` uses -- so the compressibility and
               energy routes share one hard-sphere reference;
        'py'   the Percus-Yevick (Baxter/Lebowitz) block that `get_chat_k`
               carries, i.e. the reference FMSA formally perturbs around.

Quantities provided
-------------------
  chat0(model, hs)            c_hat_ij(0), shape (N, N)
  dbetaP_drho(model, hs)      (d beta P / d rho)_{T,x} = 1 - rho sum_ij x_i x_j c_hat_ij(0)
  stability_matrix(model, hs) M_ij = delta_ij/rho_i - c_hat_ij(0); the mixture is
                              locally stable while M is positive definite, and
                              det M = 0 is the spinodal (mechanical AND demixing)
  ray_pressure(model, ...)    beta P along the constant-composition ray 0 -> rho
  ray_free_energy(model, ...) beta F_ex/V along the same ray

Why only along a ray.  The compressibility relation is a derivative at FIXED
composition.  Integrating it in total density at fixed x is exactly its inverse,
so P and F_ex follow uniquely along that ray.  Leaving the ray (chemical
potentials, or F at another composition) would need c_hat(0) to be the exact
Hessian of one free energy, i.e. d c_hat_11/d rho_2 = d c_hat_12/d rho_1, which
no approximate closure satisfies identically; the answer would then depend on
the path.  That is a property of MSA-class closures, not of this code.

The free energy uses one integral rather than two.  With g(t) = dbetaP/drho - 1
along the ray,  beta P_ex(rho) = Int_0^rho g dt  and

    beta F_ex/V (rho) = rho Int_0^rho beta P_ex(s)/s^2 ds = Int_0^rho g(t) (rho - t)/t dt,

the second form by exchanging the order of integration.  g(t) = O(t) as t -> 0,
so the integrand is finite at the origin.
"""
from __future__ import annotations

import numpy as np

from fmsa_double_prop import _fmt_A_phi

#: k used for the k -> 0 limit of the closed-form transform (flat to ~1e-8 below 1e-4)
K0 = 1e-5


def _hs_block_fmt(rho: np.ndarray, diam: np.ndarray, version: str) -> np.ndarray:
    """c_hat^HS_ij(0) = -d^2(beta F_HS/V)/d rho_i d rho_j, exactly.

    In the bulk the vector densities vanish and
        beta F_HS/V = Phi(n) = -n0 ln(1-n3) + n1 n2 A(n3) + n2^3 phi(n3),
    with the weighted densities LINEAR in rho:
        n_alpha = sum_k rho_k w_alpha(R_k),   w = (1, R, 4 pi R^2, 4 pi R^3/3).
    Hence d^2 Phi/d rho_i d rho_j = sum_ab w_a(R_i) Phi_ab w_b(R_j).
    """
    R = 0.5 * np.asarray(diam, dtype=float)
    W = np.vstack([np.ones_like(R), R, 4.0 * np.pi * R**2,
                   (4.0 / 3.0) * np.pi * R**3])            # (4, N)
    n0, n1, n2, n3 = (W @ rho)
    (A, A1, A2), (ph, ph1, ph2) = _fmt_A_phi(float(n3), version)
    v = 1.0 - n3
    H = np.zeros((4, 4))
    H[0, 3] = H[3, 0] = 1.0 / v
    H[1, 2] = H[2, 1] = A
    H[1, 3] = H[3, 1] = n2 * A1
    H[2, 3] = H[3, 2] = n1 * A1 + 3.0 * n2**2 * ph1
    H[2, 2] = 6.0 * n2 * ph
    H[3, 3] = n0 / v**2 + n1 * n2 * A2 + n2**3 * ph2
    return -(W.T @ H @ W)


def chat0(model, hs: str = 'fmt') -> np.ndarray:
    """c_hat_ij(k=0), shape (N, N).  `hs` = 'fmt' or 'py' (see module docstring)."""
    k = np.array([K0])
    c1 = np.real(np.asarray(model.get_c1hat_k(k))[0])
    if hs == 'fmt':
        chs = _hs_block_fmt(np.asarray(model.rho, float), np.asarray(model.diag_r, float),
                            getattr(model, 'hs_fmt', 'wb1'))
    elif hs == 'py':
        chs = np.real(np.asarray(model.get_chat_k(k))[0]) - c1
    else:
        raise ValueError(f"hs must be 'fmt' or 'py', got {hs!r}")
    return chs + c1


def dbetaP_drho(model, hs: str = 'fmt') -> float:
    """(d beta P / d rho)_{T,x} = 1 - rho * sum_ij x_i x_j c_hat_ij(0) = 1/S_xx(0)."""
    rho = np.asarray(model.rho, float)
    x = rho / rho.sum()
    return 1.0 - float(rho.sum() * x @ chat0(model, hs) @ x)


def stability_matrix(model, hs: str = 'fmt') -> np.ndarray:
    """M_ij = delta_ij/rho_i - c_hat_ij(0).  Positive definite = locally stable;
    det M = 0 is the spinodal.  Use `np.linalg.eigvalsh(M)` to see which
    channel (density-like or concentration-like) goes soft first."""
    rho = np.asarray(model.rho, float)
    return np.diag(1.0 / rho) - chat0(model, hs)


def _ray(model, n_gl: int):
    rho = np.asarray(model.rho, float)
    rt = float(rho.sum())
    x = rho / rt
    t, w = np.polynomial.legendre.leggauss(n_gl)
    s = 0.5 * rt * (t + 1.0)                    # nodes on (0, rho)
    return rt, x, s, 0.5 * rt * w


def ray_pressure(model, hs: str = 'fmt', n_gl: int = 24) -> float:
    """beta P at the model's state, integrated along the constant-composition
    ray from rho = 0:  beta P = rho + Int_0^rho [dbetaP/drho - 1] d rho'.
    Gauss-Legendre with `n_gl` nodes; each node is one closed-form evaluation."""
    rt, x, s, w = _ray(model, n_gl)
    g = np.array([dbetaP_drho(model._make_at_rho(x * si), hs) - 1.0 for si in s])
    return rt + float(w @ g)


def ray_free_energy(model, hs: str = 'fmt', n_gl: int = 24) -> float:
    """beta F_ex/V at the model's state, along the constant-composition ray:
        beta F_ex/V = Int_0^rho g(t) (rho - t)/t dt,   g = dbetaP/drho - 1.
    Excess only; the ideal term sum_i rho_i (ln rho_i Lambda_i^3 - 1) is analytic."""
    rt, x, s, w = _ray(model, n_gl)
    g = np.array([dbetaP_drho(model._make_at_rho(x * si), hs) - 1.0 for si in s])
    return float(w @ (g * (rt - s) / s))
