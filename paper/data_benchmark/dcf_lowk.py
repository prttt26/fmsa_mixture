#!/usr/bin/env python3
"""Repair the long-wavelength (small-k / large-r) part of a simulated DCF.

The problem
-----------
A DCF obtained by OZ-inverting a simulated h_ij(r) is wrong at large r, and the
error is not noise: at the validation state c22(r) sits at +0.50 near r = 3
sigma22 where the exact asymptote -beta*u22 is +0.003 -- 180x too large, smooth,
and monotone.  The same defect shows up as a 47 % error in chat22(0).

The cause is the one place where a fixed-N simulation cannot represent the bulk:
the k -> 0 limit.  Long-wavelength density fluctuations are suppressed in a
finite canonical box, so S_ij(k) is biased there, and the inversion
chat = I - S^{-1} amplifies that bias by ||S^{-1}||^2 -- 453x at rho* = 0.139.
The system-size ladder measures it directly: from N = 2048 to N = 8192 the
validation-state S22 moves 1729 % at k = 0.15, 117 % at k = 0.30, and 1.9 % at
k = 0.50 (all three pairs are inside 1 % for k >= 1).

The cure (Zhao & Wu, Mol. Phys. 109, 2553 (2011), for site-site water)
----------------------------------------------------------------------
Take each part of the k axis from whatever source is reliable there:

  k >= k_split   simulation.  The inversion is well conditioned and the box no
                 longer matters.
  k  = 0         bulk thermodynamics.  chat_ij(0) is a macroscopic object -- it
                 is fixed by the equation of state and the concentration
                 fluctuations, and needs no structural input at all.
  0 < k < k_split  a C1 even-polynomial bridge between the two.  chat_ij(k) is
                 even and analytic at k = 0, so a + b k^2 + d k^4 is its own
                 leading behaviour, not an ansatz (their Equation (12)).

For a simple mixture the k = 0 anchor is three convention-free equations:

    d(betaP)/d(rho_i) = 1 - sum_j rho_j chat_ij(0)                    (i = 1, 2)
    S_CC(0)           = x1 x2 (x2 S11 + x1 S22 - 2 sqrt(x1 x2) S12)
    with  S(0) = [I - sqrt(rho_i rho_j) chat(0)]^{-1}

The first two come from the measured P(rho_1, rho_2) surface.  The third needs a
free-energy curvature in composition; S_CC is used because it is the one small-k
combination the size ladder shows is robust (-3.3 % from N = 2048 to N -> inf,
against +67.8 % for S22), so the simulation may supply it.

Verification is by an observable that is nowhere in the construction: with the
small-k band replaced, c_ij(r) in the outer region must collapse onto the exact
asymptote c_ij -> -beta*u_ij.  That collapse is not imposed anywhere below.

Usage
-----
    python dcf_lowk.py --label validation                    # one MD run
    python dcf_lowk.py --literature T1_eps0.50_x0.2500       # one TN 1331 table
    python dcf_lowk.py --all                                 # everything anchorable
    python dcf_lowk.py --label validation --write            # emit c_lowk.csv

Write-up: numerical_notes/results/benchmark/repair_lowk/dcf_lowk_repair.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from structure_transforms import (PAIR_TAG, PAIRS, dst_backward, dst_forward,
                                  make_grid)

HERE = Path(__file__).parent
RUNS = HERE / 'md' / 'runs'

# Where the box stops mattering, in units of 1/sigma_11.  Measured on the
# N = 512..8192 ladder at the validation state: all three S_ij agree to <1 %
# between N = 2048 and 8192 for k >= 1, and to 1.9 % at k = 0.5.  Taking the
# conservative end costs nothing -- the bridge is exact where the two agree.
K_SPLIT = 1.0

# Beyond this the OZ-inverted c is discarded outright when reporting the outer
# region, in units of sigma_ij.  Chosen where -beta*u still dominates the
# O(h^2) correction to the asymptote at liquid densities.
R_ASYMPTOTE = 2.5

# Flat fraction of the level-correcting bump: constant out to LEVEL_FLAT * R,
# raised cosine to zero at R.  See the note in repair() for why flat.
LEVEL_FLAT = 0.7

# Reject a state where the UNCERTAIN part of the level correction reaches this
# fraction of the local rms of c.  Gating on the relative error of chat22(0)
# instead would be the wrong test: where the taper has already removed the
# artifact the leftover bump is small, and then even a poorly resolved
# chat22(0) moves c by nothing.
LEVEL_MAX_REL_SE = 0.05


def lj_matrices(ratio: float, eps_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Lorentz-Berthelot sigma_ij and eps_ij for the binary, sigma_11 = eps_11 = 1."""
    s = np.array([1.0, ratio])
    e = np.array([1.0, eps_ratio])
    return 0.5 * (s[:, None] + s[None, :]), np.sqrt(e[:, None] * e[None, :])


def beta_u(r: np.ndarray, sigma: float, eps: float, T: float,
           rc: float | None = None) -> np.ndarray:
    """beta * u_ij for the LJ 12-6, truncated (not shifted) at rc if given."""
    x = sigma / np.maximum(r, 1e-30)
    u = 4.0 * eps * (x ** 12 - x ** 6) / T
    return np.where(r > rc, 0.0, u) if rc is not None else u


def tail_chat0(R: float, sigma: float, eps: float, T: float,
               rc: float | None = None) -> float:
    """4 pi Int_R^inf r^2 (-beta u_ij) dr, in closed form."""
    def prim(a: float) -> float:
        return -(16.0 * np.pi * eps / T) * (sigma ** 12 / (9 * a ** 9)
                                            - sigma ** 6 / (3 * a ** 3))
    if rc is None:
        return prim(R)
    return 0.0 if R >= rc else prim(R) - prim(rc)


# --------------------------------------------------------------------------
# the k = 0 anchor, from bulk thermodynamics
# --------------------------------------------------------------------------

@dataclass
class Anchor:
    """chat_ij(0) and where each of its three inputs came from."""
    chat0: np.ndarray
    dP_drho: tuple[float, float]
    S_CC: float
    S0: np.ndarray
    residual: float
    sigma: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    provenance: dict = field(default_factory=dict)


def pressure_surface(ratio: float, eps_ratio: float,
                     runs: Path = RUNS) -> list[tuple[str, float, float, float, float]]:
    """(label, rho_1, rho_2, T, betaP) for every LAMMPS run at this (ratio, eps).

    All temperatures together: the fit below carries T explicitly, so pooling
    them is what makes the thin cells (four runs at some T) usable at all, and
    it is what lets a T* = 3 literature state be anchored.
    """
    out = []
    for d in sorted(p for p in runs.iterdir() if p.is_dir()):
        pj, tj = d / 'params.json', d / 'thermo.json'
        if not (pj.exists() and tj.exists()):
            continue
        par = json.loads(pj.read_text())
        if (abs(par['ratio'] - ratio) > 1e-9
                or abs(par.get('eps_ratio', -1) - eps_ratio) > 1e-9):
            continue
        rho, x1, T = par['rho'], par['x1'], par['T']
        out.append((d.name, rho * x1, rho * (1 - x1), T,
                    json.loads(tj.read_text())['P_star'] / T))
    return out


def _terms(deg: int, n_inv: int):
    return [(a, b, p) for p in range(n_inv)
            for a in range(deg + 1) for b in range(deg + 1) if 0 < a + b <= deg]


def fit_dP_drho(rows, rho1: float, rho2: float, T: float,
                deg: int = 4, n_inv: int = 3):
    """Both partials of betaP(rho_1, rho_2, T) at one state.

        betaP = sum_{0 < a+b <= deg} (alpha + gamma/T + delta/T^2) rho1^a rho2^b

    No constant term, so betaP -> 0 as rho -> 0 is built in rather than fitted.
    The 1/T powers are the high-temperature expansion of the attractive part,
    so the temperature dependence is shaped by physics, not free-form.

    Validated at ratio 2 / eps 0.5 (71 runs, T* = 1, 1.5, 2, 3): rms 0.0017 in
    betaP, the fixed-composition ray derivative 18.35 against the dedicated
    eos_rho* series' measured 18.12 (1.2 %), and leave-one-out on the seven eos
    points inside 0.005 except at the top density (0.026).
    """
    d1 = np.array([r[1] for r in rows])
    d2 = np.array([r[2] for r in rows])
    Ts = np.array([r[3] for r in rows])
    P = np.array([r[4] for r in rows])
    tm = _terms(deg, n_inv)
    A = np.column_stack([d1 ** a * d2 ** b / Ts ** p for a, b, p in tm])
    coef, *_ = np.linalg.lstsq(A, P, rcond=None)
    resid = P - A @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    val = float(sum(c * rho1 ** a * rho2 ** b / T ** p
                    for c, (a, b, p) in zip(coef, tm)))
    # the two partials are linear functionals of the coefficients, so their
    # standard errors follow from the fit covariance sigma^2 (A^T A)^-1.  This
    # is what separates a state the surface actually resolves from one it is
    # merely extrapolating over -- the convex hull alone does not, because
    # chat22(0) = (1 - M22)/rho_2 divides by a small number at dilute x2.
    g1 = np.array([c_ * a * rho1 ** (a - 1) * rho2 ** b / T ** p if a else 0.0
                   for c_, (a, b, p) in zip(np.ones(len(tm)), tm)])
    g2 = np.array([c_ * b * rho1 ** a * rho2 ** (b - 1) / T ** p if b else 0.0
                   for c_, (a, b, p) in zip(np.ones(len(tm)), tm)])
    d_1 = float(g1 @ coef)
    d_2 = float(g2 @ coef)
    dof = max(len(P) - len(tm), 1)
    cov = float(np.sum(resid ** 2)) / dof * np.linalg.pinv(A.T @ A)
    se1 = float(np.sqrt(max(g1 @ cov @ g1, 0.0)))
    se2 = float(np.sqrt(max(g2 @ cov @ g2, 0.0)))
    return val, d_1, d_2, rms, se1, se2


def inside_coverage(rows, rho1: float, rho2: float, T: float,
                    margin: float = 1e-9) -> str:
    """'' if this state is inside the pressure surface's own data, else why not.

    A polynomial surface fitted on 71 runs says nothing reliable outside their
    hull, and chat22(0) = (1 - M22)/rho_2 divides by a small number at dilute
    compositions -- so an extrapolated pressure derivative arrives amplified.
    Every anchor that failed physically in the first full pass failed this test
    first, which is why it is applied before, not after.
    """
    d1 = np.array([r[1] for r in rows])
    d2 = np.array([r[2] for r in rows])
    Ts = np.array([r[3] for r in rows])
    if not (Ts.min() - margin <= T <= Ts.max() + margin):
        return f'T* {T:g} outside the sampled {Ts.min():g}..{Ts.max():g}'
    try:
        from scipy.spatial import Delaunay
        hull = Delaunay(np.column_stack([d1, d2]))
        if hull.find_simplex(np.array([[rho1, rho2]])) < 0:
            return (f'(rho_1, rho_2) = ({rho1:.4f}, {rho2:.4f}) is outside the '
                    f'hull of the {len(rows)} runs')
    except Exception:                       # degenerate point set: fall back to a box
        if not (d1.min() <= rho1 <= d1.max() and d2.min() <= rho2 <= d2.max()):
            return f'(rho_1, rho_2) = ({rho1:.4f}, {rho2:.4f}) outside the run box'
    return ''


def affordable_fit(n_rows: int, deg_max: int = 4, margin: int = 5
                   ) -> tuple[int, int]:
    """The richest (degree, 1/T order) this many runs can carry.

    The eps cells are very unequally sampled (71 runs at eps 0.5, 8 at 1.5), so
    a fixed degree either wastes the dense cells or overfits the thin ones.
    Two 1/T powers are the floor: with none, a fit pooled over T* = 1..3 has no
    way to represent the attraction at all.
    """
    for deg in range(deg_max, 1, -1):
        for n_inv in (3, 2):
            if n_rows >= len(_terms(deg, n_inv)) + margin:
                return deg, n_inv
    raise ValueError(f'only {n_rows} runs -- not enough for any pressure surface '
                     f'(a degree-2 surface with 1/T^1 needs {len(_terms(2, 2)) + margin})')


def solve_anchor(dP1: float, dP2: float, S_CC: float,
                 rho: np.ndarray) -> tuple[np.ndarray, float]:
    """The three equations of the module docstring, in closed form.

    Write M = I - sqrt(rho_i rho_j) chat(0), so that S(0) = M^{-1}.  Both
    pressure equations are linear in M,

        M11 + sqrt(x2/x1) M12 = dP1,    M22 + sqrt(x1/x2) M12 = dP2,

    which leaves one unknown, M12.  Two cancellations then finish it.  In
    det M = M11 M22 - M12^2 the quadratic term cancels exactly, so det M is
    LINEAR in M12; and in the S_CC numerator the M12 terms cancel exactly, so

        S_CC = x1 x2 (x1 dP1 + x2 dP2) / det M

    with the bracket equal to the fixed-composition ray derivative.  So S_CC
    fixes det M outright, and M12 follows by one division:

        det M = x1 x2 (d betaP/d rho)_x / S_CC
        M12   = sqrt(x1 x2) [dP1 dP2 - det M] / (d betaP/d rho)_x

    No iteration, hence nothing to diverge at the dilute compositions where a
    Newton solve on the original three equations runs away.
    """
    x = rho / rho.sum()
    x1, x2 = float(x[0]), float(x[1])
    ray = x1 * dP1 + x2 * dP2
    det_M = x1 * x2 * ray / S_CC
    m12 = np.sqrt(x1 * x2) * (dP1 * dP2 - det_M) / ray
    m11 = dP1 - np.sqrt(x2 / x1) * m12
    m22 = dP2 - np.sqrt(x1 / x2) * m12
    c = np.array([[(1 - m11) / rho[0], -m12 / np.sqrt(rho[0] * rho[1])],
                  [-m12 / np.sqrt(rho[0] * rho[1]), (1 - m22) / rho[1]]])
    M = np.eye(2) - np.sqrt(np.outer(rho, rho)) * c
    resid = max(abs(M[0, 0] * M[1, 1] - M[0, 1] ** 2 - det_M),
                abs(1 - (rho[0] * c[0, 0] + rho[1] * c[0, 1]) - dP1),
                abs(1 - (rho[0] * c[0, 1] + rho[1] * c[1, 1]) - dP2))
    return c, float(resid)


def anchor_is_physical(c0: np.ndarray, rho: np.ndarray) -> str:
    """'' if S(0) = [I - sqrt(rho rho) chat(0)]^{-1} is a legal structure factor.

    S(0) must be positive definite (it is a fluctuation covariance) and its
    diagonal positive.  A macroscopic input that violates this says the
    pressure surface has been extrapolated somewhere it does not hold, not
    that the fluid is strange -- so it is a gate, not a warning.
    """
    M = np.eye(2) - np.sqrt(np.outer(rho, rho)) * c0
    d = np.linalg.det(M)
    if d <= 0:
        return f'det[I - sqrt(rho rho) chat(0)] = {d:.3g} <= 0'
    S = np.linalg.inv(M)
    if S[0, 0] <= 0 or S[1, 1] <= 0:
        return f'S(0) diagonal not positive: {S[0,0]:.3g}, {S[1,1]:.3g}'
    if S[0, 0] * S[1, 1] - S[0, 1] ** 2 <= 0:
        return 'S(0) is not positive definite'
    return ''


def macro_anchor(ratio: float, eps_ratio: float, T: float, rho: float,
                 x1: float, S_CC: float, deg: int = 4,
                 dP_ray: float | None = None, runs: Path = RUNS,
                 S_CC_se: float = 0.0) -> Anchor:
    """chat_ij(0) from the pressure surface plus one concentration fluctuation.

    `dP_ray`, if given, is a directly measured d(betaP)/d(rho) along the fixed-
    composition ray; both fitted partials are rescaled to reproduce it, which
    replaces the fit's weakest number by a measured one without disturbing the
    composition dependence the fit is there to supply.
    """
    rows = pressure_surface(ratio, eps_ratio, runs)
    deg, n_inv = affordable_fit(len(rows), deg)
    rho_p = np.array([x1, 1 - x1]) * rho
    bad = inside_coverage(rows, rho_p[0], rho_p[1], T)
    if bad:
        raise ValueError(f'no macroscopic anchor here -- {bad}')
    P, dP1, dP2, rms, se1, se2 = fit_dP_drho(rows, rho_p[0], rho_p[1], T, deg, n_inv)
    x = np.array([x1, 1 - x1])
    scale = 1.0
    if dP_ray is not None:
        scale = dP_ray / (x[0] * dP1 + x[1] * dP2)
        dP1, dP2 = dP1 * scale, dP2 * scale
    c0, res = solve_anchor(dP1, dP2, S_CC, rho_p)
    # Propagate (dP1, dP2, S_CC) through the closed form.  S_CC's own error is
    # taken as the larger of its fit scatter and the 3.3 % finite-size shift the
    # size ladder measured, since that shift is systematic and does not average
    # away over states.
    sc_se = max(S_CC_se, 0.033 * abs(S_CC))
    sig = np.zeros((2, 2))
    for dv, se in ((np.array([1.0, 0.0, 0.0]), se1),
                   (np.array([0.0, 1.0, 0.0]), se2),
                   (np.array([0.0, 0.0, 1.0]), sc_se)):
        if se <= 0:
            continue
        up, _ = solve_anchor(dP1 + dv[0] * se, dP2 + dv[1] * se,
                             S_CC + dv[2] * se, rho_p)
        sig += (up - c0) ** 2
    sig = np.sqrt(sig)
    # The covariance sees the scatter of the runs about the surface; it does not
    # see the surface's own form.  At the dilute-x2 states those differ by an
    # order of magnitude, so add the spread over every distinct affordable
    # (degree, 1/T order) as an independent model-form term.
    alt = []
    for dg in range(2, deg + 1):
        for ni in (2, 3):
            try:
                if len(_terms(dg, ni)) + 5 > len(rows):
                    continue
                _, a1, a2, _, _, _ = fit_dP_drho(rows, rho_p[0], rho_p[1], T, dg, ni)
                if dP_ray is not None:
                    a1, a2 = (a1 * dP_ray / (x[0] * a1 + x[1] * a2),
                              a2 * dP_ray / (x[0] * a1 + x[1] * a2))
                alt.append(solve_anchor(a1, a2, S_CC, rho_p)[0])
            except Exception:
                continue
    if len(alt) > 1:
        stack = np.stack(alt)
        sig = np.sqrt(sig ** 2 + (0.5 * (stack.max(0) - stack.min(0))) ** 2)
    if not np.all(np.isfinite(sig)):
        raise ValueError('anchor uncertainty is not finite here')
    bad = anchor_is_physical(c0, rho_p)
    if bad:
        raise ValueError(f'macroscopic anchor is unphysical here -- {bad}')
    S0 = np.linalg.inv(np.eye(2) - np.sqrt(np.outer(rho_p, rho_p)) * c0)
    return Anchor(chat0=c0, dP_drho=(dP1, dP2), S_CC=S_CC, S0=S0, residual=res,
                  sigma=sig,
                  provenance=dict(n_runs=len(rows), fit_degree=deg,
                                  fit_n_inv=n_inv, fit_rms=rms, P_fit=P,
                                  ray_rescale=scale, dP_ray=dP_ray,
                                  se_dP=(se1, se2), se_S_CC=sc_se))


# --------------------------------------------------------------------------
# the k-space repair
# --------------------------------------------------------------------------

def transform_h(r: np.ndarray, k: np.ndarray, h: np.ndarray) -> np.ndarray:
    """hhat_ij(k), the one place the simulated structure enters."""
    hk = np.empty((2, 2, k.size))
    for i, j in PAIRS:
        hk[i, j] = hk[j, i] = dst_forward(h[i, j], r, k)
    return hk


def invert(hk: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Chat = Hhat (I + D Hhat)^{-1}, the raw OZ inversion."""
    d = np.diag(rho)
    ck = np.empty_like(hk)
    for n in range(hk.shape[2]):
        ck[:, :, n] = hk[:, :, n] @ np.linalg.inv(np.eye(2) + d @ hk[:, :, n])
    return ck


def _taper(r: np.ndarray, r1: float, r2: float) -> np.ndarray:
    """1 below r1, 0 above r2, raised cosine between -- C1 at both ends."""
    w = np.ones_like(r)
    mid = (r > r1) & (r < r2)
    w[mid] = 0.5 * (1 + np.cos(np.pi * (r[mid] - r1) / (r2 - r1)))
    w[r >= r2] = 0.0
    return w


def artifact_level(r: np.ndarray, c_raw: np.ndarray, sigma: np.ndarray,
                   eps: np.ndarray, T: float, rc: float | None,
                   far: float = 4.0) -> dict[str, float]:
    """How large the long-wavelength artifact is, per pair, read off the far field.

    Beyond `far` sigma_11 the exact asymptote is negligible, so whatever the
    inverted c_ij still shows there IS the artifact, and -- because it is a
    long-wavelength object -- roughly that size everywhere.  This is the number
    that says how far in the inversion can still be trusted.
    """
    out = {}
    m = (r >= far) & (r <= r[-1])
    for i, j in PAIRS:
        d = c_raw[i, j][m] + beta_u(r[m], sigma[i, j], eps[i, j], T, rc)
        out[PAIR_TAG[(i, j)]] = float(np.sqrt(np.mean(d ** 2)))
    return out


def keep_radius(nu: float, sigma: float, eps: float, T: float,
                hi_factor: float = 8.0) -> float:
    """Where the exact tail -beta*u_ij drops to the artifact level nu.

    Inside this radius the inversion still carries more signal than artifact;
    outside it the measurement has nothing left to say and the asymptote is the
    better estimator.  Per pair, and set by the data rather than by a round
    multiple of sigma_ij.

    The search starts at the potential minimum: |beta u| is not monotone below
    it (it passes through zero at r = sigma), so bisecting from smaller r finds
    the repulsive branch and reports a radius inside the core.
    """
    lo = 2.0 ** (1 / 6) * sigma
    hi = hi_factor * sigma
    f = lambda x: abs(beta_u(np.array([x]), sigma, eps, T, None)[0]) - nu
    if f(lo) <= 0:                       # even the well depth is below the noise
        return lo
    if f(hi) > 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def bridge(k: np.ndarray, ck: np.ndarray, c0: float, k_split: float) -> np.ndarray:
    """Replace ck below k_split by the even quartic through (0, c0), C1 at k_split.

    chat is even and analytic at k = 0, so a + b k^2 + d k^4 is its own leading
    behaviour there, not an ansatz.  Matching value and slope at k_split leaves
    no free parameter.
    """
    n = int(np.searchsorted(k, k_split))
    if n < 3 or n >= k.size - 2:
        raise ValueError(f'k_split {k_split} is off the grid')
    K, V = k[n], ck[n]
    sl = np.polyfit(k[n - 2:n + 3] ** 2, ck[n - 2:n + 3], 1)[0] * 2 * K
    w = (K * sl - 2 * (V - c0)) / 2.0
    u = (V - c0) - w
    out = ck.copy()
    t = (k[:n] / K) ** 2
    out[:n] = c0 + u * t + w * t ** 2
    return out


def repair(r: np.ndarray, k: np.ndarray, c_raw: np.ndarray, chat0: np.ndarray,
           sigma: np.ndarray, eps: np.ndarray, T: float, rc: float | None,
           k_split: float = K_SPLIT, far: float = 4.0, width: float = 1.5,
           n_iter: int = 40, tol: float = 1e-6, nu_override=None,
           chat0_sigma: np.ndarray | None = None):
    """Impose the two things the simulation cannot supply, and nothing else.

    (A) in r: outside the pair's keep radius, c_ij is the exact asymptote
        -beta*u_ij, tapered in over [R, width*R] so no step is introduced.
    (B) in k: below k_split -- where the size ladder shows the box still
        matters -- chat_ij is the even quartic anchored at the macroscopic
        chat_ij(0) and matched to the simulation at k_split.

    Neither constraint alone survives the other: (A) changes every k and (B)
    changes every r.  They are convex sets, so alternating the two projections
    converges to a c satisfying both, and the correction is then shaped like the
    error actually is -- long-wavelength -- rather than by a guessed bump.

    Returns (c, info) with info carrying the per-pair keep radii, the iteration
    count, and the residuals of both constraints at the end.
    """
    nu = nu_override or artifact_level(r, c_raw, sigma, eps, T, rc, far)
    R = {PAIR_TAG[(i, j)]: keep_radius(nu[PAIR_TAG[(i, j)]], sigma[i, j],
                                       eps[i, j], T) for i, j in PAIRS}
    asym, w, phi, quality, applied = {}, {}, {}, {}, {}
    for i, j in PAIRS:
        t = PAIR_TAG[(i, j)]
        asym[t] = -beta_u(r, sigma[i, j], eps[i, j], T, rc)
        w[t] = _taper(r, R[t], width * R[t])
        p = _taper(r, LEVEL_FLAT * R[t], R[t])
        phi[t] = p / (4 * np.pi * np.trapz(r ** 2 * p, r))
        m = (r > 0) & (r < R[t])
        scale = float(np.sqrt(np.mean(c_raw[i, j][m] ** 2)))
        se = 0.0 if chat0_sigma is None else float(chat0_sigma[i, j])
        quality[t] = (abs(se) * float(phi[t].max()) / scale) if scale > 0 else np.inf
        applied[t] = bool(quality[t] <= LEVEL_MAX_REL_SE)

    # Whether the anchor is usable is decided HERE, before anything uses it.
    # Deciding it afterwards would be a bug with a visible signature: the k-space
    # bridge (B) would still have injected a rejected chat_ij(0) into the core,
    # and the pooled inner NRMSE of every model doubled because of exactly that.
    c = c_raw.copy()
    for it in range(1, n_iter + 1):
        prev = c.copy()
        for i, j in PAIRS:                                    # (A)
            t = PAIR_TAG[(i, j)]
            c[i, j] = c[j, i] = w[t] * c[i, j] + (1 - w[t]) * asym[t]
        for i, j in PAIRS:                                    # (B), where anchored
            t = PAIR_TAG[(i, j)]
            if not applied[t]:
                continue
            ck = bridge(k, dst_forward(c[i, j], r, k), chat0[i, j], k_split)
            c[i, j] = c[j, i] = dst_backward(ck, r, k)
        if max(np.max(np.abs(c[i, j] - prev[i, j])) for i, j in PAIRS) < tol:
            break
    for i, j in PAIRS:                                        # (A) has the last word
        t = PAIR_TAG[(i, j)]
        c[i, j] = c[j, i] = w[t] * c[i, j] + (1 - w[t]) * asym[t]

    # The two projections do not quite commute, so a few per cent of the level
    # is still outstanding.  Close it exactly with a smooth bump supported
    # inside the pair's own keep radius: that leaves (A) untouched, and the
    # radius is set by the data rather than chosen.
    #
    # The bump is FLAT-TOPPED, not peaked at the origin, for two reasons that
    # agree.  Physically, a correction whose spectrum sits below k_split is
    # d c(r) = D k^3/(6 pi^2) + O(r^2) -- constant over r < 1/k_split -- so flat
    # is the shape the error actually has.  Numerically, chat(0) weights by r^2,
    # so a bump peaked at r = 0 buys almost no moment there and has to be made
    # large: at the TN 1331 reference state an r=0-peaked quartic needed -4.76
    # at the core against a flat top's -1.74, i.e. it moved c hardest exactly
    # where c is largest and best measured.
    # The two repairs are separable, and only the second one needs the anchor:
    # (A) the asymptote costs nothing and is always valid, (B) the level
    # correction is applied PER PAIR, and only where the anchor determines it.
    # Rejecting a whole state because chat11(0) is loose would throw away the
    # (1,2) and (2,2) repairs, which are the ones that were broken.
    got = chat0_of(r, c, sigma, eps, T, rc)
    left = {}
    for i, j in PAIRS:
        t = PAIR_TAG[(i, j)]
        A = float(chat0[i, j] - got[i, j]) if applied[t] else 0.0
        c[i, j] = c[j, i] = c[i, j] + A * phi[t]
        left[t] = A

    # How much of the level correction is guesswork.  What matters is not the
    # relative error on chat_ij(0) -- where the taper has already removed the
    # artifact, the leftover bump is small and a loose anchor is harmless -- but
    # the uncertainty of the SHIFT the bump actually applies, measured against
    # the c it is applied to.
    got = chat0_of(r, c, sigma, eps, T, rc)
    info = dict(nu=nu, R=R, iters=it, bump=left, level_rel_se=quality,
                level_applied=applied,
                chat0_residual={PAIR_TAG[(i, j)]: float(got[i, j] - chat0[i, j])
                                for i, j in PAIRS})
    return c, info


def chat0_of(r: np.ndarray, c: np.ndarray, sigma: np.ndarray, eps: np.ndarray,
             T: float, rc: float | None) -> np.ndarray:
    """4 pi Int_0^inf r^2 c_ij dr, the grid part plus the analytic remainder."""
    out = np.zeros((2, 2))
    for i, j in PAIRS:
        v = 4 * np.pi * np.trapz(r ** 2 * c[i, j], r) + tail_chat0(
            r[-1], sigma[i, j], eps[i, j], T, rc)
        out[i, j] = out[j, i] = v
    return out


def outer_scale(r: np.ndarray, c: np.ndarray, sigma: np.ndarray,
                eps: np.ndarray, T: float, rc: float | None,
                window=(1.6, R_ASYMPTOTE)) -> dict[str, float]:
    """s_ij = projection of c_ij onto -beta*u_ij over the outer window.

    The same statistic build_dcf_pool.py grades every source with; theory says
    s -> 1, and how far it strays measures how much of the outer region is
    artifact.  Reported here as the verification observable -- nothing in the
    repair targets it.
    """
    out = {}
    for i, j in PAIRS:
        s = sigma[i, j]
        m = (r >= window[0] * s) & (r <= window[1] * s)
        ref = -beta_u(r[m], s, eps[i, j], T, rc)
        out[PAIR_TAG[(i, j)]] = float(np.sum(c[i, j][m] * ref) / np.sum(ref * ref))
    return out


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def read_csv(path: Path, as_str: bool = False):
    rows = [ln for ln in path.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith('#')]
    names = rows[0].split(',')
    if as_str:
        return [dict(zip(names, ln.split(','))) for ln in rows[1:]]
    data = np.array([[float(v) for v in ln.split(',')] for ln in rows[1:]])
    return {n: data[:, i] for i, n in enumerate(names)}


def h_on_grid(g_table: dict[str, np.ndarray], r: np.ndarray) -> np.ndarray:
    """h_ij on the common grid: -1 inside the core, 0 past the end of the table."""
    h = np.zeros((2, 2, r.size))
    rs = g_table['r']
    for i, j in PAIRS:
        col = g_table[f'g{PAIR_TAG[(i, j)]}'] - 1.0
        v = np.zeros(r.size)
        inside = (r >= rs[0]) & (r <= rs[-1])
        v[inside] = np.interp(r[inside], rs, col)
        v[r < rs[0]] = -1.0
        h[i, j] = h[j, i] = v
    return h


def S_CC_zero(k: np.ndarray, hk: np.ndarray, rho: np.ndarray,
              k_max: float = 1.6) -> tuple[float, float]:
    """S_CC(k -> 0) by the even fit S_CC = S0 + a k^2 over small k.

    The size ladder puts S_CC's own finite-size shift at -3.3 % (N = 2048 ->
    inf), an order of magnitude below S22's; that is why this one small-k
    combination may come from the simulation while the other two may not.  Same
    window as md/finite_size.py, so the number is the one the ladder graded.
    """
    rt = np.sqrt(np.outer(rho, rho))
    S = np.eye(2)[:, :, None] + rt[:, :, None] * hk
    x = rho / rho.sum()
    scc = x[0] * x[1] * (x[1] * S[0, 0] + x[0] * S[1, 1]
                         - 2 * np.sqrt(x[0] * x[1]) * S[0, 1])
    m = (k > 0) & (k <= k_max)
    slope, s0 = np.polyfit(k[m] ** 2, scc[m], 1)
    return float(s0), float(np.std(scc[m] - (s0 + slope * k[m] ** 2)))


def run_one(label: str, k_split: float = K_SPLIT, r_max: float = 16.0,
            n_grid: int = 4096, deg: int = 4, verbose: bool = True) -> dict:
    par = json.loads((RUNS / label / 'params.json').read_text())
    ratio, eps_ratio, T = par['ratio'], par['eps_ratio'], par['T']
    x1, rho_tot, rc = par['x1'], par['rho'], par['rc']
    rho = np.array([x1, 1 - x1]) * rho_tot
    sigma, eps = lj_matrices(ratio, eps_ratio)

    eos = RUNS / 'eos.json'
    dP_ray = None
    if eos.exists() and label == 'validation':
        dP_ray = json.loads(eos.read_text())['dPdrho_rho0'] / T

    r, k = make_grid(r_max, n_grid)
    g = read_csv(RUNS / label / 'g.csv')
    hk = transform_h(r, k, h_on_grid(g, r))

    S_CC, S_CC_res = S_CC_zero(k, hk, rho)
    anc = macro_anchor(ratio, eps_ratio, T, rho_tot, x1, S_CC, deg, dP_ray)

    ck_raw = invert(hk, rho)
    c_raw = np.empty_like(ck_raw)
    for i, j in PAIRS:
        c_raw[i, j] = c_raw[j, i] = dst_backward(ck_raw[i, j], r, k)
    c_fix, info = repair(r, k, c_raw, anc.chat0, sigma, eps, T, rc, k_split,
                         chat0_sigma=anc.sigma)

    s_raw = outer_scale(r, c_raw, sigma, eps, T, rc)
    s_fix = outer_scale(r, c_fix, sigma, eps, T, rc)
    c0_raw = chat0_of(r, c_raw, sigma, eps, T, rc)

    if verbose:
        print(f'\n=== {label}: ratio {ratio}, eps_r {eps_ratio}, T* {T}, '
              f'x1 {x1}, rho* {rho_tot:.4f}, r_g {g["r"][-1]:.2f} ===')
        p = anc.provenance
        print(f'  pressure surface: {p["n_runs"]} runs, degree {p["fit_degree"]}, '
              f'rms {p["fit_rms"]:.4f}; d(betaP)/d(rho_i) = '
              f'{anc.dP_drho[0]:.3f}, {anc.dP_drho[1]:.3f}'
              + (f' (ray rescaled x{p["ray_rescale"]:.4f})'
                 if p['dP_ray'] is not None else ''))
        print(f'  S_CC(0) = {S_CC:.4f} (fit scatter {S_CC_res:.4f});  '
              f'anchor residual {anc.residual:.1e}')
        print('           chat11(0)  chat12(0)  chat22(0)')
        print(f'  anchor  {anc.chat0[0,0]:10.3f} {anc.chat0[0,1]:10.3f} '
              f'{anc.chat0[1,1]:10.3f}')
        print(f'  +-      {anc.sigma[0,0]:10.3f} {anc.sigma[0,1]:10.3f} '
              f'{anc.sigma[1,1]:10.3f}')
        print(f'  raw inv {c0_raw[0,0]:10.3f} {c0_raw[0,1]:10.3f} '
              f'{c0_raw[1,1]:10.3f}   (4 pi Int r^2 c_raw dr, as published data)')
        print(f'  residual{info["chat0_residual"]["11"]:10.2e} '
              f'{info["chat0_residual"]["12"]:9.2e} '
              f'{info["chat0_residual"]["22"]:9.2e}   after {info["iters"]} iterations')
        print('  level correction applied: ' + ', '.join(
            f"{t} {'yes' if info['level_applied'][t] else 'NO'} "
            f"({100*info['level_rel_se'][t]:.0f} % uncertain)"
            for t in ('11', '12', '22')))
        print('  artifact level nu_ij and the radius where -beta u drops to it:')
        print('    ' + '   '.join(
            f'{t}: nu {info["nu"][t]:.4f}, R {info["R"][t]:.2f} '
            f'(= {info["R"][t]/sigma[p_]:.2f} sigma_ij)'
            for t, p_ in (('11', (0, 0)), ('12', (0, 1)), ('22', (1, 1)))))
        print('  outer-region scale s_ij = <c_ij, -beta u_ij> / <-beta u, -beta u>'
              '   (exact value 1)')
        print(f'    raw      11 {s_raw["11"]:8.3f}  12 {s_raw["12"]:8.3f}  '
              f'22 {s_raw["22"]:8.3f}')
        print(f'    repaired 11 {s_fix["11"]:8.3f}  12 {s_fix["12"]:8.3f}  '
              f'22 {s_fix["22"]:8.3f}')
    return dict(label=label, r=r, k=k, c_raw=c_raw, c_fix=c_fix, info=info,
                ck_raw=ck_raw, anchor=anc, sigma=sigma, eps=eps,
                T=T, rc=rc, rho=rho, s_raw=s_raw, s_fix=s_fix, S_CC=S_CC)


def _provenance(out: dict) -> str:
    a = out['anchor']
    i = out['info']
    who = out.get('label') or out.get('stem')
    return (
        f"c_ij(r) with its long-wavelength part repaired -- dcf_lowk.py, source {who}\n"
        f"chat_ij(0) from bulk thermodynamics (NOT from this structure):\n"
        f"  chat11(0) = {a.chat0[0,0]:.4f}  chat12(0) = {a.chat0[0,1]:.4f}  "
        f"chat22(0) = {a.chat0[1,1]:.4f}\n"
        f"  d(betaP)/d(rho_i) = {a.dP_drho[0]:.4f}, {a.dP_drho[1]:.4f} from a "
        f"{a.provenance['n_runs']}-run LAMMPS pressure surface "
        f"(degree {a.provenance['fit_degree']}, rms {a.provenance['fit_rms']:.4f})\n"
        f"  S_CC(0) = {a.S_CC:.4f}\n"
        "exact asymptote c_ij = -beta u_ij imposed outside the keep radius\n"
        "  artifact level nu_ij = " + ', '.join(
            f"{t} {i['nu'][t]:.4f}" for t in ('11', '12', '22')) + "\n"
        "  keep radius R_ij   = " + ', '.join(
            f"{t} {i['R'][t]:.3f}" for t in ('11', '12', '22')) + "\n"
        "r is absolute, in units of sigma_11")


def write_c(path: Path, r: np.ndarray, c: np.ndarray, header: str,
            r_out: float = 9.0) -> None:
    m = r <= r_out
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as fh:
        for line in header.splitlines():
            fh.write(f'# {line}\n')
        fh.write('r,c11,c12,c22\n')
        for n in np.nonzero(m)[0]:
            fh.write(f'{r[n]:.4f},' + ','.join(
                f'{c[i, j][n]:.6f}' for i, j in PAIRS) + '\n')


# --------------------------------------------------------------------------
# the literature path: a tabulated c, no g, no far field of its own
# --------------------------------------------------------------------------

def run_literature(stem: str, ratio: float = 2.0, deg: int = 4,
                   k_split: float = K_SPLIT, r_max: float = 16.0,
                   n_grid: int = 4096, verbose: bool = True) -> dict:
    """Same repair applied to a published c_ij table (TN 1331 Appendix C).

    Two differences from the MD path.  The table is per-pair in R* = r/sigma_ij
    and stops at R* = 2.5, so past its end the continuation is the exact
    asymptote rather than zero -- which is a repair in itself, since chat(0)
    from a table zero-padded at 2.5 sigma_ij is missing the whole outer moment.
    And the artifact level has to be read off the outer part of the table
    instead of a far field the table does not have.
    """
    idx = {}
    for row in read_csv(HERE / 'tn1331' / 'index.csv', as_str=True):
        if row['kind'] == 'dcf' and Path(row['file']).stem == stem:
            idx = row
            break
    if not idx:
        raise KeyError(f'{stem} is not a dcf entry of tn1331/index.csv')
    if not idx.get('rho_star', '').strip():
        raise ValueError('the index carries no rho* for this state')
    T, x1, rho_tot = float(idx['tstar']), float(idx['x1']), float(idx['rho_star'])
    eps_ratio = float(idx['eps'])
    rho = np.array([x1, 1 - x1]) * rho_tot
    sigma, eps = lj_matrices(ratio, eps_ratio)

    tab = read_csv(HERE / 'tn1331' / 'dcf' / f'{stem}.csv')
    r, k = make_grid(r_max, n_grid)
    c_raw = np.zeros((2, 2, r.size))
    for i, j in PAIRS:
        t = PAIR_TAG[(i, j)]
        rs = tab['R_star'] * sigma[i, j]
        v = np.zeros(r.size)
        inside = (r >= rs[0]) & (r <= rs[-1])
        v[inside] = np.interp(r[inside], rs, tab[f'c{t}'])
        v[r < rs[0]] = tab[f'c{t}'][0]
        v[r > rs[-1]] = -beta_u(r[r > rs[-1]], sigma[i, j], eps[i, j], T)
        c_raw[i, j] = c_raw[j, i] = v

    nu = {}
    for i, j in PAIRS:                       # outer part of the table, not a far field
        t = PAIR_TAG[(i, j)]
        m = (tab['R_star'] >= 1.6) & (tab['R_star'] <= 2.5)
        d = tab[f'c{t}'][m] + beta_u(tab['R_star'][m] * sigma[i, j],
                                     sigma[i, j], eps[i, j], T)
        nu[t] = float(np.sqrt(np.mean(d ** 2)))

    S_CC = None
    rdf = HERE / 'tn1331' / 'rdf' / f'{stem}.csv'
    if rdf.exists():
        gt = read_csv(rdf)
        h = np.zeros((2, 2, r.size))
        for i, j in PAIRS:
            t = PAIR_TAG[(i, j)]
            rs = gt['R_star'] * sigma[i, j]
            v = np.zeros(r.size)
            ins = (r >= rs[0]) & (r <= rs[-1])
            v[ins] = np.interp(r[ins], rs, gt[f'g{t}'] - 1.0)
            v[r < rs[0]] = -1.0
            h[i, j] = h[j, i] = v
        S_CC, _ = S_CC_zero(k, transform_h(r, k, h), rho)
    if S_CC is None:                         # ideal-mixture value, flagged as such
        S_CC = x1 * (1 - x1)

    anc = macro_anchor(ratio, eps_ratio, T, rho_tot, x1, S_CC, deg,
                       S_CC_se=0.033 * S_CC)
    c_fix, info = repair(r, k, c_raw, anc.chat0, sigma, eps, T, None, k_split,
                         nu_override=nu, chat0_sigma=anc.sigma)
    if verbose:
        print(f'\n=== {stem}: T* {T}, x1 {x1}, rho* {rho_tot}, eps_r {eps_ratio} ===')
        print(f'  S_CC(0) = {S_CC:.4f}'
              + ('' if rdf.exists() else '  (no App. A g for this state -- '
                                         'ideal-mixture value used)'))
        print('           chat11(0)  chat12(0)  chat22(0)')
        print(f'  anchor  {anc.chat0[0,0]:10.3f} {anc.chat0[0,1]:10.3f} '
              f'{anc.chat0[1,1]:10.3f}')
        print(f'  +-      {anc.sigma[0,0]:10.3f} {anc.sigma[0,1]:10.3f} '
              f'{anc.sigma[1,1]:10.3f}')
        c0 = chat0_of(r, c_raw, sigma, eps, T, None)
        print(f'  table   {c0[0,0]:10.3f} {c0[0,1]:10.3f} {c0[1,1]:10.3f}')
        print('  artifact level nu_ij / keep radius: ' + '  '.join(
            f'{t} {nu[t]:.4f} / {info["R"][t]:.2f}' for t in ('11', '12', '22')))
    return dict(stem=stem, r=r, k=k, c_raw=c_raw, c_fix=c_fix, info=info,
                anchor=anc, sigma=sigma, eps=eps, T=T, rho=rho, S_CC=S_CC)


def run_all(k_split: float, deg: int) -> int:
    """Repair every MD run and every literature table that can be anchored."""
    index = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        if not (d / 'g.csv').exists():
            continue
        try:
            out = run_one(d.name, k_split, deg=deg, verbose=False)
        except Exception as exc:                     # no anchor for this cell
            print(f'  skip md {d.name}: {exc}')
            continue
        write_c(HERE / 'converted' / f'md_{d.name}' / 'c_lowk.csv',
                out['r'], out['c_fix'], _provenance(out))
        index.append(_index_row('md', d.name, out))
        print(f'  md {d.name:34s} chat0 = '
              f'{out["anchor"].chat0[0,0]:8.3f} {out["anchor"].chat0[0,1]:9.3f} '
              f'{out["anchor"].chat0[1,1]:10.3f}')

    lit = HERE / 'tn1331' / 'dcf'
    if lit.exists():
        for f in sorted(lit.glob('*.csv')):
            try:
                out = run_literature(f.stem, deg=deg, k_split=k_split, verbose=False)
            except Exception as exc:
                print(f'  skip lit {f.stem}: {exc}')
                continue
            write_c(HERE / 'tn1331' / 'dcf_lowk' / f'{f.stem}.csv',
                    out['r'], out['c_fix'], _provenance(out))
            index.append(_index_row('tn1331_dcf', f.stem, out))
            print(f'  lit {f.stem:33s} chat0 = '
                  f'{out["anchor"].chat0[0,0]:8.3f} {out["anchor"].chat0[0,1]:9.3f} '
                  f'{out["anchor"].chat0[1,1]:10.3f}')

    cols = list(index[0]) if index else []
    with (HERE / 'dcf_lowk_index.csv').open('w') as fh:
        fh.write(','.join(cols) + '\n')
        for row in index:
            fh.write(','.join(str(row[c]) for c in cols) + '\n')
    print(f'\n{len(index)} states repaired -> dcf_lowk_index.csv')
    return 0


def _index_row(source: str, label: str, out: dict) -> dict:
    a, i = out['anchor'], out['info']
    rho = out['rho']
    return dict(
        label=label, source=source, T_star=out['T'],
        rho_star=f'{rho.sum():.5f}', x1=f'{rho[0]/rho.sum():.4f}',
        chat0_11=f'{a.chat0[0,0]:.4f}', chat0_12=f'{a.chat0[0,1]:.4f}',
        chat0_22=f'{a.chat0[1,1]:.4f}',
        dP_drho1=f'{a.dP_drho[0]:.4f}', dP_drho2=f'{a.dP_drho[1]:.4f}',
        S_CC=f'{a.S_CC:.4f}',
        se_chat0_11=f'{a.sigma[0,0]:.4f}', se_chat0_12=f'{a.sigma[0,1]:.4f}',
        se_chat0_22=f'{a.sigma[1,1]:.4f}',
        nu_11=f'{i["nu"]["11"]:.4f}', nu_12=f'{i["nu"]["12"]:.4f}',
        nu_22=f'{i["nu"]["22"]:.4f}',
        R_11=f'{i["R"]["11"]:.3f}', R_12=f'{i["R"]["12"]:.3f}',
        R_22=f'{i["R"]["22"]:.3f}',
        level_11=int(i['level_applied']['11']),
        level_12=int(i['level_applied']['12']),
        level_22=int(i['level_applied']['22']))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default='validation', help='an md/runs/<label>')
    ap.add_argument('--literature', metavar='STEM',
                    help='a tn1331/dcf/<STEM>.csv instead of an MD run')
    ap.add_argument('--all', action='store_true',
                    help='every MD run and every literature table that can be anchored')
    ap.add_argument('--k-split', type=float, default=K_SPLIT)
    ap.add_argument('--deg', type=int, default=4)
    ap.add_argument('--write', action='store_true', help='write the repaired tables')
    args = ap.parse_args()

    if args.all:
        return run_all(args.k_split, args.deg)
    if args.literature:
        out = run_literature(args.literature, deg=args.deg, k_split=args.k_split)
        if args.write:
            write_c(HERE / 'tn1331' / 'dcf_lowk' / f'{args.literature}.csv',
                    out['r'], out['c_fix'], _provenance(out))
        return 0
    out = run_one(args.label, args.k_split, deg=args.deg)
    if args.write:
        write_c(HERE / 'converted' / f'md_{args.label}' / 'c_lowk.csv',
                out['r'], out['c_fix'], _provenance(out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
