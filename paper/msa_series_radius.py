#!/usr/bin/env python3
"""Radius of convergence of the MSA coupling expansion — measured, not bounded.

The paper defines the order-`γ` term as a DERIVATIVE at zero coupling,

    c^(γ) := (1/γ!) d^γ/dτ^γ [ c_MSA(τ·K) ] |_{τ=0},

and is careful never to claim that Σ_γ c^(γ) converges at the physical point
`τ = 1`.  That caution has never been tested.  The one quantitative statement
ever made about the series -- `R_c ~ e^{−zR} ≈ 1e-6` -- was RETRACTED: it was an
artifact of the singly-propagated `(1−G²)/A²` split, not of perturbation theory.

This measures `R_c` directly.  Exact MSA is a polynomial system (Blum & Høye), so
`u(τ) = (D̃_t, G_t)(τ)` is an ALGEBRAIC function of `τ`, analytic except where two
roots collide.  Two independent readings of the same object:

  1. CAUCHY / FFT.  Sample `u` on the circle `|τ| = r` and transform:

         c_n = (1/M) Σ_k u(r e^{2πik/M}) e^{−2πikn/M} / r^n

     the standard stable way to take high derivatives -- finite differences lose
     to roundoff by order ~4, this does not.  Then `R_c = lim |c_n|^{−1/n}`.

  2. MONODROMY.  Walk the same circle by Newton continuation and compare the
     endpoint with the start.  If a branch point lies INSIDE, the walk returns on
     a different sheet and `u(2π) ≠ u(0)`.  So the largest `r` that still closes
     brackets `R_c` from below, and the smallest that fails brackets it above.

Reading 2 is what makes reading 1 trustworthy: an FFT over a circle that
encircles a branch point returns coefficients of a function that is not
single-valued there, i.e. numerical noise dressed as a Taylor series.  Neither
reading alone would be safe.

⚠ WHY NOT JUST TRACE THE REAL AXIS.  `trace_coupling` already walks `τ ∈ [0,1]`
and reports where the branch folds.  A clean real-axis trace proves NOTHING about
convergence: the radius is set by the nearest singularity in the COMPLEX plane,
and the branch can be perfectly smooth on the real segment with `R_c < 1`.  A
real fold inside `[0,1]` is sufficient for divergence, never necessary.

    python msa_series_radius.py
"""

from __future__ import annotations

import numpy as np

import msa_exact as M


def residuals(u, xi: float, z: np.ndarray, K: np.ndarray):
    """(29′) and (33), UNNORMALISED and complex-safe.

    `_bhn_residuals` divides by `2πK/z`, which is exactly zero at the expansion
    point, and casts `u` to float.  Both have to go: the series is expanded about
    `K = 0`, and every step of this is complex.
    """
    n = len(z)
    Dt, G = np.asarray(u[:n]), np.asarray(u[n:])
    p = M._bhn_pieces(xi, z, Dt, G)
    zu = z[:, None]
    src = (p['A'] + z * p['qp']
           + np.sum(zu ** 2 * z[None, :] / (zu + z[None, :])
                    * (p['gam'] * Dt)[None, :], axis=1)) / z ** 2
    return np.concatenate([Dt * p['F'] - 2.0 * np.pi * K / z,
                           2.0 * np.pi * G * p['F'] - src])


def jac(u, xi, z, K, h=1e-7):
    """Complex Jacobian by central differences.

    The residual is holomorphic in `u` -- every operation in it is `+ − × ÷ exp`
    -- so an ordinary central difference with a REAL step is the derivative in the
    complex sense.  (The complex-step trick is unavailable for the same reason it
    is usually attractive: the function is already complex.)
    """
    m = len(u)
    J = np.empty((m, m), dtype=complex)
    for j in range(m):
        e = np.zeros(m, dtype=complex)
        e[j] = h
        J[:, j] = (residuals(u + e, xi, z, K) - residuals(u - e, xi, z, K)) / (2 * h)
    return J


def newton(u0, xi, z, K, tol=1e-12, itmax=60):
    u = np.asarray(u0, dtype=complex).copy()
    for _ in range(itmax):
        r = residuals(u, xi, z, K)
        if np.linalg.norm(r) < tol:
            return u, float(np.linalg.norm(r))
        try:
            du = np.linalg.solve(jac(u, xi, z, K), -r)
        except np.linalg.LinAlgError:
            return None, np.inf
        # damped: near a branch point the full step overshoots onto the partner
        # root, which is precisely the event being measured and must not be
        # reached by accident
        for damp in (1.0, 0.5, 0.25, 0.1):
            v = u + damp * du
            if np.linalg.norm(residuals(v, xi, z, K)) < np.linalg.norm(r):
                u = v
                break
        else:
            u = u + du
    r = float(np.linalg.norm(residuals(u, xi, z, K)))
    return (u, r) if r < 1e-9 else (None, r)


def seed(xi: float, z: np.ndarray):
    """`u(0)`: at zero coupling `D̃ = 0` and `G` is the hard-sphere moment."""
    G0 = np.array([M.hs_laplace_moment_scaled(xi, zz) for zz in z], dtype=complex)
    return np.concatenate([np.zeros(len(z), dtype=complex), G0])


def walk_circle(xi, z, Kphys, r, M_pts=256):
    """Newton-continue once around `|τ| = r`; return the samples and the gap.

    The gap `‖u(2π) − u(0)‖` is the monodromy defect: zero means the circle
    encloses no branch point.
    """
    th = 2.0 * np.pi * np.arange(M_pts) / M_pts
    lam = r * np.exp(1j * th)
    u = seed(xi, z)
    # step in from tau = 0 to the circle along the real axis first
    for t in np.linspace(0, r, 24)[1:]:
        u, res = newton(u, xi, z, t * Kphys)
        if u is None:
            return None, np.inf, None
    u_start = u.copy()
    out = []
    for L in lam:
        u, res = newton(u, xi, z, L * Kphys)
        if u is None:
            return None, np.inf, None
        out.append(u.copy())
    # close the loop: one more step back to theta = 0
    u_end, res = newton(u, xi, z, lam[0] * Kphys)
    if u_end is None:
        return None, np.inf, None
    return np.array(out), float(np.linalg.norm(u_end - u_start)), u_start


def coefficients(samples, r, n_max=40):
    """Taylor coefficients from the circle samples, by FFT, plus their noise floor.

    Returns `(c, floor)`.  The floor matters as much as the coefficients: the
    transform resolves `C[n]` only down to roundoff on the samples, and the
    division by `r^n` carries that floor with it.  For `r < 1` it is AMPLIFIED
    like `r^{−n}`, so beyond some `n` every coefficient returned is noise --
    growing noise, which a growth-rate fit will happily read as a tiny radius.
    """
    Mp = samples.shape[0]
    C = np.fft.fft(samples, axis=0) / Mp
    n = np.arange(min(n_max, Mp // 2))
    c = np.array([C[k] / r ** k for k in n])
    eps = np.finfo(float).eps * float(np.abs(samples).max())
    return c, eps / r ** n.astype(float)


def radius_from_growth(c, floor, lo=3):
    """`R_c` from a LEAST-SQUARES slope of `log|c_n|` against `n`.

    Not `median |c_n|^{−1/n}`: that estimator is biased at small `n` by the
    prefactor of the asymptotic `|c_n| ~ A·R_c^{−n}`, which is exactly the
    regime where the coefficients are still large enough to trust.  A slope
    absorbs `A` into the intercept.

    ⚠ The usable window is bounded at BOTH ends.  `c_n = C[n]/r^n`, so for a
    circle of radius `r < 1` the division amplifies the FFT's roundoff floor by
    `r^{−n}`; for `r > 1` it damps it.  Either way, coefficients that have fallen
    to the floor carry no information and must be dropped, or the fitted slope is
    the slope of the noise.
    """
    nrm = np.array([np.linalg.norm(x) for x in c])
    # keep only coefficients standing clear of their OWN amplified floor, and
    # stop at the first one that does not -- past that point the sequence is
    # noise and its growth rate is meaningless
    good = nrm > 30.0 * floor
    cut = np.argmin(good) if not good.all() else len(good)
    ok = np.zeros(len(nrm), bool)
    ok[lo:cut] = True
    if ok.sum() < 4:
        return np.nan, nrm
    n = np.arange(len(nrm))[ok]
    # THREE parameters, not two: log|c_n| = A + B*n + C*log n.
    #
    # The singularity here is an algebraic branch point, where two roots of the
    # Blum-Hoye system collide, so the coefficients carry an ALGEBRAIC prefactor
    # as well as the geometric one -- |c_n| ~ A*n^{-3/2}*R_c^{-n} for a square
    # root.  A straight line through log|c_n| absorbs that prefactor into the
    # slope, biasing R_c high by about 1.5/n_bar ~ 15% over this window.  The
    # tell was unmistakable: tau_fold/R_c came out 0.912 for EVERY state and
    # both models, which is not a physical coincidence, and it violated
    # R_c <= tau_fold, which a real fold makes mandatory.
    #
    # ⚠ The K-invariance gate cannot catch this.  A multiplicative bias scales
    # out of `R_c*K` exactly, so K_c stayed invariant to 0.00% while being ~9%
    # wrong.  Invariance is a consistency check, not an accuracy one.
    Amat = np.column_stack([np.ones_like(n, dtype=float), n.astype(float),
                            np.log(n.astype(float))])
    B = np.linalg.lstsq(Amat, np.log(nrm[ok]), rcond=None)[0][1]
    # `R_c = exp(-B)` for EITHER sign of B.  An earlier version guarded on
    # `B < 0`, i.e. accepted only decaying coefficients -- which rejects
    # precisely the case this study exists to find, since `R_c < 1` means
    # `|c_n| ~ R_c^{-n}` GROWS.  Divergence was undetectable by construction.
    return float(np.exp(-B)), nrm


def gate_first_coefficient(c, xi, z, Kphys):
    """`c_1` of the `D̃` block has a closed form; check the FFT against it.

    From `D̃·F = 2πK/z` at zero coupling, `dD̃/dτ = 2πK/(z F₀)` with `F₀` the
    hard-sphere Baxter factor.  If this fails, the radius being reported belongs
    to some other function and nothing downstream means anything.
    """
    n = len(z)
    A0, B0, qp0 = M._hs_baxter(xi)
    rho = 6.0 * xi / np.pi
    F0 = 1.0 - rho * (M._phi1(z) * qp0 + M._phi2(z) * A0)
    want = 2.0 * np.pi * Kphys / (z * F0)
    got = np.real(c[1][:n])
    return float(np.max(np.abs(got - want) / np.abs(want)))


def study(label, xi, z, Kphys, radii, verbose=True):
    """Return `(R_c in tau, K_c, gate)`.

    `K_c = R_c · K_phys` is the transferable number: `R_c` alone depends on how
    large a `K` happened to be called "physical", since `tau` only ever enters as
    `tau·K`.  `K_c` is the coupling at which the expansion about zero stops
    reaching the physical point at all.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    Kphys = np.atleast_1d(np.asarray(Kphys, dtype=float))
    if verbose:
        print(f'\n{label}:  eta={xi:.3f}  z={list(z)}  K={list(Kphys)}')
        print(f"  {'r':>7} {'loop gap':>11} {'closes':>7} {'R_c est':>9} {'c1 gate':>9}")
    best_closed, first_open, Rc, gate = 0.0, np.inf, np.nan, np.nan
    for r in radii:
        s, gap, _ = walk_circle(xi, z, Kphys, r)
        if s is None:
            if verbose:
                print(f'  {r:7.3f} {"no solve":>11} {"-":>7} {"-":>9} {"-":>9}')
            first_open = min(first_open, r)
            continue
        closed = gap < 1e-6
        c, floor = coefficients(s, r)
        est, _ = radius_from_growth(c, floor) if closed else (np.nan, None)
        g = gate_first_coefficient(c, xi, z, Kphys) if closed else np.nan
        if closed:
            best_closed = max(best_closed, r)
            if np.isfinite(est):
                Rc, gate = est, g
        else:
            first_open = min(first_open, r)
        if verbose:
            print(f'  {r:7.3f} {gap:11.2e} {"yes" if closed else "NO":>7} '
                  f'{est if closed else float("nan"):9.4f} {g:9.2e}')
    Kc = Rc * float(Kphys[0])
    if verbose:
        print(f'  -> monodromy brackets R_c in ({best_closed:.3f}, '
              f'{first_open if np.isfinite(first_open) else float("inf"):.3f}); '
              f'growth gives R_c = {Rc:.4f}, i.e. K_c = {Kc:.4f}')
    return Rc, Kc, gate


def main() -> int:
    print(__doc__.split('\n')[0])
    print('=' * 78)
    LAD = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    STATES = [
        ('HSY z=1.8, eta=0.10', 0.10, [1.8],  [0.5]),
        ('HSY z=1.8, eta=0.20', 0.20, [1.8],  [0.5]),
        ('HSY z=1.8, eta=0.30', 0.30, [1.8],  [0.5]),
        ('HSY z=1.8, eta=0.40', 0.40, [1.8],  [0.5]),
        ('HSY z=1.8, eta=0.45', 0.45, [1.8],  [0.5]),
        ('HSY z=5,   eta=0.30', 0.30, [5.0],  [1.0]),
        ('HSY z=10,  eta=0.30', 0.30, [10.0], [2.0]),
    ]
    rows = []
    for lab, xi, z, K in STATES:
        Rc, Kc, gate = study(lab, xi, z, K, LAD)
        rows.append((lab, xi, z[0], K[0], Rc, Kc, gate))

    print('\n' + '=' * 78)
    print('SUMMARY -- K_c is the coupling at which the expansion about K=0 stops')
    print('reaching the physical point.  A state is expandable iff its K < K_c.')
    print(f"\n  {'state':22s} {'K used':>8} {'R_c(tau)':>9} {'K_c':>9} "
          f"{'c1 gate':>9}")
    for lab, xi, z, K, Rc, Kc, gate in rows:
        print(f'  {lab:22s} {K:8.3f} {Rc:9.4f} {Kc:9.4f} {gate:9.2e}')
    worst = max((r[6] for r in rows if np.isfinite(r[6])), default=np.nan)
    print(f'\n  worst first-coefficient gate over all states: {worst:.2e}')
    print('  (this is the exact check dD~/dtau = 2*pi*K/(z*F0); if it were not')
    print('   at roundoff, the radii above would belong to some other function)')

    # ---- gate: K_c must not depend on which K was called "physical" ----------
    # tau enters only as tau*K, so R_c(K)*K is a property of (eta, z) alone.  If
    # this drifted, R_c would be an artifact of the circle ladder rather than a
    # singularity location.
    print('\n' + '=' * 78)
    print('GATE -- K_c is invariant under the choice of physical K  (eta=0.30, z=1.8)')
    print(f"\n  {'K used':>8} {'R_c(tau)':>10} {'K_c':>10}")
    kcs = []
    for K in (0.25, 0.5, 1.0, 2.0):
        Rc, Kc, _ = study('', 0.30, [1.8], [K], LAD, verbose=False)
        kcs.append(Kc)
        print(f'  {K:8.3f} {Rc:10.4f} {Kc:10.4f}')
    kcs = [k for k in kcs if np.isfinite(k)]
    spread = (max(kcs) - min(kcs)) / np.mean(kcs) if kcs else np.nan
    print(f'  -> spread {spread * 100:.2f}%')

    # ---- the physically readable form ---------------------------------------
    # The coupling multiplier IS the inverse temperature: `K = kv/T*`
    # (msa_exact_phase_lines.py).  So setting `Kphys = kv` makes tau = 1/T*, and
    # the radius converts directly into a temperature below which the expansion
    # about K = 0 no longer reaches the physical state:  T*_div = 1/R_c.
    print('\n' + '=' * 78)
    print('THE ANSWER, AS A TEMPERATURE.  tau = 1/T*, so T*_div = 1/R_c: below')
    print('this temperature the coupling series DIVERGES at the physical point.')
    MODELS = (('HSY  z=1.8',  [1.8],             [1.0]),
              ('2YK  LJ',     [2.9637, 14.0167], [2.1714, -2.1714]))
    for name, z, kv in MODELS:
        print(f'\n  {name}     (K = kv/T*)')
        print(f"    {'eta':>6} {'R_c(tau)':>10} {'T*_div':>9}")
        for xi in (0.10, 0.20, 0.30, 0.40, 0.45):
            Rc, _, _ = study('', xi, z, kv,
                             [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], verbose=False)
            print(f'    {xi:6.2f} {Rc:10.4f} {1.0 / Rc if np.isfinite(Rc) else float("nan"):9.4f}')
    print('\n  For reference the HSY critical point sits near T* = 1.03 (fold apex')
    print('  1.0176, A=0 spinodal apex 1.0331), so compare T*_div against that.')

    # ---- and the direct demonstration ---------------------------------------
    print('\n' + '=' * 78)
    print('DOES THE SERIES DIVERGE AT PHYSICAL COUPLING?  Push K past K_c.')
    print(f"\n  {'state':22s} {'K':>7} {'K_c':>8} {'R_c':>8}   verdict")
    for lab, xi, z, K in (('HSY z=1.8, eta=0.20', 0.20, [1.8], [0.5]),
                          ('HSY z=1.8, eta=0.20', 0.20, [1.8], [1.0]),
                          ('HSY z=1.8, eta=0.20', 0.20, [1.8], [2.0]),
                          ('HSY z=1.8, eta=0.20', 0.20, [1.8], [4.0]),
                          ('HSY z=1.8, eta=0.30', 0.30, [1.8], [4.0])):
        Rc, Kc, _ = study('', xi, z, K, [0.15, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0],
                          verbose=False)
        v = ('DIVERGES at tau=1' if Rc < 1 else 'converges at tau=1') \
            if np.isfinite(Rc) else '(no reading)'
        print(f'  {lab:22s} {K[0]:7.2f} {Kc:8.3f} {Rc:8.4f}   {v}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# ─────────────────────────────────────────────────────────────────────────────
#  Is the fold the binding singularity?
# ─────────────────────────────────────────────────────────────────────────────
def sigma_min(u, xi, z, K):
    """Smallest singular value of the Jacobian — the scaled distance to a fold.

    Not `det J`: for `n` tails the determinant is a product of `2n` singular
    values, so it inherits their scale and can be tiny far from any fold and
    large near one.  `σ_min` vanishes exactly at the turning point and nowhere
    else nearby.
    """
    return float(np.linalg.svd(jac(u, xi, z, K), compute_uv=False)[-1])


def fold_in_tau(xi, z, kv, tau_hi=8.0, n_steps=400):
    """Locate the real turning point in `τ`, if there is one below `tau_hi`.

    Walks the physical branch in real `τ` and extrapolates `σ_min² → 0`.  The
    SQUARE is what makes this accurate: at a quadratic fold the branch behaves
    like `√(τ_fold − τ)`, so `σ_min ~ (τ_fold − τ)^{1/2}` and `σ_min²` is LINEAR
    in `τ` — a straight-line extrapolation from points that still converge,
    rather than a march into the region where Newton cannot.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    kv = np.atleast_1d(np.asarray(kv, dtype=float))
    u = seed(xi, z)
    t, dt = 0.0, tau_hi / n_steps
    taus, sig = [], []
    while t < tau_hi and dt > 1e-7:
        u2, _ = newton(u, xi, z, (t + dt) * kv)
        if u2 is None:
            dt /= 2.0
            continue
        u, t = u2, t + dt
        taus.append(t)
        sig.append(sigma_min(u, xi, z, t * kv))
    if len(taus) < 12:
        return np.nan, np.nan, len(taus)
    last_t, last_s = np.array(taus[-10:]), np.array(sig[-10:])
    if last_s[-1] > 0.5 * max(sig):          # σ_min never fell: no fold reached
        return np.nan, float(taus[-1]), len(taus)
    a, b = np.polyfit(last_t, last_s ** 2, 1)
    return (float(-b / a) if a < 0 else np.nan), float(taus[-1]), len(taus)


def cmd_fold() -> int:
    print('=' * 78)
    print('IS THE FOLD THE BINDING SINGULARITY?')
    print('=' * 78)
    print('`R_c <= tau_fold` always -- a real fold IS a singularity, so it can never')
    print('sit closer than the radius.  The question is whether equality holds, i.e.')
    print('whether the nearest singularity is that real turning point or a complex')
    print('pair that binds first.\n')
    LAD = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    for name, z, kv in (('HSY  z=1.8', [1.8], [1.0]),
                        ('2YK  LJ', [2.9637, 14.0167], [2.1714, -2.1714])):
        print(f'  {name}   (tau = 1/T*)')
        print(f"    {'eta':>6} {'R_c':>9} {'tau_fold':>10} {'ratio':>8} "
              f"{'traced to':>10}   verdict")
        for xi in (0.10, 0.20, 0.30, 0.40, 0.45):
            Rc, _, _ = study('', xi, z, kv, LAD, verbose=False)
            tf, tmax, npts = fold_in_tau(xi, z, kv)
            if np.isfinite(tf) and np.isfinite(Rc):
                ratio = tf / Rc
                v = ('FOLD binds' if abs(ratio - 1.0) < 0.05
                     else 'complex pair binds first')
            elif np.isfinite(Rc):
                ratio, v = np.nan, f'no real fold below tau={tmax:.2f}'
            else:
                ratio, v = np.nan, 'no radius reading'
            print(f'    {xi:6.2f} {Rc:9.4f} {tf:10.4f} {ratio:8.3f} '
                  f'{tmax:10.3f}   {v}')
        print()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  Inside the radius, does the sum land on the RIGHT solution?
# ─────────────────────────────────────────────────────────────────────────────
def exact_at(xi, z, kv, tau=1.0, n_steps=80):
    """The physical branch at `τ`, by real-axis continuation from `τ = 0`."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    kv = np.atleast_1d(np.asarray(kv, dtype=float))
    u = seed(xi, z)
    for t in np.linspace(0.0, tau, n_steps)[1:]:
        u, _ = newton(u, xi, z, t * kv)
        if u is None:
            return None
    return u


def cmd_resum() -> int:
    """Sum the series at `τ = 1` and compare with the solution it should equal.

    Analyticity already answers this -- inside `R_c` a Taylor series converges TO
    its function, and the branch is pinned to PY at `τ = 0`, so the limit can only
    be the physical branch.  What the check adds is that it exercises EVERY
    coefficient at once, where the `c_1` gate exercises one; a sign error or a
    missed factor at order 7 would show here and nowhere else.
    """
    print('=' * 78)
    print('DOES THE SUM CONVERGE TO THE RIGHT SOLUTION?  Partial sums at tau = 1.')
    print('=' * 78)
    CASES = [('HSY eta=0.40', 0.40, [1.8], [1.0], 1.50),
             ('HSY eta=0.30', 0.30, [1.8], [1.0], 1.10),
             ('HSY eta=0.10', 0.10, [1.8], [1.0], 1.02),
             ('HSY eta=0.20', 0.20, [1.8], [1.0], 0.95),   # R_c = 1.003: marginal
             ('HSY eta=0.30, K=2', 0.30, [1.8], [2.0], 0.55)]  # R_c < 1: divergent
    for lab, xi, z, kv, r in CASES:
        zz = np.atleast_1d(np.asarray(z, dtype=float))
        Rc, _, _ = study('', xi, zz, kv, [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                         verbose=False)
        s, gap, _ = walk_circle(xi, zz, np.atleast_1d(np.asarray(kv, float)), r)
        if s is None:
            print(f'  {lab:20s} circle r={r} did not solve')
            continue
        c, _ = coefficients(s, r, n_max=60)
        ref = exact_at(xi, zz, kv, 1.0)
        tag = 'INSIDE' if Rc > 1 else 'OUTSIDE'
        print(f'\n  {lab}   R_c = {Rc:.4f}  -> tau=1 is {tag} the disc'
              + ('' if ref is not None else '   [no real solution at tau=1]'))
        print(f"    {'N':>4} {'|S_N - exact|':>15}")
        run = np.zeros_like(c[0])
        for n in range(len(c)):
            run = run + c[n]                       # tau = 1, so tau^n = 1
            if n in (2, 5, 10, 20, 30, 40, 50) and ref is not None:
                print(f'    {n:4d} {np.linalg.norm(run - ref):15.3e}')
            elif n in (2, 5, 10, 20, 30, 40, 50):
                print(f'    {n:4d} {"|S_N| = " + format(np.linalg.norm(run), ".3e"):>15}')
    return 0
