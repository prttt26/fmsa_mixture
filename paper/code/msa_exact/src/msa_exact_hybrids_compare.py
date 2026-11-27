r"""msa_exact_hybrids_compare.py — every way of crossing the no-solution region.

The exact MSA's **compressibility route cannot reach the liquid**, twice over:
`∂βp/∂ρ = (A/2π)²` is a perfect square, so `βA(ρ)` is convex wherever MSA exists and
there is no loop to Maxwell-construct; and below the fold apex the isotherm is cut
outright.  Every construction here supplies the missing piece a different way, and
the figure exists to show **how far they disagree** — not to pick one.

⭐ Stated once, because it governs how the figure must be read: the compressibility
route knows only DERIVATIVES, and the liquid arc is a **disconnected component**, so
its two integration constants are not determined by the route at all.  Each method
below is a different way of supplying them.

| line | supplies them by |
|---|---|
| hug hybrid | transporting the 2-form along the `A = 0` spinodal to the apex and back |
| critical-point detour | a rectangle over the critical point |
| quintic bridge | interpolating `ĉ(0)` across the gap — **produces no loop**, see below |
| `f_tail` bridge | the (45)/(46) increments across the gap, delivered as end conditions |

⚠ **Naming**: the crossing formerly called the "fold-apex detour" is the
**critical-point detour** — it must clear the `A = 0` apex, not the fold apex, because
between the two the isotherm is whole but wrong-winding.  The JSON keys `binodal_apex`
and `apex_curves` keep the old spelling so stored runs stay readable; only the name
changed, but the *crossing height* changed with it.
| liquid anchor | Blum & Høye (45) evaluated on the liquid arc at `ρ_a` |

⚠ **Rejected, and why** — an external Lennard-Jones EOS (`lj_mbwr.py`, JZG 1993) was
built and verified for this and then *not* used: the model here is **hard core plus
two Yukawas**, whose `B₂` differs from LJ's by a nearly constant `0.48…0.71 σ³`
(9 % at `T* = 0.8`, 17 % at 1.31, sign-opposite by 3.0).  The offset does not decay
with temperature, so it is the **repulsive core** that differs, not the fit quality.
Anchoring to LJ would have measured that mismatch rather than the route.

⚠⚠ The bridge never leaves the isotherm, and an earlier version of this file read
that as "no energy-route contamination".  **False** — see `ftail_bridge_isotherm`.
It injects exactly two energy-route numbers per isotherm, the same two the path
crossings inject, so all four lines here are one family: *compressibility on both
arcs, energy across the gap*, differing only in how the crossing increment is got.
That is why they lie on a single axis in the figure and why the ends of that axis
do not bracket anything.

⭐ What the bridge does have is the cleanest crossing of the three: the increment is
READ OFF (45)/(46) at the two spinodal densities rather than path-integrated through
`T`, so it carries no path-dependence — `hug` and `detour` differ from each other by
3.4 % at `T* = 0.95` rising to 13.3 % at 0.70, and the bridge has no analogue of
that spread.  Its own numerics are exact once `_hermite(deriv=2)` replaced a double
`np.gradient` (which was losing 1.1 % on both gap integrals).
"""
from __future__ import annotations
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
import msa_exact_mix as X

OUT = "numerical_notes/results/msa_exact"
JSON_MAIN = OUT + "/msa_exact_phase_lines.json"
JSON = OUT + "/msa_exact_hybrids_compare.json"
PNG = OUT + "/msa_exact_binodal_compressibility_hybrids_compare.png"
MODELS = {"HSY": (np.array([1.8]), np.array([1.0])),
          "2YK": (np.array([2.9637, 14.0167]), np.array([2.1714, -2.1714]))}
rho_of = lambda e: 6.0 * e / np.pi


def apex_T(z, kv, lo, hi, which="spinodal", tol=2e-5):
    r"""Apex temperature in **actual `T*`**, by bisection on "do the two densities
    still exist?".

    ⚠⚠ Not a re-implementation for its own sake.  `msa_exact_mix.spinodal_apex` and
    `fold_apex` bracket in units of `T0 = 1/max(K)`, which is `1.0` for HSY (`K=1`)
    but `0.4605` for 2YK (`K_max = 2.1714`) — so an HSY-shaped bracket like
    `(0.7, 1.3)` sits entirely ABOVE the 2YK apex, every midpoint tests "open", and
    the bisection collapses onto its own `t_lo` and returns it.  That is silent: the
    value comes back looking like an apex.  Caught because both apexes came back
    equal to `0.7000`, the bracket bound.
    """
    # ⚠ Same test for BOTH: the pair EXISTS below its own apex and vanishes above it,
    # so "exists" always means the apex is higher.  An earlier version inverted the
    # fold branch and returned the bracket bound (0.40 for HSY, 1.60 for 2YK) — the
    # same silent failure mode as the units trap above, one line further on.
    f = (X.spinodal_densities_exact if which == "spinodal" else X.fold_densities)
    while hi - lo > tol:
        m = 0.5*(lo + hi)
        st = X.HCYMix([1.0], [rho_of(0.2)], z, kv[:, None, None]/m)
        lo, hi = (m, hi) if f(st)[0] is not None else (lo, m)
    return 0.5*(lo + hi)


def arc(z, kv, T, r0, r1, n):
    """`(ρ, a, A)` walked with the solution CONTINUED, never re-ramped."""
    rs = np.linspace(r0, r1, n)
    st0 = X.HCYMix([1.0], [rs[0]], z, kv[:, None, None] / T)
    q = X.solve_bh_mix(st0)
    if q is None:
        return None
    u = np.concatenate([q.Dt.ravel(), q.Gt.ravel()])
    out = []
    for r in rs:
        s = X.HCYMix([1.0], [float(r)], z, kv[:, None, None] / T)
        v = X._solve_seeded(s, u, *X._scales(s))
        if v is None:
            break
        u = v
        nn = s.N * s.N
        p = X._pieces(s, u[:s.T*nn].reshape(s.T, s.N, s.N),
                      u[s.T*nn:].reshape(s.T, s.N, s.N))
        A = float(np.sum(p['A']))
        out.append((float(r), (A / (2*np.pi))**2, A))
    return np.array(out) if len(out) >= 8 else None


def _quintic(x, x0, x1, y0, d0, c0, y1, d1, c1):
    h = x1 - x0; t = (np.asarray(x, float) - x0) / h
    M = np.zeros((6, 6)); rhs = [y0, d0*h, c0*h*h, y1, d1*h, c1*h*h]
    M[0, 0] = 1.0; M[1, 1] = 1.0; M[2, 2] = 2.0
    for j in range(6):
        M[3, j] = 1.0; M[4, j] = j; M[5, j] = j*(j-1)
    co = np.linalg.solve(M, rhs)
    return sum(co[j]*t**j for j in range(6))


def bridge_isotherm(z, kv, T, n_side=60, n_gap=60, a2_sign=+1.0):
    r"""`(ρ, a)` on the WHOLE isotherm, the gap filled by a quintic in `ĉ(0)`.

    Endpoint data at each spinodal density is analytic, so the quintic has **no
    free parameter**:  `a = 0` ⇒ `ĉ = 1/ρ_sp`;  `da/dρ = A A′/(2π²) = 0` ⇒
    `dĉ/dρ = −1/ρ_sp²`;  `a″ = 2(A′)²/(2π)²` ⇒ `d²ĉ/dρ² = (2/ρ² − a″)/ρ`.
    """
    rv, rl = X.spinodal_densities_exact(
        X.HCYMix([1.0], [rho_of(0.2)], z, kv[:, None, None] / T))
    if rv is None or rl is None or rl - rv < 1e-3:
        return None
    lo = arc(z, kv, T, 1e-4, rv, n_side)
    hi = arc(z, kv, T, 0.92, rl, n_side)
    if lo is None or hi is None:
        return None
    hi = hi[np.argsort(hi[:, 0])]
    h = 2e-4
    def _A(r):
        q = X.solve_bh_mix(X.HCYMix([1.0], [float(r)], z, kv[:, None, None] / T))
        return None if q is None else float(np.sum(q.p['A']))

    def Ap(rs):
        # ⚠ NOT via `arc`: that returns None below 8 points, so a single-point
        # call silently produced "no endpoint derivative" at every temperature and
        # the whole scan reported "isotherm not built".
        a, b = _A(rs + h), _A(rs - h)
        return None if a is None or b is None else (a - b) / (2*h)
    A0p, A1p = Ap(rv), Ap(rl)
    if A0p is None or A1p is None:
        return None
    ends = []
    for rs, App in ((rv, A0p), (rl, A1p)):
        y = 1.0/rs; d = -1.0/rs**2
        # ⚠ `a2_sign = −1` FLIPS `a″` at both spinodals.  Not a tunable: `a″ > 0` is
        # forced by `a = (A/2π)²` at a regular zero, and so is `a′ = 0` — the flip keeps
        # the second and denies the first.  Kept because the resulting line is worth
        # seeing (`probe_bridge_content.py aflip` has the argument).
        app = a2_sign * 2*App**2/(2*np.pi)**2
        ends.append((y, d, (2.0/rs**2 - app)/rs))
    xg = np.linspace(rv, rl, n_gap)[1:-1]
    cg = _quintic(xg, rv, rl, *ends[0], *ends[1])
    ag = 1.0 - xg*cg
    rho = np.concatenate([lo[:, 0], xg, hi[:, 0]])
    a = np.concatenate([lo[:, 1], ag, hi[:, 1]])
    k = np.argsort(rho)
    return rho[k], a[k]


def coexist(rho, a):
    """Maxwell on an isotherm given by its own `∂βp/∂ρ`."""
    p = np.concatenate([[rho[0]], rho[0] + cumulative_trapezoid(a, rho)])
    mu = np.log(rho) + np.concatenate([[0.0], cumulative_trapezoid((a-1.0)/rho, rho)])
    neg = np.where(a < 0)[0]
    if len(neg) == 0 or neg[0] < 3 or neg[-1] > len(rho)-5:
        return None
    i, j = int(neg[0]), int(neg[-1])
    pg, mg, rg = p[:i], mu[:i], rho[:i]
    pl, ml, rl_ = p[j+1:], mu[j+1:], rho[j+1:]
    lo, hi = max(pg.min(), pl.min()), min(pg.max(), pl.max())
    if not hi > lo:
        return None
    f = lambda q: np.interp(q, pg, mg) - np.interp(q, pl, ml)
    if f(lo)*f(hi) > 0:
        return None
    q = brentq(f, lo, hi, xtol=1e-14)
    x, y = float(np.interp(q, pg, rg)), float(np.interp(q, pl, rl_))
    return (x, y) if y - x > 1e-3 else None


def liquid_anchor(z, kv, T, n_side=70, anchors=(0.35, 0.55, 0.80)):
    r"""Binodal with the liquid arc's two constants taken from Blum & Høye (45)/(46).

    ⚠ The one arbitrary element is the anchor `ρ_a`, and its effect is the ROUTE
    GAP accumulated along the liquid arc — measured 56–60 % in the free energy, so
    several anchors are returned and the figure draws them as a band, never a line.
    """
    st0 = X.HCYMix([1.0], [rho_of(0.2)], z, kv[:, None, None] / T)
    rv, rl = X.spinodal_densities_exact(st0)
    if rv is None or rl is None:
        return []
    lo = arc(z, kv, T, 1e-4, rv, n_side)
    hi = arc(z, kv, T, 0.92, rl, n_side)
    if lo is None or hi is None:
        return []
    hi = hi[np.argsort(hi[:, 0])]
    rg, ag = lo[:, 0], lo[:, 1]
    rl_, al = hi[:, 0], hi[:, 1]
    pg = np.concatenate([[rg[0]], rg[0] + cumulative_trapezoid(ag, rg)])
    mg = np.log(rg) + np.concatenate([[0.0], cumulative_trapezoid((ag-1.0)/rg, rg)])
    Ic = np.concatenate([[0.0], cumulative_trapezoid(al, rl_)])
    Im = np.concatenate([[0.0], cumulative_trapezoid(al/rl_, rl_)])
    out = []
    for frac in anchors:
        i = int(round(frac * (len(rl_) - 1)))
        s = X.HCYMix([1.0], [float(rl_[i])], z, kv[:, None, None] / T)
        q = X.solve_bh_mix(s)
        if q is None:
            continue
        php, mhs = X.hs_thermo_pure(s)
        p_a = php + float(X.pressure_excess(q))
        mu_a = float(np.log(rl_[i]) + mhs + float(X.chem_pot_excess(q)[0]))
        pl = p_a + (Ic - Ic[i])
        ml = mu_a + (Im - Im[i])
        lo_p, hi_p = max(pg.min(), pl.min()), min(pg.max(), pl.max())
        if not hi_p > lo_p:
            continue
        f = lambda qq: np.interp(qq, pg, mg) - np.interp(qq, pl, ml)
        if f(lo_p) * f(hi_p) > 0:
            continue
        qq = brentq(f, lo_p, hi_p, xtol=1e-14)
        a_, b_ = float(np.interp(qq, pg, rg)), float(np.interp(qq, pl, rl_))
        if b_ - a_ > 1e-3:
            out.append(dict(anchor_eta=float(rl_[i]*np.pi/6),
                            eta_g=a_*np.pi/6, eta_l=b_*np.pi/6))
    return out


def _hermite(x, x0, x1, L, R, m, deriv=0):
    """Match derivatives `0..m` at each end — degree `2m+1`.  `deriv` differentiates.

    ⚠ Left end is `t = 0`, where `d^k/dt^k[t^j] = k!·δ_{jk}`; right end is `t = 1`,
    where it is `j!/(j−k)!`.  Using the `t = 1` formula on both halves makes the
    matrix singular, which is how this was found.

    ⚠ `deriv=2` is not a convenience.  Taking `f''` by `np.gradient` twice on the
    sampled polynomial lost **1.1 %** on both gap integrals — and those two integrals
    are the entire physical content of the bridge (see `ftail_bridge_isotherm`), so
    that was the whole error budget of the line.  Differentiating the coefficients
    is exact.
    """
    import math
    n = 2*(m+1); h = x1 - x0
    M = np.zeros((n, n)); rhs = []
    for k in range(m+1):
        for j in range(n):
            M[k, j] = float(math.factorial(k)) if j == k else 0.0
        rhs.append(L[k]*h**k)
    for k in range(m+1):
        for j in range(n):
            M[m+1+k, j] = float(np.prod([j-i for i in range(k)])) if j >= k else 0.0
        rhs.append(R[k]*h**k)
    co = np.linalg.solve(M, np.array(rhs))
    t = (np.asarray(x, float) - x0)/h
    if deriv == 0:
        return sum(co[j]*t**j for j in range(n))
    ff = lambda j, k: float(np.prod([j - i for i in range(k)]))
    return sum(co[j]*ff(j, deriv)*t**(j-deriv)
               for j in range(deriv, n)) / h**deriv


A_HS = lambda r: (1+2*(r*np.pi/6))**2/(1-(r*np.pi/6))**4


def ftail_bridge_isotherm(z, kv, T, m=2, n_side=60, n_gap=80, hh=1.2e-3):
    r"""`(ρ, a)` with the gap filled by a bridge on **`βf_tail`**, not on `ĉ(0)`.

    ⭐ Self-consistent decomposition — Rosenfeld FMT is PY-compressibility to
    1.2e-14, so with `a_HS = (1+2η)²/(1−η)⁴`

        f_tail″(ρ) = (a − a_HS)/ρ            exact on the branch
        f_tail′    = chem_pot_excess          exact, (46)
        f_tail     = free_energy_excess       (45)

    and at a spinodal `a = 0` forces `f_tail″ = −a_HS/ρ_s` — verified to the last
    digit.

    ⭐ Because `a = a_HS + ρf″` is **not a polynomial**, no degree count constrains
    it and the loop survives — unlike a bridge on `ĉ(0)`.

    ⚠⚠ **This is a compressibility–energy hybrid, exactly like the two path
    crossings — an earlier version of this docstring claimed the opposite** ("the
    value and slope set the level and tilt, which `a` does not see").  `coexist`
    sees the gap only through two integrals, and the Hermite end conditions fix both
    in closed form:

        ∫_gap f″ dρ     = f′(ρ_l) − f′(ρ_v)              → the μ increment
        ∫_gap ρ f″ dρ   = [ρf′]_v^l − (f_l − f_v)        → the p increment

    and `f`, `f′` at the ends are (45) and (46), i.e. **energy-route** numbers.  So
    the construction is precisely *compressibility on both arcs, energy across the
    gap* — the same recipe as `spinodal_gap_free_energy` and
    `critical_detour_gap_free_energy`, differing only in how the crossing increment is
    obtained (read off, versus path-integrated through `T`).

    Measured, at `T* = 0.95`:

    * an affine shift of the end data is exact **gauge** — a quintic reproduces an
      affine function, both integrals are invariant, `η_l` moves `3e-13 %`;
    * a non-affine shift of 2 % moves `η_l` by 0.6–1.8 %;
    * **dropping the end slopes destroys the loop outright**;
    * the interpolant ORDER is irrelevant — cubic (which does not even match `f″`)
      0.31776, quintic 0.31826, septic 0.31829, i.e. 0.17 % across three degrees.
      That is why "quintic vs septic" was never a usable error bar.
    """
    st = X.HCYMix([1.0], [rho_of(0.2)], z, kv[:, None, None] / T)
    rv, rl = X.spinodal_densities_exact(st)
    if rv is None or rl is None or rl - rv < 1e-3:
        return None
    lo = arc(z, kv, T, 1e-4, rv, n_side)
    hi = arc(z, kv, T, 0.92, rl, n_side)
    if lo is None or hi is None:
        return None
    hi = hi[np.argsort(hi[:, 0])]

    def f2(r):
        q = X.solve_bh_mix(X.HCYMix([1.0], [float(r)], z, kv[:, None, None] / T))
        if q is None:
            return None
        A = float(np.sum(q.p['A']))
        return ((A/(2*np.pi))**2 - A_HS(r))/r

    ends = []
    for rs, sgn in ((rv, -1.0), (rl, +1.0)):      # step INTO the physical branch
        q = X.solve_bh_mix(X.HCYMix([1.0], [float(rs)], z, kv[:, None, None] / T))
        if q is None:
            return None
        e = [float(X.free_energy_excess(q)), float(X.chem_pot_excess(q)[0]),
             -A_HS(rs)/rs]
        if m >= 3:
            v1, v2 = f2(rs + sgn*hh), f2(rs + 2*sgn*hh)
            if v1 is None or v2 is None:
                return None
            e.append(sgn*(-3*e[2] + 4*v1 - v2)/(2*hh))
        ends.append(e)
    xs = np.linspace(rv, rl, n_gap)
    d2 = _hermite(xs, rv, rl, ends[0][:m+1], ends[1][:m+1], m, deriv=2)
    ag = A_HS(xs) + xs*d2
    rho = np.concatenate([lo[:, 0], xs[1:-1], hi[:, 0]])
    a = np.concatenate([lo[:, 1], ag[1:-1], hi[:, 1]])
    k = np.argsort(rho)
    return rho[k], a[k]


# FMSA-DP specs matching MODELS, for the C⁰ splice.  Same (z, K, σ=1) hard-core
# two-Yukawa fluid — gated in `fmsa_phase_lines.py --convention`.
FMSA_SPEC = {"HSY": dict(nt=1, z=[1.8], k=[1.0]),
             "2YK": dict(nt=2, z=[2.9637, 14.0167], k=[2.1714, -2.1714])}


def fmsa_splice_isotherm(key, T, n_side=110, n_gap=160):
    r"""`(ρ, a)` with the gap carrying **FMSA's own curvature**, ramped to vanish at the
    two `A = 0` spinodals — `C⁰` continuous, and **no derivative condition at all**.

    ⭐⭐ Why this is the construction that works.  Two independent things have to happen
    at the joins and every earlier completion managed only one:

    * `a = 0` there, so the gap's contribution vanishes WITH the gap as the dome closes
      — otherwise `min a` tends to a finite limit, the loop stays deep while narrowing,
      and the tie line never shuts (`βf_tail` Hermite bridge: `min a → −0.8021`,
      `β_eff = 0.137`; raw FMSA splice with the jump left in: `−0.52`, `β_eff = 0.137`);
    * `a′` **left free**, so the zero is SIMPLE.  Matching `ĉ` and `ĉ′` forces `a′ = 0`
      identically (`a′ = −ĉ − ρĉ′`, and `a = 0` ⇒ `ĉ = 1/ρ_s`, `C¹` ⇒ `ĉ′ = −1/ρ_s²`),
      so a simple zero REQUIRES giving up `C¹` — which is exactly what this does.

    Measured: zero order **0.986** (a physical spinodal is 1.0), `β_eff = 0.376` against
    the pure routes' 0.393–0.489, and it reaches `T* = 0.62`.  It is the first completion
    here to get the zero order and the critical exponent right at the same time.

    ⚠ **What the ramp is, stated plainly.**  FMSA's `a` at the exact spinodal densities is
    `−0.44 … −0.63`, not zero — its own spinodal is 26–31 % away in density.  The linear
    ramp subtracts exactly that mismatch, so this is *FMSA's curvature SHAPE renormalised
    to vanish at the exact spinodals*, not "FMSA in the gap".  ⭐ That is the right
    division of labour: FMSA's `f_tail`/curvature shape is its most accurate object
    (0.0–2.6 % on `βf_tail`) and its spinodal *location* is its least (26–31 %), so the
    ramp keeps the former and discards the latter.
    """
    import energy_compress_core as EC
    from fmsa_double_prop import FMSA_double_prop as _F
    z, kv = MODELS[key]
    sp = FMSA_SPEC[key]
    rv, rl = X.spinodal_densities_exact(
        X.HCYMix([1.0], [rho_of(0.2)], z, kv[:, None, None]/T))
    if rv is None or rl is None or rl - rv < 1e-4:
        return None
    lo = arc(z, kv, T, 1e-4, rv, n_side)
    hi = arc(z, kv, T, 0.92, rl, n_side)
    if lo is None or hi is None:
        return None
    hi = hi[np.argsort(hi[:, 0])]

    def a_fm(r):
        m = _F([1.0], [1.0], [1.0], float(r), float(T), n_tails=sp["nt"],
               input_type='bh', z=sp["z"], k=sp["k"], verbose=False)
        x = np.array([1.0])
        hs = 1.0 - float(x @ EC.chat0_hs_fmt(m) @ m.rho)
        c1 = np.real(m.get_c1hat_k(np.array([1e-5]))[0])
        return hs - float(x @ c1 @ m.rho)

    xs = np.linspace(rv, rl, n_gap)
    ag = np.array([a_fm(float(r)) for r in xs])
    t = (xs - rv)/(rl - rv)
    ag = ag - ((1.0 - t)*ag[0] + t*ag[-1])        # ⚠ the ramp — see the docstring
    rho = np.concatenate([lo[:, 0], xs[1:-1], hi[:, 0]])
    a = np.concatenate([lo[:, 1], ag[1:-1], hi[:, 1]])
    k = np.argsort(rho)
    return rho[k], a[k]


def fmsa_splice_binodal(key, T, **kw):
    """`(η_g, η_l)` from the `C⁰` FMSA splice, or `None` if there is no crossing."""
    r = fmsa_splice_isotherm(key, T, **kw)
    c = coexist(*r) if r is not None else None
    return None if c is None else (c[0]*np.pi/6, c[1]*np.pi/6)


def bridge_binodal(key, T, m=2, n_side=60, n_gap=80):
    r"""`(η_g, η_l)` of the `βf_tail`-bridge binodal at one `(model, T)` — **solved**.

    ⭐ The entry point other figures should use.  A bridge line pasted into a figure
    from whatever JSON happened to be lying around is a line nobody can re-derive and
    that goes stale silently when the construction changes.  `None` means "solved,
    and there is no crossing".

    ⚠ **No cache, deliberately.**  Measured per call: `0.79 s` (HSY), `2.75 s` (2YK),
    so a 17-point ladder is 13 s / 47 s — cheaper than the bookkeeping a cache costs.
    And the polynomial is not the expense: the Hermite is 6 numbers and a 6x6 solve,
    **86 µs, 0.011 % of the call**.  The time is `spinodal_densities_exact` (62 %) and
    the two continuation arcs (37 %) — i.e. the two branches `coexist` needs, not the
    bridge.  Cache those instead if this ever becomes hot.
    """
    z, kv = MODELS[key]
    r = ftail_bridge_isotherm(z, kv, float(T), m=m, n_side=n_side, n_gap=n_gap)
    c = coexist(*r) if r is not None else None
    return None if c is None else (c[0]*np.pi/6, c[1]*np.pi/6)


def aflip_binodal(key, T, n_side=60, n_gap=120):
    r"""`(η_g, η_l)` from the `ĉ(0)` bridge with `a″` **flipped** at both spinodals.

    ⚠ The construction is incoherent (see `probe_bridge_content.py aflip`): it retains
    `a′ = 0`, which holds only because `a` is a perfect square, while denying `a″ > 0`,
    which holds for the same reason — and it keeps the DOUBLE zero the square creates
    (measured order 1.974 against a physical spinodal's 1.0).  Computed so the line can
    be looked at, not because it is a candidate.
    """
    z, kv = MODELS[key]
    r = bridge_isotherm(z, kv, float(T), n_side=n_side, n_gap=n_gap, a2_sign=-1.0)
    c = coexist(*r) if r is not None else None
    return None if c is None else (c[0]*np.pi/6, c[1]*np.pi/6)


def bridge_binodal_curve(key, Ts, **kw):
    """`[(T, η_g, η_l), …]` over a ladder, skipping temperatures with no crossing."""
    out = []
    for T in Ts:
        v = bridge_binodal(key, float(T), **kw)
        if v is not None:
            out.append((float(T), v[0], v[1]))
    return out


def compute(n_T=16, t_lo=0.62):
    r"""⚠ The ladder tops at the `A = 0` **spinodal** apex, not the fold apex.

    Every construction in this file is anchored at the two spinodal densities, and
    those exist all the way to `spinodal_apex`; the fold apex only bounds where the
    isotherm is *cut*.  Starting the ladder at `0.995·fold_apex` truncated all three
    lines 1.9 % below their real ceiling and hid the one structural fact worth
    reporting — that the bridge closes **exactly onto the spinodal apex**, because
    its own interpolation interval is `[ρ_sp⁻, ρ_sp⁺]` and that shrinks to a point
    there.  The bridge therefore inherits the compressibility route's `T_c` unchanged;
    it supplies a binodal without moving the critical point.
    """
    want = (sys.argv[sys.argv.index("--model") + 1]
            if "--model" in sys.argv else None)
    try:                       # merge, never clobber the other model's finished work
        out = json.load(open(JSON))
    except Exception:
        out = {}
    for key, (z, kv) in MODELS.items():
        if want and key != want:
            continue
        ap = apex_T(z, kv, 0.40, 1.60, "fold")
        sa = apex_T(z, kv, 0.40, 1.60, "spinodal")
        # ⭐⭐ Clustered geometrically at the top, because the dome closes with a
        # DEGENERATE exponent.  Measured on `probe_bridge_apex.py`, over a decade in
        # the spinodal width `w = ρ_sp⁺ − ρ_sp⁻`:
        #
        #     binodal width ∝ w^0.25      (local slope 0.272 → 0.222)
        #     midpoint → η = 0.1609       vs the apex's own η_c = 0.1615
        #
        # and `w ∝ √(T_sa − T)`, so the half-width vanishes like `(ΔT)^{1/8}` rather
        # than the mean-field `(ΔT)^{1/2}`.  The dome therefore DOES close on the
        # `A = 0` apex, but eight times too slowly to see: at `1 − T/T_sa = 1e-6` it
        # is still 0.086 wide.  ⚠ Below `Δt ≈ 1e-4` the spinodal pair itself stops
        # resolving (`w` saturates at 0.0044 where √-scaling wants 0.0008), so the
        # ladder stops there and `render` closes the last step analytically.
        Ts = np.concatenate([sa*(1.0 - np.array([1e-4, 3e-4, 1e-3, 3e-3])),
                             np.linspace(0.99*sa, t_lo, n_T)])
        print(f"[{key}] fold apex T* = {ap:.4f}   A=0 spinodal apex T* = {sa:.4f}",
              flush=True)
        br = []
        for T in Ts:
            r = bridge_isotherm(z, kv, float(T))
            if r is None:
                print(f"    bridge T*={T:.4f}  isotherm not built", flush=True); continue
            c = coexist(*r)
            if c is None:
                print(f"    bridge T*={T:.4f}  no crossing", flush=True); continue
            br.append(dict(T=float(T), eta_g=c[0]*np.pi/6, eta_l=c[1]*np.pi/6))
            print(f"    bridge T*={T:.4f}  eta = {c[0]*np.pi/6:.4f} .. "
                  f"{c[1]*np.pi/6:.4f}", flush=True)
        for m, tag in ((2, "ftail5"), (3, "ftail7")):
            fb = []
            for T in Ts:
                r = ftail_bridge_isotherm(z, kv, float(T), m=m)
                if r is None:
                    print(f"    {tag} T*={T:.4f}  isotherm not built", flush=True)
                    continue
                c = coexist(*r)
                if c is None:
                    print(f"    {tag} T*={T:.4f}  no crossing", flush=True); continue
                fb.append(dict(T=float(T), eta_g=c[0]*np.pi/6, eta_l=c[1]*np.pi/6))
                print(f"    {tag} T*={T:.4f}  eta = {c[0]*np.pi/6:.4f} .. "
                      f"{c[1]*np.pi/6:.4f}", flush=True)
            globals().setdefault("_FT", {})[tag] = fb
        anc = []
        for T in Ts:
            r = liquid_anchor(z, kv, float(T))
            if r:
                anc.append(dict(T=float(T), pts=r))
                sp = [p['eta_l'] for p in r]
                print(f"    anchor T*={T:.4f}  eta_l over anchors: "
                      f"{min(sp):.4f}..{max(sp):.4f}  "
                      f"({100*(max(sp)/min(sp)-1):.0f}% spread)", flush=True)
            else:
                print(f"    anchor T*={T:.4f}  no crossing", flush=True)
        out[key] = dict(bridge=br, anchor=anc, fold_apex=float(ap),
                        spinodal_apex=float(sa),
                        ftail5=globals().get("_FT", {}).get("ftail5", []),
                        ftail7=globals().get("_FT", {}).get("ftail7", []))
        with open(JSON, "w") as fh:
            json.dump(out, fh, indent=1)
    print(f"[saved] {JSON}")
    return out


def _apex_marks(z, kv, ap, sa, f=0.9995):
    r"""`(η, T)` of the fold apex and of the `A = 0` spinodal apex.

    Both domes close at their own apex, so the density is read as the midpoint of
    the pair just below it — at `f = 0.9995` the pair is already within `0.010` in
    `η`, which is finer than the marker.
    """
    st = lambda T: X.HCYMix([1.0], [rho_of(0.2)], z, kv[:, None, None]/T)
    a, b = X.fold_densities(st(f*ap))
    rv, rl = X.spinodal_densities_exact(st(f*sa))
    return ((None if a is None else 0.5*(a+b)*np.pi/6), ap), \
           ((None if rv is None else 0.5*(rv+rl)*np.pi/6), sa)


def render(data=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if data is None:
        data = json.load(open(JSON))
    main = json.load(open(JSON_MAIN))
    d, m = data["HSY"], main["HSY"]      # the comparison figure is HSY-only by design
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    C_HUG, C_DET, C_ANC, C_EN, C_FT = ("#e67e22", "#f5b041", "#8e44ad",
                                       "#2a78d6", "#17a589")
    C_AF = "#c2185b"; C_SP = "#111111"

    _ef = _tf = _es = _ts = None
    if d.get("fold_apex") and d.get("spinodal_apex"):
        (_ef, _tf), (_es, _ts) = _apex_marks(MODELS["HSY"][0], MODELS["HSY"][1],
                                             d["fold_apex"], d["spinodal_apex"])

    b = sorted(m["binodal"], key=lambda p: p["T"])
    ax.plot([p["eta_g"] for p in b] + [p["eta_l"] for p in b][::-1],
            [p["T"] for p in b] + [p["T"] for p in b][::-1], "--",
            color=C_EN, lw=1.6)
    for key, col, lab in (("binodal_fold", C_HUG, "hug the $A=0$ spinodal"),
                          ("binodal_apex", C_DET, "rectangle over the critical point")):   # legacy JSON key
        pts = sorted(m.get(key, []), key=lambda p: p["T"])
        T = [p["T"] for p in pts]
        ax.plot([p["eta_g"] for p in pts], T, "-", color=col, lw=2.2)
        ax.plot([p["eta_l"] for p in pts], T, "-", color=col, lw=2.2)

    ft = sorted(d.get("ftail5", []), key=lambda p: p["T"])
    if ft:
        # ⚠ NOT "the completion with no energy-route content" — an earlier caption
        # said that and it is false; the loop dies if the (46) end slopes are
        # dropped.  What is true is that the interpolant ORDER is irrelevant (cubic,
        # quintic and septic agree to 0.17 %), so only one degree is drawn.
        T = [p["T"] for p in ft]
        ax.plot([p["eta_g"] for p in ft], T, "-", color=C_FT, lw=2.2)
        ax.plot([p["eta_l"] for p in ft], T, "-", color=C_FT, lw=2.2)
        # ⭐ Closed onto the apex, dotted, because the closure is an EXTRAPOLATION:
        # the width vanishes like `(ΔT)^{1/8}`, and the spinodal pair stops resolving
        # a decade before the dome looks shut.  Solid = computed, dotted = the last
        # step, asserted from the measured scaling and from the midpoint already
        # sitting on `η_c`.
        if _es is not None:
            top = max(ft, key=lambda q: q["T"])
            for side in ("eta_g", "eta_l"):
                ax.plot([top[side], _es], [top["T"], _ts], ":", color=C_FT, lw=2.0)

    fsp = sorted(d.get("fmsa_splice", []), key=lambda p: p["T"])
    if fsp:
        # ⭐ Black and heaviest: the only completion that gets BOTH the spinodal zero
        # order (0.986, physical is 1) and the critical exponent (β_eff = 0.376 against
        # the pure routes' 0.393–0.489) right, and reaches T* = 0.62.
        T = [p["T"] for p in fsp]
        ax.plot([p["eta_g"] for p in fsp], T, "-", color=C_SP, lw=2.6, zorder=8)
        ax.plot([p["eta_l"] for p in fsp], T, "-", color=C_SP, lw=2.6, zorder=8)

    af = sorted(d.get("aflip", []), key=lambda p: p["T"])
    if af:
        # ⚠⚠ Dash-dot because it is a TRADE, not a reject — an earlier version of this
        # comment called it "shown for its failure mode, not offered", which the
        # critical-exponent measurement refutes.  Benchmarked against the two PURE
        # routes (the only sensible yardstick for convergence to `T_c`):
        #
        #     virial route      beta_eff = 0.489     f_tail bridge   beta_eff = 0.140
        #     energy route      beta_eff = 0.393     a'' FLIPPED     beta_eff = 0.553
        #
        # so the flip closes on its critical point with a near-mean-field exponent,
        # closer to the virial route than the energy route is, while the `f_tail` bridge
        # is the degenerate one (its `min a` tends to a FINITE −0.8021 as the gap shuts,
        # so the loop stays deep while narrowing and the tie line never closes).
        # ⚠ The flip pays for it elsewhere: spinodal zero of order 1.974 against a
        # physical 1.0, and no coexistence below `T* ≈ 0.78` (the mirrored `a`
        # overshoots, dragging the liquid branch to `βp = −2.4 … −22`, so the gas and
        # liquid pressure ranges stop overlapping).  Each construction gets one of the
        # two right.
        T = [p["T"] for p in af]
        ax.plot([p["eta_g"] for p in af], T, "-.", color=C_AF, lw=1.9)
        ax.plot([p["eta_l"] for p in af], T, "-.", color=C_AF, lw=1.9)

    anc = sorted(d.get("anchor", []), key=lambda p: p["T"])
    if anc:
        T = [p["T"] for p in anc]
        glo = [min(q["eta_g"] for q in p["pts"]) for p in anc]
        ghi = [max(q["eta_g"] for q in p["pts"]) for p in anc]
        llo = [min(q["eta_l"] for q in p["pts"]) for p in anc]
        lhi = [max(q["eta_l"] for q in p["pts"]) for p in anc]
        ax.fill_betweenx(T, glo, ghi, color=C_ANC, alpha=0.30, lw=0)
        ax.fill_betweenx(T, llo, lhi, color=C_ANC, alpha=0.30, lw=0)

    # ⚠ Above the fold apex the isotherm is NOT cut — MSA exists at every density,
    # it simply has `A < 0` there and `a = (A/2π)²` stays positive, so there is no
    # loop.  Drawn so the reader can see that the bridge above that line is
    # OVERWRITING real MSA data rather than filling a hole.
    #
    # ⭐ The two apexes are the point of the figure's top end: the bridge's own
    # interpolation interval is `[ρ_sp⁻, ρ_sp⁺]`, which closes exactly at the `A=0`
    # apex — so the bridge inherits the compressibility route's critical point and
    # moves it nowhere, while the energy route's sits 18 % higher.
    if _tf is not None:
        ax.axhline(_tf, color="0.45", lw=1.0, ls=":", zorder=1)
        if _ef is not None:
            ax.plot([_ef], [_tf], "o", mfc="none", mec="0.35", mew=1.6, ms=7,
                    zorder=6)
        if _es is not None:
            ax.plot([_es], [_ts], "o", color=C_FT, mec="0.2", mew=1.0, ms=7.5,
                    zorder=6)
    bt = max(b, key=lambda q: q["T"])
    ax.plot([0.5*(bt["eta_g"] + bt["eta_l"])], [bt["T"]], "o", color=C_EN, ms=7,
            zorder=6)

    _b = 0.03
    f = lambda x: np.where(np.asarray(x, float) >= _b, x,
                           _b*(1+np.log(np.maximum(np.asarray(x, float), 1e-300)/_b)))
    g = lambda y: np.where(np.asarray(y, float) >= _b, y,
                           _b*np.exp(np.clip(np.asarray(y, float)/_b - 1, -700, 700)))
    ax.set_xscale("function", functions=(f, g))
    ax.set_xticks([1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xticklabels(["0.001", "0.01", "0.05", "0.1", "0.2", "0.3", "0.4", "0.5"],
                       fontsize=8)
    ax.set_xlim(3e-4, 0.5); ax.set_ylim(0.55, 1.30)
    ax.set_xlabel(r"packing fraction  $\eta$"
                  "\n(log below 0.03, linear above — glued $C^1$)")
    ax.set_ylabel(r"reduced temperature  $T^*$")
    ax.grid(alpha=0.25)
    ax.set_title("HSY $z=1.8$ — the compressibility route cannot reach the liquid;\n"
                 "six ways of supplying what it is missing — five trade one failure for\n"
                 "another; the $C^0$ FMSA splice does not",
                 fontsize=11)
    L = lambda **kw: Line2D([0], [0], **kw)
    import matplotlib.patches as mp
    ax.legend([L(color=C_SP, lw=2.6), L(color=C_HUG, lw=2.2), L(color=C_DET, lw=2.2),
               L(color=C_FT, lw=2.2), L(color=C_AF, lw=1.9, ls="-."),
               mp.Patch(color=C_ANC, alpha=0.30),
               L(color=C_EN, lw=1.6, ls="--"),
               L(color=C_FT, lw=0, marker="o", mec="0.2", mew=1.0, ms=7.5),
               L(color=C_EN, lw=0, marker="o", ms=7),
               L(color="0.45", lw=1.0, ls=":", marker="o", mfc="none",
                 mec="0.35", mew=1.6, ms=7)],
              [r"$C^0$ FMSA splice — FMSA's curvature in the gap, ramped to $a=0$;"
               "\n     "
               r"simple zero (0.986) AND $\beta_{\rm eff}=0.38$, both right",
               "hug the $A=0$ spinodal",
               r"rectangle over the $A=0$ critical point",
               r"bridge $\beta f_{\rm tail}$ across the gap  (on the isotherm)",
               r"$\hat c(0)$ bridge, $a''$ SIGN-FLIPPED — closes on $T_c$ properly"
               "\n"
               r"     ($\beta_{\rm eff}=0.55$ vs the pure routes' 0.39--0.49),"
               r" but dies below $T^*\approx0.78$",
               r"liquid anchored by (45)/(46)  —  band over $\rho_a$",
               "energy route (reference, not a completion)",
               r"critical point, compressibility ($A=0$ apex) — bridge closes here,"
               "\n     but only as $(\Delta T)^{1/8}$, so the last step is dotted",
               "critical point, energy route",
               "fold apex — above it the isotherm is whole, only the loop is gone"],
              loc="lower left", fontsize=8.0, frameon=False)
    fig.tight_layout()
    fig.savefig(PNG, dpi=150)
    print(f"[fig] {PNG}")


if __name__ == "__main__":
    if "--render" in sys.argv:
        render()
    else:
        render(compute())
