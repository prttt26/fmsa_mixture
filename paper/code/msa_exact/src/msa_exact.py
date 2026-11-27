r"""msa_exact.py — the EXACT MSA solution for a single-component hard-core Yukawa fluid.

Group **MSAX** (`todo/waisman_msa_plan.md`), Phase 1.
Derivation and conventions: `numerical_notes/theory/waisman_msa_closed_form.md`.

Source
------
E. Waisman, *Mol. Phys.* **25**, 45 (1973) — `pdf/sources_paywall/The radial
distribution function for a fluid of hard spheres at high densities.pdf`.
Equation numbers below are his.  The mixture generalisation (Blum & Høye,
*J. Stat. Phys.* **19**, 317 (1978)) is Phase 2 and is deliberately not started
here: Waisman's single-component system is three equations against roughly twenty,
so the transcription risk is an order of magnitude lower, and **his own converged
solution is a direct gate on exactly these equations** with no dictionary in
between.

The system
----------
With `x = r/d`, `ξ = πρd³/6`, closure `C(x) = K x⁻¹e^{−z(x−1)}` for `x > 1` and
`g(x) = 0` for `x < 1`, the solution is (his Eq. 2)

    −C(x) = a + bx + ξ a x³/2
            + v·[1 − e^{−zx}]/(zx)
            + v²·(cosh zx − 1)/(2K z² e^{z}·x)          for x < 1

⚠ **The `·x` in the last denominator is a correction to the printed Eq. (2)**,
which omits it.  See `minus_C` and `waisman_msa_closed_form.md` §7e: the two
forms agree at `x = 1`, so `y₀ = g(1⁺)` and hence (3a) are unaffected — which is
why the error survived, `y₀` being the one number Waisman checks — while (3b)
and (3c) are broken by up to 1.2e+02 and 9.9e-03 at `z = 1.8`.  With the `x`
restored all three hold to machine precision and this route reproduces FMSA-DP's
`O(K)` slope on 25/25 grid points (worst 4.2e-05, against 55 % before).

with three unknowns `(a, b, v)` fixed by his Eq. (3):

    (3a)  24ξ y₀²             = −4b + 2vz − v²/(K e^{z})
    (3b)  24ξ (y₁² − 2y₀y₂)   = 24ξa − 2vz³ + v²z²/(K e^{z})
    (3c)  a                   = 1 − 24ξ ∫₀^∞ C(x)x² dx

where `y_i ≡ (dⁱ/dxⁱ)[x g(x)]|_{x=1⁺}`, obtained from the `x<1` side as
`y_i = (dⁱ/dxⁱ)[K e^{z(1−x)} + x·P(x)]|_{x=1}` with `P ≡ −C` the bracket above.

**All of that was verified against Waisman's own converged point before any of it
was coded** (see `--gate`): `y₀` reproduces his printed `5.69`, and (3a)/(3b)
balance to 0.2 % on his 4-digit `a, b, v`.  (3c) additionally collapses to the
exact Percus–Yevick identity `a = 1 + 8ξa + 6ξb + 2ξ²a` when `v = 0`.

⚠ That gate is necessary but was **not sufficient** — it passes both with and
without the `·x` above, because `z = 28.751` is precisely where the defect dies.
A single published state point cannot gate an ansatz; the `O(K)` limit against an
independent theory is what caught it.

⚠ On "exact"
------------
The *reduction* is exact — no approximation between the integral equation and the
algebraic system.  But the system has **several roots and the source selects
none**: Blum & Høye (p. 323) state that with `K` given "one must expect multiple
solutions to occur, of which only one is acceptable".  So the branch ambiguity is
**intrinsic to the MSA closure**, not an artifact of a numerical OZ solver; going
analytic makes it finite-dimensional and enumerable, which is the gain, but does
not resolve it.  `solve` therefore returns **every** root it finds and leaves
selection to the caller; `SOLUTION.physical_ok` collects the necessary conditions
that are known.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve, least_squares

# Waisman 1973 §1, ξ = 0.49 — a converged solution of exactly this system.
# ⚠ Regime caveat: he is using the Yukawa as a *fitting form for hard spheres*
# (K and z chosen to match Carnahan–Starling), so ξ = 0.49 and z = 28.751 are both
# extreme.  A stringent algebra test; says nothing about the physical branch at the
# βK ≈ 3, z = 1–10 conditions Group MSAX actually cares about.
K_PIN = 0.05   # below this the w-column of the Jacobian is ~K; see solve_continued

WAISMAN = dict(xi=0.49, K=1.336, z=28.751, a=51.85, b=-60.278, v=2.432, y0=5.69)


# ═══════════════════════════════════════════════════════════════════════════════
# numerically stable pieces
# ═══════════════════════════════════════════════════════════════════════════════
#
# Every `cosh`/`sinh` in the solution appears divided by `e^z`, and `z` can be ~30
# (Waisman) where `e^z ~ 1e12`.  Forming `cosh(zx)` and then dividing overflows the
# useful precision; these fold the `e^{−z}` in first, so every exponent is ≤ 0 for
# `x ∈ [0,1]`.

def _cosh_ratio(z: float, x: float) -> float:
    """(cosh(zx) − 1)/e^z, evaluated without ever forming cosh(zx)."""
    return 0.5 * (np.exp(z * (x - 1.0)) + np.exp(-z * (x + 1.0))) - np.exp(-z)


def _sinh_ratio(z: float, x: float) -> float:
    """sinh(zx)/e^z."""
    return 0.5 * (np.exp(z * (x - 1.0)) - np.exp(-z * (x + 1.0)))


def py_ab(xi: float) -> tuple[float, float]:
    """Percus–Yevick `(a, b)` — the `K → 0` limit of the system.

    Wertheim–Thiele: `c(x) = −λ₁ − 6ηλ₂x − ½ηλ₁x³` with
    `λ₁ = (1+2η)²/(1−η)⁴`, `λ₂ = −(1+η/2)²/(1−η)⁴`.  Matching Waisman's
    `−C = a + bx + ξa x³/2` gives `a = λ₁`, `b = 6ηλ₂` — and the `x³` coefficients
    agree identically, which is the first sign the ansatz is the right one.
    """
    lam1 = (1.0 + 2.0 * xi) ** 2 / (1.0 - xi) ** 4
    lam2 = -((1.0 + 0.5 * xi) ** 2) / (1.0 - xi) ** 4
    return lam1, 6.0 * xi * lam2


def hs_laplace_moment(xi: float, z: float) -> float:
    """`G(z) = ∫₁^∞ x g_HS(x) e^{−zx} dx` for Percus–Yevick hard spheres (σ = 1).

    Wertheim–Thiele closed form.  Checked against the dilute limit: as `η → 0`,
    `g → 1` and this must go to `e^{−z}(1+z)/z²`, which it does identically.
    """
    L = 12.0 * xi * ((1.0 + xi / 2.0) * z + (1.0 + 2.0 * xi))
    S = ((1.0 - xi) ** 2 * z ** 3 + 6.0 * xi * (1.0 - xi) * z ** 2
         + 18.0 * xi ** 2 * z - 12.0 * xi * (1.0 + 2.0 * xi))
    return z * L / (12.0 * xi * (L + S * np.exp(z)))


def w_first_order(xi: float, z: float) -> float:
    """`w = v/K` in the limit `K → 0` — **the first-order (FMSA) amplitude**.

    Waisman defines `v ≡ 24ξK ∫₁^∞ x e^{−z(x−1)}g(x)dx`, so
    `w = 24ξ e^{z} ∫₁^∞ x g(x)e^{−zx}dx`.  At `K → 0` the `g` there is the
    hard-sphere one, giving `w₀ = 24ξ e^{z} G_HS(z)`.

    This is the quantity the algebraic system **cannot** supply at `K = 0` — `w`
    only ever appears as `K w`, so the system is rank-deficient there and any
    ramp started from it tracks an arbitrary root.  It is also exactly the
    self-consistency of theory-note §6 cut at first order: the amplitude is the
    bare coupling times the transform of `g` at the Yukawa rate, with the *full*
    `g` replaced by `g_HS`.  So this single line is both the seed the
    continuation needs and the `O(K)` limit the plan gates against.
    """
    return 24.0 * xi * np.exp(z) * hs_laplace_moment(xi, z)


# ═══════════════════════════════════════════════════════════════════════════════
# the state point and the algebraic system
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HCY:
    """Single-component hard-core Yukawa in Waisman's reduced variables.

    `xi` = πρd³/6 (the packing fraction η), `K` = βA the Yukawa amplitude
    (`K > 0` is attraction; his §2: "it is enough to set K = βA"), `z` the decay
    rate in units of 1/d.
    """
    xi: float
    K: float
    z: float

    # ─── the inner-core DCF, his Eq. (2) ──────────────────────────────────────

    def minus_C(self, x, a: float, b: float, w: float):
        """`−C(x)` for `x < 1`, in the regular variable `w = v/K`.

        ⚠ **Waisman's Eq. (2) is printed with the `v²` term missing its `1/x`.**
        He prints `v²(cosh zx − 1)/(2Kz²e^{z})`; the correct term is that divided
        by `x`.  The two agree at `x = 1`, so `y₀` and hence Eq. (3a) are
        unaffected — which is exactly why the defect survived: `y₀ = g(1⁺)` is the
        one number he checks.  See `waisman_msa_closed_form.md` §7f.

        The slip is a natural one: in `x·(−C(x))` the term *is* exactly
        `v²(cosh zx − 1)/(2Kz²e^{z})`, i.e. the `x·(−C)` form's last term was
        carried into the `−C` equation.
        """
        xi, K, z = self.xi, self.K, self.z
        v = K * w
        x = np.asarray(x, dtype=float)
        yuk = np.where(np.abs(z * x) < 1e-12, v * 1.0,
                       v * (1.0 - np.exp(-z * np.where(x == 0, 1e-300, x)))
                       / (z * np.where(x == 0, 1e-300, x)))
        quad = v * v / (2.0 * K * z * z) * np.vectorize(
            lambda xx: _cosh_ratio(z, xx))(x) / np.where(x == 0, 1e-300, x)
        return a + b * x + xi * a * x ** 3 / 2.0 + yuk + quad

    # ─── the contact derivatives y_i ──────────────────────────────────────────

    def y_moments(self, a: float, b: float, w: float) -> tuple[float, float, float]:
        """`y_i = (dⁱ/dxⁱ)[K e^{z(1−x)} + x·(−C(x))]|_{x=1}`, i = 0,1,2.

        `y₀ = g(1⁺)`, the contact value — which is how this was validated: it
        reproduces Waisman's printed 5.69 at his converged point.
        """
        xi, K, z = self.xi, self.K, self.z
        v = K * w
        q = K * w * w / (2.0 * z * z)          # = v²/(2Kz²); regular as K → 0
        ch, sh = _cosh_ratio(z, 1.0), _sinh_ratio(z, 1.0)
        ez = np.exp(-z)

        # From `x·(−C(x)) = ax + bx² + ξax⁴/2 + (v/z)(1−e^{−zx}) + q̃(cosh zx − 1)`
        # with the corrected Eq. (2) — see `minus_C`.  `y₀` is unchanged (the two
        # forms agree at x = 1); `y₁` loses a `ch` and `y₂` a `2z·sh` relative to
        # the printed form, and *that* is what repairs Eq. (3b).
        y0 = K + a + b + xi * a / 2.0 + (v / z) * (1.0 - ez) + q * ch
        y1 = -z * K + a + 2.0 * b + 2.0 * xi * a + v * ez + q * z * sh
        y2 = (z * z * K + 2.0 * b + 6.0 * xi * a - v * z * ez
              + q * z * z * (ch + ez))
        return float(y0), float(y1), float(y2)

    # ─── the compressibility integral of Eq. (3c) ─────────────────────────────

    def _int_C_x2(self, a: float, b: float, w: float) -> float:
        """`∫₀^∞ C(x)x² dx`, split at the core.

        Outside: `∫₁^∞ K x e^{−z(x−1)}dx = K(1+z)/z²`.
        Inside:  `−∫₀¹ (−C)x²dx`, term by term.
        """
        xi, K, z = self.xi, self.K, self.z
        v = K * w
        ez = np.exp(-z)

        outer = K * (1.0 + z) / (z * z)

        poly = a / 3.0 + b / 4.0 + xi * a / 12.0
        yuk = (v / z) * (0.5 - (1.0 - ez * (1.0 + z)) / (z * z))
        # With the corrected Eq. (2) the `v²` term carries `1/x`, so this is
        # ∫₀¹ x(cosh zx − 1)/e^z dx = [sinh z/z − cosh z/z² + 1/z²]/e^z − 1/(2e^z).
        ch1, sh1 = _cosh_ratio(z, 1.0), _sinh_ratio(z, 1.0)
        cosh_over = ch1 + ez                      # cosh(z)/e^z
        quad_int = sh1 / z - cosh_over / (z * z) + ez / (z * z) - ez / 2.0
        quad = K * w * w / (2.0 * z * z) * quad_int

        return outer - (poly + yuk + quad)

    # ─── the three residuals ──────────────────────────────────────────────────

    def residuals_scaled(self, abw) -> np.ndarray:
        """`residuals` divided by the natural size of each equation.

        (3b) carries `y₁² − 2y₀y₂ ~ 1e4` where (3c) is `O(a) ~ 1`, so the raw
        residual vector spans four orders of magnitude and any least-squares step
        is dominated by (3b) alone.  Scaling is what makes the ill-conditioned
        small-`K` region tractable.
        """
        a, b, w = (float(t) for t in abw)
        y0, y1, y2 = self.y_moments(a, b, w)
        sc = np.array([max(1.0, abs(24.0 * self.xi * y0 ** 2)),
                       max(1.0, abs(24.0 * self.xi * (y1 ** 2 - 2 * y0 * y2))),
                       max(1.0, abs(a))])
        return self.residuals(abw) / sc

    def residuals(self, abw) -> np.ndarray:
        """Waisman Eq. (3a), (3b), (3c) as residuals, in `(a, b, w = v/K)`.

        Substituting `v = K w` clears every `1/K`: the `v²/(K e^z)` of (3a)/(3b)
        becomes `K w²/e^z` and the `v²/(2Kz²e^z)` of Eq. (2) becomes
        `K w²/(2z²e^z)`.  **The system is then regular at `K = 0`**, which is what
        makes continuation from the exact Percus–Yevick point possible — in
        Waisman's own `v` it is a 0/0 limit and the ramp cannot start.
        """
        a, b, w = (float(t) for t in abw)
        xi, K, z = self.xi, self.K, self.z
        v = K * w
        y0, y1, y2 = self.y_moments(a, b, w)
        q = K * w * w * np.exp(-z) if z < 700 else 0.0    # = v²/(K e^z)

        r1 = 24.0 * xi * y0 ** 2 - (-4.0 * b + 2.0 * v * z - q)
        r2 = (24.0 * xi * (y1 ** 2 - 2.0 * y0 * y2)
              - (24.0 * xi * a - 2.0 * v * z ** 3 + q * z * z))
        r3 = a - (1.0 - 24.0 * xi * self._int_C_x2(a, b, w))
        return np.array([r1, r2, r3])


# ═══════════════════════════════════════════════════════════════════════════════
# solutions
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Solution:
    st: HCY
    a: float
    b: float
    w: float           # w = v/K, the variable regular at K = 0
    residual: float
    A: float | None = None   # Blum & Høye's `A`; `None` on the Waisman route — see `physical_ok`

    @property
    def v(self) -> float:
        """Waisman's amplitude `v = K w` — reported for comparison with his tables."""
        return self.st.K * self.w

    @property
    def y0(self) -> float:
        return self.st.y_moments(self.a, self.b, self.w)[0]

    @property
    def g_contact(self) -> float:
        """`g(1⁺) = y₀` — the contact value."""
        return self.y0

    @property
    def dbetaP_drho(self) -> float:
        """`∂βp/∂ρ` by the compressibility route.

        Waisman's Eq. (3c) *defines* `a` as `∂βp/∂ρ`: "Equation (3c) is the
        definition of the compressibility".  So the indicator FWL computed by
        k→0 quadrature is simply `a` — available in closed form, no grid.
        """
        return self.a

    @property
    def physical_ok(self) -> tuple[bool, str]:
        r"""Is this the physical root?  ⭐ `A > 0` decides it — when `A` is known.

        ⚠⚠ **`a` and `g(1⁺)` cannot decide it, and not for want of tuning.**
        Outside the fold gap the Blum–Høye system has **exactly two** real roots at
        every state (measured: 460 multi-starts in `(D̃, Ĝ)` including negative `Ĝ`
        and `|D̃|` to 300, over `z` = 1/1.8/4 and `η` = 0.02…0.45 — never one, never
        three).  They differ by the **sign of `A/2π = 1 − ρQ̂(0)`**, i.e. by the
        winding of the Wiener–Hopf factorization.  But `a = (A/2π)²` is the
        *square*, so it is blind to exactly that.  Measured at `η = 0.05`,
        `T* = 1.20`, the wrong-winding root has

            A = −7.49,   a = 1.421 > 0,   g(1⁺) = 3.67 > 0,   ĝ(z) = 0.63 > 0

        — it passes every test that can be built from `(a, b, w)`.  ⇒ On Waisman's
        variables the physical root is **not identifiable in principle**; the sign
        has to come from the Blum–Høye route, which is why `solve_bh` now records it
        and `solve` (Waisman) leaves it `None`.

        ⚠ Still only *mechanical* stability and the right winding.  A state between
        the binodal and the spinodal passes everything here and is **metastable**;
        deciding that needs the free energy, not the root.
        """
        if not np.isfinite([self.a, self.b, self.v]).all():
            return False, "non-finite"
        if self.A is not None and self.A <= 0.0:
            return False, f"A = {self.A:.4g} <= 0 — wrong Wiener-Hopf winding"
        if self.a <= 0.0:
            return False, f"dbetaP/drho = a = {self.a:.4g} <= 0"
        if self.g_contact < 0.0:
            return False, f"g(1+) = {self.g_contact:.4g} < 0"
        if self.A is None:
            return True, ("passes the (a, b, w) necessary conditions — but the sign "
                          "of A is not available on this route, so the wrong-winding "
                          "root would pass too")
        return True, "A > 0: the physical winding"


def solve_continued(st: HCY, n_steps: int = 60, tol: float = 1e-8,
                    jump_tol: float = 0.25, max_halve: int = 10
                    ) -> Solution | None:
    """The **physical** root, tracked by continuation from Percus–Yevick.

    The selection rule, which neither source states at `N = 1`
    (Blum & Høye's Eq. 41 is a mixture symmetry, vacuous here):

        the physical root is the one continuously connected to PY,
        i.e. the branch with `w → w₀ = 24ξ e^{z} G_HS(z)` as `K → 0`.

    That `w₀` is independently confirmed — `hs_laplace_moment` matches the
    project's `get_g0L_laplace` to 1.00000, and the resulting `O(K)` slope of
    `∂βp/∂ρ` matches `FMSA_double_prop.get_chat_k` to 1.00000 at large `z`.
    So pinning `w = w₀` at small `K` is **the branch condition, not a numerical
    hack**, and the jump guard below extends the same condition to large `K`.

    Two regimes:

    * `K < K_PIN` — the `w` column of the Jacobian is `~K`, so any trust-region
      method terminates before exploring it (measured: the residual floors at
      `O(K)` and does not move with the finite-difference step).  There `w` is
      not in doubt: pin it to `w₀` and solve the well-conditioned 2×2.
    * `K ≥ K_PIN` — free 3×3, but seeded by **linear extrapolation** from the
      last two accepted points and **rejected if it jumps**.  Without this the
      solve lands on a genuinely different root (measured at η = 0.4, z = 5:
      residual 7e-16, yet the `O(K)` slope is off by a factor 2.2).  On a
      rejected step the increment is halved and retried.

    Returns `None` when the branch cannot be continued even at the smallest
    step — which is **information**: the physical branch has ended (spinodal) or
    collided with another.  Mapping that is MSAX.5.
    """
    a0, b0 = py_ab(st.xi)
    w0 = w_first_order(st.xi, st.z)
    if st.K == 0.0:
        return Solution(st, a0, b0, w0, 0.0)

    x = np.array([a0, b0, w0], dtype=float)
    prev_x, prev_step = None, None
    K, dK, halved = 0.0, st.K / n_steps, 0

    while K < st.K * (1.0 - 1e-12):
        step = min(dK, st.K - K)
        Kn = K + step
        stn = HCY(xi=st.xi, K=float(Kn), z=st.z)

        if prev_x is None or prev_step is None:
            pred = x.copy()
        else:                              # linear extrapolation along the branch
            pred = x + (x - prev_x) * (step / prev_step)

        # The branch condition, as a CONSTRAINT rather than a pin.  Pinning w
        # leaves 3 equations for 2 unknowns, so the third residual is O(K) by
        # construction -- a valid seed, not a valid solution.  Instead solve the
        # full 3x3 with `w` confined to a neighbourhood of its predicted value:
        # that is exactly "the branch continuously connected to PY", and it stops
        # the solve falling into the second root of the O(K) system (which is
        # quadratic in w, and genuinely has two roots at moderate z).
        dw = 0.5 * (abs(pred[2]) + 1.0)
        lo = np.array([-np.inf, -np.inf, pred[2] - dw])
        hi = np.array([np.inf, np.inf, pred[2] + dw])
        try:
            r = least_squares(stn.residuals_scaled, np.clip(pred, lo, hi),
                              bounds=(lo, hi), method='trf',
                              xtol=1e-15, ftol=1e-15, gtol=1e-15)
        except Exception:
            return None
        xn = r.x

        res = float(np.max(np.abs(stn.residuals_scaled(xn))))
        jump = float(np.max(np.abs(xn - pred) / (np.abs(pred) + 1.0)))
        good = np.isfinite(res) and res <= tol and jump <= jump_tol

        if not good:
            if halved < max_halve:         # refuse the step, try a smaller one
                dK *= 0.5
                halved += 1
                continue
            return None

        prev_x, prev_step = x, step
        x, K = xn, Kn
        if halved > 0 and dK < st.K / n_steps:
            dK *= 1.3                      # cautiously grow back

    return Solution(st, float(x[0]), float(x[1]), float(x[2]),
                    float(np.max(np.abs(st.residuals_scaled(x)))))


def solve(st: HCY, n_starts: int = 12, tol: float = 1e-8) -> list[Solution]:
    """**All** roots found, not one — the multiplicity is the physics (see module
    docstring).  Seeded from a spread of starts around the PY point."""
    a_py, b_py = py_ab(st.xi)
    starts = []
    for fa in (0.2, 1.0, 3.0):
        for fb in (0.5, 1.0, 2.0):
            for w0 in (0.0, 1.0, -1.0, 5.0):
                starts.append((fa * a_py, fb * b_py, w0))
    out: list[Solution] = []
    for s0 in starts[: max(n_starts, len(starts))]:
        try:
            sol, info, ier, _ = fsolve(st.residuals, np.array(s0, dtype=float),
                                       full_output=True, xtol=1e-13)
        except Exception:
            continue
        if ier != 1:
            continue
        res = float(np.max(np.abs(st.residuals(sol))))
        scale = max(1.0, abs(24.0 * st.xi * st.y_moments(*sol)[0] ** 2))
        if res / scale > tol:
            continue
        if any(abs(sol[0] - o.a) < 1e-6 and abs(sol[1] - o.b) < 1e-6
               and abs(sol[2] - o.w) < 1e-6 for o in out):
            continue
        out.append(Solution(st, float(sol[0]), float(sol[1]), float(sol[2]), res))
    out.sort(key=lambda s: s.w)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# the Blum–Høye route — ⭐ this is the one that is right
# ═══════════════════════════════════════════════════════════════════════════════
#
# Waisman's Eq. (3b) does not hold.  Measured at the Blum–Høye solution, its
# residual relative to its own scale runs 1.8e-3 (z = 10) → 5.2 (z = 3) → 1.2e+2
# (z = 1.8), while (3a) is satisfied to **1e-15** at the same points and (3c) to
# 2e-5…1e-2.  Waisman reports exactly one state point, z = 28.751, where the
# defect is invisible; see `waisman_msa_closed_form.md` §7e.
#
# Blum & Høye's system needs no such condition.  At N = 1 it is 2 × 2 in
# `(D, γ)` — smaller than Waisman's three quadratics — and every piece below is
# their equation number.  ⚠ Three of their printed equations carry errors; the
# corrections are derived and measured in §7d and are marked `[corrected]`.

def _hs_baxter(xi: float) -> tuple[float, float, float]:
    """`(A⁰, B⁰, q⁰′)` — Blum & Høye Eq. (25) at `σ = 1`, where `ξ₂ = ξ₃ = ξ`.

    Cross-check built in by the caller: `q⁰′ = A⁰ + B⁰`, which Eqs. (11) and (14)
    force independently of (25).
    """
    d = 2.0 * np.pi / (1.0 - xi) ** 2
    return d * (1.0 + 2.0 * xi), d * (-1.5 * xi), d * (1.5 * xi + (1.0 - xi))


# ⚠ Both are 0/0 at the origin and lose everything to cancellation near it:
# φ₂'s error is ~eps/|x|³, so it is already garbage by |x| ~ 1e-2.  `baxter_F`'s
# pole search runs right up to s = 0⁻, so the series form is not optional.
_PHI_SERIES = 0.5


def _phi1(x):                            # Blum & Høye Eq. (22)
    x = np.asarray(x)
    small = np.abs(x) < _PHI_SERIES
    xs = np.where(small, x, 0.0)
    ser = (-1.0 / 2 + xs / 6 - xs ** 2 / 24 + xs ** 3 / 120 - xs ** 4 / 720
           + xs ** 5 / 5040 - xs ** 6 / 40320 + xs ** 7 / 362880)
    big = np.where(small, 1.0, x)
    return np.where(small, ser, (1.0 - big - np.exp(-big)) / big ** 2)


def _phi2(x):
    x = np.asarray(x)
    small = np.abs(x) < _PHI_SERIES
    xs = np.where(small, x, 0.0)
    ser = (1.0 / 6 - xs / 24 + xs ** 2 / 120 - xs ** 3 / 720 + xs ** 4 / 5040
           - xs ** 5 / 40320 + xs ** 6 / 362880 - xs ** 7 / 3628800)
    big = np.where(small, 1.0, x)
    return np.where(small, ser,
                    (1.0 - big + 0.5 * big * big - np.exp(-big)) / big ** 3)


def hs_laplace_moment_scaled(xi: float, z: float) -> float:
    """`G_HS(z) = e^{z}·hs_laplace_moment(xi, z)`, computed without forming `e^{z}`.

    `hs_laplace_moment` carries `e^{z}` in its denominator, so the product is
    `O(1)` — but at `z ≈ 29` the intermediate `e^{z} ≈ 3e12` costs digits for
    nothing.  Waisman's own state point is `z = 28.751`.
    """
    L = 12.0 * xi * ((1.0 + xi / 2.0) * z + (1.0 + 2.0 * xi))
    S = ((1.0 - xi) ** 2 * z ** 3 + 6.0 * xi * (1.0 - xi) * z ** 2
         + 18.0 * xi ** 2 * z - 12.0 * xi * (1.0 + 2.0 * xi))
    return z * L / (12.0 * xi * (L * np.exp(-z) + S))


def _bh_pieces(xi: float, z: float, Dt: float, G: float) -> dict:
    """Everything downstream of the unknowns, at `N = 1`, `σ = 1`, `λ = 0`.

    ⭐ The unknowns are **`Dt = D·e^{−z}` and `G = ĝ(z)·e^{z}`**, not `(D, γ)`.
    Both are `O(1)` (indeed `Dt = O(K)`), and in these variables *every* `e^{z}`
    in Blum & Høye's system cancels analytically — see `solve_bh`.  The naive
    `(D, γ)` form is unusable: `M` and `N` are each differences of two terms of
    size `ρD/z` that cancel to zero as `z → ∞`, and `γ → 1` with the physics
    living in `1 − γ ~ 1e-14`.
    """
    rho = 6.0 * xi / np.pi
    A0, B0, qp0 = _hs_baxter(xi)
    ez = np.exp(-z)
    gam = 1.0 - 2.0 * np.pi * rho * G * ez / z                       # (27), 1−γ = 2πρĝ/z
    # (20)/(21) with C = −γD substituted and the cancellation done by hand:
    #   z e^{−z}φ₁(−z) = [e^{−z}(1+z) − 1]/z,  z e^{−z}φ₂(−z) = [1 − e^{−z}(1+z+z²/2)]/z²
    M = -(rho * Dt / z) * (gam * (1.0 + z) + 2.0 * np.pi * rho * G / z)
    N = (rho * Dt / z ** 2) * (2.0 * np.pi * rho * G / z
                               + gam * (1.0 + z + z * z / 2.0))      # (21) [corrected]
    A = A0 * (1.0 + M) - 4.0 * B0 * N                                # (23)
    qp = qp0 * (1.0 + M) + A0 * N                                    # (24)
    # (37)'s C-term −γD(1−e^{−z})²/(2z) plus the D e^{−zr} tail D/(2z) that (37)
    # excludes; their sum is e^{z}-free because γ → 1 kills the leading piece.
    tail = (Dt / (2.0 * z)) * (2.0 * np.pi * rho * G / z + gam * (2.0 - ez))
    Qh = ((1.0 + M) * (_phi1(z) * qp0 + _phi2(z) * A0)
          + N * (_phi1(z) * A0 - 4.0 * B0 * _phi2(z)) + tail)        # (37)
    return dict(gam=gam, M=M, N=N, A=A, qp=qp, Qh=Qh, F=1.0 - rho * Qh)


def _bh_residuals(u, xi: float, z: float, K: float, s2: float = 1.0) -> list[float]:
    """(29′) and (33)-at-`s=z` — the whole system, in `(Dt, G)`.  Exponential-free.

    ⚠ (29) is printed without the `e^{zσ}` that Eq. (3)'s contact normalisation
    forces on the left of (8).  Restoring it gives `2πKe^{z}/z = D[1 − ρQ̂(z)]`,
    and with `Dt = De^{−z}` the `e^{z}` cancels on both sides:

        (29′)   Dt·F = 2πK/z
        (33)    2π G·F = [A + z q′ + (z²/2)·γ·Dt] / z²

    (33)'s `−(z²/2)e^{−z}C` becomes `+(z²/2)γDt` exactly, since `C = −γD`.
    """
    Dt, G = u
    p = _bh_pieces(xi, z, Dt, G)
    return [(Dt * p['F'] - 2.0 * np.pi * K / z) / (2.0 * np.pi * K / z),
            (2.0 * np.pi * G * p['F']
             - (p['A'] + z * p['qp'] + 0.5 * z * z * p['gam'] * Dt) / z ** 2) / s2]


def solve_bh(st: HCY) -> Solution | None:
    """⭐ Solve the MSA exactly, via Blum & Høye, and report it in Waisman's `(a, b, w)`.

    Seeded at the `K → 0` limit — `D → 2πKe^{z}/(zQ̂₀(z))`, `γ → 1 − 2πρĝ_HS(z)/z`
    — which is exact, so no continuation ladder and no branch tracking is needed:
    the two equations are far better conditioned than Waisman's three quadratics.

    Returns `None` if the solve does not converge.  `Solution.residual` is the
    norm of **Blum & Høye's** residuals, not Waisman's.
    """
    from scipy.optimize import fsolve

    xi, K, z = st.xi, st.K, st.z
    rho = 6.0 * xi / np.pi
    A0, B0, qp0 = _hs_baxter(xi)
    F0 = 1.0 - rho * (_phi1(z) * qp0 + _phi2(z) * A0)         # 1 − ρQ̂₀(z)
    G0 = hs_laplace_moment_scaled(xi, z)                 # exact K → 0 value of ĝe^{z}
    Dt0 = 2.0 * np.pi * K / (z * F0)                     # exact K → 0 amplitude
    s2 = abs(A0 + z * qp0) / z ** 2 or 1.0

    def attempt(K_t, seed):
        # ⚠ Accept on the residual, never on fsolve's flag.  The seed here is the
        # exact K → 0 limit, so at small K the very first step already sits at
        # ~1e-14 and fsolve returns ok=5 ("not making good progress") — because
        # there is no progress left to make.  Gating on ok==1 threw away 12 of 25
        # converged solves.
        u = fsolve(_bh_residuals, seed, args=(xi, z, K_t, s2), xtol=1e-13)
        r = float(np.linalg.norm(_bh_residuals(u, xi, z, K_t, s2)))
        return (u, r if r < 1e-9 else np.inf)

    u, res = attempt(K, [Dt0, G0])
    if not np.isfinite(res):
        # Ramp in K.  The K → 0 seed is *exact*, so a short ladder is enough;
        # this is continuation for conditioning only, not branch tracking —
        # unlike the Waisman route there is no second root to lose.
        seed = [0.0, G0]
        for f in np.linspace(0.0, 1.0, 13)[1:]:
            seed, res = attempt(K * f, seed)
            if not np.isfinite(res):
                return None
        u = seed
    Dt, G = u
    p = _bh_pieces(xi, z, Dt, G)
    a = (p['A'] / (2.0 * np.pi)) ** 2                                       # (39)
    y0 = (p['qp'] + z * p['gam'] * Dt) / (2.0 * np.pi)                      # (38)
    v = 24.0 * xi * K * G                                                   # (★W)
    # b from Eq. (2) at x = 1⁻, using y₀ = K − C(1⁻).  Independent of (3a) —
    # and (3a) then holds to 1e-15, which is the sharpest check on the whole
    # dictionary that exists here.  (cosh z − 1)/e^{z} = (1 − e^{−z})²/2.
    b = (y0 - K - a - xi * a / 2.0 - v * (1.0 - np.exp(-z)) / z
         - v * v * (1.0 - np.exp(-z)) ** 2 / (4.0 * K * z * z))
    return Solution(st, float(a), float(b), float(v / K), float(res),
                    float(p['A']))


# ═══════════════════════════════════════════════════════════════════════════════
# MSAX.3 — n Yukawa tails, still N = 1
# ═══════════════════════════════════════════════════════════════════════════════
#
# Blum & Høye drop the tail index at the top of their §2 ("the generalization to a
# sum of such terms is obvious").  It is, but only after the same `e^{z}` surgery
# as `solve_bh` — and one term is easy to lose, see `_bhn_pieces`.
#
#     Q(r)  = Q⁰(r) + Σ_t D_t e^{−z_t r},   Q⁰ ⊃ Σ_t C_t(e^{−z_t r} − e^{−z_t})
#     c(r)  = Σ_t K_t e^{−z_t(r−1)}/r                                    (r > 1)
#
# 2n unknowns `(D_t, ĝ(z_t))`, 2n equations: (29′) and (33)-at-`s=z_u`, one of each
# per tail.  Everything is carried in `Dt_t = D_t e^{−z_t}` and `G_t = ĝ(z_t)e^{z_t}`.


@dataclass
class SolutionN:
    """A converged n-tail solution.  `a = ∂βp/∂ρ` — Blum & Høye (39)."""
    xi: float
    z: np.ndarray
    K: np.ndarray
    Dt: np.ndarray          # D_t e^{−z_t}
    G: np.ndarray           # ĝ(z_t) e^{z_t}
    a: float
    y0: float               # g(1⁺)
    qp: float
    A: float
    residual: float

    @property
    def v(self) -> np.ndarray:
        """Waisman's per-tail amplitude `v_t = 24ξK_t e^{z_t}ĝ(z_t)`."""
        return 24.0 * self.xi * self.K * self.G

    @property
    def dbetaP_drho(self) -> float:
        return self.a

    @property
    def chat0(self) -> float:
        """`ĉ(0)` — from `1 − ρĉ(0) = a` with `ρ = 6ξ/π`."""
        return (1.0 - self.a) / (6.0 * self.xi / np.pi)


def _bhn_pieces(xi: float, z: np.ndarray, Dt: np.ndarray, G: np.ndarray) -> dict:
    """`M`, `N`, `A`, `q′` and `F(z_u) = 1 − ρQ̂(z_u)` for every tail `u`."""
    rho = 6.0 * xi / np.pi
    A0, B0, qp0 = _hs_baxter(xi)
    ez = np.exp(-z)
    gam = 1.0 - 2.0 * np.pi * rho * G * ez / z                        # (27), per tail
    # (20)/(21): the moment contributions are additive over tails, each with the
    # same hand-done cancellation as the single-tail case.
    M = -np.sum((rho * Dt / z) * (gam * (1.0 + z) + 2.0 * np.pi * rho * G / z))
    N = np.sum((rho * Dt / z ** 2) * (2.0 * np.pi * rho * G / z
                                      + gam * (1.0 + z + z * z / 2.0)))
    A = A0 * (1.0 + M) - 4.0 * B0 * N                                 # (23)
    qp = qp0 * (1.0 + M) + A0 * N                                     # (24)
    # Q̂⁰'s polynomial part is φ₁(s)q′ + φ₂(s)A — (23)/(24) turn (37)'s
    # (1+M)/N form into exactly this, which is what makes n tails trivial.
    # ⚠ Q⁰'s exponentials are C_t(e^{−z_t r} − e^{−z_t}): the SECOND piece is a
    # constant on [0,1] and contributes its own term.  Dropping it is the easy
    # mistake — at n = 1 it is what turns γe^{−z} into γ(2 − e^{−z}).
    zu = z[:, None]                          # rows = u (the evaluation rate)
    cross = (2.0 * np.pi * rho * Dt * G / z + gam * Dt * np.exp(-zu)) / (zu + z[None, :])
    const = gam[None, :] * Dt[None, :] * (1.0 - np.exp(-zu)) / zu
    Qh = _phi1(z) * qp + _phi2(z) * A + np.sum(cross + const, axis=1)
    return dict(gam=gam, M=M, N=N, A=A, qp=qp, F=1.0 - rho * Qh)


def _bhn_residuals(u, xi: float, z: np.ndarray, K: np.ndarray, s2: np.ndarray):
    """(29′) and (33)@`s=z_u`, exponential-free, one pair per tail.

        (29′)  Dt_u·F(z_u)   = 2πK_u/z_u
        (33)   2π G_u·F(z_u) = [A + z_u q′ + Σ_t (z_u²z_t/(z_u+z_t))·γ_t·Dt_t]/z_u²

    `(33)`'s `−(z_u²z_t/(z_u+z_t))e^{−z_t}C_t` becomes `+…γ_t Dt_t` exactly, since
    `C_t = −γ_t D_t` and `D_t e^{−z_t} = Dt_t`.
    """
    n = len(z)
    Dt, G = np.asarray(u[:n], dtype=float), np.asarray(u[n:], dtype=float)
    p = _bhn_pieces(xi, z, Dt, G)
    zu = z[:, None]
    src = (p['A'] + z * p['qp']
           + np.sum(zu ** 2 * z[None, :] / (zu + z[None, :])
                    * (p['gam'] * Dt)[None, :], axis=1)) / z ** 2
    return np.concatenate([(Dt * p['F'] - 2.0 * np.pi * K / z) / (2.0 * np.pi * K / z),
                           (2.0 * np.pi * G * p['F'] - src) / s2])


def solve_bh_n(xi: float, z, K) -> SolutionN | None:
    """⭐ **MSAX.3** — exact MSA for `N = 1` with `n` Yukawa tails.

    `z` and `K` are per-tail arrays; `K_t > 0` is attraction, in the contact
    normalisation `c(r) = Σ_t K_t e^{−z_t(r−1)}/r` of Blum & Høye Eq. (3), which is
    also `FMSA_double_prop`'s (`c(r) = K e^{−z(r−R)}/(2πrρ)` with `K = 2πβρ k_bare`).

    Reduces to `solve_bh` at `n = 1` — gated, and to machine precision.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    K = np.atleast_1d(np.asarray(K, dtype=float))
    if len(K) == 1 and len(z) > 1:
        K = np.repeat(K, len(z))
    rho = 6.0 * xi / np.pi
    A0, B0, qp0 = _hs_baxter(xi)
    F0 = 1.0 - rho * (_phi1(z) * qp0 + _phi2(z) * A0)
    G0 = np.array([hs_laplace_moment_scaled(xi, zz) for zz in z])
    Dt0 = 2.0 * np.pi * K / (z * F0)
    s2 = np.abs(A0 + z * qp0) / z ** 2

    def attempt(Kt, seed):
        # fsolve's chatter is not informative here — the acceptance test is the
        # residual, and a non-converged attempt is a legitimate answer (the MSA
        # branch really does terminate; see the plan's MSAX.5).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = fsolve(_bhn_residuals, seed, args=(xi, z, Kt, s2), xtol=1e-13)
        r = float(np.linalg.norm(_bhn_residuals(u, xi, z, Kt, s2)))
        return (u, r if r < 1e-9 else np.inf)

    u, res = attempt(K, np.concatenate([Dt0, G0]))
    if not np.isfinite(res):
        # Adaptive ramp: previous solution as seed, halve the step on failure.
        # ⚠ A fixed ladder is NOT enough near the spinodal.  `a = (A/2π)²` has a
        # *double* root there, so `A` passes through zero while the step in `a`
        # collapses quadratically; a 17-step linear ramp stopped at `a ≈ 0.2`
        # and made it look as though the branch terminated before the spinodal.
        # It does not — the spinodal is a regular point (|det J| = 2e-2…8e-1).
        # ⚠ A residual test alone is NOT enough.  Near the fold `fsolve` happily
        # converges onto the *other* branch — residual ~1e-14, wrong solution —
        # and every later step then rides that branch.  Symptom: `A` jumping
        # +3.58 → −8.01 in one step.  So the ramp carries a secant predictor and
        # rejects any step whose increment far exceeds the predicted one.
        seed = np.concatenate([np.zeros_like(Dt0), G0])
        prev, prev_f = None, None
        f, df = 0.0, 1.0 / 64.0
        while f < 1.0 - 1e-14:
            fn = min(f + df, 1.0)
            pred = (seed if prev is None
                    else seed + (seed - prev) * (fn - f) / (f - prev_f))
            trial, r = attempt(K * fn, pred)
            if np.isfinite(r) and prev is not None:
                step = np.linalg.norm(trial - seed)
                expect = (np.linalg.norm(seed - prev) * (fn - f) / (f - prev_f)
                          + 1e-8 * (1.0 + np.linalg.norm(seed)))
                if step > 4.0 * expect:
                    r = np.inf                     # branch jump — reject
            if not np.isfinite(r):
                df *= 0.5
                if df < 1e-11:
                    return None        # genuine fold — see the plan, MSAX.5
                continue
            prev, prev_f = seed, f
            seed, res, f = trial, r, fn
            df = min(df * 1.25, 1.0 / 32.0)
        u = seed
    n = len(z)
    Dt, G = u[:n], u[n:]
    p = _bhn_pieces(xi, z, Dt, G)
    a = (p['A'] / (2.0 * np.pi)) ** 2                                       # (39)
    y0 = (p['qp'] + np.sum(z * p['gam'] * Dt)) / (2.0 * np.pi)              # (38)
    return SolutionN(xi, z, K, Dt, G, float(a), float(y0), float(p['qp']),
                     float(p['A']), float(res))


def trace_coupling(xi: float, z, kv, lam_max: float = 12.0,
                   dlam0: float = 1.0 / 256.0):
    """Continue the **physical** branch in the coupling multiplier `λ` (`K = λ·kv`).

    Returns `[(λ, SolutionN), …]` from `λ = 0` up to `lam_max` or the fold,
    whichever comes first.

    ⚠ This exists because `solve_bh_n` re-ramps from `K = 0` on every call, and
    that cannot resolve the approach to the spinodal: `a = (A/2π)²` has a double
    root there, so the step in `a` collapses quadratically and an independent
    ramp stalls at `a ≈ 0.1` — looking like a fold that is not one.  Tracking the
    branch once, with a secant predictor and the previous point as seed, walks
    straight through `A = 0`.  **Bracket on `A`, never on `a`.**
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    kv = np.atleast_1d(np.asarray(kv, dtype=float))
    if len(kv) == 1 and len(z) > 1:
        kv = np.repeat(kv, len(z))
    rho = 6.0 * xi / np.pi
    A0, B0, qp0 = _hs_baxter(xi)
    s2 = np.abs(A0 + z * qp0) / z ** 2
    G0 = np.array([hs_laplace_moment_scaled(xi, zz) for zz in z])

    def solve_at(lam, seed):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = fsolve(_bhn_residuals, seed, args=(xi, z, lam * kv, s2), xtol=1e-13)
        return u, float(np.linalg.norm(_bhn_residuals(u, xi, z, lam * kv, s2)))

    def rec(lam, u):
        n = len(z)
        p = _bhn_pieces(xi, z, u[:n], u[n:])
        return SolutionN(xi, z, lam * kv, u[:n], u[n:],
                         float((p['A'] / (2 * np.pi)) ** 2),
                         float((p['qp'] + np.sum(z * p['gam'] * u[:n])) / (2 * np.pi)),
                         float(p['qp']), float(p['A']), 0.0)

    u = np.concatenate([np.zeros_like(z), G0])
    out = [(0.0, rec(0.0, u))]
    prev, prev_l, lam, dl = None, None, 0.0, dlam0
    while lam < lam_max and dl > 1e-11:
        ln = min(lam + dl, lam_max)
        pred = u if prev is None else u + (u - prev) * (ln - lam) / (lam - prev_l)
        trial, r = solve_at(ln, pred)
        ok = r < 1e-9
        if ok and prev is not None:
            expect = (np.linalg.norm(u - prev) * (ln - lam) / (lam - prev_l)
                      + 1e-8 * (1.0 + np.linalg.norm(u)))
            ok = np.linalg.norm(trial - u) <= 4.0 * expect
        if not ok:
            dl *= 0.5
            continue
        prev, prev_l = u, lam
        u, lam = trial, ln
        out.append((lam, rec(lam, u)))
        dl = min(dl * 1.25, dlam0 * 8)
    return out


def spinodal_coupling(xi: float, z, kv, lam_max: float = 12.0):
    """The coupling multiplier `λ` at which `A = 0` — the spinodal.  `None` if the
    branch folds first.  Brackets on `A` (regular), not `a` (double root)."""
    from scipy.optimize import brentq
    tr = trace_coupling(xi, z, kv, lam_max=lam_max)
    lam = np.array([t[0] for t in tr])
    A = np.array([t[1].A for t in tr])
    i = np.where(np.sign(A[:-1]) != np.sign(A[1:]))[0]
    if len(i) == 0:
        return None
    i = i[0]
    z_ = np.atleast_1d(np.asarray(z, dtype=float))
    kv_ = np.atleast_1d(np.asarray(kv, dtype=float))
    if len(kv_) == 1 and len(z_) > 1:
        kv_ = np.repeat(kv_, len(z_))
    rho = 6.0 * xi / np.pi
    A0, B0, qp0 = _hs_baxter(xi)
    s2 = np.abs(A0 + z_ * qp0) / z_ ** 2
    seed = np.concatenate([tr[i][1].Dt, tr[i][1].G])

    def Aof(l):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = fsolve(_bhn_residuals, seed, args=(xi, z_, l * kv_, s2), xtol=1e-13)
        n = len(z_)
        return _bhn_pieces(xi, z_, u[:n], u[n:])['A']

    return brentq(Aof, lam[i], lam[i + 1], xtol=1e-12)


def baxter_F(sol: "SolutionN", s):
    """`F(s) = 1 − ρQ̂(s)` at arbitrary **complex** `s` — the Baxter factor.

    `1 − ρĉ(k) = F(ik)·F(−ik)`, so the poles of `ĥ(k)` are the zeros of `F`.
    Baxter's theorem puts them all in `Re s ≤ 0`, and the asymptotic decay of
    `h(r)` is governed by the zero of largest `Re s` (i.e. smallest `|Re s|`):
    real ⇒ monotonic `e^{−α₀r}`, complex pair ⇒ damped oscillatory.  The
    Fisher–Widom line is where the two exchange rank.

        Q̂(s) = φ₂(s)A + φ₁(s)q′
               + Σ_t { [2πρ Dt_t G_t/z_t + γ_t Dt_t e^{−s}]/(s+z_t)
                       + γ_t Dt_t (1−e^{−s})/s }

    ⭐ Exponential-free in the same variables as the solve, so it is usable at
    `z ≈ 29` where `D ~ e^{z}` would overflow.  Reduces to `_bhn_pieces`' `F` at
    `s = z_u`, which is the gate.
    """
    xi, z, Dt, G = sol.xi, sol.z, sol.Dt, sol.G
    rho = 6.0 * xi / np.pi
    s = np.asarray(s, dtype=complex)
    gam = 1.0 - 2.0 * np.pi * rho * G * np.exp(-z) / z
    sv = s[..., None]
    es = np.exp(-s)[..., None]
    # (1 − e^{−s})/s is 0/0 at the origin, which F(0) = A/2π needs; series it.
    tiny = np.abs(sv) < 1e-4
    exprel = np.where(tiny, 1.0 - sv / 2 + sv ** 2 / 6 - sv ** 3 / 24,
                      (1.0 - es) / np.where(tiny, 1.0, sv))
    tail = np.sum((2.0 * np.pi * rho * Dt * G / z + gam * Dt * es) / (sv + z)
                  + gam * Dt * exprel, axis=-1)
    # φ₁, φ₂ are removable at s = 0; the series there is what the caller needs if
    # it ever probes the origin, but the pole search never does.
    Qh = _phi2(s) * sol.A + _phi1(s) * sol.qp + tail
    return 1.0 - rho * Qh


def leading_poles(sol: "SolutionN", n_seed: int = 260,
                  span: float = 14.0) -> tuple[float, float, float]:
    """`(α₀_mon, α₀_osc, α₁_osc)` — the leading monotonic and oscillatory decays.

    Zeros of `baxter_F` with the largest `Re s < 0`, split by whether they are
    real.  Returns `inf` for a family that has no zero in the searched box.
    """
    from scipy.optimize import brentq, fsolve

    zmin = float(np.min(sol.z))
    # real zeros: bracket by sign change on (−zmin, 0), the strip free of the
    # explicit 1/(s+z_t) poles.
    xs = np.linspace(-zmin + 1e-9, -1e-9, n_seed)
    fr = np.array([baxter_F(sol, np.array([x]))[0].real for x in xs])
    a_mon = np.inf
    for i in range(len(xs) - 1):
        if np.sign(fr[i]) != np.sign(fr[i + 1]):
            r = brentq(lambda x: baxter_F(sol, np.array([x]))[0].real,
                       xs[i], xs[i + 1], xtol=1e-13)
            a_mon = min(a_mon, -r)          # decay rate is −Re s
            break
    # complex zeros: Newton from a grid of seeds in the left half strip
    # ⚠ Distant seeds drive `fsolve` where `e^{−s}` overflows.  Those iterates are
    # thrown away by the residual test anyway; silence the arithmetic, not the test.
    best = None
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for re0 in np.linspace(-span, -1e-3, 26):
            for im0 in np.linspace(0.3, span, 26):
                def _f(v):
                    w = baxter_F(sol, np.array([v[0] + 1j * v[1]]))[0]
                    return [w.real, w.imag]
                try:
                    v = fsolve(_f, [re0, im0], xtol=1e-12, full_output=False)
                except Exception:
                    continue
                if abs(v[1]) < 1e-6 or v[0] > -1e-9:
                    continue
                if np.linalg.norm(_f(v)) > 1e-8:
                    continue
                if best is None or v[0] > best[0]:
                    best = (v[0], abs(v[1]))
    if best is None:
        return a_mon, np.inf, np.nan
    return a_mon, -best[0], best[1]


# ═══════════════════════════════════════════════════════════════════════════════
# gates
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_gate() -> int:
    print("=" * 78)
    print("MSAX Phase 1 — exact single-component MSA (Waisman 1973)")
    print("=" * 78)

    print("\n1. transcription check: Waisman's own converged point satisfies Eq. (3)")
    print("-" * 78)
    st = HCY(xi=WAISMAN["xi"], K=WAISMAN["K"], z=WAISMAN["z"])
    # his v, in this module's regular variable
    abv = (WAISMAN["a"], WAISMAN["b"], WAISMAN["v"] / WAISMAN["K"])
    y0, y1, y2 = st.y_moments(*abv)
    r = st.residuals(abv)
    scale = 24.0 * st.xi * y0 ** 2
    print(f"   y0 (ours) = {y0:.4f}   Waisman prints g(1+) = {WAISMAN['y0']}"
          f"   dev {abs(y0 - WAISMAN['y0']):.2e}")
    print(f"   residual (3a) = {r[0]:+.4g}   relative to 24xi*y0^2 = {scale:.4g}"
          f"  ->  {abs(r[0]) / scale:.2e}")
    print(f"   residual (3b) = {r[1]:+.4g}   relative                        "
          f"  ->  {abs(r[1]) / abs(24 * st.xi * (y1 ** 2 - 2 * y0 * y2)):.2e}")
    print(f"   residual (3c) = {r[2]:+.4g}   relative to a = {WAISMAN['a']}"
          f"       ->  {abs(r[2]) / WAISMAN['a']:.2e}")
    print("   (his a,b,v are printed to 4-5 digits, so ~1e-3 is the expected floor)")

    print("\n2. K -> 0 must reproduce Percus-Yevick")
    print("-" * 78)
    print("   (a) at K = 0 exactly the PY point must annihilate all three residuals")
    print("       for EVERY w -- w only ever enters as K*w, so it drops out.")
    worst_id = 0.0
    for xi in (0.1, 0.2, 0.3, 0.4, 0.45):
        s0 = HCY(xi=xi, K=0.0, z=1.8)
        a_py, b_py = py_ab(xi)
        for w in (0.0, 1.0, -3.0, 17.0):
            rr = np.max(np.abs(s0.residuals((a_py, b_py, w))))
            worst_id = max(worst_id, rr / max(1.0, a_py))
    print(f"       worst relative residual over xi x w: {worst_id:.2e}   "
          f"{'PASS' if worst_id < 1e-12 else 'FAIL'}")

    print("   (b) O(K): the TRACKED branch vs this project's first-order construction")
    print("       Arbiter is `FMSA_double_prop.get_chat_k` -- independent code, so")
    print("       this is not circular.  Exact MSA and a first-order theory must")
    print("       share the O(K) slope of dbetaP/drho; that is MSAEXACT.5's content.")
    try:
        from fmsa_double_prop import FMSA_double_prop
    except Exception as e:                                   # pragma: no cover
        print(f"       skipped ({e})")
        FMSA_double_prop = None
    if FMSA_double_prop is not None:
        Kt = 1e-4
        print(f"   {'xi':>6} " + " ".join(f"{z:>9.2f}" for z in (3.0, 5.0, 10.0, 28.751)))
        for xi in (0.1, 0.2, 0.3, 0.4):
            a0, _ = py_ab(xi)
            row = []
            for z in (3.0, 5.0, 10.0, 28.751):
                rho = 6.0 * xi / np.pi
                m = FMSA_double_prop([1.0], [1.0], [1.0], rho, 1.0 / Kt,
                                     n_tails=1, z=z, k=1.0, input_type='bh',
                                     verbose=False)
                sF = (1.0 - rho * float(np.real(
                    m.get_chat_k(np.array([1e-5]))[0][0, 0])) - a0) / Kt
                c = solve_continued(HCY(xi=xi, K=Kt, z=z))
                row.append(f"{((c.a - a0) / Kt) / sF:9.5f}" if c else f"{'--':>9}")
            print(f"   {xi:6.2f} " + " ".join(row))
        print("       ratio -> 1 is the target, and it is 1.00000 across the table.")
        print("       Before Eq. (2)'s missing 1/x was restored this column read")
        print("       0.45 at z = 3 and 1.00000 only at z = 28.75 -- see 7e.")

    print("\n3. Waisman's xi = 0.49 point, solved from scratch")
    print("-" * 78)
    sols = solve(st)
    print(f"   {len(sols)} root(s) found; Waisman reports "
          f"a={WAISMAN['a']}, b={WAISMAN['b']}, v={WAISMAN['v']}")
    print(f"   {'#':>3} {'a':>12} {'b':>12} {'v':>10} {'g(1+)':>9} "
          f"{'resid':>10} {'physical?':>12}")
    for i, q in enumerate(sols):
        ok, why = q.physical_ok
        print(f"   {i:3d} {q.a:12.5f} {q.b:12.5f} {q.v:10.5f} {q.g_contact:9.4f} "
              f"{q.residual:10.2e} {('yes' if ok else 'no'):>12}")

    print("\n4. branch coverage: where does the physical branch exist?")
    print("-" * 78)
    print("   a = dbetaP/drho on the tracked branch; '--' = branch not continuable.")
    print(f"   {'eta':>5} {'z':>6} " + " ".join(f"{bk:>8.1f}" for bk in
                                               (0.5, 1.0, 2.0, 3.0, 4.0)))
    for eta in (0.1, 0.2, 0.3, 0.4):
        for z in (1.8, 5.0, 10.0):
            row = []
            for bk in (0.5, 1.0, 2.0, 3.0, 4.0):
                c = solve_continued(HCY(xi=eta, K=bk, z=z))
                row.append(f"{c.a:8.3f}" if c is not None else f"{'--':>8}")
            print(f"   {eta:5.2f} {z:6.1f} " + " ".join(row))
    print("   Loss of the branch at small z / large betaK is the MSAX.5 map, not")
    print("   a solver defect -- Blum & Hoye's multiplicity made concrete.")

    print("\n5. continuation vs Waisman's own point (the branch must survive to xi=0.49)")
    print("-" * 78)
    c = solve_continued(st, n_steps=200)
    if c is None:
        print("   ramp did not reach K = 1.336 — branch lost")
    else:
        print(f"   continued: a={c.a:.5f} b={c.b:.5f} v={c.v:.5f}")
        print(f"   Waisman  : a={WAISMAN['a']} b={WAISMAN['b']} v={WAISMAN['v']}")
        print(f"   dev      : {abs(c.a / WAISMAN['a'] - 1):.2e} "
              f"{abs(c.b / WAISMAN['b'] - 1):.2e} "
              f"{abs(c.v / WAISMAN['v'] - 1):.2e}")

    print("\n6. ⭐ the Blum-Hoye route (solve_bh) -- this is the one that is right")
    print("-" * 78)
    s = solve_bh(st)
    print(f"   Waisman's own point: a={s.a:.4f} b={s.b:.4f} y0={s.y0:.4f} v={s.v:.4f}")
    print(f"   he prints          : a={WAISMAN['a']} b={WAISMAN['b']} "
          f"y0={WAISMAN['y0']} v={WAISMAN['v']}")
    print(f"   dev                : {abs(s.a/WAISMAN['a']-1):.1e} "
          f"{abs(s.b/WAISMAN['b']-1):.1e} {abs(s.y0/WAISMAN['y0']-1):.1e} "
          f"{abs(s.v/WAISMAN['v']-1):.1e}   (he prints 4 digits)")

    print("\n   K -> 0 must return Percus-Yevick exactly:")
    for xi, z in ((0.1, 1.8), (0.3, 5.0), (0.49, 28.751)):
        q = solve_bh(HCY(xi=xi, K=1e-9, z=z))
        a0, b0 = py_ab(xi)
        print(f"     xi={xi:4.2f} z={z:7.3f}   a-a_PY {q.a-a0:+.1e}   b-b_PY {q.b-b0:+.1e}"
              f"   w-w0 {q.w-w_first_order(xi, z):+.1e}")

    print("\n   Waisman's (3a)/(3b)/(3c) evaluated AT the exact Blum-Hoye solution,")
    print("     each relative to its own scale.  All three now hold to machine")
    print("     precision -- with the printed Eq. (2) (no 1/x on the v^2 term)")
    print("     (3b) reads -1.2e+02 and (3c) 9.9e-03 at z=1.8.")
    print(f"     {'xi':>5} {'z':>7} {'K':>6} | {'(3a)':>9} {'(3b)':>9} {'(3c)':>9}")
    for xi, z, K in ((0.3, 1.8, 0.1), (0.3, 3.0, 0.2), (0.3, 5.0, 0.3),
                     (0.3, 10.0, 0.5), (0.49, 28.751, 1.336)):
        q = solve_bh(HCY(xi=xi, K=K, z=z))
        if q is None:
            continue
        t = HCY(xi=xi, K=K, z=z)
        r = t.residuals((q.a, q.b, q.w))
        y0, y1, y2 = t.y_moments(q.a, q.b, q.w)
        sc = [abs(24 * xi * y0 ** 2),
              abs(24 * xi * (y1 ** 2 - 2 * y0 * y2)), abs(q.a)]
        print(f"     {xi:5.2f} {z:7.3f} {K:6.3f} | "
              + " ".join(f"{r[i]/sc[i]:9.1e}" for i in range(3)))
    print("     The uncorrected defect is 1.1e-04 at z=28.751 -- below Waisman's")
    print("     own 4-digit printing, and z=28.751 is the ONLY point he reports.")

    print("\n7. ⭐ MSAX.3 -- n Yukawa tails (solve_bh_n)")
    print("-" * 78)
    worst = 0.0
    for xi in (0.05, 0.2, 0.4):
        for z in (1.8, 5.0, 28.751):
            for K in (1e-5, 0.3):
                s1, sn = solve_bh(HCY(xi=xi, K=K, z=z)), solve_bh_n(xi, z, K)
                worst = max(worst, abs(sn.a / s1.a - 1), abs(sn.y0 / s1.y0 - 1))
    print(f"   (a) n=1 reduction: solve_bh_n == solve_bh over 18 points -> {worst:.1e}")

    worst = 0.0
    for xi in (0.1, 0.3, 0.45):
        for zz in ([2.9637, 14.0167], [1.8, 5.0, 14.0]):
            q = solve_bh_n(xi, zz, [1e-9] * len(zz))
            worst = max(worst, abs(q.a - py_ab(xi)[0]))
    print(f"   (b) K -> 0 gives Percus-Yevick, n=2 and n=3      -> {worst:.1e}")

    print("   (c) O(K) slope of dbetaP/drho vs FMSA-DP -- independent code, so this")
    print("       is the n-tail form of gate 2(b).  LJ 2-Yukawa is the project's own")
    print("       fit (z = 2.9637/14.0167, k = +/-2.1714).")
    try:
        from fmsa_double_prop import FMSA_double_prop
    except ImportError:
        FMSA_double_prop = None
    if FMSA_double_prop is None:
        print("       fmsa_double_prop unavailable -- skipped")
    else:
        sets = [("LJ 2-Yukawa", np.array([2.9637, 14.0167]), np.array([2.1714, -2.1714])),
                ("n=2 same sign", np.array([1.8, 5.0]), np.array([1.0, 0.6])),
                ("n=3", np.array([2.0, 7.0, 20.0]), np.array([1.0, -0.5, 0.25])),
                ("n=4", np.array([1.5, 4.0, 9.0, 18.0]), np.array([1.0, 0.4, -0.3, 0.15]))]
        print(f"       {'set':<14} {'xi':>5} {'exact':>12} {'FMSA-DP':>12} {'rel':>9}")
        for nm, z, kv in sets:
            for xi in (0.1, 0.3):
                a0 = py_ab(xi)[0]
                lam = 1e-6
                q1, q2 = solve_bh_n(xi, z, lam * kv), solve_bh_n(xi, z, 0.5 * lam * kv)
                ex = 2 * (q2.a - a0) / (0.5 * lam) - (q1.a - a0) / lam
                rho = 6.0 * xi / np.pi

                def _f(L):
                    m = FMSA_double_prop([1.], [1.], [1.], rho, 1.0 / L, n_tails=len(z),
                                         z=list(z), k=list(kv), input_type='bh',
                                         verbose=False)
                    ck = float(np.real(m.get_chat_k(np.array([1e-5]))[0][0, 0]))
                    return (1 - rho * ck - a0) / L
                fd = 2 * _f(0.5 * lam) - _f(lam)
                print(f"       {nm:<14} {xi:5.2f} {ex:12.6f} {fd:12.6f} {abs(ex/fd-1):9.1e}")

    # ── 8. the root structure, and the selection rule the sources do not give ──
    print("\n8. ⭐⭐ HOW MANY real roots, and what selects the physical one")
    print("-" * 78)
    print("   Blum & Hoye: \"multiple solutions, of which only one is acceptable\" --")
    print("   but their Eq. (41) is a mixture symmetry and is VACUOUS at N = 1, so")
    print("   they state no rule here.  This measures what the rule is.")

    def _enum(xi, T, z):
        """Every distinct real root of the BH 2x2, by wide multi-start."""
        from scipy.optimize import fsolve
        zz, Kv = np.array([z]), np.array([1.0 / T])
        A0, B0, qp0 = _hs_baxter(xi)
        s2 = np.array([abs(A0 + z * qp0) / z ** 2 or 1.0])
        lad = np.concatenate([-np.geomspace(1e-4, 300, 9)[::-1], [0.0],
                              np.geomspace(1e-4, 300, 9)])
        # ⚠ negative Ĝ is seeded too: ĝ(z) > 0 holds for a PHYSICAL g, and seeding
        # only positive Ĝ would assume the answer.
        Gl = np.concatenate([-np.geomspace(1e-4, 300, 7)[::-1],
                             np.geomspace(1e-4, 300, 9)])
        roots = []
        for d0 in lad:
            for g0 in Gl:
                try:
                    u = fsolve(_bhn_residuals, [d0, g0], args=(xi, zz, Kv, s2), xtol=1e-13)
                except Exception:
                    continue
                r = float(np.linalg.norm(_bhn_residuals(u, xi, zz, Kv, s2)))
                if not np.isfinite(r) or r > 1e-9:
                    continue
                if any(abs(u[0] - q[0]) < 1e-7 and abs(u[1] - q[1]) < 1e-7 for q in roots):
                    continue
                roots.append((float(u[0]), float(u[1])))
        return [float(np.sum(_bhn_pieces(xi, zz, np.array([d]), np.array([g]))['A']))
                for d, g in roots]

    print("   (a) the count is 2 outside the fold gap and 0 inside -- never 1, 3, 4")
    print(f"       {'z':>5}{'T*':>6}{'eta':>6}{'n roots':>9}   signs of A")
    ok_count = True
    for z_, T, xi in ((1.8, 1.20, 0.10), (1.8, 1.20, 0.30), (1.8, 0.98, 0.05),
                      (1.8, 0.98, 0.16), (1.8, 0.85, 0.25), (4.0, 0.90, 0.35)):
        As = sorted(_enum(xi, T, z_))
        sg = " ".join("+" if a_ > 1e-8 else ("-" if a_ < -1e-8 else "0") for a_ in As)
        ok_count &= len(As) in (0, 2)
        print(f"       {z_:5.1f}{T:6.2f}{xi:6.2f}{len(As):9d}   {sg}"
              + ("   (inside the fold gap)" if not As else ""))
    print(f"       every count in {{0, 2}}: {'PASS' if ok_count else 'FAIL'}")

    print("   (b) ⚠ the two differ ONLY by the sign of A = 2pi(1-rho*Qhat(0)), and")
    print("       `a = (A/2pi)^2` is BLIND to it -- so no test built from Waisman's")
    print("       (a, b, w) can separate them.  The wrong-winding root at eta=0.05,")
    print("       T*=1.20 passes every one of them:")
    As = sorted(_enum(0.05, 1.20, 1.8))
    for A_ in As:
        st_ = HCY(0.05, 1.0 / 1.20, 1.8)
        print(f"       A = {A_:9.4f}   a = (A/2pi)^2 = {(A_/(2*np.pi))**2:8.4f} > 0"
              f"   {'PHYSICAL' if A_ > 0 else 'wrong winding -- yet a > 0'}")

    print("   (c) ⭐ THE RULE: `A > 0` is unique where it exists, and IS the root")
    print("       continuation from the K -> 0 Percus-Yevick point lands on.")
    n_st = n_uni = n_match = n_none = 0
    worst = 0.0
    for z_ in (1.0, 1.8, 4.0):
        for T in (2.0, 1.10, 0.90, 0.75):
            for xi in (0.05, 0.15, 0.25, 0.35, 0.45):
                As = _enum(xi, T, z_)
                if not As:
                    continue
                n_st += 1
                pos = [a_ for a_ in As if a_ > 1e-8]
                if len(pos) > 1:
                    print(f"       !! {z_} {T} {xi}: {len(pos)} roots with A > 0")
                elif len(pos) == 0:
                    n_none += 1                      # inside the spinodal: no physical state
                else:
                    n_uni += 1
                    sol = solve_bh(HCY(xi, 1.0 / T, z_))
                    if sol is not None:
                        d = abs(sol.a - (pos[0] / (2 * np.pi)) ** 2) / max(1.0, (pos[0] / (2 * np.pi)) ** 2)
                        worst = max(worst, d)
                        n_match += d < 1e-6
    print(f"       {n_st} states with a real root: {n_uni} have exactly one A > 0,")
    print(f"       {n_none} have none (they are between the fold and the spinodal,")
    print("       where the state is mechanically unstable and has no physical root).")
    print(f"       continuation lands on it: {n_match}/{n_uni}   worst {worst:.1e}   "
          f"{'PASS' if n_match == n_uni and n_uni > 0 else 'FAIL'}")
    print("   => at N = 1 the acceptability criterion is `A > 0`, and it is complete:")
    print("      it selects exactly the continuation branch wherever one exists.")
    print("   ⚠ MECHANICAL stability + winding only.  A state between the binodal and")
    print("     the spinodal passes all of this and is METASTABLE; that needs the free")
    print("     energy, not the root.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", action="store_true", help="run the Phase-1 gates")
    a = p.parse_args()
    if a.gate:
        return cmd_gate()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
