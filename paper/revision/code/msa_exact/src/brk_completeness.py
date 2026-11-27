r"""brk_completeness.py — BRK.6's open half: is `{±lam_ij, ±R_ij}` ALL the knots of the
exact-MSA `c_ij(r)`, or merely a list that contains them?

BRK.2 (Lean, std-3) shows the four crossings are `sigma_l`-free.  BRK.5 shows the exact MSA
is smooth at the mediated candidates `r* = R[a,b] + R[i,a]`.  Neither shows the enumeration
is EXHAUSTIVE -- and "r* is not in a list" is weaker than "c is smooth at r*", which is what
`hzero` needs.  This file closes that by scanning for kinks WHEREVER they are and comparing
the detected set against the predicted one.

STRUCTURAL ARGUMENT (why the count is complete).  The exact-MSA DCF is

    c_ij(r) = -Q'_ij(r) + sum_l rho_l * INT dt Q'_il(t+r) Q_jl(t)

`Q_jl(t)` is piecewise with exactly TWO knots, at `t = lam_jl` and `t = R_jl` (zero below the
first, poly+exp between, pure exp above).  `Q'_il(t+r)` has the same two, shifted: at
`t = lam_il - r` and `t = R_il - r`.  A convolution of two piecewise-smooth factors can kink
only where a MOVING knot meets a FIXED one -- BRK.1's closure lemma -- so the knot set of the
integral is exactly the 2x2 = FOUR crossings, and BRK.2 evaluates all four:

    lam_il - r = lam_jl  ->  r = lam_ij       R_il - r = R_jl   ->  r = lam_ij
    R_il  - r = lam_jl   ->  r = R_ij         lam_il - r = R_jl ->  r = -R_ij

⇒ the enumeration is exhaustive BY COUNTING, and the direct term `-Q'_ij(r)` contributes its
own two knots at the same `{lam_ij, R_ij}`.  No `sigma_l` survives, so no mediated knot can
exist at any coupling.  ⚠ What the counting cannot rule out is an ERROR in that piecewise
form or in the quadrature's breakpoint list -- which is what the scan below tests.

Run:  python brk_completeness.py [scan|all]
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import quad
from msa_exact_mix import HCYMix, solve_bh_mix


def machinery(sigma, rho, K, z=1.8):
    r"""`(c_of_r, sig, R, lam, N)` — the exact-MSA DCF from the Baxter `Q`.

    ⚠⚠ `c_of_r` returns **`2*pi*r*c_ij(r)`**, not `c_ij(r)` — see its own docstring.

    `z` is a scalar for one tail or a sequence for several; `K` has shape `(T, N, N)`.
    """
    # ⚠ `z` may be a scalar (one tail, as BRK always used it) or a list (multi-tail).
    # The original wrapped it unconditionally as `[z]`, which turns a 2-tail list into a
    # single nested entry and then fails the `K` reshape.
    zs = [float(z)] if np.isscalar(z) else [float(v) for v in z]
    st = HCYMix(sigma=sigma, rho=rho, z=zs, K=K)
    N = st.N; rho = st.rho; sig = st.sigma; R = st.SIG
    sol = solve_bh_mix(st); p = sol.p
    qp = p['qp']; A = p['A']
    zs = np.asarray(st.z, float)                    # ⚠ ALL tails, not z[0]
    Wt, Ct = p['Wt'], p['Ct']                       # shape (T, N, N)
    nt = len(zs)
    lam = lambda i, j: (sig[j] - sig[i])/2

    def Q(i, j, r):
        r"""⚠ Multi-tail: the Baxter `Q` carries a SUM over tails (`qhat`'s `Sum_nu`).

        The original hard-coded tail 0 — correct for BRK, which only ever ran `z = 1.8`,
        and silently wrong for 2YK.  Gate 2 of `dcf_region_atlas.py` (`4pi Int r^2 c dr`
        against `(1-a)/rho`, an independent path through the Baxter algebra) is what
        checks this form.
        """
        l = lam(i, j); si = sig[i]; sij = R[i, j]; x = r - l
        if r < l:
            return 0.0
        tail = sum(Wt[t][i, j]*np.exp(-zs[t]*(r-l)) for t in range(nt))
        if r <= sij:
            return (qp[i, j]*(x-si) + 0.5*A[j]*(x-si)**2 + tail
                    + sum(Ct[t][i, j] for t in range(nt)))
        return tail + sum(Ct[t][i, j]*np.exp(-zs[t]*(r-sij)) for t in range(nt))

    def Qp(i, j, r):
        l = lam(i, j); si = sig[i]; sij = R[i, j]; x = r - l
        if r < l:
            return 0.0
        dtail = sum(-zs[t]*Wt[t][i, j]*np.exp(-zs[t]*(r-l)) for t in range(nt))
        if r <= sij:
            return qp[i, j] + A[j]*(x-si) + dtail
        return dtail + sum(-zs[t]*Ct[t][i, j]*np.exp(-zs[t]*(r-sij)) for t in range(nt))

    LO = min(lam(a, b) for a in range(N) for b in range(N)) - 1.0
    HI = max(R[a, b] for a in range(N) for b in range(N)) + 40.0

    def c_of_r(i, j, r):
        r"""⚠⚠ Returns `2*pi*r*c_ij(r)`, NOT `c_ij(r)` — divide by `2*pi*r`.

        Harmless for THIS file, whose subject is where the kinks ARE: an overall factor of
        `r` moves no breakpoint.  It is wrong for any quantitative consumer, and was caught
        by `dcf_region_atlas.py`'s gate 2 (`4pi Int r^2 c dr` came out 69.0 against a known
        2.28).  Verified against the MSA closure's own tail: outside the core
        `c(r) = -beta u(r)`, and `g / (2*pi*r*(-beta u))` is `1.00000` at every `r`.

        ⚠⚠ **The expression below is complete only when the ROW species is the LARGER
        one**, and it is silently wrong the other way round.  `qhat`'s convention is
        `Qhat_ij(s) = Int_{lam_ji}^inf e^{-st} Q_ij(t) dt` with `lam_ji = (sig_j-sig_i)/2`,
        and the factorization `I - sqrt(rho) chat sqrt(rho) = F(ik)F(-ik)^T` carries TWO
        linear terms, `Qhat_ij(ik) + Qhat_ji(-ik)`.  The second inverts to `Q_ji(-r)`,
        supported on `r <= lam_ji`, so it contributes on `0 < r < lam_ji` while this form
        keeps only `-Q'_ij(r)`.  `lam_ji > 0` exactly when `sig_j > sig_i`, i.e. when the
        row is the SMALLER species; ordering the row to the larger diameter empties that
        interval, and the form is then exact.  `c` is symmetric, so the swap costs nothing.

        ⚠ What is ESTABLISHED is the interval and the cure, not the missing term's form:
        the error lives entirely on `0 < r < lam_ji` (both orders agree to round-off above
        it), and row = larger is exact.  Adding a bare `-Q'_ji(-r)` to the small-row case
        improves it from `2.9e-02` to `7.8e-03` but does NOT close it, so the dropped piece
        is not simply that derivative.  Nobody needs it — swap and move on.

        ⚠ Invisible at equal diameters (`lam_ii = 0`), which is why the pure-fluid gate
        passed for a year.  Measured on a binary `sigma = (1, 1.5)` against `chat0` — an
        independent path through the same algebra — `c_10` (row = big) came out at
        `2.0e-07` and `c_01` (row = small) at `2.9e-02`, with a spurious `1/r` blow-up
        below `lam = 0.25`.  Both agree to round-off for `r > lam`.
        """
        if sig[i] < sig[j]:                 # ⚠ row must be the LARGER species
            i, j = j, i
        conv = 0.0
        for l in range(N):
            pts = [b for b in sorted({lam(j, l), R[j, l], lam(i, l)-r, R[i, l]-r})
                   if LO < b < HI]
            conv += rho[l]*quad(lambda t: Qp(i, l, t+r)*Q(j, l, t), LO, HI,
                                points=pts, limit=600)[0]
        return -Qp(i, j, r) + conv
    return c_of_r, sig, R, lam, N


def predicted_knots(i, j, R, lam):
    """`{|lam_ij|, R_ij}` — everything BRK.2 + the direct term allow, on `r > 0`."""
    return sorted({abs(lam(i, j)), float(R[i, j])})


def scan_pair(c_of_r, i, j, R, lam, n=340, pad=1.6):
    r"""Kink indicator on a uniform grid: the raw 4th finite difference.

    ⚠ Deliberately NOT divided by `h^4`.  For a smooth `c` the 4th difference is
    `O(h^4 c'''')` — vanishing — while at a break of ANY order `<= 3` it is `O(jump)`.
    Dividing would restore the noise the small `h` is suppressing.
    """
    hi = float(R[i, j]) + pad
    rs = np.linspace(1e-3, hi, n)
    y = np.array([c_of_r(i, j, r) for r in rs])
    d4 = np.abs(np.diff(y, n=4))
    scale = max(float(np.max(np.abs(y))), 1e-300)
    ind = d4/scale
    return rs, y, rs[2:-2], ind


def peaks(rc, ind, thresh_mult=25.0, min_sep=0.06):
    """Locations where the indicator stands `thresh_mult` x above its own median."""
    med = float(np.median(ind)) or 1e-300
    hits = np.flatnonzero(ind > thresh_mult*med)
    out = []
    for k in hits:
        if not out or rc[k] - out[-1][0] > min_sep:
            out.append((float(rc[k]), float(ind[k]/med)))
        elif ind[k]/med > out[-1][1]:
            out[-1] = (float(rc[k]), float(ind[k]/med))
    return out


def cmd_scan():
    CASES = [("N=3  [1,4,8]", [1., 4., 8.], [0.10, 0.005, 0.0005],
              [[.5, .3, .2], [.3, .4, .25], [.2, .25, .35]]),
             ("N=4  [1,2,3,6]", [1., 2., 3., 6.], [0.10, 0.02, 0.006, 0.0004],
              [[.5, .3, .2, .15], [.3, .4, .25, .18], [.2, .25, .35, .2],
               [.15, .18, .2, .3]])]
    worst_unexplained = []
    for tag, sigma, rho, K in CASES:
        print("=" * 92)
        print(f"{tag}")
        c_of_r, sig, R, lam, N = machinery(sigma, rho, K)
        for i in range(N):
            for j in range(N):
                rc_all, y, rc, ind = scan_pair(c_of_r, i, j, R, lam)
                pk = peaks(rc, ind)
                pred = predicted_knots(i, j, R, lam)
                unexplained = [(r, h) for r, h in pk
                               if min(abs(r - q) for q in pred) > 0.05]
                tick = "ok" if not unexplained else "*** UNEXPLAINED ***"
                print(f"  c_({i},{j})  predicted {[f'{q:.3f}' for q in pred]}"
                      f"   detected {[f'{r:.3f}' for r, _ in pk]}   {tick}")
                if unexplained:
                    worst_unexplained.append((tag, i, j, unexplained))
    print("=" * 92)
    if worst_unexplained:
        print("⚠⚠ UNEXPLAINED KINKS — the enumeration is NOT complete:")
        for tag, i, j, u in worst_unexplained:
            print(f"   {tag}  c_({i},{j})  at {[f'{r:.4f} (x{h:.0f})' for r, h in u]}")
    else:
        print("⭐ every detected kink lies in {|lam_ij|, R_ij} — no other breakpoints found")
    return 1 if worst_unexplained else 0


if __name__ == "__main__":
    raise SystemExit(cmd_scan())
