"""FMSA_double_prop: multi-species Yukawa DCF via the doubly-propagated
first-order MSA k-space assembly (exact first order, no OZ iteration).

The inner-core first-order correction is computed from the exact first-order
solution of the OZ + MSA equations in Fourier space ([LN] lecture notes
`pdf/lecture_notes_OZ_Yukawa.tex`, Eqs. OZ1_Baxter / B1_def / bij; the
spectral-amplitude residue is Lean-proven in `YukawaDCF/SpectralAmplitude.lean`
Y1.5 `bMulti_residue_Qinv` and `YukawaWienerHopf.lean` Y1.6 `Hhat1_residue`):

    b_ij(s)    = Σ_{t,m,n} [δ+A]_im(z_tmn) · K[t,m,n] · [δ+A]_jn(z_tmn) / (s + z_tmn)
    {B₁(k)}_ij = exp(−ik·R_ij) · b_ij(ik)
    Ĉ₁(k)      = Q̂₀(−k) · B₁(k) · Q̂₀ᵀ(−k)               ← NO Q̂₀ inverses

The last identity follows algebraically from the two [LN] equations
(eq:OZ1_Baxter and Q̂₀ᵀĤ₁Q̂₀ = B₁) and is the structural key: Q̂₀ is entire and
B₁ carries only the Yukawa poles, so the DCF has NO hard-sphere poles at all
(the zeros of det Q̂₀ never enter) and c⁽¹⁾ is a FINITE closed form.  In real
space it is a triple convolution of elementary kernels,

    𝒲 = 𝒬⁻ ⋆ ℬ ⋆ (𝒬⁻)ᵀ,   𝒬⁻_ij(t) = δ_ijδ(t) − 2π√(ρᵢρⱼ)·q_ij(−t)
    c⁽¹⁾_ij(r) = [𝒲_ij(r) − 𝒲_ij(−r)] / (2π√(ρᵢρⱼ)·r),   r > 0

giving a finite sum of (polynomial × exponential) on finitely many pieces —
exponential rates = the Yukawa z_q; piece edges = support-overlap breakpoints
(λ_ij first).  See `_build_closed_form` and Lean_code/to_LEAN.md §1.
(The infinite Mittag-Leffler HS-pole sum is needed only for the RDF ĥ₁, whose
Ĥ₁ = Q̂₀ᵀ(k)⁻¹B₁Q̂₀(k)⁻¹ does carry the inverses.)

[δ+A] is the doubly-propagated bare hard-sphere propagator (A without the
λ-phase), evaluated at the fixed Yukawa pole z_tmn — K is sandwiched by TWO
inverse-Baxter factors, replacing the ill-conditioned (1−G²)/A² split of
FMSA_GA_matrix_mix whose inner formula misestimates c^(1) by up to 10⁸ for
2YK pairs (see numerical_notes/results/true_first_order_probe.md).

Validation (probe_dp_assembly.py, 2026-07-16): N=1 reduces to FMSA_pure/Tang
(rel 5e-3 at d=1); mixture h₁ vanishes in-core; c^(1) matches the OZ coupling
derivative ground truth, with the residual shrinking as the OZ grid refines
(2YK unlike pair: 0.42 → 0.060 → 0.022 for n_r = 1024/4096/16384).  The closed
form agrees with the k-space assembly to ~1e-12 and reproduces the bare MSA
tail for r > R_ij to ~2e-15.

The total DCF is c_ij(r) = c^HS_ij(r) + c^(1)_ij(r) inside the core and the
strict MSA tail −βu_ij(r) outside.  Two hard-sphere references are available,
both closed form: White Bear FMT (`get_HS_FMT`, project standard) and
Baxter/PY (`get_HS_PY`, the reference c^(1) formally perturbs around, so
c^HS-PY + c^(1) + tail reproduces converged OZ+MSA to first-order accuracy).
Evaluation anywhere in (ρ, T) — no iteration, so the method remains evaluable
where OZ solvers fail to converge (near/inside the spinodal); accuracy there
degrades gracefully (first-order truncation).

Fully analytic: every public quantity — c(r), 𝓕[c](k), c^HS(r) — is closed
form.  No radial grid, no quadrature, no iteration anywhere.
Self-contained — imports only numpy (scipy only for the optional `get_c1_exact`
cross-check).  No inheritance; all helpers are inlined.  Suitable for direct
C++ translation (plain array arithmetic throughout).
"""

from __future__ import annotations
import warnings

import math

import numpy as np

try:                      # numpy >= 2.0 renamed trapz -> trapezoid
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# ══════════════════════════════════════════════════════════════════════════════
# Piecewise poly-exponential algebra  (closed under convolution)
#
#   f(t) = Σ_α P_α(t)·exp(−α·t)   on each piece [a, b)
#
# stored as pieces = [(a, b, {α: poly})], poly = increasing-order coeff array.
# Everything below is plain array arithmetic — direct C++ translation.
# ══════════════════════════════════════════════════════════════════════════════

_PX_TOL = 1e-11


def _px_binom(n: int, k: int) -> float:
    return float(math.comb(n, k))


def _px_trim(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    nz = np.nonzero(np.abs(p) > 1e-300)[0]
    return p[:nz[-1] + 1] if len(nz) else np.zeros(1)


def _px_bipoly_u(P: np.ndarray) -> np.ndarray:
    """P(u) → A[i,j] (coefficient of u^i·r^j)."""
    A = np.zeros((len(P), 1))
    A[:, 0] = P
    return A


def _px_bipoly_r_minus_u(Q: np.ndarray) -> np.ndarray:
    """Q(r−u) → A[i,j]."""
    n_max = len(Q)
    A = np.zeros((n_max, n_max))
    for n in range(n_max):
        if Q[n] == 0.0:
            continue
        for k in range(n + 1):
            A[n - k, k] += Q[n] * _px_binom(n, k) * ((-1.0) ** (n - k))
    return A


def _px_bimul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    out = np.zeros((A.shape[0] + B.shape[0] - 1, A.shape[1] + B.shape[1] - 1))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 0.0:
                continue
            out[i:i + B.shape[0], j:j + B.shape[1]] += A[i, j] * B
    return out


def _px_du(A: np.ndarray) -> np.ndarray:
    if A.shape[0] <= 1:
        return np.zeros((1, A.shape[1]))
    out = np.zeros((A.shape[0] - 1, A.shape[1]))
    for i in range(1, A.shape[0]):
        out[i - 1] = i * A[i]
    return out


def _px_antideriv_u(A: np.ndarray, gamma: float):
    """Antiderivative in u of A(u,r)·e^{−γu}.

    γ=0 → ('poly', F),  F(u,r) = ∫A du.
    γ≠0 → ('exp',  B),  F = −e^{−γu}·B(u,r),  B = Σ_m ∂ᵐ_u A / γ^{m+1}
          (check: γB − B_u = A).
    """
    if abs(gamma) < _PX_TOL:
        F = np.zeros((A.shape[0] + 1, A.shape[1]))
        for i in range(A.shape[0]):
            F[i + 1] = A[i] / (i + 1)
        return ('poly', F)
    B = np.zeros_like(A)
    D = A.copy()
    m = 0
    while np.any(np.abs(D) > 1e-300) and m <= A.shape[0] + 2:
        B[:D.shape[0]] += D / gamma ** (m + 1)
        D = _px_du(D)
        m += 1
    return ('exp', B)


def _px_subst_const(B: np.ndarray, c: float) -> np.ndarray:
    """B(c, r) → poly in r."""
    out = np.zeros(B.shape[1])
    for i in range(B.shape[0]):
        out += B[i] * (c ** i)
    return out


def _px_subst_r_minus_c(B: np.ndarray, c: float) -> np.ndarray:
    """B(r−c, r) → poly in r."""
    out = np.zeros(B.shape[0] + B.shape[1] - 1)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if B[i, j] == 0.0:
                continue
            for k in range(i + 1):
                out[k + j] += B[i, j] * _px_binom(i, k) * ((-c) ** (i - k))
    return out


def _px_add_term(d: dict, alpha: float, poly: np.ndarray) -> None:
    key = None
    for k in d:
        if abs(k - alpha) < _PX_TOL:
            key = k
            break
    if key is None:
        d[alpha] = np.array(poly, dtype=float)
    else:
        n = max(len(d[key]), len(poly))
        acc = np.zeros(n)
        acc[:len(d[key])] += d[key]
        acc[:len(poly)] += poly
        d[key] = acc


def px_eval(pieces: list, t: float) -> float:
    v = 0.0
    for (a, b, terms) in pieces:
        if a <= t < b:
            for al, P in terms.items():
                v += float(np.polyval(P[::-1], t)) * math.exp(-al * t)
    return v


def px_convolve(f_pieces: list, g_pieces: list) -> list:
    """(f ⋆ g)(r) = ∫ f(u)·g(r−u) du, exactly, in the same function class.

    Output breakpoints are the pairwise sums of the input edges; on each output
    piece the limits lo=max(a1, r−b2), hi=min(b1, r−a2) are each either a
    constant or (r − const), and the u-integral of
    P_α(u)Q_β(r−u)·e^{−αu}e^{−β(r−u)} is closed form, producing only the rates
    α and β.  Semi-infinite g (b2=∞) is supported.
    """
    bps = set()
    for (a1, b1, _) in f_pieces:
        for (a2, b2, _) in g_pieces:
            for x in (a1 + a2, a1 + b2, b1 + a2, b1 + b2):
                if np.isfinite(x):
                    bps.add(round(x, 10))
    if not bps:
        return []
    edges = sorted(bps) + [np.inf]

    out_pieces = []
    for q in range(len(edges) - 1):
        r0, r1 = edges[q], edges[q + 1]
        rmid = (r0 + r1) / 2.0 if np.isfinite(r1) else r0 + 1.0
        terms: dict = {}
        for (a1, b1, tf) in f_pieces:
            for (a2, b2, tg) in g_pieces:
                lo_lin = (rmid - b2) if np.isfinite(b2) else -np.inf
                hi_lin = rmid - a2
                lo_is_lin = lo_lin > a1
                hi_is_lin = hi_lin < b1
                lo_val = lo_lin if lo_is_lin else a1
                hi_val = hi_lin if hi_is_lin else b1
                if hi_val <= lo_val + 1e-13:
                    continue
                for al, P in tf.items():
                    for be, Q in tg.items():
                        A = _px_bimul(_px_bipoly_u(P), _px_bipoly_r_minus_u(Q))
                        gam = al - be
                        kind, F = _px_antideriv_u(A, gam)
                        for (is_lin, cval, sgn) in (
                                (hi_is_lin, (a2 if hi_is_lin else b1), +1.0),
                                (lo_is_lin, (b2 if lo_is_lin else a1), -1.0)):
                            if kind == 'poly':
                                pr = (_px_subst_r_minus_c(F, cval) if is_lin
                                      else _px_subst_const(F, cval))
                                _px_add_term(terms, be, sgn * pr)
                            elif is_lin:
                                # −e^{−γ(r−c)}·B(r−c,r) → rate β+γ = α
                                pr = _px_subst_r_minus_c(F, cval)
                                _px_add_term(terms, al,
                                             -sgn * math.exp(gam * cval) * pr)
                            else:
                                pr = _px_subst_const(F, cval)
                                _px_add_term(terms, be,
                                             -sgn * math.exp(-gam * cval) * pr)
        terms = {k: _px_trim(v) for k, v in terms.items()
                 if np.any(np.abs(v) > 1e-300)}
        if terms:
            out_pieces.append((r0, r1, terms))
    return out_pieces


def px_scale_add(acc: list, pieces: list, scale: float) -> list:
    """acc += scale·pieces (piecewise, re-gridding onto the merged edges)."""
    if not pieces:
        return acc
    edges = set()
    for (a, b, _) in acc + pieces:
        edges.add(round(a, 10))
        if np.isfinite(b):
            edges.add(round(b, 10))
    edges = sorted(edges) + [np.inf]
    out = []
    for q in range(len(edges) - 1):
        r0, r1 = edges[q], edges[q + 1]
        rmid = (r0 + r1) / 2.0 if np.isfinite(r1) else r0 + 1.0
        terms: dict = {}
        for (src, sc) in ((acc, 1.0), (pieces, scale)):
            for (a, b, t) in src:
                if a <= rmid < b:
                    for al, P in t.items():
                        _px_add_term(terms, al, sc * P)
        terms = {k: _px_trim(v) for k, v in terms.items()
                 if np.any(np.abs(v) > 1e-300)}
        if terms:
            out.append((r0, r1, terms))
    return out


def px_clip(pieces: list, lo: float, hi: float) -> list:
    """The same function restricted to [lo, hi] — pieces outside are dropped.

    Needed before `px_scale_add` whenever one operand is semi-infinite on the
    LEFT: that routine forms `rmid = (r0+r1)/2` per output piece, which is −inf
    for a piece starting at −inf, and then `a <= rmid < b` selects nothing.
    Clipping first keeps every edge finite.
    """
    out = []
    for (a, b, terms) in pieces:
        a2, b2 = max(a, lo), min(b, hi)
        if b2 > a2 + 1e-13:
            out.append((a2, b2, {al: np.array(P, dtype=float)
                                 for al, P in terms.items()}))
    return sorted(out, key=lambda p: p[0])


def px_reflect(pieces: list) -> list:
    """t ↦ −t:  f(−t), still in the same class.

    A piece (a, b, {α: P}) supports Σ_α P(t)e^{−αt} on [a, b); after reflection
    the same values sit on (−b, −a] as Σ_α P(−t)e^{+αt}, i.e. rate −α and
    coefficients with alternating signs.  Half-open edges flip side, which is a
    measure-zero difference and so invisible to `px_integrate`.
    """
    out = []
    for (a, b, terms) in pieces:
        new: dict = {}
        for al, P in terms.items():
            Q = np.array([P[k] * (-1.0) ** k for k in range(len(P))])
            _px_add_term(new, -al, Q)
        out.append((-b if np.isfinite(b) else -np.inf, -a, new))
    return sorted(out, key=lambda p: p[0])


def px_mul(f_pieces: list, g_pieces: list) -> list:
    """Pointwise product f(t)·g(t) — exact, and still in the same class.

    Not a convolution: `px_convolve` is ∫f(u)g(r−u)du, this is f·g at the same
    t.  Both stay closed under the class because P_α(t)e^{−αt}·Q_β(t)e^{−βt} =
    (P_αQ_β)(t)·e^{−(α+β)t}, i.e. polynomials multiply and rates ADD.
    """
    edges = set()
    for (a, b, _) in list(f_pieces) + list(g_pieces):
        edges.add(round(a, 10))
        if np.isfinite(b):
            edges.add(round(b, 10))
    if not edges:
        return []
    edges = sorted(edges) + [np.inf]
    out = []
    for q in range(len(edges) - 1):
        r0, r1 = edges[q], edges[q + 1]
        rmid = (r0 + r1) / 2.0 if np.isfinite(r1) else r0 + 1.0
        terms: dict = {}
        for (a1, b1, tf) in f_pieces:
            if not (a1 <= rmid < b1):
                continue
            for (a2, b2, tg) in g_pieces:
                if not (a2 <= rmid < b2):
                    continue
                for al, P in tf.items():
                    for be, Q in tg.items():
                        _px_add_term(terms, al + be, np.convolve(P, Q))
        terms = {k: _px_trim(v) for k, v in terms.items()
                 if np.any(np.abs(v) > 1e-300)}
        if terms:
            out.append((r0, r1, terms))
    return out


def px_mul_poly(pieces: list, poly: np.ndarray) -> list:
    """Multiply by a single polynomial (increasing order) — exact, in-class."""
    poly = np.asarray(poly, dtype=float)
    out = []
    for (a, b, terms) in pieces:
        new = {al: np.convolve(P, poly) for al, P in terms.items()}
        out.append((a, b, new))
    return out


def px_integrate(pieces: list, lo: float, hi: float) -> float:
    """∫_lo^hi of a piecewise (polynomial × exponential) function, in closed form.

    Per term ∫ t^n e^{−αt} dt: for α = 0 the polynomial antiderivative; for
    α ≠ 0 the standard −e^{−αt}·Σ_m P^{(m)}(t)/α^{m+1} (differentiate it back to
    check).  No quadrature anywhere, so this is the exactness gate every
    panel-quadrature route in this file is measured against.
    """
    total = 0.0
    for (a, b, terms) in pieces:
        a2, b2 = max(a, lo), min(b, hi)
        if b2 <= a2:
            continue
        for al, P in terms.items():
            P = np.asarray(P, dtype=float)
            if abs(al) < _PX_TOL:
                F = np.concatenate(([0.0], P / np.arange(1, len(P) + 1)))
                total += (np.polyval(F[::-1], b2) - np.polyval(F[::-1], a2))
                continue
            # B = Σ_m P^(m)/α^{m+1};  d/dt[−e^{−αt}B] = e^{−αt}(αB − B') = P e^{−αt}
            B = np.zeros(len(P))
            D = P.copy()
            m = 0
            while len(D) and np.any(np.abs(D) > 1e-300):
                B[:len(D)] += D / al ** (m + 1)
                D = np.array([D[k] * k for k in range(1, len(D))])
                m += 1
            total += (-math.exp(-al * b2) * np.polyval(B[::-1], b2)
                      + math.exp(-al * a2) * np.polyval(B[::-1], a2))
    return float(total)


# ══════════════════════════════════════════════════════════════════════════════
# Second-order jet (truncated Taylor number) — exact first/second derivatives
# of the FMT auxiliary functions without hand algebra.  Plain operator
# overloading; translates directly to a small C++ struct.
# ══════════════════════════════════════════════════════════════════════════════


class _J2:
    """(f, f', f'') under +, −, ×, ÷ and log."""

    __slots__ = ('a', 'b', 'c')

    def __init__(self, a: float, b: float = 0.0, c: float = 0.0):
        self.a, self.b, self.c = a, b, c

    @staticmethod
    def _w(o):
        return o if isinstance(o, _J2) else _J2(float(o))

    def __add__(self, o):
        o = _J2._w(o)
        return _J2(self.a + o.a, self.b + o.b, self.c + o.c)
    __radd__ = __add__

    def __neg__(self):
        return _J2(-self.a, -self.b, -self.c)

    def __sub__(self, o):
        return self + (-_J2._w(o))

    def __rsub__(self, o):
        return _J2._w(o) + (-self)

    def __mul__(self, o):
        o = _J2._w(o)
        return _J2(self.a * o.a,
                   self.a * o.b + self.b * o.a,
                   self.a * o.c + 2.0 * self.b * o.b + self.c * o.a)
    __rmul__ = __mul__

    def __truediv__(self, o):
        o = _J2._w(o)
        q  = self.a / o.a
        q1 = (self.b - q * o.b) / o.a
        q2 = (self.c - 2.0 * q1 * o.b - q * o.c) / o.a
        return _J2(q, q1, q2)

    def __rtruediv__(self, o):
        return _J2._w(o) / self

    def log(self):
        return _J2(math.log(self.a), self.b / self.a,
                   self.c / self.a - (self.b / self.a)**2)


def _fmt_A_phi(x: float, version: str):
    """FMT auxiliary functions of n₃ = η, with exact derivatives, for

        Φ = −n₀·ln(1−n₃) + (n₁n₂ − n_v1·n_v2)·A(n₃)
                         + (n₂³ − 3n₂ n_v2·n_v2)·φ(n₃)

    'rosenfeld' — Rosenfeld (1989), the original FMT:
        A = 1/(1−n₃),  φ = 1/(24π(1−n₃)²)
    'wb1' — White Bear I (Roth, Evans, Lang & Kahl 2002):
        A = 1/(1−n₃),  φ = [n₃ + (1−n₃)²ln(1−n₃)] / (36π n₃²(1−n₃)²)
    'wb2' — White Bear II (Hansen-Goos & Roth 2006):
        A = (1 + φ₂(n₃)/3)/(1−n₃),   φ = (1 − φ₃(n₃)/3) / (24π(1−n₃)²)
        φ₂ = [2n₃ − n₃² + 2(1−n₃)ln(1−n₃)] / n₃
        φ₃ = [2n₃ − 3n₃² + 2n₃³ + 2(1−n₃)²ln(1−n₃)] / n₃²

    Rosenfeld is exactly the φ₂ = φ₃ = 0 case of 'wb2', and shares 'wb1'’s A.
    Both White Bears reduce to it as n₃ → 0 (φ₂ → n₃²/3, φ₃ → 4n₃/3), which is
    why all three agree in the dilute limit and differ at finite density.

    The one-component limits are NOT all the same, and that is the point:
    'wb1'/'wb2' give Carnahan–Starling, 'rosenfeld' gives the Percus–Yevick
    COMPRESSIBILITY result βP = ρ(1+η+η²)/(1−η)³.  Rosenfeld is also the
    flavour whose DCF is the PY/Lebowitz one, so it is the consistent partner
    for a PY-based hard-sphere reference.

    Returns ((A, A′, A″), (φ, φ′, φ″)).
    """
    X = _J2(x, 1.0, 0.0)
    v = 1.0 - X
    if version == 'rosenfeld':
        A   = 1.0 / v
        phi = 1.0 / (24.0 * np.pi * v * v)
    elif version == 'wb1':
        A   = 1.0 / v
        phi = (X + v * v * v.log()) / (36.0 * np.pi * X * X * v * v)
    elif version == 'wb2':
        p2  = (2.0*X - X*X + 2.0 * v * v.log()) / X
        p3  = (2.0*X - 3.0*X*X + 2.0*X*X*X + 2.0 * v * v * v.log()) / (X * X)
        A   = (1.0 + p2 / 3.0) / v
        phi = (1.0 - p3 / 3.0) / (24.0 * np.pi * v * v)
    else:
        raise ValueError(f"unknown FMT version {version!r} "
                         f"(use 'rosenfeld', 'wb1' or 'wb2')")
    return (A.a, A.b, A.c), (phi.a, phi.b, phi.c)



class FMSA_double_prop:
    """Multi-species Yukawa DCF from the doubly-propagated first-order assembly.

    Parameters
    ----------
    sig_pure   : pure-species LJ diameters (input_type='lj') or BH diameters
                 (input_type='bh'), shape (N,)
    eps_pure   : pure-species LJ well-depths, shape (N,)
    eps_ij     : optional explicit cross-interaction matrix, shape (N,N),
                 REPLACING the Lorentz-Berthelot geometric mean
                 eps_ij = sqrt(eps_i * eps_j).  Its diagonal must equal
                 `eps_pure` (eps_ii IS the pure-species well depth) and it must
                 be symmetric; both are checked.  Needed for any mixture whose
                 unlike interaction is not the Berthelot mean -- in particular
                 the symmetric hard-core Yukawa mixture of Caccamo et al.,
                 JCP 109, 4498 (1998) and Mkanya et al., JPCM 29, 365102 (2017),
                 which is defined by sigma_1 = sigma_2, eps_11 = eps_22 and a
                 free ratio alpha = eps_12/eps_11.  Under Berthelot those
                 constraints force alpha = 1, which is exactly the degenerate
                 case where the mixture collapses to a one-component fluid and
                 has NO demixing, so alpha != 1 is unreachable without this.
    x          : mole fractions, shape (N,); must sum to 1
    rhostar    : total reduced density; rho = x * rhostar
    tstar      : reduced temperature
    n_tails    : number of Yukawa tails (1 = HSY, 2 = 2YK LJ fit; default 2)
    z, k       : optional Yukawa range/amplitude overrides (same conventions as
                 FMSA_GA_matrix_mix / FMSA_poly_term_species)
    input_type : 'lj' (default) or 'bh'
    hs_fmt     : which FMT functional `get_HS_FMT` evaluates —
                 'rosenfeld' Rosenfeld (1989), whose DCF is exactly the
                 PY/Lebowitz one this class also computes in `get_HS_PY`,
                 'wb1' (default) White Bear I, matching the other classes in the
                 project (`fmsa_ga_matrix_mix`, `FMSA_pure`,
                 `FMSA_MC_cleaned_2cpp`), so cross-route numbers stay comparable;
                 'wb2' White Bear II (Hansen-Goos & Roth 2006), which adds
                 consistency with the exact scaled-particle relation and is the
                 better choice for inhomogeneous/confined work.  Both reproduce
                 Carnahan–Starling exactly for the pure fluid; they differ by
                 ~1–2% in the c(r) *shape* and, at bulk ĉ_ij(0), by ~0.4% at
                 σ-ratio 2 and only ~0.05% at CO₂–H₂-like σ-ratio 1.28.
    n_k, r_max_factor : retained for API compatibility; unused since the
                 closed form replaced the spectral table (2026-07-16)
    verbose    : print a construction summary
    """

    def __init__(
        self,
        sig_pure: np.ndarray,
        eps_pure: np.ndarray,
        x: np.ndarray,
        rhostar: float,
        tstar: float,
        n_tails: int = 2,
        z=None,
        k=None,
        eps_ij=None,
        input_type: str = 'lj',
        d_pure=None,
        n_k: int = 16384,
        r_max_factor: float = 15.0,
        hs_fmt: str = 'wb1',
        verbose: bool = True,
    ) -> None:
        sig_pure = np.asarray(sig_pure, dtype=float)
        eps_pure = np.asarray(eps_pure, dtype=float)
        x        = np.asarray(x,        dtype=float)
        N = len(sig_pure)

        self.sig_pure = sig_pure
        self.eps_pure = eps_pure
        self.x        = x
        self.rhostar  = rhostar
        self.tstar    = tstar
        self.beta     = beta = 1.0 / tstar
        self.N        = N
        self.n_tails  = n_tails
        if hs_fmt not in ('rosenfeld', 'wb1', 'wb2'):
            raise ValueError(
                f"hs_fmt must be 'rosenfeld', 'wb1' or 'wb2', got {hs_fmt!r}")
        self.hs_fmt   = hs_fmt

        # Mixing rules (Lorentz-Berthelot), unless an explicit cross matrix is
        # supplied.  Berthelot is not merely an approximation here: for equal
        # diameters with eps_11 = eps_22 it FORCES eps_12 = eps_11, i.e. the
        # one-component limit, in which the mixture cannot demix at all.
        sig_mix = 0.5 * (sig_pure[:, None] + sig_pure[None, :])
        if eps_ij is None:
            eps_mix = np.sqrt(eps_pure[:, None] * eps_pure[None, :])
        else:
            eps_mix = np.asarray(eps_ij, dtype=float)
            if eps_mix.shape != (N, N):
                raise ValueError(
                    f"eps_ij must have shape ({N}, {N}), got {eps_mix.shape}")
            if not np.allclose(eps_mix, eps_mix.T, rtol=0.0, atol=1e-12):
                raise ValueError("eps_ij must be symmetric")
            if not np.allclose(np.diag(eps_mix), eps_pure, rtol=1e-12, atol=1e-12):
                raise ValueError(
                    "diag(eps_ij) must equal eps_pure -- eps_ii IS the "
                    f"pure-species well depth; got diag {np.diag(eps_mix)} "
                    f"against eps_pure {eps_pure}")
        self.sig_mix = sig_mix
        self.eps_mix = eps_mix

        # BH diameters and contact-distance matrix
        if d_pure is not None:
            # Hard cores given explicitly while sig_pure/eps_pure stay LJ, so the
            # TAIL is still parameterised off sigma_LJ.  input_type='bh' cannot do
            # this: there sig_mix is the diameter matrix, so z = z_fac/d and the
            # amplitude loses its exp(z(sigma-d)) shift -- z comes out 3.4 % high
            # and k_bare 12.2 % low for an LJ fluid.  That reading is right for a
            # hard-core Yukawa model (which is what 'bh' is for) and wrong for an
            # LJ whose core simply is not the BH correlation's -- the LJ + hard
            # sphere mixture, where the sphere is a WCA stand-in 2.7-4.4 % LARGER
            # than sigma and therefore unreachable by any BH formula.
            dd = np.asarray(d_pure, dtype=float)
            self.d_bh = 0.5 * (dd[:, None] + dd[None, :])
        elif input_type == 'lj':
            self.d_bh = self._bh_diameters(sig_mix, eps_mix, beta)
        else:   # 'bh': sig_pure already contains BH pure-species diameters
            self.d_bh = sig_mix.copy()
        self.sigma  = sigma = np.diag(self.d_bh)
        self.R_ij   = 0.5 * (sigma[:, None] + sigma[None, :])
        self.diag_r = sigma.copy()

        self.lambda_ij = 0.5 * (sigma[None, :] - sigma[:, None])
        self.sigma_min = float(np.min(sigma))

        self.rho = rho = x * rhostar
        self.xi  = np.array([float(np.sum(rho * sigma**n)) for n in range(4)])
        self.eta = (np.pi / 6.0) * self.xi[3]
        self.vac = 1.0 - self.eta

        # HS Baxter Q-factors  Q₀[i,j], Qpp[j]
        self.Q0 = (2.0 * np.pi / self.vac) * (
            self.R_ij
            + (np.pi * self.xi[2] / (4.0 * self.vac)) * sigma[:, None] * sigma[None, :]
        )
        self.Qpp = (2.0 * np.pi / self.vac) * (
            1.0 + (np.pi * self.xi[2] / (2.0 * self.vac)) * sigma
        )

        # Yukawa ranges and amplitudes (identical conventions to FMSA_GA_matrix_mix)
        def _to_arr(v, n: int):
            if v is None:
                return None
            a = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
            return np.full(n, a[0]) if a.size == 1 else a

        if n_tails == 1:
            z_arr = _to_arr(z, 1) if z is not None else np.array([1.791])
            self.z_poles = z_arr[:, None, None] / self.R_ij[None, :, :]
        else:
            z_arr = _to_arr(z, n_tails) if z is not None else np.array([2.9637, 14.0167])
            self.z_poles = z_arr[:, None, None] / sig_mix[None, :, :]

        k_arr  = _to_arr(k, n_tails)
        k_bare = np.zeros((n_tails, N, N), dtype=float)
        k0_fac = 2.1714
        if n_tails == 1:
            k_amp = float(k_arr[0]) if k_arr is not None else 1.0
            k_bare[0] = k_amp * eps_mix * self.R_ij
        else:
            for t in range(n_tails):
                if k_arr is not None:
                    k_bare[t] = k_arr[t] * sig_mix * eps_mix * np.exp(
                        self.z_poles[t] * (sig_mix - self.R_ij)
                    )
                else:
                    sign = 1.0 if t == 0 else -1.0
                    k_bare[t] = sign * k0_fac * sig_mix * eps_mix * np.exp(
                        self.z_poles[t] * (sig_mix - self.R_ij)
                    )
        self.k_bare  = k_bare
        self.epsilon = k_bare / self.R_ij

        rho_sq = np.sqrt(rho[:, None] * rho[None, :])
        self.K = 2.0 * np.pi * beta * rho_sq[None, :, :] * k_bare

        # Doubly-propagated pole/residue tables (four-term bMulti structure)
        self._build_b_tables()

        # Store constructor inputs for _make_at_rho
        self._z_input    = z
        self._k_input    = k
        self._eps_ij     = None if eps_ij is None else np.asarray(eps_ij, float)
        self._input_type = input_type
        self._d_pure     = None if d_pure is None else np.asarray(d_pure, float)
        self._n_k        = n_k
        self._rmaxf      = r_max_factor

        # Explicit closed form for c^(1) (finite poly×exp pieces; no grid)
        self._build_closed_form()

        if verbose:
            npc = sum(len(self._W[i][j]) for i in range(N) for j in range(N))
            print(f"FMSA_double_prop: N={N}, n_tails={n_tails}, eta={self.eta:.4f}, "
                  f"closed form: {npc} poly-exp pieces")

    # ------------------------------------------------------------------
    # Barker-Henderson diameters
    # ------------------------------------------------------------------

    @staticmethod
    def _bh_diameters(sig_mix, eps_mix, beta, cbh=None):
        """d_bh[i,j] = 2^{1/6}·σ_ij/(1+√f)^{1/6}, f = 1 + c₀t + c₁t² + c₂t⁴, t = T*/ε."""
        if cbh is None:
            # WARNING -- these coefficients are BIASED, and knowingly kept.
            # The form is d/sigma = 2^(1/6)/(1+sqrt(1+c0 T*+c1 T*^2+c2 T*^4))^(1/6), which
            # inverts to the statement beta*u(d) = c0 + c1 T* + c2 T*^3.  At the TRUE
            # Barker-Henderson diameter that quantity tends to 1 as T*->0 (near contact
            # u ~ -24 eps (r-sigma)/sigma, so d/sigma -> 1 - T*/24) and falls monotonically
            # from there, so 1 is its supremum -- c0 = 1.1287 is above it and cannot come
            # from fitting this quantity over any range.  Measured against the integral:
            # -0.32% at T* = 0.5, -0.90% at 2.12, -3.72% at 10, always low.  It is not a
            # different DEFINITION either (integrating to r_min, to infinity, or over the
            # WCA split is 3-200% off).  Inherited from the original MATLAB (FMSA_MC.m,
            # "empirical fit coefficients", no citation).
            # LEFT ALONE ON PURPOSE: this project studies hard-core Yukawa systems, where
            # input_type='bh' is used and this LJ mapping never runs.  The C++ port
            # (cpp/FMSA_model) does run it, and uses Cotterman, Schwarz & Prausnitz,
            # AIChE J 32, 1787 (1986), d/sigma = (1+0.2977 T*)/(1+0.33163 T*+0.0010477 T*^2),
            # which is within +-0.12% of the integral over T* in [0.5, 10].
            cbh = np.array([1.1287, -0.05536, 0.0007278])
        t   = 1.0 / (beta * eps_mix)
        fac = 1.0 + cbh[0]*t + cbh[1]*t**2 + cbh[2]*t**4
        return 2.0**(1.0/6.0) * sig_mix / (1.0 + np.sqrt(fac))**(1.0/6.0)

    # ------------------------------------------------------------------
    # Baxter Q̂₀(s) — complex Laplace argument
    # ------------------------------------------------------------------

    def _build_Qhat(self, s):
        """Q̂₀_ij(s) = δ_ij − √(ρᵢρⱼ)·e^{−sλ_ij}·[Q₀_ij·φ₁(σ_i,s) + Qpp_j·φ₂(σ_i,s)].

        Accepts scalar (real or complex) s; returns (N, N).
        """
        sigma, rho = self.sigma, self.rho
        lam, Q0, Qpp = self.lambda_ij, self.Q0, self.Qpp
        p1 = (1.0 - s*sigma - np.exp(-s*sigma)) / s**2
        zs = s * sigma
        p2 = (1.0 - zs + 0.5*zs**2 - np.exp(-zs)) / s**3
        sqrt_rr = np.sqrt(rho[:, None] * rho[None, :])
        inner   = p1[:, None] * Q0 + p2[:, None] * Qpp[None, :]
        Qhat    = -sqrt_rr * np.exp(-s * lam) * inner
        Qhat   += np.eye(self.N)
        return Qhat

    def _build_M(self, s: float) -> np.ndarray:
        """The density-free kernel of Q̂₀:  Q̂₀ = I − D^{1/2}·M(s)·D^{1/2}, D = diag(ρ).

        Same expression as `_build_Qhat` with the √(ρᵢρⱼ) stripped off.  It exists
        so that the propagator can be de-normalised WITHOUT dividing by a √ρ that
        may be zero -- see `_reduced_contact_sums`.
        """
        sigma, lam, Q0, Qpp = self.sigma, self.lambda_ij, self.Q0, self.Qpp
        s = float(s)
        p1 = (1.0 - s*sigma - np.exp(-s*sigma)) / s**2
        zs = s * sigma
        p2 = (1.0 - zs + 0.5*zs**2 - np.exp(-zs)) / s**3
        inner = p1[:, None] * Q0 + p2[:, None] * Qpp[None, :]
        return np.exp(-s * lam) * inner

    def _build_Qhat_vec(self, s_arr: np.ndarray) -> np.ndarray:
        """Vectorised Q̂₀ over an array of complex s → shape (len(s), N, N).

        Uses the Taylor series of φ₁/φ₂ for |s·σ| < 1e-3 (removable s→0
        singularity: φ₁ → −σ²/2, φ₂ → σ³/6)."""
        sigma, rho = self.sigma, self.rho
        lam, Q0, Qpp = self.lambda_ij, self.Q0, self.Qpp
        s = s_arr[:, None]                                   # (nk, 1) broadcast on species
        zs = s * sigma[None, :]
        small = np.abs(zs) < 1e-3
        zs_safe = np.where(small, 1.0, zs)                   # avoid 0/0 in the exact branch
        s_safe  = np.where(small, 1.0, s * np.ones_like(zs))
        p1 = (1.0 - zs_safe - np.exp(-zs_safe)) / s_safe**2                 # (nk, N)
        p2 = (1.0 - zs_safe + 0.5*zs_safe**2 - np.exp(-zs_safe)) / s_safe**3
        sg = sigma[None, :]
        p1_ser = -sg**2/2.0 + s*sg**3/6.0 - s**2*sg**4/24.0 + s**3*sg**5/120.0
        p2_ser =  sg**3/6.0 - s*sg**4/24.0 + s**2*sg**5/120.0 - s**3*sg**6/720.0
        p1 = np.where(small, p1_ser, p1)
        p2 = np.where(small, p2_ser, p2)
        sqrt_rr = np.sqrt(rho[:, None] * rho[None, :])                      # (N, N)
        inner = (p1[:, :, None] * Q0[None, :, :]
                 + p2[:, :, None] * Qpp[None, None, :])                     # (nk, N, N)
        Qhat = -sqrt_rr[None, :, :] * np.exp(-s_arr[:, None, None]
                                             * lam[None, :, :]) * inner
        Qhat += np.eye(self.N)[None, :, :]
        return Qhat

    # ------------------------------------------------------------------
    # Doubly-propagated pole/residue tables (bMulti four-term structure)
    # ------------------------------------------------------------------

    def _build_b_tables(self) -> None:
        """Fill _b_poles[i][j], _b_coeffs[i][j]: b_ij(s) = Σ_q coeffs_q/(s+poles_q).

        Four term classes per tail t ([LN] eq:bij; Lean bMulti):
          1. direct   K[t,i,j]                                  at z_t[i,j]
          2. second   Σ_n K[t,i,n]·A[t,i,n,j,n]                 at z_t[i,n]
          3. third    Σ_m K[t,m,j]·A[t,m,j,i,m]                 at z_t[m,j]
          4. fourth   Σ_{m,n} K[t,m,n]·A[t,m,n,i,m]·A[t,m,n,j,n] at z_t[m,n]
        A = bare hard-sphere propagator: A_prop[t,m,n,i,j] =
            (Q̂₀(z_tmn)⁻¹)_ij·e^{+z λ_ij} off-diagonal, (Q̂₀⁻¹)_ii − 1 diagonal.
        """
        N, nt = self.N, self.n_tails
        # Q̂₀⁻¹ and bare A at every pole slot
        A_prop = np.zeros((nt, N, N, N, N))
        for t in range(nt):
            for m in range(N):
                for n in range(N):
                    zv = float(self.z_poles[t, m, n])
                    if zv <= 0.0:
                        continue
                    Qinv = np.linalg.inv(self._build_Qhat(zv))
                    for i in range(N):
                        for j in range(N):
                            if i == j:
                                A_prop[t, m, n, i, i] = Qinv[i, i] - 1.0
                            else:
                                A_prop[t, m, n, i, j] = (
                                    Qinv[i, j] * math.exp(zv * self.lambda_ij[i, j])
                                )
        self.A_prop = A_prop

        K, zp = self.K, self.z_poles
        self._b_poles:  list = [[None] * N for _ in range(N)]
        self._b_coeffs: list = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                pole_map: dict = {}

                def _add(zv: float, c: float) -> None:
                    if zv > 0.0:
                        pole_map[zv] = pole_map.get(zv, 0.0) + c

                for t in range(nt):
                    _add(zp[t, i, j], K[t, i, j])
                    for nn in range(N):
                        _add(zp[t, i, nn], K[t, i, nn] * A_prop[t, i, nn, j, nn])
                    for mm in range(N):
                        _add(zp[t, mm, j], K[t, mm, j] * A_prop[t, mm, j, i, mm])
                    for mm in range(N):
                        for nn in range(N):
                            _add(zp[t, mm, nn],
                                 K[t, mm, nn] * A_prop[t, mm, nn, i, mm]
                                              * A_prop[t, mm, nn, j, nn])
                self._b_poles[i][j]  = np.array(list(pole_map.keys()))
                self._b_coeffs[i][j] = np.array(list(pole_map.values()))
        self._reduced_contact_sums()

    def _reduced_contact_sums(self) -> None:
        """Σ_q b-coeffs_ij divided by √(ρᵢρⱼ) — computed, never divided out.

        `get_g_contact` needs (Σ_q coeffs)/(2π√(ρᵢρⱼ)R).  Both parts vanish when a
        component has zero density, and the ratio is finite -- the √ρ is only the
        symmetric normalisation H = D^{1/2} h D^{1/2}, not physics.  Evaluated as
        written it is 0/0, which is what makes `FMSAFreeEnergy.get_excess_free_energy`
        fail on the ρ → 0 end of its own λ-integration path.

        The factor comes out analytically.  With Q̂₀ = I − D^{1/2}M D^{1/2},

            Q̂₀⁻¹ − I = D^{1/2}·Ñ·D^{1/2},      Ñ = (I − M·D)⁻¹·M

        (verified against (Q̂₀⁻¹−I)/√(ρᵢρⱼ) to 2×10⁻¹⁵), so the bare propagator is
        A_ij = √(ρᵢρⱼ)·Ñ_ij·e^{zλ_ij} for every i, j including i = j.  Feeding that
        and K̃ = K/√(ρᵢρⱼ) = 2πβ·k̄ into the four term classes leaves an explicit
        ρ on each internal summation index and nothing under a root:

            direct   K̃_ij
            second   Σ_n K̃_in·Ã_jn·ρ_n
            third    Σ_m K̃_mj·Ã_im·ρ_m
            fourth   Σ_{m,n} K̃_mn·Ã_im·Ã_jn·ρ_m·ρ_n

        This runs alongside the existing table rather than replacing it, so no
        number this class already produced can move.
        """
        N, nt = self.N, self.n_tails
        rho, zp = self.rho, self.z_poles
        Kr = 2.0 * np.pi * self.beta * self.k_bare          # K with √(ρᵢρⱼ) removed
        eye = np.eye(N)
        A_red = np.zeros((nt, N, N, N, N))
        for t in range(nt):
            for m in range(N):
                for n in range(N):
                    zv = float(zp[t, m, n])
                    if zv <= 0.0:
                        continue
                    M = self._build_M(zv)
                    Ntil = np.linalg.solve(eye - M * rho[None, :], M)
                    A_red[t, m, n] = Ntil * np.exp(zv * self.lambda_ij)
        red = np.zeros((N, N))
        self._b_coeffs_red: list = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                pmap: dict = {}

                def _addr(zv: float, c: float) -> None:
                    if zv > 0.0:
                        pmap[zv] = pmap.get(zv, 0.0) + c

                for t in range(nt):
                    _addr(zp[t, i, j], Kr[t, i, j])
                    for nn in range(N):
                        _addr(zp[t, i, nn],
                              Kr[t, i, nn] * A_red[t, i, nn, j, nn] * rho[nn])
                    for mm in range(N):
                        _addr(zp[t, mm, j],
                              Kr[t, mm, j] * A_red[t, mm, j, i, mm] * rho[mm])
                    for mm in range(N):
                        for nn in range(N):
                            _addr(zp[t, mm, nn],
                                  Kr[t, mm, nn] * A_red[t, mm, nn, i, mm]
                                  * A_red[t, mm, nn, j, nn] * rho[mm] * rho[nn])
                # keep the same pole ordering as the un-reduced table
                self._b_coeffs_red[i][j] = np.array(
                    [pmap.get(float(z), 0.0) for z in self._b_poles[i][j]])
                red[i, j] = float(sum(pmap.values()))
        self._b_csum_red = red

    # -- de-normalised kernels and closed form, built only if a density is zero
    def _P_kernel_red(self, i: int, m: int) -> list:
        """P_im with the √(ρᵢρ_m) prefactor removed."""
        c = self._q_coeffs(i, m)
        poly = 2.0 * np.pi * np.array([c[0], -c[1], c[2]])
        return [(-float(self.R_ij[i, m]), -float(self.lambda_ij[i, m]),
                 {0.0: poly})]

    def _B_kernel_red(self, m: int, n: int) -> list:
        """ℬ_mn with the √(ρ_mρ_n) carried by its coefficients removed."""
        R = float(self.R_ij[m, n])
        terms = {float(z): np.array([float(c) * math.exp(float(z) * R)])
                 for z, c in zip(self._b_poles[m][n], self._b_coeffs_red[m][n])}
        return [(R, np.inf, terms)]

    def _ensure_W_red(self) -> None:
        """Lazily build 𝒲/√(ρᵢρⱼ), the closed form a zero density needs.

        Same four terms as `_build_closed_form`; every √ρ that cancelled against
        the 1/√(ρᵢρⱼ) of the definition is gone, and what the cancellation leaves
        behind is an explicit ρ on each internal index -- exactly as in
        `_reduced_contact_sums`.  Built on demand because it costs as much as the
        ordinary closed form, and only a state with an empty component needs it.
        """
        if getattr(self, '_W_red', None) is not None:
            return
        N, rho = self.N, self.rho
        Wr: list = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                W = self._B_kernel_red(i, j)
                for n in range(N):
                    W = px_scale_add(W, px_convolve(self._B_kernel_red(i, n),
                                                    self._P_kernel_red(j, n)),
                                     -float(rho[n]))
                for m in range(N):
                    W = px_scale_add(W, px_convolve(self._P_kernel_red(i, m),
                                                    self._B_kernel_red(m, j)),
                                     -float(rho[m]))
                for m in range(N):
                    for n in range(N):
                        W = px_scale_add(
                            W, px_convolve(px_convolve(self._P_kernel_red(i, m),
                                                       self._B_kernel_red(m, n)),
                                           self._P_kernel_red(j, n)),
                            float(rho[m]) * float(rho[n]))
                Wr[i][j] = W
        self._W_red = Wr

    # ------------------------------------------------------------------
    # One-time spectral assembly → c^(1) lookup table
    # ------------------------------------------------------------------

    def _kspace_C1t_F(self, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Closed-form k-space assembly at arbitrary k > 0.

        Returns
        -------
        C1t : (nk, N, N) — symmetrized 3D sine transform C̃₁(k) of c^(1)
        F   : (nk, N, N) — I − C̃₀(k) (Baxter HS factorization, real)
        """
        N = self.N
        n_k = len(k)
        s  = 1j * k

        # B₁(k)_ij = e^{−ikR_ij}·b_ij(ik)
        B1 = np.zeros((n_k, N, N), dtype=complex)
        for i in range(N):
            for j in range(N):
                b = np.zeros(n_k, dtype=complex)
                for z_q, c_q in zip(self._b_poles[i][j], self._b_coeffs[i][j]):
                    b += c_q / (s + z_q)
                B1[:, i, j] = np.exp(-1j * k * self.R_ij[i, j]) * b

        Q  = self._build_Qhat_vec(s)                     # Q̂₀(ik)
        QT_inv = np.linalg.inv(np.transpose(Q, (0, 2, 1)))
        Q_inv  = np.linalg.inv(Q)
        H1hat = np.einsum('kij,kjl,klm->kim', QT_inv, B1, Q_inv)

        H1t = -2.0 * np.imag(H1hat) / k[:, None, None]   # 3D sine transform of h₁
        F = np.einsum('kij,klj->kil', Q, np.conj(Q)).real   # I − C̃₀
        C1t = np.einsum('kij,kjl,klm->kim', F, H1t, F)
        return C1t, F

    def get_c1hat_k(self, k) -> np.ndarray:
        """Closed-form ĉ^(1)_ij(k) (3D transform of the full first-order DCF,
        inner + MSA tail) at arbitrary wavevector(s) — NO grid, NO iteration.

        Returns shape (N, N) for scalar k, else (len(k), N, N).
        """
        k_arr = np.atleast_1d(np.asarray(k, dtype=float))
        C1t, _ = self._kspace_C1t_F(k_arr)
        pref = np.sqrt(self.rho[:, None] * self.rho[None, :])
        out = C1t / pref[None, :, :]
        return out[0] if np.isscalar(k) or np.ndim(k) == 0 else out

    def get_chat_k(self, k) -> np.ndarray:
        """Closed-form TOTAL ĉ_ij(k) = ĉ^HS-PY_ij(k) + ĉ^(1)_ij(k) at arbitrary k.

        Fully analytic (PY/Baxter HS reference from the same factorization:
        ĉ₀ = (I − F)/√(ρᵢρⱼ)).  This is 𝓕[c](k) with no radial grid anywhere.
        Returns shape (N, N) for scalar k, else (len(k), N, N).
        """
        k_arr = np.atleast_1d(np.asarray(k, dtype=float))
        C1t, F = self._kspace_C1t_F(k_arr)
        pref = np.sqrt(self.rho[:, None] * self.rho[None, :])
        out = (np.eye(self.N)[None, :, :] - F + C1t) / pref[None, :, :]
        return out[0] if np.isscalar(k) or np.ndim(k) == 0 else out

    def get_c1_exact(self, r: float, i: int, j: int,
                     epsabs: float = 1e-9, epsrel: float = 1e-8,
                     check: bool = True, return_err: bool = False):
        """Grid-free c^(1)_ij(r): adaptive oscillatory quadrature of the
        closed-form ĉ^(1)(k) — QUADPACK QAWF, no table, no DST.

            c^(1)(r) = 1/(2π²r) ∫₀^∞ k·ĉ^(1)_ij(k)·sin(kr) dk

        ⚠⚠ **NOT the high-accuracy path, despite what this docstring used to claim
        ("free of any radial-grid or Gibbs artifacts; use ... for isolated
        high-accuracy evaluations").**  `c^(1)` has a BREAKPOINT at `r = R_ij`
        (Group BRK), so `ĉ^(1)(k)` decays too slowly for QAWF's tail extrapolation
        just inside the core edge.  Measured against the exact closed form
        (`get_c1_inner`) at `ρ = 0.4452`, `T* = 0.95`, `z = 1.8`:

            r      closed form      QAWF          diff       QUADPACK abserr
            0.30   1.078072         1.078072      −8e-12     1.2e-10
            0.97   0.752922         0.751352      −1.6e-3    6.4e-2
            0.99   0.736003         0.739690      +3.7e-3    3.6e-2
            1.00   1.052632         0.889980      −0.163     8.3e-5   (AT the break)
            2.50   0.028297         0.028297      −1e-13     4.6e-12

        Away from the core edge it is excellent (1e-12); in `0.9 ≲ r < R_ij` it is
        wrong in the third or fourth digit, and AT the breakpoint by 15 %.

        ⚠⚠ **The old version discarded QUADPACK's error estimate** (`val, _ = quad(...)`),
        so those failures were silent even though the routine reports them — abserr
        comes back as `3e-2 … 6e-2` against a requested `1e-9`.  It is now checked, and
        `check=True` warns.  ⚠ At `r = R_ij` exactly the estimate is NOT reliable either
        (`8.3e-5` while the true error is `0.163`), because there the function jumps and
        the "value" is one-sided — use `get_c1_inner`, which returns the limit.

        ⇒ **Use `get_c1_inner` for production**; it is the exact closed form (finite sum
        of polynomial × exponential, no quadrature at all).  Keep this one as an
        INDEPENDENT cross-check of that closed form, which is what it is good for — the
        two agree to `1e-12` wherever QAWF converges.
        """
        from scipy.integrate import quad

        pref = math.sqrt(self.rho[i] * self.rho[j])

        def integrand(kk: float) -> float:
            kk = max(kk, 1e-8)     # H̃₁ = −2Im(Ĥ₁)/k has a removable 0/0 at k=0
            C1t, _ = self._kspace_C1t_F(np.array([kk]))
            return kk * C1t[0, i, j] / pref

        val, abserr = quad(integrand, 0.0, np.inf, weight='sin', wvar=r,
                           epsabs=epsabs, epsrel=epsrel, limlst=200, limit=400)
        scale = 2.0 * np.pi**2 * r
        out, err = val / scale, abserr / scale
        if check:
            tol = max(epsabs, epsrel * abs(out))
            if not (err <= 1e3 * tol):
                warnings.warn(
                    f"get_c1_exact: QAWF did not converge at r={r:.6g} "
                    f"(abserr {err:.2e} vs tol {tol:.2e}); c^(1) has a breakpoint at "
                    f"R_ij and the Fourier tail is slow there. Use get_c1_inner.",
                    RuntimeWarning, stacklevel=2)
        return (out, err) if return_err else out

    # ------------------------------------------------------------------
    # Explicit closed form for c^(1) (no grid, no quadrature)
    # ------------------------------------------------------------------

    def _P_kernel(self, i: int, m: int) -> list:
        """P_im(u) = 2π√(ρᵢρ_m)·q_im(−u)  on u ∈ [−R_im, −λ_im]  (quadratic)."""
        c = self._q_coeffs(i, m)
        pref = 2.0 * np.pi * math.sqrt(self.rho[i] * self.rho[m])
        poly = pref * np.array([c[0], -c[1], c[2]])     # q(−u) = c0 − c1u + c2u²
        return [(-float(self.R_ij[i, m]), -float(self.lambda_ij[i, m]),
                 {0.0: poly})]

    def _B_kernel(self, m: int, n: int) -> list:
        """ℬ_mn(v) = Σ_q c_q·e^{−z_q(v−R_mn)}  on v ∈ [R_mn, ∞)."""
        R = float(self.R_ij[m, n])
        terms = {float(z): np.array([float(c) * math.exp(float(z) * R)])
                 for z, c in zip(self._b_poles[m][n], self._b_coeffs[m][n])}
        return [(R, np.inf, terms)]

    def _build_closed_form(self) -> None:
        """Explicit closed form for c^(1) via the real-space triple convolution.

        Ĉ₁ = Q̂₀(−k)·B₁(k)·Q̂₀ᵀ(−k)  (no Q̂₀ inverses — see the module docstring)
        transforms to  𝒲 = 𝒬⁻ ⋆ ℬ ⋆ (𝒬⁻)ᵀ  with 𝒬⁻_ij(t) = δ_ijδ(t) − P_ij(t).
        Expanding the two deltas gives four terms:

            𝒲_ij = ℬ_ij − Σ_n [ℬ_in ⋆ P_jn] − Σ_m [P_im ⋆ ℬ_mj]
                        + Σ_{m,n} [P_im ⋆ ℬ_mn ⋆ P_jn]

        𝒲 is supported on [−R_ij, ∞) while 𝒞_ij(r) := 2π√(ρᵢρⱼ)·r·c^(1)_ij(r)
        is one-sided.  The k-space identity pins only the odd part
        (𝒲(r) − 𝒲(−r) = 𝒞(r) − 𝒞(−r)), so for r > 0, where 𝒞(−r) = 0:

            𝒞_ij(r) = 𝒲_ij(r) − 𝒲_ij(−r)                                   (†)

        (†) automatically reproduces the bare MSA tail for r > R_ij, where
        𝒲(−r) = 0.  Everything is a finite sum of (polynomial × exponential)
        on finitely many pieces — the exponential rates are exactly the Yukawa
        z_q, and the piece edges are the support-overlap breakpoints (λ_ij
        first among them).  No HS poles appear anywhere.
        """
        N = self.N
        self._W: list = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                W = self._B_kernel(i, j)                                  # T1
                for n in range(N):                                        # T2
                    W = px_scale_add(W, px_convolve(self._B_kernel(i, n),
                                                    self._P_kernel(j, n)), -1.0)
                for m in range(N):                                        # T3
                    W = px_scale_add(W, px_convolve(self._P_kernel(i, m),
                                                    self._B_kernel(m, j)), -1.0)
                for m in range(N):                                        # T4
                    for n in range(N):
                        W = px_scale_add(
                            W, px_convolve(px_convolve(self._P_kernel(i, m),
                                                       self._B_kernel(m, n)),
                                           self._P_kernel(j, n)), +1.0)
                self._W[i][j] = W

    def closed_form_pieces(self, i: int, j: int) -> list:
        """The explicit closed form of 𝒲_ij as [(a, b, {rate: poly_coeffs})].

        c^(1)_ij(r) = [𝒲_ij(r) − 𝒲_ij(−r)] / (2π√(ρᵢρⱼ)·r)   for r > 0.
        """
        return self._W[i][j]

    # ------------------------------------------------------------------
    # Public DCF interface (sc_checks duck-type + dense-grid)
    # ------------------------------------------------------------------

    def get_c1_inner(self, r: float, i: int, j: int) -> float:
        """First-order MSA correction c^(1)_ij(r), from the explicit closed form.

        c^(1)_ij(r) = [𝒲_ij(r) − 𝒲_ij(−r)] / (2π√(ρᵢρⱼ)·r),  see `_build_closed_form`.
        Exact — no grid, no quadrature.  Valid for all r > 0 (for r > R_ij it
        returns the bare MSA tail automatically).
        """
        if r <= 0.0:
            r = 1e-12
        rr = math.sqrt(self.rho[i] * self.rho[j])
        if rr > 0.0:
            W = self._W[i][j]
            return (px_eval(W, r) - px_eval(W, -r)) / (2.0 * np.pi * rr * r)
        # an empty component makes this 0/0; the de-normalised form is the limit
        self._ensure_W_red()
        W = self._W_red[i][j]
        return (px_eval(W, r) - px_eval(W, -r)) / (2.0 * np.pi * r)

    def _c1_pieces(self, i: int, j: int) -> tuple[list, float]:
        """𝒞_ij = 𝒲(r) − 𝒲(−r) on r ∈ (0, R_ij], and its 1/(2π√(ρᵢρⱼ)) divisor.

        Returns (pieces, denom) with c^(1)_ij(r) = px_eval(pieces, r)/(denom·r),
        i.e. exactly what `get_c1_inner` evaluates pointwise — but as a closed
        form the px algebra can multiply and integrate.
        """
        rr = math.sqrt(self.rho[i] * self.rho[j])
        if rr > 0.0:
            W, denom = self._W[i][j], 2.0 * np.pi * rr
        else:
            self._ensure_W_red()
            W, denom = self._W_red[i][j], 2.0 * np.pi
        R = float(self.R_ij[i, j])
        C = px_scale_add(px_clip(W, 0.0, R),
                         px_clip(px_reflect(W), 0.0, R), -1.0)
        return C, denom

    def c1_incore_moment(self, i: int, j: int, n: int = 2) -> float:
        """∫₀^{R_ij} rⁿ·c^(1)_ij(r) dr in closed form — no quadrature.

        Since r·c^(1) is the piecewise (polynomial × exponential) object 𝒞/2π√ρρ,
        the integrand rⁿ⁻¹·𝒞(r) stays in that class and `px_integrate` closes it.

        n = 2 is the moment the free energy and ĉ(0) both want: the in-core part
        of ĉ^(1)_ij(0) = 4π∫₀^∞ r²c^(1)dr.  n ≥ 1 only — rⁿ⁻¹ must be polynomial.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1 (r^(n-1) must be polynomial), got {n}")
        C, denom = self._c1_pieces(i, j)
        pw = np.zeros(n); pw[n - 1] = 1.0            # r^(n−1), increasing order
        return px_integrate(px_mul_poly(C, pw), 0.0, float(self.R_ij[i, j])) / denom

    def c1_incore_sq_moment(self, i: int, j: int) -> float:
        """∫₀^{R_ij} r²·[c^(1)_ij(r)]² dr in closed form — no quadrature.

        The second-order companion of `c1_incore_moment`.  It exists because the
        PY closure pins the FIRST-order cavity function the same way it pins the
        zeroth-order one: c = f·y expanded to first order is c₁ = f·y₁ + f₁·y₀,
        and inside a hard core f = −1 while f₁ = 0 (the perturbation does not
        move the core), so

            y¹_ij(r) = −c^(1)_ij(r),     r < R_ij

        exactly as y⁰ = −c^HS-PY.  The second-order energy-route term therefore
        carries c^(1) twice — once as the effective potential, once as the
        weight — and is this squared moment.

        r²c² = 𝒞²/(2π√(ρᵢρⱼ))², and 𝒞² stays a piecewise polynomial × exponential
        (`px_mul`: polynomials multiply, rates add), so `px_integrate` closes it.
        """
        C, denom = self._c1_pieces(i, j)
        return px_integrate(px_mul(C, C), 0.0,
                            float(self.R_ij[i, j])) / (denom * denom)

    def incore_breakpoints(self, i: int, j: int) -> np.ndarray:
        """Every kink of c^(1)_ij and c^HS-PY_ij inside (0, R_ij), sorted.

        Panel edges for any quadrature over the core: between consecutive
        breakpoints both DCFs are a single smooth (polynomial × exponential),
        so fixed-order Gauss-Legendre per panel is effectively exact, while a
        quadrature that straddles a kink is not.

        Sources: the closed-form piece edges of 𝒲_ij and of its reflection
        (c^(1)); λ_ij (where q'_ij switches on and q'_ji(−r) switches off) and,
        from `_conv_int`'s lo/hi crossings, λ_im−λ_jm, R_im−R_jm, λ_im−R_jm and
        R_im−λ_jm for every internal species m (c^HS-PY).
        """
        R = float(self.R_ij[i, j])
        bps = {float(self.lambda_ij[i, j])}
        for (a, b, _) in self._W[i][j]:
            for e in (a, b):
                if np.isfinite(e):
                    bps.add(abs(float(e)))
        lam, Rm = self.lambda_ij, self.R_ij
        for m in range(self.N):
            bps.update((float(lam[i, m] - lam[j, m]), float(Rm[i, m] - Rm[j, m]),
                        float(lam[i, m] - Rm[j, m]), float(Rm[i, m] - lam[j, m])))
        # `px_convolve` rounds its output edges to 10 decimals, so an edge that
        # is analytically R_ij arrives as R_ij ± 1e-10 -- two entries straddling
        # the endpoint rather than one at it.  Merge on an absolute tolerance
        # scaled to that noise, and drop whatever lands on either endpoint.
        tol = 1e-9 * max(R, 1.0)
        inner: list[float] = []
        for b in sorted(bps):
            if b <= tol or b >= R - tol:
                continue
            if not inner or b - inner[-1] > tol:
                inner.append(b)
        return np.array(inner, dtype=float)

    # ------------------------------------------------------------------
    # Closed-form PY (Baxter/Lebowitz) hard-sphere mixture DCF
    # ------------------------------------------------------------------

    def _q_coeffs(self, i: int, j: int) -> np.ndarray:
        """Baxter real-space q_ij(t): increasing-order polynomial coeffs in t.

        Inverting the Laplace form used by `_build_Qhat` (φ₁ ↔ (r−R_ij),
        φ₂ ↔ (r−R_ij)²/2 — both verified exactly):

            q_ij(t) = (1/2π)·[Q₀_ij·(t − R_ij) + Qpp_j·(t − R_ij)²/2],
                      λ_ij ≤ t ≤ R_ij,  0 otherwise

        so that Q̂₀_ij(s) = δ_ij − 2π√(ρᵢρⱼ)·∫_{λ_ij}^{R_ij} e^{−st} q_ij(t) dt.
        q is NOT symmetric: supp q_ij = [λ_ij, R_ij] has width σ_i.
        """
        A = float(self.Q0[i, j])
        B = float(self.Qpp[j])
        R = float(self.R_ij[i, j])
        tp = 1.0 / (2.0 * np.pi)
        return np.array([tp * (-A * R + 0.5 * B * R**2),   # t⁰
                         tp * (A - B * R),                  # t¹
                         tp * 0.5 * B])                     # t²

    def _qprime_coeffs(self, i: int, j: int) -> np.ndarray:
        """d/dt q_ij(t): increasing-order coeffs in t."""
        c = self._q_coeffs(i, j)
        return np.array([c[1], 2.0 * c[2]])

    def _q_eval(self, i: int, j: int, t: float) -> float:
        """q_ij(t), zero outside the support [λ_ij, R_ij]."""
        if t < float(self.lambda_ij[i, j]) or t > float(self.R_ij[i, j]):
            return 0.0
        c = self._q_coeffs(i, j)
        return float(c[0] + c[1] * t + c[2] * t**2)

    def q_jump(self, i: int, j: int) -> float:
        """The jump of q_ij at the inner support edge:  q_ij(λ_ij⁺) = −σ_iσ_j/(2Δ).

        q_ij is DISCONTINUOUS at t = λ_ij (it is zero below and
        (1/2π)[−Q₀_ij σ_i + Qpp_j σ_i²/2] above), so q'_ij carries a Dirac
        delta  q_ij(λ_ij)·δ(t − λ_ij)  on top of the polynomial derivative.
        The ξ₂-dependent parts cancel exactly, leaving the symmetric closed
        form −σ_iσ_j/(2Δ) — symmetry is what makes the deltas of the
        −q'_ij(r) and +q'_ji(−r) terms of Baxter's relation cancel each other
        (so c(r) itself has no delta), while the delta inside the convolution
        term survives and must be kept.
        """
        return -float(self.sigma[i] * self.sigma[j]) / (2.0 * self.vac)

    def _qprime_eval(self, i: int, j: int, t: float) -> float:
        """Polynomial part of q'_ij(t) (delta at λ_ij handled separately)."""
        if t < float(self.lambda_ij[i, j]) or t > float(self.R_ij[i, j]):
            return 0.0
        d = self._qprime_coeffs(i, j)
        return float(d[0] + d[1] * t)

    def _conv_int(self, i: int, j: int, m: int, r: float) -> float:
        """∫ q_im(t)·q'_jm(t−r) dt, including the delta part of q'_jm.

        Polynomial part: supp q_im(t) = [λ_im, R_im] ∩ supp q'_jm(t−r) =
        [r+λ_jm, r+R_jm]; the integrand is a cubic in t — integrated exactly.
        Delta part: q_jm(λ_jm)·δ(t − r − λ_jm) picks out q_im(r + λ_jm).
        """
        val = 0.0

        lo = max(float(self.lambda_ij[i, m]), r + float(self.lambda_ij[j, m]))
        hi = min(float(self.R_ij[i, m]),      r + float(self.R_ij[j, m]))
        if hi > lo:
            c = self._q_coeffs(i, m)          # q_im(t)  = c0 + c1 t + c2 t²
            e = self._qprime_coeffs(j, m)     # q'_jm(u) = e0 + e1 u,  u = t − r
            f0 = e[0] - e[1] * r              # q'_jm(t−r) = f0 + e1 t
            f1 = e[1]
            p = np.array([c[0] * f0,                 # cubic product, incr. order
                          c[0] * f1 + c[1] * f0,
                          c[1] * f1 + c[2] * f0,
                          c[2] * f1])
            for n in range(4):
                val += p[n] * (hi**(n + 1) - lo**(n + 1)) / (n + 1)

        # delta of q'_jm at its inner edge λ_jm
        val += self.q_jump(j, m) * self._q_eval(i, m, r + float(self.lambda_ij[j, m]))
        return float(val)

    def get_HS_PY(self, r: float, i: int, j: int) -> float:
        """Closed-form PY (Baxter/Lebowitz) hard-sphere mixture DCF c^HS-PY_ij(r).

        Baxter's real-space relation, obtained by inverting the factorization
        I − C̃₀(k) = Q̂₀(ik)·Q̂₀ᵀ(−ik) term by term (the same factorization the
        first-order assembly uses):

            r·c_ij(r) = −q'_ij(r) + q'_ji(−r)
                        − 2π Σ_m ρ_m ∫ q_im(t)·q'_jm(t−r) dt

        Fully analytic — no grid, no quadrature (the q's are quadratics, so the
        convolution integrand is a cubic, integrated in closed form).  The
        q'_ji(−r) term supplies the r < λ_ij piece of the classic Lebowitz
        piecewise structure; c vanishes identically for r > R_ij.

        This is the reference c^(1) formally perturbs around, so
        c^HS-PY + c^(1) + MSA tail reproduces converged OZ+MSA to first order.
        """
        if r <= 0.0:
            r = 1e-12
        # ⚠ The two direct terms TILE the core: -q'_ij(r) on r >= λ_ij, q'_ji(-r) on
        # r < λ_ij (λ_ji = -λ_ij).  Written as one half-open split over r, so a double
        # count at the knot cannot arise -- it used to, whenever λ_ij > 0: -12.28 against
        # -7.26 on either side at the reference binary.  c is continuous there (a C0 kink).
        # The half-openness lives HERE and not in `_qprime_eval`: the knot is the LOWER end
        # of both terms' supports in t (t = r for one, t = -r for the other), so no single
        # convention on q' counts exactly one of them.  `_qprime_eval` keeps its closed
        # support for the convolution.  Mirrors the C++ port (cpp main 3e79a04).
        lam = float(self.lambda_ij[i, j])
        val = -self._qprime_eval(i, j, r) if r >= lam else self._qprime_eval(j, i, -r)
        for m in range(self.N):
            val -= 2.0 * np.pi * float(self.rho[m]) * self._conv_int(i, j, m, r)
        return val / r

    def get_HS_FMT(self, r: float, i: int, j: int) -> float:
        """White Bear FMT hard-sphere mixture pair DCF c^HS_ij(r), closed form.

        Built directly from the literature definitions (Roth, Evans, Lang & Kahl
        2002; weights and Φ as stated e.g. in arXiv:2508.14642):

            ω₀ = δ(R−r)/(4πR²),  ω₁ = δ(R−r)/(4πR),  ω₂ = δ(R−r),
            ω₃ = Θ(R−r),         ω_v2 = r̂·δ(R−r),    ω_v1 = ω_v2/(4πR)
            n₀ = Σρᵢ,  n₁ = Σρᵢ Rᵢ,  n₂ = Σρᵢ·4πRᵢ²,  n₃ = Σρᵢ(4π/3)Rᵢ³ = η
            Φ = −n₀ln(1−n₃) + (n₁n₂ − n_v1·n_v2)/(1−n₃)
                + (n₂³ − 3n₂ n_v2·n_v2)·φ(n₃)
            c_ij(r) = −Σ_αβ (∂²Φ/∂n_α∂n_β)·(ω_α^i ⊛ ω_β^j)(r)

        The vector weights satisfy ŵ_v2(k) = −i·k·ŵ₃(k), so their convolution is
        −∇²(ω₃^i ⊛ ω₃^j) — no separate geometry is needed.  The scalar weight
        convolutions are elementary and piecewise, with the inner break at
        |Rᵢ−Rⱼ| = |λ_ij| (the smaller sphere entirely inside the larger).

        Verified (2026-07-17): −ĉ_ij(0) = ∂²Φ/∂ρᵢ∂ρⱼ for every mixture pair
        (~1e-5, quadrature noise); one-component 1 − ρĉ(0) reproduces
        Carnahan–Starling (1+4η+4η²−4η³+η⁴)/(1−η)⁴ to 0.0000% at every η;
        c(r) → −1 as ρ → 0.

        [Corrected 2026-07-17.  The previous χ₀..χ₃/χ₂₂ formula (inherited
        verbatim from FMSA_GA_matrix_mix, FMSA_pure and FMSA_MC_cleaned_2cpp)
        used n₂ = π Σρσ²/4 — a factor 4 below the literature n₂ = Σρ·4πR² —
        with χ-coefficients that did not compensate: it undershot the
        Carnahan–Starling compressibility by 8.8% at η=0.10 and 51% at η=0.42.]
        """
        Ri = float(self.sigma[i]) / 2.0
        Rj = float(self.sigma[j]) / 2.0
        rho, R = self.rho, self.sigma / 2.0

        n0 = float(np.sum(rho))
        n1 = float(np.sum(rho * R))
        n2 = float(np.sum(rho * 4.0 * np.pi * R**2))
        n3 = float(np.sum(rho * (4.0 * np.pi / 3.0) * R**3))     # = η
        v  = 1.0 - n3
        (A, A1, A2), (ph, ph1, ph2) = _fmt_A_phi(n3, self.hs_fmt)

        # Phi_ab := ∂²Φ/∂n_a∂n_b, the Hessian of Φ at the bulk
        # (vector densities vanish there); subscripts are weight indices.
        Phi03 = 1.0 / v
        Phi12 = A
        Phi13 = n2 * A1
        Phi23 = n1 * A1 + 3.0 * n2**2 * ph1
        Phi22 = 6.0 * n2 * ph
        Phi33 = n0 / v**2 + n1 * n2 * A2 + n2**3 * ph2
        Phi_v1v2 = -A
        Phi_v2v2 = -6.0 * n2 * ph

        lam = abs(Ri - Rj)
        S   = Ri + Rj
        if r >= S:
            return 0.0
        if r < lam:
            # smaller sphere entirely inside the larger one
            A    = 0.0
            Bij  = 4.0 * np.pi * Ri**2 if Ri < Rj else 0.0
            Bji  = 4.0 * np.pi * Rj**2 if Rj < Ri else 0.0
            V    = (4.0 * np.pi / 3.0) * min(Ri, Rj)**3
            lapV = 0.0
        else:
            Dm   = (Ri - Rj)**2
            A    = 2.0 * np.pi * Ri * Rj / r                       # ω₂ⁱ ⊛ ω₂ʲ
            Bij  = np.pi * Ri * (Rj**2 - (r - Ri)**2) / r          # ω₂ⁱ ⊛ ω₃ʲ
            Bji  = np.pi * Rj * (Ri**2 - (r - Rj)**2) / r          # ω₃ⁱ ⊛ ω₂ʲ
            V    = np.pi * (S - r)**2 * (r**2 + 2.0*r*S - 3.0*Dm) / (12.0 * r)
            # ∇²V = (1/r)·d²(rV)/dr² with rV = (π/12)·p(r),
            # p = r⁴ − 3(S²+Dm)r² + (2S³+6S·Dm)r − 3S²Dm  ⟹ p″ = 12r² − 6(S²+Dm)
            lapV = (np.pi / 12.0) * (12.0 * r**2 - 6.0 * (S**2 + Dm)) / r

        scal = (Phi03 * (Bij / (4.0*np.pi*Ri**2) + Bji / (4.0*np.pi*Rj**2))
                + Phi12 * (A / (4.0*np.pi*Ri) + A / (4.0*np.pi*Rj))
                + Phi13 * (Bij / (4.0*np.pi*Ri) + Bji / (4.0*np.pi*Rj))
                + Phi23 * (Bij + Bji) + Phi22 * A + Phi33 * V)
        vec = lapV * (Phi_v1v2 * (1.0/(4.0*np.pi*Ri) + 1.0/(4.0*np.pi*Rj)) + Phi_v2v2)
        return float(-scal + vec)

    def get_c_total(self, r_grid: np.ndarray, hs_ref: str = 'fmt') -> np.ndarray:
        """Total DCF c_ij(r) on a radial grid → shape (N, N, len(r_grid)).

        Inner (r < R_ij): c^HS + c^(1) (spectral table).
        Outer (r ≥ R_ij): strict MSA tail  Σ_t K_t e^{−z_t(r−R)} / (2π√(ρᵢρⱼ) r).

        hs_ref : 'fmt' (default) — White Bear FMT reference (project standard,
                 more accurate HS thermodynamics vs GCMC);
                 'py' — Baxter/PY reference, the one c^(1) formally perturbs
                 around; with 'py', the total reproduces converged OZ+MSA to
                 first-order accuracy (1–2% in ĉ at the 2YK state point).
        """
        N = self.N
        r_grid = np.asarray(r_grid, dtype=float)
        c_total = np.zeros((N, N, len(r_grid)))
        for i in range(N):
            for j in range(N):
                rho_ij = math.sqrt(self.rho[i] * self.rho[j])
                R = float(self.R_ij[i, j])
                for q, rv in enumerate(r_grid):
                    if rv <= 0.0:
                        continue
                    if rv < R:
                        c_hs = (self.get_HS_FMT(rv, i, j) if hs_ref == 'fmt'
                                else self.get_HS_PY(rv, i, j))
                        c_total[i, j, q] = c_hs + self.get_c1_inner(rv, i, j)
                    else:
                        tail = 0.0
                        Kcol = (self.K if rho_ij > 0.0
                                else 2.0 * np.pi * self.beta * self.k_bare)
                        for t in range(self.n_tails):
                            zv = float(self.z_poles[t, i, j])
                            if zv > 0.0:
                                tail += float(Kcol[t, i, j]) * math.exp(-zv * (rv - R))
                        # with an empty component both K and the divisor carry the
                        # same √(ρᵢρⱼ); K was replaced by K/√(ρᵢρⱼ) just above, so
                        # the divisor drops it too and the ratio is the limit.
                        c_total[i, j, q] = tail / (2.0 * np.pi * rv
                                                   * (rho_ij if rho_ij > 0.0 else 1.0))
        return c_total

    # ==================================================================
    # RDF  —  two inversion routes out of ONE closed form
    # ==================================================================
    #
    # The analytic content is a single object.  Write
    #
    #     𝓗(s)_ij = [Q̂₀ᵀ(s)⁻¹ · B₁(s) · Q̂₀(s)⁻¹]_ij ,   B₁(s)_ij = e^{−sR_ij} b_ij(s)
    #
    # Then, in the symmetrised normalisation used throughout this class,
    #
    #     𝓗(s) = 2π√(ρᵢρⱼ) · L[r·h⁽¹⁾_ij(r)](s)                           (†)
    #
    # and the k-space quantity the DCF assembly already uses is the SAME
    # function evaluated on the imaginary axis:
    #
    #     h̃⁽¹⁾(k) = −2 Im 𝓗(ik) / k                                       (‡)
    #
    # so `_kspace_C1t_F` and the Laplace methods below are two views of one
    # closed form.  What differs is only the contour along which it is inverted
    # back to r:
    #
    #   route 'fourier'  — invert along the imaginary axis (s = ik): a sine
    #                      transform of ĥ(k).  Whole curve at once by DST-I.
    #   route 'laplace'  — invert along a vertical Bromwich line (s = a + iω,
    #                      a > 0) by the de Hoog Fourier-series ILT.  Pointwise
    #                      and grid-free: one r costs 2M+1 evaluations of (†).
    #
    # Two inversion methods that look natural here do NOT work, and both
    # failures are structural rather than numerical (see
    # numerical_notes/results/fmsa_dp_rdf_routes.md):
    #
    #   Stehfest (real axis only) cannot represent the contact jump or the
    #   packing oscillations — it returns a non-zero g inside the core.
    #   Talbot (contour deformed into Re s < 0) overflows, because the
    #   hard-core delay factor e^{−sR_ij} in B₁ grows without bound there.
    #   Any ILT for a hard-core fluid must stay in Re s ≥ 0.
    #
    # Both are implemented anyway, behind `method=`, because being able to
    # reproduce the failure is what makes the statement checkable.
    # ------------------------------------------------------------------

    def _b_of_s(self, s_arr: np.ndarray) -> np.ndarray:
        """B₁(s)_ij = e^{−sR_ij}·b_ij(s) at arbitrary complex s → (ns, N, N)."""
        N = self.N
        s_arr = np.atleast_1d(np.asarray(s_arr, dtype=complex))
        B1 = np.zeros((len(s_arr), N, N), dtype=complex)
        for i in range(N):
            for j in range(N):
                b = np.zeros(len(s_arr), dtype=complex)
                for z_q, c_q in zip(self._b_poles[i][j], self._b_coeffs[i][j]):
                    b += c_q / (s_arr + z_q)
                B1[:, i, j] = np.exp(-s_arr * self.R_ij[i, j]) * b
        return B1

    def _H_of_s(self, s_arr: np.ndarray) -> np.ndarray:
        """𝓗(s) = Q̂₀ᵀ(s)⁻¹·B₁(s)·Q̂₀(s)⁻¹ — the closed form of (†), complex s."""
        s_arr = np.atleast_1d(np.asarray(s_arr, dtype=complex))
        Q = self._build_Qhat_vec(s_arr)
        return np.einsum('kij,kjl,klm->kim',
                         np.linalg.inv(np.transpose(Q, (0, 2, 1))),
                         self._b_of_s(s_arr), np.linalg.inv(Q))

    # ------------------------------------------------------------------
    # k-space closed forms
    # ------------------------------------------------------------------

    def get_h1hat_k(self, k) -> np.ndarray:
        """Closed-form ĥ⁽¹⁾_ij(k), the first-order TCF — NO grid, NO iteration.

        Obtained from (‡); this is the object whose OZ image is ĉ⁽¹⁾, so
        `get_c1hat_k` and this method are algebraically tied by C̃₁ = F·H̃₁·F.
        """
        k_arr = np.atleast_1d(np.asarray(k, dtype=float))
        C1t, F = self._kspace_C1t_F(k_arr)
        Finv = np.linalg.inv(F)
        H1t = np.einsum('kij,kjl,klm->kim', Finv, C1t, Finv)
        pref = np.sqrt(self.rho[:, None] * self.rho[None, :])
        out = H1t / pref[None, :, :]
        return out[0] if np.isscalar(k) or np.ndim(k) == 0 else out

    def get_hhat_k(self, k, order: str = 'first') -> np.ndarray:
        """Closed-form TOTAL ĥ_ij(k) = ĥ⁰_ij(k) + ĥ⁽¹⁾_ij(k) at arbitrary k.

        ĥ⁰ is the PY/Baxter hard-sphere reference from the same factorisation,
        I + H̃₀ = F⁻¹ with F = I − C̃₀ — analytic, no iteration anywhere.

        order : 'first' (default) — H̃ = (F⁻¹ − I) + H̃₁, the first-order-consistent
                sum, i.e. the partner of `get_c_total(hs_ref='py')`;
                'oz' — H̃ = (F − C̃₁)⁻¹ − I, the full OZ resummation of the same
                ĉ.  The two differ at second order in the tail; 'oz' can amplify
                the first-order error near a spinodal, 'first' cannot.
        """
        k_arr = np.atleast_1d(np.asarray(k, dtype=float))
        C1t, F = self._kspace_C1t_F(k_arr)
        eye = np.eye(self.N)[None, :, :]
        if order == 'oz':
            Ht = np.linalg.inv(F - C1t) - eye
        elif order == 'first':
            Finv = np.linalg.inv(F)
            H1t = np.einsum('kij,kjl,klm->kim', Finv, C1t, Finv)
            Ht = (Finv - eye) + H1t
        else:
            raise ValueError(f"order must be 'first' or 'oz', got {order!r}")
        pref = np.sqrt(self.rho[:, None] * self.rho[None, :])
        out = Ht / pref[None, :, :]
        return out[0] if np.isscalar(k) or np.ndim(k) == 0 else out

    def get_S_k(self, k, order: str = 'first') -> np.ndarray:
        """Ashcroft–Langreth partial structure factors
        S_ij(k) = δ_ij + √(ρᵢρⱼ)·ĥ_ij(k), from the closed form above."""
        k_arr = np.atleast_1d(np.asarray(k, dtype=float))
        hk = self.get_hhat_k(k_arr, order=order)
        pref = np.sqrt(self.rho[:, None] * self.rho[None, :])
        out = np.eye(self.N)[None, :, :] + pref[None, :, :] * hk
        return out[0] if np.isscalar(k) or np.ndim(k) == 0 else out

    # ------------------------------------------------------------------
    # Laplace closed forms
    # ------------------------------------------------------------------

    def get_h1L_laplace(self, s) -> np.ndarray:
        """L[r·h⁽¹⁾_ij(r)](s) in closed form, from (†).  Complex s allowed.

        Verified against a numerical Laplace transform of the Fourier-route
        h⁽¹⁾(r) to 4–5 digits at the reference state.
        """
        s_arr = np.atleast_1d(np.asarray(s, dtype=complex))
        pref = 2.0 * np.pi * np.sqrt(self.rho[:, None] * self.rho[None, :])
        out = self._H_of_s(s_arr) / pref[None, :, :]
        return out[0] if np.isscalar(s) or np.ndim(s) == 0 else out

    def get_g0L_laplace(self, s) -> np.ndarray:
        """L[r·g⁰_ij(r)](s) for the PY/Baxter hard-sphere reference mixture.

            L[r g⁰_ij](s) = e^{−sR_ij}/(vac·detQ̂₀(s)) ·
                            [ A/s² + (R_ij + πξ₂σᵢσⱼ/(4vac))/s
                              + (π/(2vac))Σ_m ρ_m φ₁(σ_m,s)(σ_m−σᵢ)(σ_m−σⱼ)/s ]

        with A = 1 + πξ₃/(2vac) and φ₁(σ,s) = (1 − sσ − e^{−sσ})/s².

        This is the classical Baxter mixture result; `detQ̂₀(s)` is taken
        directly from `_build_Qhat_vec`, which agrees with the explicit
        determinant expansion in FMSA_MC_cleaned_2cpp to 1e-15.

        NOTE the normalisation.  `FMSA_MC_cleaned_2cpp.get_g0ij_r` multiplies
        this by a further 2π√(ρᵢρⱼ), which makes its g⁰ tend to 2π√(ρᵢρⱼ)
        instead of 1 at large r; that routine is wrong on two counts and is
        superseded here (see the module note referenced above).
        """
        s_arr = np.atleast_1d(np.asarray(s, dtype=complex))
        sig, rho, vac = self.sigma, self.rho, self.vac
        s_c = s_arr[:, None]
        zs = s_c * sig[None, :]
        small = np.abs(zs) < 1e-3
        zs_safe = np.where(small, 1.0, zs)
        s_safe = np.where(small, 1.0, s_c * np.ones_like(zs))
        phi1 = (1.0 - zs_safe - np.exp(-zs_safe)) / s_safe**2
        phi1 = np.where(small, -sig[None, :]**2 / 2.0 + s_c * sig[None, :]**3 / 6.0, phi1)

        detQ = np.linalg.det(self._build_Qhat_vec(s_arr))            # (ns,)
        A = 1.0 + np.pi * self.xi[3] / (2.0 * vac)
        term1 = R = self.R_ij
        quad = (np.pi * self.xi[2] / (4.0 * vac)) * sig[:, None] * sig[None, :]
        # Σ_m ρ_m φ₁(σ_m,s)·(σ_m−σᵢ)(σ_m−σⱼ)  — shape (ns, N, N)
        mix = np.einsum('m,km,mij->kij', rho, phi1,
                        (sig[:, None, None] - sig[None, :, None])
                        * (sig[:, None, None] - sig[None, None, :]))
        s3 = s_arr[:, None, None]
        num = (A / s3**2
               + (term1 + quad)[None, :, :] / s3
               + (np.pi / (2.0 * vac)) * mix / s3)
        out = np.exp(-s3 * R[None, :, :]) * num / (vac * detQ[:, None, None])
        return out[0] if np.isscalar(s) or np.ndim(s) == 0 else out

    def get_gL_laplace(self, s) -> np.ndarray:
        """L[r·g_ij(r)](s) = L[r·g⁰_ij](s) + L[r·h⁽¹⁾_ij](s) — the first-order
        RDF as one closed-form Laplace transform (g₁ ≡ h₁)."""
        s_arr = np.atleast_1d(np.asarray(s, dtype=complex))
        out = self.get_g0L_laplace(s_arr) + self.get_h1L_laplace(s_arr)
        return out[0] if np.isscalar(s) or np.ndim(s) == 0 else out

    # ------------------------------------------------------------------
    # route A — invert along the imaginary axis
    # ------------------------------------------------------------------

    def get_g_fourier(self, r_grid=None, k_max: float = 1500.0, n_k: int = 200000,
                      order: str = 'first', return_r: bool = False):
        """RDF g_ij(r) by inverting the closed-form ĥ(k) — the k-space route.

            h_ij(r) = 1/(2π²r) ∫₀^∞ k·ĥ_ij(k)·sin(kr) dk

        Evaluated as a DST-I on a uniform k grid, which gives the whole curve
        in one shot on the conjugate grid r_m = mπ/(n_k·Δk); if `r_grid` is
        given the result is interpolated onto it.

        The integrand decays only as 1/k (g has a jump at contact), so k_max is
        the accuracy knob: at the reference state the core violation |g(r<R)|
        falls 5e-3 → 4e-4 going from k_max = 400 to 4000, while the cost is
        linear.  Values within ~2 % of R_ij are smeared by that jump in either
        route — take the contact value from `get_g_contact` instead.

        Returns (N, N, n_r) — or (r, g) if return_r.
        """
        from scipy.fft import dst
        dk = k_max / n_k
        k = dk * np.arange(1, n_k + 1)
        hk = np.atleast_3d(self.get_hhat_k(k, order=order))            # (nk, N, N)
        r = np.pi / ((n_k + 1) * dk) * np.arange(1, n_k + 1)
        integ = k[:, None, None] * hk
        # scipy's DST-I carries a factor 2:  y_m = 2 Σ_n x_n sin(π n m/(n_k+1))
        h_r = (0.5 * dk) * dst(integ, type=1, axis=0) / (2.0 * np.pi**2 * r[:, None, None])
        g = 1.0 + h_r
        for i in range(self.N):
            for j in range(self.N):
                g[:, i, j] = np.where(r < self.R_ij[i, j], 0.0, g[:, i, j])
        g = np.transpose(g, (1, 2, 0))
        if r_grid is None:
            return (r, g) if return_r else g
        r_grid = np.asarray(r_grid, dtype=float)
        out = np.zeros((self.N, self.N, len(r_grid)))
        for i in range(self.N):
            for j in range(self.N):
                out[i, j] = np.interp(r_grid, r, g[i, j])
        return (r_grid, out) if return_r else out

    # ------------------------------------------------------------------
    # route B — invert along a vertical Bromwich line
    # ------------------------------------------------------------------

    @staticmethod
    def _dehoog(fs: np.ndarray, t: float, T: float, M: int, a: float) -> float:
        """de Hoog–Knight–Stokes ILT: the Bromwich Fourier series accelerated by
        a quotient-difference continued fraction.  `fs` = F(a + inπ/T), n=0..2M."""
        fs = np.asarray(fs, dtype=complex).copy()
        fs[0] *= 0.5
        n2 = 2 * M
        e = np.zeros((n2 + 2, M + 2), dtype=complex)
        q = np.zeros((n2 + 2, M + 2), dtype=complex)
        with np.errstate(divide='ignore', invalid='ignore'):
            for i in range(n2):
                q[i, 1] = fs[i + 1] / fs[i] if fs[i] != 0 else 0.0
            for rr in range(1, M + 1):
                for i in range(n2 - 2 * rr + 2):
                    e[i, rr] = q[i + 1, rr] - q[i, rr] + e[i + 1, rr - 1]
                for i in range(n2 - 2 * rr + 1):
                    if e[i, rr] != 0:
                        q[i, rr + 1] = q[i + 1, rr] * e[i + 1, rr] / e[i, rr]
        d = np.zeros(n2 + 1, dtype=complex)
        d[0] = fs[0]
        for rr in range(1, M + 1):
            d[2 * rr - 1] = -q[0, rr]
            if 2 * rr <= n2:
                d[2 * rr] = -e[0, rr]
        z = np.exp(1j * np.pi * t / T)
        Am2, Am1 = 0.0 + 0j, d[0]
        Bm2, Bm1 = 1.0 + 0j, 1.0 + 0j
        A, B = d[0], 1.0 + 0j
        for n in range(1, n2 + 1):
            A = Am1 + d[n] * z * Am2
            B = Bm1 + d[n] * z * Bm2
            Am2, Am1 = Am1, A
            Bm2, Bm1 = Bm1, B
        return float(np.exp(a * t) / T * np.real(A / B))

    @staticmethod
    def _dehoog_batch(fs: np.ndarray, t: np.ndarray, T: np.ndarray,
                      M: int, a: np.ndarray) -> np.ndarray:
        """`_dehoog` for many points at once — same algorithm, batch axis first.

        `fs` is (nb, 2M+1); t, T and a are (nb,).  The quotient-difference table
        is built with the i-loops as numpy slices over the batch, leaving 3M
        Python iterations instead of ~M² per point.  `_dehoog` is left untouched
        as the reference the S.11 gate compares against, and is still what
        `get_g_laplace` uses.
        """
        fs = np.array(fs, dtype=complex)
        fs[:, 0] *= 0.5
        nb, n2 = fs.shape[0], 2 * M
        e = np.zeros((nb, n2 + 2, M + 2), dtype=complex)
        q = np.zeros((nb, n2 + 2, M + 2), dtype=complex)
        with np.errstate(divide='ignore', invalid='ignore'):
            den = fs[:, :n2]
            q[:, :n2, 1] = np.where(den != 0,
                                    fs[:, 1:n2 + 1] / np.where(den == 0, 1.0, den),
                                    0.0)
            for rr in range(1, M + 1):
                k = n2 - 2 * rr + 2
                if k > 0:
                    e[:, :k, rr] = (q[:, 1:k + 1, rr] - q[:, :k, rr]
                                    + e[:, 1:k + 1, rr - 1])
                k2 = n2 - 2 * rr + 1
                if k2 > 0:
                    ee = e[:, :k2, rr]
                    q[:, :k2, rr + 1] = np.where(
                        ee != 0,
                        q[:, 1:k2 + 1, rr] * e[:, 1:k2 + 1, rr]
                        / np.where(ee == 0, 1.0, ee),
                        0.0)
        d = np.zeros((nb, n2 + 1), dtype=complex)
        d[:, 0] = fs[:, 0]
        for rr in range(1, M + 1):
            d[:, 2 * rr - 1] = -q[:, 0, rr]
            if 2 * rr <= n2:
                d[:, 2 * rr] = -e[:, 0, rr]
        z = np.exp(1j * np.pi * t / T)
        Am2 = np.zeros(nb, dtype=complex)
        Am1 = d[:, 0].copy()
        Bm2 = np.ones(nb, dtype=complex)
        Bm1 = np.ones(nb, dtype=complex)
        A, B = d[:, 0].copy(), np.ones(nb, dtype=complex)
        for n in range(1, n2 + 1):
            A = Am1 + d[:, n] * z * Am2
            B = Bm1 + d[:, n] * z * Bm2
            Am2, Am1 = Am1, A
            Bm2, Bm1 = Bm1, B
        return np.exp(a * t) / T * np.real(A / B)

    @staticmethod
    def _stehfest_V(N: int) -> np.ndarray:
        """Stehfest weights (N even).  Kept for the failure demonstration."""
        V = np.zeros(N)
        for n in range(1, N + 1):
            tot = 0.0
            for k in range((n + 1) // 2, min(n, N // 2) + 1):
                tot += (k ** (N // 2) * math.factorial(2 * k)
                        / (math.factorial(N // 2 - k) * math.factorial(k)
                           * math.factorial(k - 1) * math.factorial(n - k)
                           * math.factorial(2 * k - n)))
            V[n - 1] = (-1) ** (n + N // 2) * tot
        return V

    def get_g_laplace(self, r_grid, method: str = 'dehoog', M: int = 24,
                      n_stehfest: int = 12, tol: float = 1e-9) -> np.ndarray:
        """RDF g_ij(r) by inverting the closed-form L[r·g](s) — the Laplace route.

        Pointwise and grid-free: each r is independent, so a single r costs
        2M+1 evaluations of the closed form and nothing else.  That is the one
        thing the k-space route cannot do.

        method : 'dehoog'   (default) vertical Bromwich line, Re s = a > 0.
                             Agrees with the Fourier route to 3–4 digits and is
                             EXACTLY zero inside the core, which it gets from
                             causality rather than from masking.
                 'stehfest' real axis only — DOES NOT WORK for an RDF; kept so
                             the failure is reproducible (it returns a non-zero
                             g inside the core and smooths the packing
                             oscillations away, at every density tested).

        Returns (N, N, len(r_grid)).
        """
        r_grid = np.atleast_1d(np.asarray(r_grid, dtype=float))
        out = np.zeros((self.N, self.N, len(r_grid)))
        if method == 'stehfest':
            V = self._stehfest_V(n_stehfest)
            ln2 = math.log(2.0)
            for n, t in enumerate(r_grid):
                if t <= 0:
                    continue
                s = np.array([m * ln2 / t for m in range(1, n_stehfest + 1)])
                G = self.get_gL_laplace(s).real
                out[:, :, n] = (ln2 / t**2) * np.einsum('k,kij->ij', V, G)
            return out
        if method != 'dehoog':
            raise ValueError(f"method must be 'dehoog' or 'stehfest', got {method!r}")
        for n, t in enumerate(r_grid):
            if t <= 0:
                continue
            T = 2.0 * t
            a = -math.log(tol) / (2.0 * T)
            s = a + 1j * np.arange(0, 2 * M + 1) * np.pi / T
            G = self.get_gL_laplace(s)
            for i in range(self.N):
                for j in range(self.N):
                    out[i, j, n] = self._dehoog(G[:, i, j], t, T, M, a) / t
        return out

    def get_g0_r(self, r_grid, i: int, j: int, M: int = 40,
                 tol: float = 1e-12) -> np.ndarray:
        """Hard-sphere reference RDF g⁰_ij(r) — de Hoog inversion of `get_g0L_laplace`.

        Same machinery as `get_g_laplace`, but on the PY/Baxter HS transform
        alone, so this is the RDF of the reference system rather than of the
        first-order fluid.  Pointwise and grid-free.

        Accurate away from r = R_ij; ON the contact discontinuity the inversion
        returns the mid-value of the jump and rings over a band ~1/M wide (the
        Gibbs phenomenon of a truncated Bromwich sum).  `get_y0_r` handles that
        band explicitly rather than pretending it is not there.
        """
        return self._invert_rL(self.get_g0L_laplace, r_grid, i, j, M, tol)

    def _invert_rL(self, Lfun, r_grid, i: int, j: int, M: int, tol: float):
        """de Hoog inversion of L[r·f](s) at every r in one batched transform call.

        Each r needs its own Bromwich line (T = 2r, a = −ln tol/2T), so the
        s-grids differ from point to point — but `get_g0L_laplace` and
        `get_h1L_laplace` are vectorised over s, so all nr·(2M+1) nodes go in a
        SINGLE call instead of nr calls of 2M+1.  Worth doing rather than
        micro-optimisation: it is what makes the shell's cavity weight
        affordable inside a critical-point bisection.  Together with
        `_dehoog_batch` it takes `betaf_incore_lj('cavity')` on the binary
        reference state from 620 ms to 127 ms.

        Not bit-for-bit: the quotient-difference continued fraction is
        ill-conditioned, so re-associating its loops moves the answer by up to
        ~4e-9 relative near the contact jump (1e-15 to 1e-11 away from it).
        That is five orders below the ~2e-3 `guard` uncertainty this weight
        already carries, and S.11 measures it rather than assuming it.
        """
        rs  = np.atleast_1d(np.asarray(r_grid, dtype=float))
        out = np.zeros(len(rs))
        good = np.flatnonzero(rs > 0.0)
        if len(good):
            T = 2.0 * rs[good]
            a = -math.log(tol) / (2.0 * T)
            s = (a[:, None]
                 + 1j * np.arange(0, 2 * M + 1)[None, :] * np.pi / T[:, None])
            G = Lfun(s.ravel())[:, i, j].reshape(len(good), 2 * M + 1)
            out[good] = self._dehoog_batch(G, rs[good], T, M, a) / rs[good]
        return out if np.ndim(r_grid) else float(out[0])

    def get_y0_r(self, r_grid, i: int, j: int, guard: float = 0.015,
                 M: int = 40, tol: float = 1e-12) -> np.ndarray:
        """Hard-sphere cavity (background) correlation function y⁰_ij(r), all r > 0.

            y_ij(r) = g_ij(r)·e^{βu_ij(r)}

        y strips the DIRECT pair interaction out of g and keeps only what the
        medium mediates, so where g⁰ is annihilated by the hard-core Boltzmann
        factor y⁰ is finite, and y⁰ is CONTINUOUS at r = R_ij while g⁰ jumps.
        This is why every in-core / soft-core perturbation term is weighted by
        y⁰ and not by g⁰ (Barker-Henderson; the contact-value approximation used
        by `FMSAFreeEnergy.betaf_incore_lj` is y⁰ frozen at y⁰(R_ij) = g⁰(R_ij⁺)).

        Two exact branches and one interpolated band:

        r < R_ij   the PY closure is c = f·y with f = e^{−βu}−1 = −1 inside a
                   hard core, so **y⁰_ij(r) = −c^HS-PY_ij(r)** exactly — the
                   closed form `get_HS_PY` already computes, from the very same
                   Q̂₀ factorization the rest of this class is built on.  No new
                   machinery, no quadrature.
        r > R_ij(1+guard)
                   y⁰ = g⁰ (the Boltzmann factor is 1 outside the core), from
                   `get_g0_r`.
        the band in between
                   linear between the EXACT contact value −c^HS-PY(R_ij⁻) and
                   the first trustworthy inversion point.  The de Hoog sum rings
                   across the contact jump, and y⁰ is smooth there, so a 1.5 %
                   chord is far more accurate than the rung it replaces; the
                   `guard` sensitivity is measured in
                   `numerical_notes/results/single_point/incore_selfcons.md`.
        """
        rs  = np.atleast_1d(np.asarray(r_grid, dtype=float))
        R   = float(self.R_ij[i, j])
        out = np.empty(len(rs))
        inside = rs < R * (1.0 - 1e-12)
        for n in np.flatnonzero(inside):
            out[n] = -self.get_HS_PY(float(rs[n]), i, j)
        rest = np.flatnonzero(~inside)
        if len(rest):
            y_R = -self.get_HS_PY(R * (1.0 - 1e-11), i, j)
            out[rest] = self._outside_contact(rs[rest], i, j, self.get_g0_r,
                                              y_R, guard, M, tol)
        return out if np.ndim(r_grid) else float(out[0])

    def _outside_contact(self, rs: np.ndarray, i: int, j: int, invert,
                         contact: float, guard: float, M: int,
                         tol: float) -> np.ndarray:
        """`invert` beyond R_ij(1+guard); a chord from the exact contact value in.

        Shared by the zeroth- and first-order shell weights: both RDFs jump at
        r = R_ij, both have an exact algebraic contact value, and the de Hoog
        sum rings across the jump over a band ~1/M wide.
        """
        R   = float(self.R_ij[i, j])
        r_g = R * (1.0 + guard)
        out = np.empty(len(rs))
        far = rs >= r_g
        if np.any(far):
            out[far] = invert(rs[far], i, j, M=M, tol=tol)
        if np.any(~far):
            v_g = float(np.atleast_1d(invert(np.array([r_g]), i, j, M=M, tol=tol))[0])
            w   = (rs[~far] - R) / (r_g - R)
            out[~far] = (1.0 - w) * contact + w * v_g
        return out

    def get_g1_r(self, r_grid, i: int, j: int, M: int = 40,
                 tol: float = 1e-12) -> np.ndarray:
        """First-order RDF correction g¹_ij(r) — de Hoog inversion of `get_h1L_laplace`.

        Same caveat as `get_g0_r`: accurate away from contact, rings across the
        jump at r = R_ij.  `get_y1_r` applies the guard.
        """
        return self._invert_rL(self.get_h1L_laplace, r_grid, i, j, M, tol)

    def get_y1_r(self, r_grid, i: int, j: int, guard: float = 0.015,
                 M: int = 40, tol: float = 1e-12) -> np.ndarray:
        """The first-order weight g¹_ij(r) OUTSIDE the core, guarded at contact.

        The companion of `get_y0_r` for the second-order shell term, whose
        contact-value weight is g¹_ij(R_ij) rather than g₀_ij(R_ij).

        Only the exterior branch exists.  Inside the core the first-order core
        condition is h¹ ≡ 0, so a "first-order cavity function"
        y¹ = g¹e^{βu_HS} is a 0·∞ limit this class does not compute.  The
        exterior is all the LJ shell needs whenever d_BH < σ — always, for an LJ
        input — so rather than return a 0 that would integrate silently, an
        interior r raises.
        """
        rs  = np.atleast_1d(np.asarray(r_grid, dtype=float))
        R   = float(self.R_ij[i, j])
        if np.any(rs < R * (1.0 - 1e-12)):
            raise ValueError(
                f"get_y1_r is defined only outside the core: r < R_{i}{j} = "
                f"{R:.6f} requested (min {float(np.min(rs)):.6f}). The "
                "first-order cavity function inside the core is a 0*inf limit "
                "this class does not compute.")
        return self._outside_contact(rs, i, j, self.get_g1_r,
                                     self.get_g1ij_contact(i, j), guard, M, tol) \
            if np.ndim(r_grid) else float(self._outside_contact(
                rs, i, j, self.get_g1_r, self.get_g1ij_contact(i, j),
                guard, M, tol)[0])

    def get_g_r(self, r_grid=None, route: str = 'fourier', **kw):
        """Dispatch to the two RDF routes: route = 'fourier' | 'laplace'."""
        if route == 'fourier':
            return self.get_g_fourier(r_grid, **kw)
        if route == 'laplace':
            if r_grid is None:
                raise ValueError('the laplace route is pointwise: give r_grid')
            return self.get_g_laplace(r_grid, **kw)
        raise ValueError(f"route must be 'fourier' or 'laplace', got {route!r}")

    def get_g_contact(self, i: int, j: int, order: str = 'first') -> float:
        """g_ij(R⁺) algebraically — no inversion, so no jump smearing.

        Zeroth order is BMCSL for additive hard spheres,
            g⁰_ij(R⁺) = 1/vac + 3ζ₂dᵢdⱼ/((dᵢ+dⱼ)vac²)
                        + 2ζ₂²(dᵢdⱼ/(dᵢ+dⱼ))²/vac³,  ζ₂ = (π/6)ξ₂,
        which reduces to Carnahan–Starling (1−η/2)/(1−η)³ for one component.

        order='first' adds the first-order contact term via the initial-value
        theorem on (†): g¹_ij(R⁺) = lim_{s→∞} s·e^{sR_ij}·L[r h⁽¹⁾_ij](s)/R_ij.
        """
        vac, sig = self.vac, self.sigma
        z2 = (np.pi / 6.0) * self.xi[2]
        di, dj = float(sig[i]), float(sig[j])
        red = di * dj / (di + dj)
        g0 = 1.0 / vac + 3.0 * z2 * red / vac**2 + 2.0 * z2**2 * red**2 / vac**3
        if order == 'zeroth':
            return g0
        # s -> infinity in (†): Q̂₀ -> I and b_ij(s) -> (Σ_q c_q)/s, so the
        # initial-value theorem closes algebraically — no limit is taken numerically.
        R = float(self.R_ij[i, j])
        # (Σ_q coeffs)/√(ρᵢρⱼ), formed without ever dividing by a √ρ that can be
        # zero -- see `_reduced_contact_sums`.  Identical to the direct quotient
        # wherever that quotient is defined.
        g1 = float(self._b_csum_red[i, j]) / (2.0 * np.pi * R)
        return g0 + g1

    # ------------------------------------------------------------------
    # FMSAFreeEnergy interface — aliases and energy-route kernels
    #
    # The energy route needs L[r·g⁰] and L[r·g¹] at the Yukawa poles s = z_t,ij;
    # both are already closed forms above (`get_g0L_laplace`, `_H_of_s`).  The
    # four methods below only restore the e^{+sR_ij} / 2π√(ρᵢρⱼ) normalisation
    # that FMSAFreeEnergy expects, so
    #     βf₁ = −2πβ Σ_ij ρᵢρⱼ Σ_t k̄_t,ij·G0red_ij(z_t,ij)
    #     βf₂ = −πβ  Σ_ij (ρᵢρⱼ/2π√(ρᵢρⱼ)) Σ_t k̄_t,ij·G1L_shift_ij(z_t,ij)
    # reproduce  βf₁ = 2π Σ ρᵢρⱼ∫r²βu g⁰dr  and  βf₂ = π Σ ρᵢρⱼ∫r²βu g¹dr.
    # ------------------------------------------------------------------

    @property
    def radii(self) -> np.ndarray:
        """Contact-distance matrix R_ij — alias expected by FMSAFreeEnergy."""
        return self.R_ij

    @property
    def z_matrices(self) -> np.ndarray:
        return self.z_poles

    @property
    def k_bare_matrices(self) -> np.ndarray:
        return self.k_bare

    def G0red_laplace_ij(self, i: int, j: int, s: float) -> float:
        """Reduced HS Laplace kernel  G0red_ij(s) = e^{sR_ij}·L[r·g⁰_ij](s).

        Identical object to `FMSA_GA_matrix_mix.G0red_laplace_ij` (classical
        PY/Baxter mixture result); taken here from this class's own
        `get_g0L_laplace`, which carries the e^{−sR_ij} factor explicitly.
        """
        s = float(s)
        g0L = float(np.real(self.get_g0L_laplace(s)[i, j]))
        return g0L * math.exp(s * float(self.R_ij[i, j]))

    def get_g0ij_contact(self, i: int, j: int) -> float:
        """BMCSL hard-sphere contact value g⁰_ij(R_ij⁺)."""
        return self.get_g_contact(i, j, order='zeroth')

    def get_g1ij_contact(self, i: int, j: int) -> float:
        """First-order RDF contact value g¹_ij(R_ij⁺).

        From the initial-value theorem on (†) — no inversion, no jump smearing:
        g¹_ij(R⁺) = (Σ_q b-coeffs_ij) / (2π√(ρᵢρⱼ)·R_ij).
        """
        return (self.get_g_contact(i, j, order='first')
                - self.get_g_contact(i, j, order='zeroth'))

    def get_G1L_ij_shift(self, s: float, i: int, j: int) -> float:
        """Shifted first-order Laplace kernel

            G1L_shift_ij(s) = e^{sR_ij}·2π√(ρᵢρⱼ)·L[r·g¹_ij](s) = e^{sR_ij}·𝓗(s)_ij

        with g¹ ≡ h⁽¹⁾ outside the core.  The 2π√(ρᵢρⱼ) is left in because
        FMSAFreeEnergy._betaf2_yk divides it back out.
        """
        s = float(s)
        H = float(np.real(self._H_of_s(np.array([s], dtype=complex))[0, i, j]))
        return H * math.exp(s * float(self.R_ij[i, j]))

    def _make_at_rho(self, rho_vec: np.ndarray) -> "FMSA_double_prop":
        """Rebuild at a new density vector (same T, composition from rho_vec)."""
        rho_vec = np.asarray(rho_vec, dtype=float)
        rt = float(np.sum(rho_vec))
        return FMSA_double_prop(
            self.sig_pure, self.eps_pure, rho_vec / rt, rt, self.tstar,
            d_pure=self._d_pure,
            n_tails=self.n_tails, z=self._z_input, k=self._k_input,
            eps_ij=self._eps_ij,
            input_type=self._input_type, n_k=self._n_k,
            r_max_factor=self._rmaxf, hs_fmt=self.hs_fmt, verbose=False,
        )


# ----------------------------------------------------------------------
# Self-test: N=1 reduction against FMSA_pure (Tang closed form, d=1 units)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from FMSA_pure import FMSA_pure

    rho1, t1 = 0.30, 1.5
    dp = FMSA_double_prop(np.array([1.0]), np.array([1.0]), np.array([1.0]),
                          rho1, t1, n_tails=2, input_type='bh', verbose=True)
    pure = FMSA_pure(1.0, 1.0, rho1, t1, input_type='bh')

    r_test = np.linspace(0.08, 0.93, 200)
    pref = 2.0 * np.pi * rho1
    c1_dp   = np.array([dp.get_c1_inner(rv, 0, 0) for rv in r_test])
    c1_tang = np.array([pure.get_rc1r(rv) / (pref * rv) for rv in r_test])
    err = float(np.max(np.abs(c1_dp - c1_tang)))
    m   = float(np.max(np.abs(c1_tang)))
    print(f"N=1 reduction vs FMSA_pure (d=1): max|diff| = {err:.3e}, "
          f"max|Tang| = {m:.3e}, rel = {err/m:.3e}  "
          f"[{'PASS' if err/m < 2e-2 else 'FAIL'}]")

    # 2YK mixture smoke test: ĉ_ij(0) via dense grid, both HS references
    dp2 = FMSA_double_prop(np.array([1.0, 2.0]), np.array([1.0, 0.5]),
                           np.array([0.25, 0.75]), 0.139, 1.0,
                           n_tails=2, verbose=True)
    r_g = np.linspace(0.005, 8.0, 8000)
    c_fmt = dp2.get_c_total(r_g, hs_ref='fmt')
    c_py  = dp2.get_c_total(r_g, hs_ref='py')
    print("  2YK ĉ_ij(0):   [expected: 'py' ≈ OZ+MSA (−4.18, −31.9, −187); "
          "'fmt' = project-standard hybrid]")
    for (i, j), lbl in zip([(0, 0), (0, 1), (1, 1)], ["(1,1)", "(1,2)", "(2,2)"]):
        v_f = float(4.0 * np.pi * _trapz(r_g**2 * c_fmt[i, j], r_g))
        v_p = float(4.0 * np.pi * _trapz(r_g**2 * c_py[i, j],  r_g))
        print(f"    {lbl}:  fmt = {v_f:10.4f}    py = {v_p:10.4f}")
