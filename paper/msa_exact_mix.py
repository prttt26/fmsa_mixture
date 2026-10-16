r"""msa_exact_mix.py — the EXACT MSA for a hard-core multi-Yukawa **mixture**.

Group **MSAX**, Phase 2 (`todo/waisman_msa_plan.md`, task MSAX.6).
Phase 1 (`N = 1`, `n` tails) is `msa_exact.py`; this module restores the species
indices and nothing else — every equation below reduces to one of that module's,
term by term, and `--gate` checks that numerically at `N = 1`.

Source
------
L. Blum & J. S. Høye, *J. Stat. Phys.* **19**, 317 (1978) —
`pdf/sources_paywall/Solution of the Ornstein-Zernike equation with Yukawa
closure for a mixture.pdf`.  Every equation number below is theirs.  Their §2 is
already the mixture; `msa_exact.py`'s `_bh_pieces` is that section evaluated at
`N = 1`, so this file is the *general* case and the scalar one is the corollary.

⚠ **Three of their printed equations carry errors**, all three already found and
measured at `N = 1` (theory note §7d) and all three re-confirmed here by the
mixture's own internal consistency:

* **(29)** drops the `e^{zσ_ij}` that Eq. (3)'s contact normalisation forces.
* **(35)** prints `γ_ij = δ_ij − 2πρ_j ĝ_ij(z)·z`; (27) forces `/z`.
* **(36)** prints `(1+zλ_jl)/z` where (21) has `(1+zλ_jl)/z²`.

⚠ **(24)'s `A_i⁰` is NOT a typo** — the superscript-zero `A` there really carries
the *row* index while `M_j`, `N_j`, `A_j`, `B_j` all carry the column one.  It is
forced: (13)/(14) give `q'_ij = B_j + σ_ij A_j`, and expanding that with (23)
turns the `N_j` coefficient into `A_j⁰ + 4B_j⁰λ_ji/σ_j²`, which collapses to
`A_i⁰` identically.  Getting this wrong is invisible at equal diameters.

The system
----------
Unknowns, per tail `ν` and per ordered pair `(i,j)`:

    Dt^ν_ij = D^ν_ij·e^{−z_ν σ_ij}         the Baxter tail amplitude, O(K)
    Gt^ν_ij = ĝ_ij(z_ν)·e^{+z_ν σ_ij}      the Laplace moment of r·g_ij at z_ν

— `2N²ω` unknowns for `N` species and `ω` tails, matching `msa_exact`'s `2ω` at
`N = 1`.  ⭐ **As at `N = 1`, these are the variables in which every `e^{zσ}`
cancels analytically**; the raw `(D, ĝ)` form is unusable because `D ~ e^{zσ_ij}`
while the physics lives in `1 − γ ~ e^{−zσ}`.  The mixture adds one wrinkle: the
cancellation that makes `M_j`/`N_j` finite is no longer term-by-term but happens
through the *off-diagonal* of `γ`, via the identity (derived in `_pieces`)

    Dt_lj − Ct_lj = e^{−zσ_l}·(2π/z)·Σ_k ρ_k Gt_lk Dt_kj                    (★)

whose right-hand side is manifestly `O(K)` and exponential-free.

The `2N²ω` equations are (29) and (33)-at-`s = z_ν`, one of each per tail and per
ordered pair, in the scaled form derived in `_residuals`.

⚠ On the ROOT
-------------
Blum & Høye (p. 323): the system "is linear in `D_ij` and thus has a unique
solution … when the `γ_ij` are considered known", but "with `K_ij` instead of
`γ_ij` as known one must expect multiple solutions to occur, of which only one is
acceptable".  So the branch ambiguity is intrinsic to the MSA closure.  Here,
unlike at `N = 1`, the source *does* state a selection rule — their Eq. (41),
which is `g_ij = g_ji` written out.  `SolutionMix.symmetry_defect` evaluates it,
and it is **not** imposed on the solve: the system is square without it, and the
physical root satisfies it to round-off, which makes it a free root test rather
than a constraint.  (At `N = 1` it is vacuous, which is why Phase 1 had none.)
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import fsolve

from msa_exact import _phi1, _phi2


# ═══════════════════════════════════════════════════════════════════════════════
# the state point
# ═══════════════════════════════════════════════════════════════════════════════

def _exprel(x):
    """`(1 − e^{−x})/x`, series-stable at the origin (its value there is 1)."""
    x = np.asarray(x)
    small = np.abs(x) < 1e-3
    xs = np.where(small, x, 0.0)
    ser = 1.0 - xs / 2 + xs ** 2 / 6 - xs ** 3 / 24 + xs ** 4 / 120
    big = np.where(small, 1.0, x)
    return np.where(small, ser, (1.0 - np.exp(-big)) / big)


@dataclass
class HCYMix:
    """A hard-core multi-Yukawa mixture in Blum & Høye's conventions.

    `sigma` (N,)   hard-core diameters; contact at `σ_ij = (σ_i+σ_j)/2`
    `rho`   (N,)   number densities (not mole fractions)
    `z`     (ω,)   Yukawa decay rates, **shared by every pair** — this is
                   Blum & Høye's covered case, and Tang & Lu's gate.  A
                   pair-specific `z_ij` is representable by adding tails whose
                   amplitude matrix is supported on the pairs that want it.
    `K`     (ω,N,N) amplitudes in the **contact** normalisation of their Eq. (3),

            c_ij(r) = Σ_ν K^ν_ij e^{−z_ν(r−σ_ij)}/r          (r > σ_ij)

    so MSA (`c = −βu`) with `u_ij = −σ_ij ε_ij e^{−z(r−σ_ij)}/r` gives
    `K_ij = β ε_ij σ_ij` — the convention `benchmark_tanglu_contact.py` uses up
    to its extra `2π√(ρ_iρ_j)`.
    """
    sigma: np.ndarray
    rho: np.ndarray
    z: np.ndarray
    K: np.ndarray
    # geometry, filled in by __post_init__
    N: int = field(init=False)
    T: int = field(init=False)
    SIG: np.ndarray = field(init=False)     # σ_ij
    LAM: np.ndarray = field(init=False)     # LAM[a,b] = (σ_a−σ_b)/2 = their λ_ab
    A0: np.ndarray = field(init=False)      # A_j⁰   (25)
    B0: np.ndarray = field(init=False)      # B_j⁰   (25)
    QP0: np.ndarray = field(init=False)     # q^{0′}_ij  (25)

    def __post_init__(self):
        self.sigma = np.atleast_1d(np.asarray(self.sigma, dtype=float))
        self.rho = np.atleast_1d(np.asarray(self.rho, dtype=float))
        self.z = np.atleast_1d(np.asarray(self.z, dtype=float))
        K = np.asarray(self.K, dtype=float)
        n = len(self.sigma)
        self.K = K.reshape(len(self.z), n, n) if K.ndim == 3 else np.broadcast_to(
            K, (len(self.z), n, n)).copy()
        self.N, self.T = n, len(self.z)
        s = self.sigma
        self.SIG = 0.5 * (s[:, None] + s[None, :])
        self.LAM = 0.5 * (s[:, None] - s[None, :])
        # (25): the Percus–Yevick hard-core mixture, in Baxter's coefficients.
        # ξ_n = (π/6)Σρσⁿ, so ξ₃ = η.  Cross-checked against the project's own
        # `_build_Q0_Qpp` (which writes ξ₂ without the π/6) in `--gate`.
        xi2 = (np.pi / 6.0) * float(np.sum(self.rho * s ** 2))
        xi3 = (np.pi / 6.0) * float(np.sum(self.rho * s ** 3))
        d = 2.0 * np.pi / (1.0 - xi3) ** 2
        self.A0 = d * ((1.0 - xi3) + 3.0 * s * xi2)
        self.B0 = d * (-1.5 * s ** 2 * xi2)
        self.QP0 = d * (1.5 * s[:, None] * s[None, :] * xi2
                        + self.SIG * (1.0 - xi3))

    @property
    def eta(self) -> float:
        return float((np.pi / 6.0) * np.sum(self.rho * self.sigma ** 3))


# ═══════════════════════════════════════════════════════════════════════════════
# everything downstream of the unknowns
# ═══════════════════════════════════════════════════════════════════════════════

def _pieces(st: HCYMix, Dt: np.ndarray, Gt: np.ndarray) -> dict:
    r"""`γ`, `Ct`, `W̃`, `M_j`, `N_j`, `A_j`, `q′_ij` — Blum & Høye (20)–(27), (35).

    Shapes: `Dt`, `Gt` are `(ω, N, N)`; `M`, `N`, `A` are `(N,)`; `q′` is `(N,N)`.

    The scaled quantities, all `O(1)` or `O(K)` with no surviving `e^{zσ}`:

        γ^ν_ij  = δ_ij − 2πρ_j Gt^ν_ij e^{−z_ν σ_ij}/z_ν                    (35)
        Ct^ν_ij = −C^ν_ij e^{−z_ν σ_ij} = Σ_l γ^ν_il Dt^ν_lj e^{z_ν(σ_l−σ_i)/2}
        W̃^ν_ij = (2π/z_ν) Σ_k ρ_k Gt^ν_ik Dt^ν_kj

    ⭐ `W̃` is the mixture form of the cancellation that makes the scalar case
    work.  `M_j` and `N_j` are built from `D_lj + C_lj`, and *each term* of that
    sum is `O(e^{+zσ_l})` while the sum is `O(K)`.  At `N = 1` the cancellation is
    visible directly (`D + C = (1−γ)D` with `1−γ = 2πρĝ/z`); for a mixture it runs
    through the off-diagonal of `γ`, and the identity is (★) of the module
    docstring — proved by writing

        δ_lk − γ_lk e^{z(σ_k−σ_l)/2} = (2π/z)ρ_k Gt_lk e^{−zσ_l}

    which holds for **both** `k = l` (where the two `δ`'s cancel) and `k ≠ l`
    (where `−σ_lk + (σ_k−σ_l)/2 = −σ_l` exactly).  Forming `M` and `N` from the
    unscaled `D`, `C` instead loses `zσ_max/ln 10` digits to cancellation.
    """
    sig, rho, z, SIG, LAM = st.sigma, st.rho, st.z, st.SIG, st.LAM
    N, T = st.N, st.T
    eye = np.eye(N)

    gam = np.empty((T, N, N))
    Ct = np.empty((T, N, N))
    Wt = np.empty((T, N, N))
    for t in range(T):
        zt = z[t]
        gam[t] = eye - 2.0 * np.pi * rho[None, :] * Gt[t] * np.exp(-zt * SIG) / zt
        # e^{z(σ_l−σ_i)/2} = e^{−z·LAM[i,l]}
        Ct[t] = (gam[t] * np.exp(-zt * LAM)) @ Dt[t]
        Wt[t] = (2.0 * np.pi / zt) * ((Gt[t] * rho[None, :]) @ Dt[t])

    M = np.zeros(N)
    Nc = np.zeros(N)
    for t in range(T):
        zt = z[t]
        # (20) with C eliminated: M_j = −(1/z)Σ_l ρ_l[W̃_lj + (1+zσ_l)Ct_lj]
        M -= (1.0 / zt) * np.einsum('l,lj->j', rho, Wt[t]
                                    + (1.0 + zt * sig)[:, None] * Ct[t])
        # (21) likewise.  λ_jl = LAM[j,l]  ⇒  transpose to index [l,j].
        lam_jl = LAM.T                              # lam_jl[l,j] = LAM[j,l]
        Nc += np.einsum('l,lj->j', rho,
                        ((1.0 + zt * lam_jl) / zt ** 2) * Wt[t]
                        + (((1.0 + zt * sig + 0.5 * (zt * sig) ** 2)
                            / zt ** 2)[:, None]
                           + lam_jl * ((1.0 + zt * sig) / zt)[:, None]) * Ct[t])

    A = st.A0 * (1.0 + M) - 4.0 * st.B0 * Nc / sig ** 2                     # (23)
    qp = st.QP0 * (1.0 + M)[None, :] + st.A0[:, None] * Nc[None, :]         # (24)
    return dict(gam=gam, Ct=Ct, Wt=Wt, M=M, Nc=Nc, A=A, qp=qp)


def qhat(st: HCYMix, p: dict, s):
    r"""`Q̂_ij(s) = ∫_{λ_ji}^∞ dt e^{−st}Q_ij(t)` at arbitrary **complex** `s`.

    From (10), (11), (16) with the `D`/`C` cancellation of `_pieces` done first:

        Q̂_ij(s) = e^{−sλ_ji}·{ σ_i²φ₁(sσ_i)·q′_ij + σ_i³φ₂(sσ_i)·A_j
                    + Σ_ν [ W̃^ν_ij/(s+z_ν)
                            + Ct^ν_ij·( e^{−sσ_i}/(s+z_ν) + σ_i·(1−e^{−sσ_i})/s ) ] }

    with `λ_ji = (σ_j−σ_i)/2` and `σ_i = σ_ij − λ_ji` (the width of `Q_ij`'s
    support, which is set by the **row** species alone).

    ✅ **Verified against their Eq. (37)**, which prints `Q̂⁰_ij(z)` — the same
    expression at `s = z` with the tails' `W̃` term absent — including its
    `−(1/2z)e^{−2zλ_ji}(1−e^{−zσ_i})²Σ_lγ_ilD_lj`, which the grouping above
    reproduces identically.  That is a full independent check of the exponent
    bookkeeping, the `A_i⁰`-vs-`A_j⁰` reading of (24), and the `ρ_l`-placement.

    Returns `(N,N)` for scalar `s`, `(len(s),N,N)` for an array.
    """
    sig, LAMT = st.sigma, st.LAM.T          # LAMT[i,j] = λ_ji
    sc = np.atleast_1d(np.asarray(s, dtype=complex))
    out = np.empty((len(sc), st.N, st.N), dtype=complex)
    for m, sv in enumerate(sc):
        x = sv * sig                                          # (N,)
        core = ((sig ** 2 * _phi1(x))[:, None] * p['qp']
                + (sig ** 3 * _phi2(x))[:, None] * p['A'][None, :])
        for t in range(st.T):
            zt = st.z[t]
            core = core + (p['Wt'][t] / (sv + zt)
                           + p['Ct'][t] * (np.exp(-x)[:, None] / (sv + zt)
                                           + (sig * _exprel(x))[:, None]))
        out[m] = np.exp(-sv * LAMT) * core
    return out[0] if np.ndim(s) == 0 else out


# ═══════════════════════════════════════════════════════════════════════════════
# the residuals
# ═══════════════════════════════════════════════════════════════════════════════

def _residuals(u, st: HCYMix, sc1: np.ndarray, sc2: np.ndarray) -> np.ndarray:
    r"""(29′) and (33)-at-`s = z_ν`, scaled, one of each per tail and pair.

        (29′)  Σ_l Dt^ν_il e^{−z_ν λ_jl}[δ_lj − ρ_l Q̂_jl(z_ν)]  =  2πK^ν_ij/z_ν

        (33′)  Σ_l 2π Gt^ν_il e^{+z_ν λ_jl}[δ_lj − ρ_l Q̂_lj(z_ν)]
                   =  [ A_j + z_ν q′_ij + Σ_μ (z_ν²z_μ/(z_ν+z_μ))·Ct^μ_ij ] / z_ν²

    ⚠ **The two carry `Q̂` with opposite index order**, and that is not a
    transcription slip: (29) descends from Eq. (8), whose convolution is
    `Σ_l ρ_l ∫Q_jl(t)Q′_il(r+t)dt`, while (33) descends from (32), whose
    convolution is `Σ_l ρ_l ∫(r−t)g_il(|r−t|)Q_lj(t)dt`.  So (29) is `D(I−PQ̂ᵀ)`
    and (33) is `2πĝ(I−PQ̂)`, with `P = diag(ρ)`.  At `N = 1` the distinction is
    invisible, so `msa_exact.py` cannot have caught it; here it is the difference
    between passing and failing the Tang & Lu gate at `σ₂/σ₁ = 1.5`.

    Both equations are the printed ones divided by `e^{−z_ν σ_ij}` — which is
    what restores (29)'s missing contact factor and cancels (33)'s.
    """
    N, T = st.N, st.T
    n2 = N * N
    Dt = u[:T * n2].reshape(T, N, N)
    Gt = u[T * n2:].reshape(T, N, N)
    p = _pieces(st, Dt, Gt)
    rho, LAM, eye = st.rho, st.LAM, np.eye(N)

    r1 = np.empty((T, N, N))
    r2 = np.empty((T, N, N))
    Q = [qhat(st, p, st.z[t]).real for t in range(T)]
    for t in range(T):
        zt = st.z[t]
        # Y[l,j] = e^{−z λ_jl}(δ_lj − ρ_l Q̂_jl)  — note Q̂ transposed
        Y = np.exp(-zt * LAM.T) * (eye - rho[:, None] * Q[t].T)
        r1[t] = Dt[t] @ Y - 2.0 * np.pi * st.K[t] / zt
        # Z[l,j] = e^{+z λ_jl}(δ_lj − ρ_l Q̂_lj)
        Z = np.exp(zt * LAM.T) * (eye - rho[:, None] * Q[t])
        src = p['A'][None, :] + zt * p['qp']
        for m in range(T):
            src = src + (zt ** 2 * st.z[m] / (zt + st.z[m])) * p['Ct'][m]
        r2[t] = 2.0 * np.pi * (Gt[t] @ Z) - src / zt ** 2
    return np.concatenate([(r1 / sc1).ravel(), (r2 / sc2).ravel()])


def _scales(st: HCYMix) -> tuple[np.ndarray, np.ndarray]:
    """Per-**tail** scales: the `K → 0` size of each residual's right-hand side.

    (33)'s source is `O(A⁰/z²)` while (29)'s is `O(K)`, so the raw residual
    vector spans however many orders of magnitude `K` happens to be small by.
    Same reason as `msa_exact.residuals_scaled`.

    ⚠ **One scale per tail, not per entry.** A pair-specific `z_ij` is encoded by
    giving a tail an amplitude matrix supported on one pair (see `--gate` §5), so
    `K^ν_ij` is *exactly zero* on the other entries — and an entrywise scale then
    divides a residual that cancels to `O(K·eps)` by a floor, inflating pure
    round-off to `1e-9` and losing the solve.  A per-tail scale is the right
    normalisation anyway: (29) is one matrix equation per tail, not `N²`
    independent ones.
    """
    sc1 = np.array([max(float(np.max(np.abs(2.0 * np.pi * st.K[t] / st.z[t]))),
                        1e-30) for t in range(st.T)])
    sc2 = np.array([max(float(np.max(np.abs(st.A0[None, :] + zt * st.QP0)
                                     / zt ** 2)), 1e-30) for zt in st.z])
    return sc1[:, None, None], sc2[:, None, None]


# ═══════════════════════════════════════════════════════════════════════════════
# the solution
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolutionMix:
    st: HCYMix
    Dt: np.ndarray          # (ω,N,N)  D^ν_ij e^{−z_ν σ_ij}
    Gt: np.ndarray          # (ω,N,N)  ĝ_ij(z_ν) e^{+z_ν σ_ij}
    p: dict
    residual: float

    @property
    def A(self) -> np.ndarray:
        """`A_j` — Blum & Høye (13)/(23).  `F(0)_j = A_j/2π`."""
        return self.p['A']

    @property
    def contact(self) -> np.ndarray:
        """`g_ij(σ_ij⁺)` — their (38), `[q′_ij − zC_ij e^{−zσ_ij}]/(2πσ_ij)`.

        In the scaled variables `−Σ_ν z_ν C^ν_ij e^{−z_ν σ_ij} = Σ_ν z_ν Ct^ν_ij`.
        """
        num = self.p['qp'] + sum(self.st.z[t] * self.p['Ct'][t]
                                 for t in range(self.st.T))
        return num / (2.0 * np.pi * self.st.SIG)

    @property
    def dbetaP_drho(self) -> float:
        """`∂βp/∂ρ = (1/ρ)Σ_j ρ_j(A_j/2π)²` — their (39).

        ⭐ A **sum of squares**, so it is ≥ 0 on every real solution and the
        spinodal is exactly where a coefficient vanishes, not where a difference
        of large numbers changes sign."""
        rho = self.st.rho
        return float(np.sum(rho * (self.A / (2.0 * np.pi)) ** 2) / np.sum(rho))

    @property
    def symmetry_defect(self) -> float:
        """Their (41), `g_ij = g_ji`, as a relative defect — **a root test**.

        Not imposed anywhere in the solve (the system is square without it), so
        this is free information: the physical root returns round-off, a spurious
        one need not.  Vacuous at `N = 1`, which is why Phase 1 had no stated
        selection rule."""
        g = self.contact
        return float(np.max(np.abs(g - g.T)) / max(np.max(np.abs(g)), 1e-30))

    @property
    def gL(self) -> np.ndarray:
        """`ĝ_ij(z_ν) = ∫₀^∞ dr e^{−z_ν r} r g_ij(r)`, shape `(ω,N,N)`."""
        return self.Gt * np.exp(-self.st.z[:, None, None] * self.st.SIG)

    def matF(self, s):
        """`F(s) = I − diag(ρ)·Q̂(s)` — the mixture Baxter factor at complex `s`.

        `I − P^{1/2}Ĉ(k)P^{1/2} = Q̃(ik)Q̃ᵀ(−ik)` with `Q̃ = I − P^{1/2}Q̂P^{1/2}`,
        which is similar to this one, so **`det F` has the same zeros** and they
        are the poles of `ĥ_ij`: real ⇒ monotonic decay, complex pair ⇒ damped
        oscillation.  This is the object a mixture Fisher–Widom study consumes
        (`msa_exact.baxter_F` is its `N = 1` case), and `det F(0)` is the mixture
        spinodal indicator — **one** condition, unlike `∂βp/∂ρ`, which (39) makes
        a sum of squares and therefore never negative.
        """
        Q = qhat(self.st, self.p, s)
        eye = np.eye(self.st.N)
        if Q.ndim == 2:
            return eye - self.st.rho[:, None] * Q
        return eye[None] - self.st.rho[None, :, None] * Q

    def detF(self, s):
        """`det F(s)` — scalar (or array); its zeros in `Re s < 0` set the decay."""
        return np.linalg.det(self.matF(s))


def _qhat_vec(st: HCYMix, p: dict, s: np.ndarray) -> np.ndarray:
    r"""`qhat` for a whole array of `s` at once — same formula, no Python loop.

    `qhat` iterates over `s` because most callers want one complex point (a pole
    search, a winding number).  Inverting to `r` wants thousands, and at
    `n_r = 16384` the loop is the whole cost of the transform rather than the
    transform itself.  Timing an unvectorised loop would have charged exact MSA
    for an implementation choice instead of for the method.

    Gated against `qhat` in `_gate_c_of_r`; the two agree to round-off.
    """
    sig, LAMT = st.sigma, st.LAM.T
    s = np.asarray(s, dtype=complex)
    x = s[:, None] * sig                                   # (S,N)
    core = ((sig ** 2 * _phi1(x))[:, :, None] * p['qp'][None]
            + (sig ** 3 * _phi2(x))[:, :, None] * p['A'][None, None, :])
    for t in range(st.T):
        zt = st.z[t]
        den = (s + zt)[:, None, None]
        core = core + (p['Wt'][t][None] / den
                       + p['Ct'][t][None] * (np.exp(-x)[:, :, None] / den
                                             + (sig * _exprel(x))[:, :, None]))
    return np.exp(-s[:, None, None] * LAMT[None]) * core


def chat_k(sol: "SolutionMix", k: np.ndarray) -> np.ndarray:
    r"""`ĉ_ij(k)`, shape `(len(k), N, N)`, from the Baxter factorisation.

        I − P^{1/2}Ĉ(k)P^{1/2} = F̃(ik)F̃ᵀ(−ik),   F̃ = I − P^{1/2}Q̂P^{1/2}

    so `ĉ_ij(k) = (δ_ij − M_ij(k))/√(ρ_iρ_j)`.  At `k = 0` this is `chat0`
    verbatim, which is one of the gates.
    """
    k = np.atleast_1d(np.asarray(k, float))
    rho = sol.st.rho
    rr = np.sqrt(rho[:, None] * rho[None, :])
    I = np.eye(sol.st.N)
    Fp = I[None] - rr[None] * _qhat_vec(sol.st, sol.p, 1j * k)
    Fm = I[None] - rr[None] * _qhat_vec(sol.st, sol.p, -1j * k)
    M = np.real(np.einsum('kij,klj->kil', Fp, Fm))
    return (I[None] - M) / rr[None]


def c_of_r(sol: "SolutionMix", n_r: int = 4096, r_max: float | None = None):
    r"""`(r, c)` with `c` of shape `(N,N,n_r)` — the real-space \DCF{}.

    ⭐ **This is the k → r inversion, and it is what the exact solution costs in
    r-space.**  Blum & Høye deliver a Baxter factor and its Laplace transforms;
    `qhat` evaluates them at any complex `s`, so `ĉ(k)` is closed form, but
    `c(r)` is not — it has to be transformed back.  Everything above this line is
    analytic; this function is the numerical step, and unlike the solve it scales
    with the radial grid.

    ⚠ **Same grid and same kernel as `oz_numerical.OZSolverMSA`** — midpoint `r`
    and `k`, `scipy.fft.dst` type 4, `dk = π/r_max`.  That is deliberate: the OZ
    legs pay for this transform on every iteration, so charging exact MSA a
    *different* transform would not be a like-for-like comparison.

    ⚠ The default `r_max` is `15 × max σ_ij`, matching `OZSolverMSA`'s
    `r_max_factor`.
    """
    from scipy.fft import dst
    if r_max is None:
        r_max = 15.0 * float(np.max(sol.st.SIG))
    dr = r_max / n_r
    r = (np.arange(n_r) + 0.5) * dr
    dk = np.pi / r_max
    k = (np.arange(n_r) + 0.5) * dk
    ck = chat_k(sol, k)                                    # (n_r,N,N)
    raw = dst(k[:, None, None] * ck, type=4, axis=0)
    c = raw * (dk / (4.0 * np.pi ** 2)) / r[:, None, None]
    return r, np.moveaxis(c, 0, -1)


def _gate_c_of_r(sol: "SolutionMix", n_r: int = 16384, quiet: bool = False) -> dict:
    r"""Four checks on `c_of_r`, because a timing on an unverified transform is noise.

    1. `_qhat_vec` against the scalar `qhat` loop it replaces.
    2. `chat_k(0)` against `chat0`, which is an independent path through the algebra.
    3. the OUTER region against the \MSA{} tail `Σ_ν K^ν_ij e^{−z_ν(r−σ_ij)}/r`, which
       the closure fixes exactly and the solver has no freedom over.
    4. `4π∫r²c dr` against `chat0`, the integrated content.

    ⚠ The tail carries `1/r`, **not** `1/(r/σ_ij)`.  With the `σ_ij` in the numerator
    check 3 reports 3--49 % and looks like a broken transform; it is the convention.

    Measured on the reference binary: `0.0`, `1.6e-16`, `6.4e-4`, `1.8e-9`.  Check 3 is
    the ringing from the contact jump and is the one that moves with `n_r`
    (`2.6e-3` at 4096), which is the same grid dependence the \OZ{} legs carry.
    """
    st, p = sol.st, sol.p
    ks = np.array([0.3, 1.7, 9.0])
    a = _qhat_vec(st, p, 1j * ks)
    b = np.array([qhat(st, p, 1j * k) for k in ks])
    g1 = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))

    ref = chat0(sol)
    g2 = float(np.max(np.abs(chat_k(sol, np.array([1e-9]))[0] - ref))
               / np.max(np.abs(ref)))

    r, c = c_of_r(sol, n_r=n_r)
    SIG, K, z = np.asarray(st.SIG, float), np.asarray(st.K, float), np.asarray(st.z, float)
    g3 = g4 = 0.0
    for i in range(st.N):
        for j in range(st.N):
            R = SIG[i, j]
            m = (r > 1.3 * R) & (r < 3.0 * R)
            u = sum(K[t, i, j] * np.exp(-z[t] * (r[m] - R)) / r[m]
                    for t in range(len(z)))
            g3 = max(g3, float(np.max(np.abs(c[i, j, m] - u)) / np.max(np.abs(u))))
            I = 4 * np.pi * np.trapz(r ** 2 * c[i, j], r)
            g4 = max(g4, float(abs(I / ref[i, j] - 1.0)))

    out = {'qhat_vec': g1, 'chat_k(0)': g2, 'outer tail': g3, 'int r^2 c': g4}
    if not quiet:
        print('  gate c_of_r: ' + '   '.join(f'{k} {v:.2e}' for k, v in out.items()))
    return out


def seed_zero_K(st: HCYMix) -> tuple[np.ndarray, np.ndarray]:
    r"""The exact `K → 0` solution — **two linear solves**, no iteration.

    At `K = 0` every `Dt → 0`, hence `Ct, W̃, M, N → 0`, `A → A⁰`, `q′ → q^{0′}`
    and `Q̂ → Q̂⁰`.  Both residuals then become linear:

        Gt⁰ = (1/2π)·[A_j⁰ + z q^{0′}_ij]/z² · Z₀⁻¹ ,   Dt⁰ = (2πK/z)·Y₀⁻¹

    ⭐ **`Gt⁰` is a non-trivial check of the whole hard-sphere transcription.**
    Its right-hand side `A_j⁰ + z q^{0′}_ij` is *not* symmetric in `(i,j)`, yet
    `ĝ⁰_ij(z)` must be — and it comes out symmetric to round-off only if `ψ₁/ψ₂`,
    the `ρ_l` placement and the `λ` convention are all right.  `--gate` also
    compares it against `FMSA_double_prop.get_g0L_laplace`, an independent closed
    form for the same object.
    """
    z, rho, LAM, eye = st.z, st.rho, st.LAM, np.eye(st.N)
    p0 = _pieces(st, np.zeros((st.T, st.N, st.N)), np.zeros((st.T, st.N, st.N)))
    Dt = np.empty((st.T, st.N, st.N))
    Gt = np.empty((st.T, st.N, st.N))
    for t in range(st.T):
        zt = z[t]
        Q0 = qhat(st, p0, zt).real
        Y = np.exp(-zt * LAM.T) * (eye - rho[:, None] * Q0.T)
        Z = np.exp(zt * LAM.T) * (eye - rho[:, None] * Q0)
        Dt[t] = np.linalg.solve(Y.T, (2.0 * np.pi * st.K[t] / zt).T).T
        src = (st.A0[None, :] + zt * st.QP0) / zt ** 2
        Gt[t] = np.linalg.solve(Z.T, src.T).T / (2.0 * np.pi)
    return Dt, Gt


def solve_bh_mix(st: HCYMix, tol: float = 1e-9) -> SolutionMix | None:
    """⭐ **MSAX.6** — solve the mixture MSA exactly, on the branch through `K = 0`.

    The seed is the exact `K → 0` limit, so at small coupling the first Newton
    step already sits at round-off; at finite coupling the solve is ramped in a
    multiplier `λ` on `K`, with a secant predictor and a step rejection, exactly
    as `msa_exact.solve_bh_n`.  The two traps that ramp guards against are the
    same ones Phase 1 measured:

    * **Accept on the residual, never on `fsolve`'s flag.** An exact seed makes
      `fsolve` return `ok = 5` ("not making good progress") *because there is no
      progress left to make*.
    * **Reject steps that jump.** Near a fold `fsolve` converges happily onto a
      different branch — small residual, wrong root — and every later step then
      rides it.

    Returns `None` if the branch cannot be continued, which is information (the
    physical root has ended), not a solver defect.
    """
    sc1, sc2 = _scales(st)
    n2 = st.N * st.N

    def attempt(K_t, seed):
        st_t = HCYMix(st.sigma, st.rho, st.z, K_t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = fsolve(_residuals, seed, args=(st_t, sc1, sc2), xtol=1e-13)
        r = float(np.linalg.norm(_residuals(u, st_t, sc1, sc2)))
        return u, (r if r < tol else np.inf)

    Dt0, Gt0 = seed_zero_K(st)
    u, res = attempt(st.K, np.concatenate([Dt0.ravel(), Gt0.ravel()]))
    if not np.isfinite(res):
        _, G00 = seed_zero_K(HCYMix(st.sigma, st.rho, st.z, 0.0 * st.K))
        seed = np.concatenate([np.zeros(st.T * n2), G00.ravel()])
        prev, prev_f, f, df = None, None, 0.0, 1.0 / 64.0
        while f < 1.0 - 1e-14:
            fn = min(f + df, 1.0)
            pred = (seed if prev is None
                    else seed + (seed - prev) * (fn - f) / (f - prev_f))
            trial, r = attempt(st.K * fn, pred)
            if np.isfinite(r) and prev is not None:
                step = np.linalg.norm(trial - seed)
                expect = (np.linalg.norm(seed - prev) * (fn - f) / (f - prev_f)
                          + 1e-8 * (1.0 + np.linalg.norm(seed)))
                if step > 4.0 * expect:
                    r = np.inf                      # branch jump — reject
            if not np.isfinite(r):
                df *= 0.5
                if df < 1e-11:
                    return None
                continue
            prev, prev_f = seed, f
            seed, res, f = trial, r, fn
            df = min(df * 1.25, 1.0 / 32.0)
        u = seed
    Dt = u[:st.T * n2].reshape(st.T, st.N, st.N)
    Gt = u[st.T * n2:].reshape(st.T, st.N, st.N)
    return SolutionMix(st, Dt, Gt, _pieces(st, Dt, Gt), float(res))


def leading_poles_mix(sol: SolutionMix, n_seed: int = 260, span: float = 14.0,
                      n_grid: int = 26) -> tuple[float, float, float]:
    r"""`(α₀^mon, α₀^osc, α₁^osc)` — the leading monotonic and oscillatory decays.

    Zeros of `det F(s)` with the largest `Re s < 0`, split by whether they are
    real; `inf` for a family with no zero in the searched box.  The mixture
    analogue of `msa_exact.leading_poles`, and the `N = 1` reduction is a gate.

    ⭐ **There is one pole set for the whole mixture, not one per pair.** Every
    `ĥ_ij` is a ratio whose denominator is `det F`, so all `N(N+1)/2` correlation
    functions decay at the **same** rate with the same wavelength — only the
    amplitudes differ.  That is Evans, Leote de Carvalho, Henderson & Hoyle's
    statement, and it is why "the" Fisher–Widom line of a mixture is well defined
    at all.  `--gate` §8 checks it against real-space `OZSolverMSA` tails.

    ⚠ The search strip is `(−min_ν z_ν, 0)`: `det F` has explicit poles at
    `s = −z_ν` (order up to `N`), so brackets must not straddle them.
    """
    from scipy.optimize import brentq, fsolve

    zmin = float(np.min(sol.st.z))
    xs = np.linspace(-zmin + 1e-9, -1e-9, n_seed)
    fr = np.array([float(np.real(sol.detF(x))) for x in xs])
    a_mon = np.inf
    for i in range(len(xs) - 1):
        if np.sign(fr[i]) != np.sign(fr[i + 1]):
            r = brentq(lambda x: float(np.real(sol.detF(x))), xs[i], xs[i + 1],
                       xtol=1e-13)
            a_mon = -r                      # decay rate is −Re s; first = leading
            break

    def _f(v):
        w = complex(sol.detF(v[0] + 1j * v[1]))
        return [w.real, w.imag]

    # ⚠ Seeds far from any zero send `fsolve` where `e^{−s}` overflows; those
    # iterates are rejected by the residual test below, but they make numpy shout.
    # Silence the arithmetic, not the acceptance test.
    best = None
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for re0 in np.linspace(-span, -1e-3, n_grid):
            for im0 in np.linspace(0.3, span, n_grid):
                try:
                    v = fsolve(_f, [re0, im0], xtol=1e-12)
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


def trace_coupling_mix(st: HCYMix, lam_max: float = 1.0,
                       dlam0: float = 1.0 / 128.0):
    """Continue the physical branch in a multiplier `λ` on `K`: `[(λ, Solution)]`.

    ⚠ **This exists for the same reason `msa_exact.trace_coupling` does.**
    `solve_bh_mix` re-ramps from `K = 0` on every call, and that cannot resolve
    the approach to the spinodal — measured here at `σ₂/σ₁ = 1.5, η = 0.30`, a
    per-point solve reports "branch lost" at `T* = 0.50` while the traced branch
    walks on to `T* ≈ 0.40`, and what looked like a fold is the tracer's step
    collapsing near `det F(0) = 0`.  Track the branch once, seeded by a secant
    predictor from the previous two points.

    ⭐ **Bracket on `det F(0)`, never on `∂βp/∂ρ`.** The mixture repeats the
    scalar lesson in matrix form: `det[I − P^{1/2}Ĉ(0)P^{1/2}] = (det F(0))²`, so
    `det F(0)` is the regular variable that crosses zero linearly, while (39)'s
    `∂βp/∂ρ` is a sum of squares and — this is the mixture-specific part —
    **is not even the right indicator**: it is only mechanical stability at fixed
    composition, and a binary can go unstable against composition fluctuations
    while `∂βp/∂ρ` is still comfortably positive (η = 0.46 below).
    """
    sc1, sc2 = _scales(st)
    n2 = st.N * st.N

    def solve_at(l, seed):
        s = HCYMix(st.sigma, st.rho, st.z, st.K * l)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = fsolve(_residuals, seed, args=(s, sc1, sc2), xtol=1e-13)
        return u, float(np.linalg.norm(_residuals(u, s, sc1, sc2))), s

    def rec(u, s):
        Dt = u[:st.T * n2].reshape(st.T, st.N, st.N)
        Gt = u[st.T * n2:].reshape(st.T, st.N, st.N)
        return SolutionMix(s, Dt, Gt, _pieces(s, Dt, Gt), 0.0)

    _, G0 = seed_zero_K(HCYMix(st.sigma, st.rho, st.z, 0.0 * st.K))
    u = np.concatenate([np.zeros(st.T * n2), G0.ravel()])
    out = [(0.0, rec(u, HCYMix(st.sigma, st.rho, st.z, 0.0 * st.K)))]
    prev, prev_l, lam, dl = None, None, 0.0, dlam0
    while lam < lam_max and dl > 1e-11:
        ln = min(lam + dl, lam_max)
        pred = u if prev is None else u + (u - prev) * (ln - lam) / (lam - prev_l)
        trial, r, s = solve_at(ln, pred)
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
        out.append((lam, rec(u, s)))
        dl = min(dl * 1.25, dlam0 * 8)
    return out


def spinodal_coupling_mix(st: HCYMix, lam_max: float = 1.0):
    """The multiplier `λ` on `K` at which `det F(0) = 0` — the mixture spinodal.

    `None` if the traced branch ends before it.  Brackets on `det F(0)`, which is
    regular there; see `trace_coupling_mix`."""
    from scipy.optimize import brentq
    tr = trace_coupling_mix(st, lam_max=lam_max)
    lam = np.array([t[0] for t in tr])
    dF = np.array([float(np.real(t[1].detF(0.0))) for t in tr])
    i = np.where(np.sign(dF[:-1]) != np.sign(dF[1:]))[0]
    if len(i) == 0:
        return None
    i = i[0]
    sc1, sc2 = _scales(st)
    n2 = st.N * st.N
    seed = np.concatenate([tr[i][1].Dt.ravel(), tr[i][1].Gt.ravel()])

    def _d(l):
        s = HCYMix(st.sigma, st.rho, st.z, st.K * l)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = fsolve(_residuals, seed, args=(s, sc1, sc2), xtol=1e-13)
        Dt = u[:st.T * n2].reshape(st.T, st.N, st.N)
        Gt = u[st.T * n2:].reshape(st.T, st.N, st.N)
        return float(np.real(SolutionMix(s, Dt, Gt, _pieces(s, Dt, Gt), 0.0).detF(0.0)))

    return brentq(_d, lam[i], lam[i + 1], xtol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════════
# thermodynamics — Blum & Høye (42)–(46), and the two routes to ΔA
# ═══════════════════════════════════════════════════════════════════════════════
#
# ⭐ MSA is special here: the excess **free energy** is closed-form, with no
# thermodynamic integration at all.  That is Høye & Stell's result, quoted by
# Blum & Høye as their (43)–(45), and it is what makes an *exact* energy-route /
# compressibility-route comparison possible — Group X could only do that
# numerically, and only at first order.
#
# In this module's scaled variables every `e^{zσ}` in those equations is already
# absorbed: (42)/(44)/(46) all contain `K_ij e^{zσ_ij}ĝ_ij(z)`, which is exactly
# `K_ij·Gt_ij`.

def excess_energy(sol: SolutionMix) -> float:
    """`βΔE/V` — excess internal energy per unit volume, their (42).

        −βΔE = 2π Σ_ij ρ_iρ_j Σ_ν K^ν_ij e^{z_νσ_ij} ĝ_ij(z_ν)
             = 2π Σ_ij ρ_iρ_j Σ_ν K^ν_ij Gt^ν_ij

    Negative for `K > 0` (attraction), as it must be.
    """
    st = sol.st
    rr = st.rho[:, None] * st.rho[None, :]
    return float(-2.0 * np.pi * np.sum(rr[None, :, :] * st.K * sol.Gt))


def qhat_deriv(st: HCYMix, p: dict, s, radius: float | None = None):
    """`dQ̂/ds` by Cauchy's formula on a small circle — spectrally accurate.

    `Q̂` is analytic except for explicit poles at `s = −z_ν`, so
    `Q̂′(s) = (1/2πi)∮ Q̂(w)/(w−s)²dw` evaluated by the trapezoidal rule on
    `|w−s| = r` converges geometrically in the number of nodes.  Preferred over
    hand-differentiating six terms (three of which are `0/0` at the origin) and
    over finite differences, which would cap the `J` integral of (44) at ~1e-10.
    """
    r = radius if radius is not None else 0.25 * float(np.min(st.z))
    m = 48
    th = 2.0 * np.pi * np.arange(m) / m
    w = np.asarray(s, dtype=complex) + r * np.exp(1j * th)
    vals = qhat(st, p, w)                                   # (m,N,N)
    return np.einsum('k,kij->ij', np.exp(-1j * th), vals) / (m * r)


def gL_deriv(sol: SolutionMix) -> np.ndarray:
    r"""`∂ĝ_ij(s)/∂s` at `s = z_ν`, shape `(ω,N,N)` — the one extra object (44) needs.

    Not a new unknown: (33) holds for **all** `s`, so differentiating it gives a
    linear system for `ĝ′` whose other entries are already known.  Writing (33) as
    `2π ĝ(s)·[I − PQ̂(s)] = R(s)` with

        R_ij(s) = e^{−sσ_ij}[ A_j/s² + q′_ij/s + Σ_ν z_ν Ct^ν_ij/(s+z_ν) ]

    (the printed form with `C^ν_ij = −Ct^ν_ij e^{+z_νσ_ij}` substituted), one has

        2π ĝ′(s)·[I − PQ̂(s)] = R′(s) + 2π ĝ(s)·P Q̂′(s)

    which is one `N × N` solve per tail.
    """
    st, p = sol.st, sol.p
    N, T = st.N, st.T
    rho, SIG, eye = st.rho, st.SIG, np.eye(N)
    out = np.empty((T, N, N))
    gL = sol.gL
    for t in range(T):
        s = st.z[t]
        Q = qhat(st, p, s).real
        dQ = qhat_deriv(st, p, s).real
        pole = sum(st.z[m] * p['Ct'][m] / (s + st.z[m]) for m in range(T))
        dpole = sum(-st.z[m] * p['Ct'][m] / (s + st.z[m]) ** 2 for m in range(T))
        bracket = p['A'][None, :] / s ** 2 + p['qp'] / s + pole
        dbracket = -2.0 * p['A'][None, :] / s ** 3 - p['qp'] / s ** 2 + dpole
        e = np.exp(-s * SIG)
        dR = e * (dbracket - SIG * bracket)
        lhs = eye - rho[:, None] * Q                        # (I − PQ̂)
        rhs = dR + 2.0 * np.pi * (gL[t] @ (rho[:, None] * dQ))
        out[t] = np.linalg.solve(lhs.T, rhs.T).T / (2.0 * np.pi)
    return out


def chat0(sol: SolutionMix) -> np.ndarray:
    """`ĉ_ij(0)` from the factorization: `δ_ij − √(ρ_iρ_j)ĉ_ij(0) = (F̃F̃ᵀ)_ij`
    with `F̃_ij = δ_ij − √(ρ_iρ_j)Q̂_ij(0)` — their (5) at `k = 0`."""
    st = sol.st
    rr = np.sqrt(st.rho[:, None] * st.rho[None, :])
    Ft = np.eye(st.N) - rr * qhat(st, sol.p, 0.0).real
    return (np.eye(st.N) - Ft @ Ft.T) / rr


def _a_hs(st: HCYMix) -> float:
    """`∂βp/∂ρ` of the hard-core reference — (39) with `A_j → A_j⁰`."""
    return float(np.sum(st.rho * (st.A0 / (2.0 * np.pi)) ** 2) / np.sum(st.rho))


def pressure_excess(sol: SolutionMix) -> float:
    r"""⭐ `βΔp/V` — the excess pressure, **closed form**.  Blum & Høye (43)+(44).

    ⚠ **Their (43) as printed carries two defects**, both found here by testing it
    against the λ-integration of (42) (`--gate` §8) and both fixed below:

    1. **`q′_ij` there is the derivative *jump*, not the coefficient of (11).**
       The object is `Q′_ij(σ_ij⁻) = q′_ij − Σ_ν z_ν C^ν_ij e^{−z_νσ_ij}`, which
       (38) identifies as `2πσ_ij g_ij(σ_ij⁺)`.  ⭐ The structure makes this
       obvious once seen: `q^{0′}_ij` **is** `2πσ_ij g⁰_ij(σ_ij)` already (the
       reference has `C ≡ 0`), so the bracket is one object evaluated for the
       Yukawa system and for its reference — the printed form breaks that
       symmetry.  Reading it literally is not a small error: at `η = 0.4`,
       `z = 5`, `K = 0.5` the polynomial term is **−0.510 instead of +0.568**.
    2. **`+ J` should be `+ J/ρ`.** (43) declares `φ^E = βp^E/ρ` and gives the
       polynomial a `1/(12πρ)`, but `J` as defined by (44) is a *pressure*: it is
       exactly the Yukawa tail's virial integral
       `−(2π/3)Σ_ijρ_iρ_j∫r³(dβu_ij/dr)g_ij dr`, which with `βu = −Ke^{−z(r−σ)}/r`
       and `ĝ(s) = ∫e^{−sr}rg dr` reduces to (44) term by term.  So the two
       halves of (43) differ by one power of `ρ`.

    With both restored the identity `βΔp = ρ ∂(βΔA/V)/∂ρ − βΔA/V` holds to 1e-5
    (the finite-difference floor) on 16 states spanning `η = 0.1–0.4`,
    `K = 0.1–0.5`, `z = 1.8, 5`; with either defect left in, the error is 14–94 %.
    """
    st, p = sol.st, sol.p
    rr = st.rho[:, None] * st.rho[None, :]
    dg = gL_deriv(sol)
    J = (2.0 * np.pi / 3.0) * float(np.sum(
        rr[None, :, :] * st.K * np.exp(st.z[:, None, None] * st.SIG[None])
        * (st.z[:, None, None] * dg - sol.gL)))
    Y = 2.0 * np.pi * st.SIG * sol.contact              # = q′_ij − Σ z C e^{−zσ}
    poly = float(np.sum(rr * st.SIG * (Y ** 2 - st.QP0 ** 2))) / (12.0 * np.pi)
    return poly + J


def pressure_virial(sol: SolutionMix) -> float:
    r"""⭐ `βp` — the **VIRIAL route**, the third one, and it is closed form here.

        βp = Σρ_i + (2π/3)Σ_ij ρ_iρ_j σ_ij³ g_ij(σ_ij⁺)  −  (2π/3)Σ_ij ρ_iρ_j ∫r³(dβu_ij/dr)g_ij dr

    Both pieces are already in hand and neither needs a quadrature: the hard-core
    term is the **contact value** `sol.contact`, and the tail integral is exactly
    `(44)`'s `J` — with `βu = −Ke^{−z(r−σ)}/r`,

        ∫_σ^∞ r³(dβu/dr)g dr = K e^{zσ}[ĝ(z) − z ĝ′(z)]

    and `ĝ′` is `gL_deriv`.  So all three routes are closed form in this solution.

    ⚠⚠ **This is NOT what `pressure_excess` returns.**  Blum & Høye's (43)+(44)
    shares the tail `J` but its hard-core part is `(π/3)Σρ_iρ_jσ_ij(Y²−Y₀²)/(4π²)`
    with `Y = 2πσ g(σ⁺)` — **quadratic** in the contact value, not linear — and the
    total is the pressure thermodynamically consistent with the Høye–Stell free
    energy (45), i.e. the **energy** route (`--gate` §8 pins (45) to the λ-charging
    integral at 3.8e-05).  Measured difference between the two at `z = 1.8, K = 1`:
    −32 % at `η = 0.10, T* = 1.20`, and sign-changing by `η = 0.20`.  Verified
    separately that `poly + J` reproduces `pressure_excess` to 1e-16, so the split
    above is the actual content of (43), not a paraphrase.

    ⭐ **Gate built in**: at `K → 0` this returns the Percus–Yevick *virial*
    equation of state `ρ(1+2η+3η²)/(1−η)²`, where `pressure_excess` + the reference
    returns the *compressibility* one `ρ(1+η+η²)/(1−η)³`.  The classic PY
    inconsistency is therefore already visible at zero coupling, which is the
    cheapest possible check that the two are different objects.
    """
    st = sol.st
    rr = st.rho[:, None] * st.rho[None, :]
    dg = gL_deriv(sol)
    J = (2.0 * np.pi / 3.0) * float(np.sum(
        rr[None, :, :] * st.K * np.exp(st.z[:, None, None] * st.SIG[None])
        * (st.z[:, None, None] * dg - sol.gL)))
    hard = (2.0 * np.pi / 3.0) * float(np.sum(rr * st.SIG ** 3 * sol.contact))
    return float(np.sum(st.rho)) + hard + J


def dbetaP_drho_virial(sol: SolutionMix, h: float = 2e-4) -> float:
    r"""`∂βp_virial/∂ρ` at fixed composition, by central difference.

    ⭐ Why this is worth having: the compressibility route's `∂βp/∂ρ = (A/2π)²` is a
    **perfect square** and therefore never negative, so that route has no van der
    Waals loop and no binodal at all.  The virial route carries no such constraint,
    so it has its own loop, its own spinodal and its own critical point — a *third*
    one, alongside the compressibility (`det F(0) = 0`) and energy routes.
    """
    st = sol.st
    out = []
    for d in (+h, -h):
        q = solve_bh_mix(_scale_rho(st, float(np.sum(st.rho)) + d))
        if q is None:
            return float("nan")
        out.append(pressure_virial(q))
    return (out[0] - out[1]) / (2.0 * h)


def free_energy_virial(st: HCYMix, n: int = 90, rho_max: float | None = None):
    r"""⭐⭐ `(ρ, βp, βμ)` along the isotherm by the **virial route** — and this is the
    one route that yields a binodal with **no crossing step at all**.

    `βp_virial` is closed form at every state (`pressure_virial`), so one quadrature
    of `p_ex/ρ²` gives both the free energy and the chemical potential:

        βΔA/V = ρ∫₀^ρ (p_ex/ρ′²)dρ′ ,   βμ = ln ρ + ∫₀^ρ (p_ex/ρ′²)dρ′ + p_ex/ρ

    with `p_ex = βp_virial − ρ` (finite as `ρ→0`, since `p_ex ~ B₂ρ²`).  Coexistence
    is then the ordinary Maxwell construction `p(ρ_v) = p(ρ_l)`, `μ(ρ_v) = μ(ρ_l)`.

    ⭐ **Why the compressibility route cannot do this and this one can.** That route
    is blocked twice over: `∂βp/∂ρ = (A/2π)²` is a *perfect square*, so `βA(ρ)` is
    convex on every isotherm and there is **no loop to construct on** even where the
    isotherm is whole; and below the fold apex the isotherm is cut as well.  The
    virial pressure carries neither constraint — it has its own van der Waals loop,
    and its critical point sits far above the fold:

        fold apex 1.0177  <  A=0 apex 1.0313  <<  virial T*_c 1.4132   (HSY, z = 1.8)

    so over the whole interval between them the isotherm is **not cut at all**.

    ⚠ **The honest window, measured** (`--gate` §13; `T/T_c` on the virial route).
    ⚠ The upper end is set by the **fold**, not by the coexistence solve: with the
    bracketed construction of `binodal_virial` the binodal tracks to `0.999·T_c`.

    | `T*` | `T/T_c` | isotherm cut? | min `A` on the loop | binodal |
    |---|---|---|---|---|
    | 1.35 … 1.05 | 0.955 … 0.743 | no | **+4.53 … +0.78** | ⭐ direct, and every state on the loop is physical |
    | 1.05 … 1.0177 | 0.743 … 0.720 | no | goes negative (−0.87 at 1.020) | direct, but the loop dips onto the wrong-winding sheet |
    | below 1.0177 | < 0.720 | **yes** | — | needs the same detour as the compressibility route |

    ⇒ The detour is removed over the top ~25 % of the dome and **not** below the fold
    apex.  The fold is common to every route — there the solution does not exist —
    so no route can remove it; what the virial route removes is the *perfect-square*
    obstruction, which was the compressibility route's own.

    ⚠ Self-consistent **within this route only.**  Its binodal does not coincide
    with the energy route's (`T*_c = 1.2237`) or with the compressibility hybrids —
    that disagreement is the route gap, now spanning **+37 %** in `T*_c` across the
    three.  ⚠ Note also that the energy route was never blocked (charging in `K` at
    fixed `ρ` does not cross the two-phase region); "needing a detour" was only ever
    the compressibility route's problem.
    """
    from scipy.interpolate import CubicSpline
    rho_max = rho_max or 6.0 * 0.47 / np.pi
    rs = np.concatenate([np.geomspace(1e-4, 0.02, 12), np.linspace(0.025, rho_max, n)])
    q = solve_bh_mix(_scale_rho(st, rs[0]))
    if q is None:
        return None
    u = np.concatenate([q.Dt.ravel(), q.Gt.ravel()])
    out = []
    for r in rs:
        s2 = _scale_rho(st, float(r))
        v = _solve_seeded(s2, u, *_scales(s2))
        if v is None:
            break                       # the fold: the isotherm is cut here
        u = v
        nn = s2.N * s2.N
        sol = SolutionMix(s2, u[:s2.T * nn].reshape(s2.T, s2.N, s2.N),
                          u[s2.T * nn:].reshape(s2.T, s2.N, s2.N),
                          _pieces(s2, u[:s2.T * nn].reshape(s2.T, s2.N, s2.N),
                                  u[s2.T * nn:].reshape(s2.T, s2.N, s2.N)), 0.0)
        out.append((float(r), pressure_virial(sol) - float(r),
                    float(np.sum(sol.p['A']))))
    if len(out) < 30:
        return None
    r, pex, A = (np.array(c) for c in zip(*out))
    prim = CubicSpline(r, pex / r ** 2).antiderivative()
    I = prim(r) - prim(r[0]) + pex[0] / r[0]     # + the ρ→0 limit of the integrand
    return r, r + pex, np.log(r) + I + pex / r, A


def binodal_virial(st: HCYMix, **kw):
    r"""`(ρ_v, ρ_l, βp_co)` by Maxwell construction on the virial isotherm, or `None`.

    ⚠⚠ **Corrects an earlier claim made here.** This used to say the construction
    "fails very close to `T_c` — measured, at `T/T_c = 0.977`" and to present that as
    a property of the method.  It was a property of *this function*: it solved
    `p(ρ_v)=p(ρ_l)`, `μ(ρ_v)=μ(ρ_l)` with a 2-D `fsolve` seeded at fixed multiples
    of the loop edges (`0.5·ρ_v`, `1.6·ρ_l`).  Near `T_c` the loop is a few
    hundredths wide and those multiples put the seed outside the roots entirely.

    ⭐ Replaced by a **bracketed bisection**, which has no seed to misplace: `p` is
    monotone on each stable branch, so tabulate `μ` against `p` there and find the
    single crossing of `μ_gas(p)` and `μ_liq(p)`.  Measured on the same isotherms
    and the same `p`, `μ`: it resolves `T/T_c = 0.999` on **both** models, and where
    the old version also worked (`0.960`) the two agree to `1e-4`.

    ⚠ Refining the `ρ` grid does **not** help the old version (`n = 90 → 300`
    changes nothing) — which is how the seed, rather than resolution, was
    identified as the cause.

    ⚠ `None` still means something real below the fold apex: there the isotherm is
    genuinely cut and no construction can bridge it.
    """
    from scipy.optimize import brentq
    res = free_energy_virial(st, **kw)
    if res is None:
        return None
    r, p, mu, A = res
    d = np.diff(p) / np.diff(r)
    neg = np.where(d < 0)[0]
    if len(neg) == 0:
        return None                      # no loop at this T
    i, j = int(neg[0]), int(neg[-1])
    if i < 3 or j > len(r) - 5:
        return None
    # ⚠⚠ WRONG-WINDING GUARD.  `a = 1 − ρĉ(0) = F(0)²` is a **square**, so every
    # compressibility-type quantity is blind to the sign of `F(0)` and a walk that
    # runs past a spinodal keeps producing perfectly ordinary numbers on a branch
    # where the Wiener–Hopf factor has the wrong winding and `h(r)` is not a valid
    # correlation function.  `A > 0` is the only detector, and it has to be applied
    # explicitly — measured: without it the liquid "branch" could consist of a
    # single wrong-winding point next to the fold and still pass the length test.
    ok = A > 0.0
    g_end = i
    while g_end > 0 and not ok[g_end - 1]:
        g_end -= 1
    l_start = j + 1
    while l_start < len(r) and not ok[l_start]:
        l_start += 1
    if g_end < 5 or len(r) - l_start < 5:
        return None
    pg, mg, rg = p[:g_end], mu[:g_end], r[:g_end]
    pl, ml, rl = p[l_start:], mu[l_start:], r[l_start:]
    lo, hi = max(pg.min(), pl.min()), min(pg.max(), pl.max())
    if not (hi > lo):
        return None
    f = lambda q: np.interp(q, pg, mg) - np.interp(q, pl, ml)
    if f(lo) * f(hi) > 0:
        return None
    q = brentq(f, lo, hi, xtol=1e-14)
    a, b = float(np.interp(q, pg, rg)), float(np.interp(q, pl, rl))
    return (a, b, float(q)) if b - a > 1e-4 else None


def free_energy_excess(sol: SolutionMix) -> float:
    r"""⭐ `βΔA/V` over hard spheres, **closed form — no thermodynamic integration**.

    Blum & Høye (45), quoting Høye & Stell:  `−βΔA = ρΔφ^E − βΔE − ½ρ[χ⁻¹−χ₀⁻¹]`,
    with `ρΔφ^E = βΔp` from `pressure_excess` and `χ⁻¹ = ∂βp/∂ρ` from (39).

    ⭐ **That MSA has a closed-form free energy at all is the point** — it is why
    an *exact* energy-route/compressibility-route comparison is possible here,
    where Group X could only do it numerically and only at first order.  Note
    (45) is not an independent statement: it is Gibbs–Duhem
    (`βΔA/V = Σ_iρ_iβμ_i^E − βΔp`) applied to (46), since
    `Σ_iρ_iβμ_i^E = ½ρ[χ⁻¹−χ₀⁻¹] + βΔE`.

    Gated in `--gate` §8 against `free_energy_charging`, the same quantity by
    λ-integration of (42).
    """
    return -(pressure_excess(sol) - excess_energy(sol)
             - 0.5 * float(np.sum(sol.st.rho)) * (sol.dbetaP_drho - _a_hs(sol.st)))


def chem_pot_excess(sol: SolutionMix) -> np.ndarray:
    r"""`βμ_i^E` — excess chemical potentials, **closed form**.  Their (46):

        −βρ_iμ_i^E = ½Σ_j ρ_iρ_j[ĉ_ij(0) − ĉ⁰_ij(0)] + 2πΣ_j ρ_iρ_j K_ij Gt_ij

    (the printed `ρ_i` cancels).  ✅ Verified directly against
    `∂(βΔA/V)/∂ρ_i` of the charging integral — this equation is printed
    correctly, unlike (43).
    """
    st = sol.st
    st0 = HCYMix(st.sigma, st.rho, st.z, 0.0 * st.K)
    dc = chat0(sol) - chat0(solve_bh_mix(st0))
    tail = 2.0 * np.pi * np.einsum('j,tij,tij->i', st.rho, st.K, sol.Gt)
    return -(0.5 * (dc @ st.rho) + tail)


def free_energy_charging(st: HCYMix, n_lam: int = 24) -> float:
    r"""`βΔA/V` the long way — λ-integration of (42). The gate on the closed form.

    `(∂βA/∂β)_{ρ,V} = E` and `K ∝ β`, so with `K(λ) = λK` and `β′ = λβ`

        βΔA/V = −2π ∫₀¹ dλ Σ_ij ρ_iρ_j Σ_ν K^ν_ij Gt^ν_ij(λ)

    The integrand is regular at `λ = 0` (`Gt → ĝ_HS e^{zσ}`), which is why it is
    written in `Gt` and not in `ĝ`.  Uses the traced branch, interpolated onto
    Gauss–Legendre nodes — a per-λ re-solve would drop the branch near the fold.
    """
    tr = trace_coupling_mix(st, lam_max=1.0)
    lam = np.array([t[0] for t in tr])
    rr = st.rho[:, None] * st.rho[None, :]
    val = np.array([-2.0 * np.pi * float(np.sum(rr[None] * st.K * q.Gt))
                    for _, q in tr])
    if lam[-1] < 1.0 - 1e-9:
        return float("nan")                     # branch ended before full coupling
    x, w = np.polynomial.legendre.leggauss(n_lam)
    xs = 0.5 * (x + 1.0)
    return float(np.sum(0.5 * w * np.interp(xs, lam, val)))


def free_energy_compressibility(st: HCYMix, n_rho: int = 40) -> float:
    r"""⭐ `βΔA/V` by the **compressibility route** — one quadrature, not two.

    The route integrates `∂βp/∂ρ` twice in density along the isopleth.  Swapping
    the order of the two integrals collapses them:

        βΔA/V = ρ∫₀^ρ dρ′ [a(ρ′) − a₀(ρ′)]·(1/ρ′ − 1/ρ)

    with `a` from (39) and `a₀` its hard-sphere value at the same `ρ′`.  The
    integrand is regular at `ρ′ = 0` because `a − a₀ = O(ρ′)`.  ⭐ **No `k`-space
    quadrature anywhere** — `a` is closed-form — so this is `n_rho` algebraic
    solves, against the grid-based `∂βp/∂ρ` the numerical route needs.

    ⚠ For a mixture this is the free energy **along one isopleth**, which is the
    correct object to compare with (45) at that state and nothing more: the
    closure is not thermodynamically consistent, so the two-dimensional
    integration in `{ρ_i}` is path-dependent.
    """
    rho_t = float(np.sum(st.rho))
    x = st.rho / rho_t
    nodes, w = np.polynomial.legendre.leggauss(n_rho)
    rs = 0.5 * rho_t * (nodes + 1.0)
    tot = 0.0
    for rp, wt in zip(rs, 0.5 * rho_t * w):
        s2 = HCYMix(st.sigma, x * rp, st.z, st.K)
        q = solve_bh_mix(s2)
        if q is None:
            return float("nan")
        a0 = float(np.sum(s2.rho * (s2.A0 / (2.0 * np.pi)) ** 2) / rp)
        tot += wt * (q.dbetaP_drho - a0) * (1.0 / rp - 1.0 / rho_t)
    return rho_t * tot


def free_energy_compressibility_via(st: HCYMix, T_ratio: float,
                                    n_rho: int = 40) -> float:
    r"""`βΔA/V` by the **supercritical detour**: ρ-integration at a weaker coupling,
    then energy-route cooling at fixed `ρ`.

    The isothermal compressibility route dies whenever the path from `ρ = 0` to the
    target crosses the two-phase region — i.e. for every liquid state below `T_c`.
    The standard repair is to integrate in `ρ` at a **supercritical** reference
    (`K → K/T_ratio`, `T_ratio > 1`), where the whole isotherm is single phase, and
    then bring the state down at fixed `ρ`.  This works, and `--gate` §10 measures
    what it costs.

    ⚠ **The answer is not unique, and the ambiguity is the whole route gap.** The
    cooling leg is `∂(βA)/∂β = E`, i.e. the *energy* route, so

        A_hybrid(ρ,T;T_ref) − A_energy(ρ,T)  =  A_compr(ρ,T_ref) − A_energy(ρ,T_ref)

    — the hybrid differs from the pure energy route by **the route gap evaluated at
    the crossing temperature**, and by nothing else.  Since that gap shrinks as
    `T_ref` rises, the family interpolates continuously from the compressibility
    route (`T_ref → T`, where it exists) to the energy route (`T_ref → ∞`).
    Measured on liquid states of HCY `z = 1.8` below `T_c`: `T_ref/T_c = 1.07`
    gives −11 % against the energy route, `T_ref/T_c = 100` gives −0.05 %, a
    **spread of ~10 %** with nothing in the thermodynamics to prefer a member.

    ⇒ **"the compressibility-route free energy of a liquid" is not a well-defined
    object below `T_c`.** What *is* well defined — and needs no path at all — is
    `free_energy_excess`, and phase equilibria follow from it and (46); see
    `compare_msa_vle_exact.py`.
    """
    stR = HCYMix(st.sigma, st.rho, st.z, st.K / T_ratio)
    ac = free_energy_compressibility(stR, n_rho=n_rho)
    q, qR = solve_bh_mix(st), solve_bh_mix(stR)
    if q is None or qR is None or not np.isfinite(ac):
        return float("nan")
    return ac + free_energy_excess(q) - free_energy_excess(qR)


# ───────────────────────────────────────────────────────────────────────────────
# ⭐ the fold route — the canonical "semi-compressibility" liquid free energy
# ───────────────────────────────────────────────────────────────────────────────
#
# Below the fold apex the isotherm is cut: the physical vapour arc ends at
# `ρ_fold_v` and the physical liquid arc begins at `ρ_fold_l`, and they are
# separate connected components of the real solution locus (the root pair is
# complex conjugate in between).  The compressibility route therefore cannot reach
# a liquid on its own — see `free_energy_compressibility`.
#
# This is the repair that gives up the least.  Transport the free energy in the
# (ρ, β) plane with
#
#     d(βΔA/V) = βΔμ dρ + ΔE dβ
#     d(βΔμ)   = [(a − a₀)/ρ] dρ + (∂ΔE/∂ρ) dβ
#
# — `a` from (39) is the **compressibility** route and supplies the `ρ` direction;
# `ΔE` from (42) is the **energy** route and supplies the `β` direction, which
# `ĉ(0)` does not constrain at all — along the path
#
#     (0, T) → (ρ_fold_v(T), T) → [up the fold curve, over the apex, down] →
#     (ρ_fold_l(T), T) → (ρ_target, T)
#
# ⭐ **Why the fold curve and not the spinodal curve.** The fold is the actual
# boundary of the domain on which `a(ρ)` exists; between spinodal and fold the
# compressibility route is perfectly well defined, just on the wrong-winding
# (`A < 0`) branch.  Hugging the fold therefore means "use the compressibility
# route on its entire domain and lift only where it genuinely has none".  It is
# also cheaper: the fold dome closes at `T*_fold = 1.0177` for HCY `z = 1.8`
# against `T_c = 1.0313` for the spinodal dome.  Measured cost of the difference:
# −22.2 % (fold) vs −17.4 % (spinodal) against the energy route at `η = 0.35`,
# `T* = 0.85`.
#
# ⚠ **This is a labelled hybrid, not "the compressibility-route free energy".**
# The `dβ` legs are energy-route input and cannot be removed.  What the fold curve
# buys is that the path is no longer chosen: it is fixed by the theory, and the
# apex gives an exact anchor (the two boundary points coincide there, so the
# free-energy difference across the gap starts at zero).  Different-but-reasonable
# paths spread over the whole route gap — see the write-up's table.

def _solve_seeded(st: HCYMix, seed, sc1, sc2, tol: float = 1e-9):
    """One fsolve on `_residuals` from an explicit seed; `None` if it misses."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = fsolve(_residuals, seed, args=(st, sc1, sc2), xtol=1e-13)
    r = float(np.linalg.norm(_residuals(u, st, sc1, sc2)))
    return u if r < tol else None


def _scale_rho(st: HCYMix, rho_tot: float) -> HCYMix:
    """The same state at a different total density, composition held fixed."""
    x = st.rho / float(np.sum(st.rho))
    return HCYMix(st.sigma, x * rho_tot, st.z, st.K)


def _jacobian(u, st: HCYMix, sc1, sc2, h: float = 1e-7) -> np.ndarray:
    """`∂residuals/∂u`, by finite differences (the system is 2N²ω-dimensional)."""
    r0 = _residuals(u, st, sc1, sc2)
    J = np.empty((len(u), len(u)))
    for j in range(len(u)):
        up = np.array(u, dtype=float)
        up[j] += h
        J[:, j] = (_residuals(up, st, sc1, sc2) - r0) / h
    return J


def fold_coupling(st: HCYMix, lam_lo: float = 1.0, seed=None,
                  tol: float = 1e-9):
    r"""⭐ The fold, as the solution of an **augmented algebraic system** — not as the
    place where a continuation dies.

    A fold is a turning point of the branch, i.e. exactly where the system's
    Jacobian is singular.  So instead of stepping until `fsolve` fails, solve

        residuals(u; λ) = 0     and     det ∂residuals/∂u (u; λ) = 0

    for `(u, λ)` together — `2N²ω + 1` equations in `2N²ω + 1` unknowns, with `λ`
    the coupling multiplier (`K → λK`, so `T* = T*_state/λ`).

    ⭐ **This removes the failure mode entirely.** Locating a fold by continuation
    is self-defeating: the continuation is *designed* to stop there, so the answer
    is only as good as the step size, and any transport that then wants to run near
    the fold hits the same wall.  The augmented system converges *onto* the fold
    from either side and returns it to solver tolerance.

    Returns `None` if no fold is bracketed below `lam_lo`-scaled couplings.
    """
    n = st.N * st.N * st.T * 2
    sc1, sc2 = _scales(st)
    T0 = 1.0 / float(np.max(st.K))

    def sys(v):
        u, lam = v[:n], float(v[n])
        s = HCYMix(st.sigma, st.rho, st.z, st.K * lam)
        r = _residuals(u, s, sc1, sc2)
        d = np.linalg.det(_jacobian(u, s, sc1, sc2))
        return np.append(r, d)

    if seed is None:
        # seed just past the spinodal, where the branch is already on `A < 0` and
        # the fold is close — the tracer gets there cheaply and only once
        tr = trace_coupling_mix(st, lam_max=8.0)
        if len(tr) < 3:
            return None, None
        lam0 = tr[-1][0]
        seed = np.append(np.concatenate([tr[-1][1].Dt.ravel(), tr[-1][1].Gt.ravel()]),
                         lam0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = fsolve(sys, seed, xtol=1e-13)
    if float(np.linalg.norm(sys(v))) > tol:
        return None, None
    return float(v[n]), v[:n]


def fold_density(st: HCYMix, rho_seed: float, u_seed=None, tol: float = 1e-9):
    r"""⭐ The **exact** fold density at this state's `T`, from `det J = 0`.

    The fold is a turning point of the branch in `ρ`, i.e. exactly where the
    Jacobian with respect to the unknowns is singular.  So solve

        residuals(u; ρ) = 0     and     det ∂residuals/∂u (u; ρ) = 0

    for `(u, ρ)` together — `2N²ω + 1` equations, `2N²ω + 1` unknowns.

    ⭐⭐ **This is what removes the fold route's failure mode.** Locating the fold
    by stepping in `ρ` until the continuation dies is self-defeating twice over:
    the answer is only as accurate as the step, and any *interpolation* of the
    resulting curve overshoots it, because the curve is convex near the apex.
    Measured consequence of the old way: an arc inset by 2 % still landed at 100.2 %
    of the true fold at `T* = 0.9869` and the transport died there — nowhere near
    the apex, which is where the trouble was assumed to be.  With the augmented
    solve the curve is exact at every temperature and the inset means what it says.

    Returns `(ρ_fold, u_fold)`, or `(None, None)` if the Newton misses.
    """
    n = 2 * st.T * st.N * st.N
    sc1, sc2 = _scales(st)
    if u_seed is None:
        q = solve_bh_mix(_scale_rho(st, rho_seed))
        if q is None:
            return None, None
        u_seed = np.concatenate([q.Dt.ravel(), q.Gt.ravel()])

    def sys(v):
        u, rho = v[:n], float(v[n])
        s = _scale_rho(st, max(rho, 1e-6))
        return np.append(_residuals(u, s, sc1, sc2),
                         np.linalg.det(_jacobian(u, s, sc1, sc2)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = fsolve(sys, np.append(u_seed, rho_seed), xtol=1e-13)
    if float(np.linalg.norm(sys(v))) > tol:
        return None, None
    return float(v[n]), v[:n]


def fold_densities(st: HCYMix, rho_max: float | None = None,
                   n: int = 700) -> tuple[float | None, float | None]:
    r"""`(ρ_fold_v, ρ_fold_l)` on this isopleth at this `T` — the ends of the two arcs.

    ⚠ **Kept for the dome map and for `fold_apex`'s "is the isotherm cut?" test**,
    where only a yes/no is needed.  For anything that has to *work near* the fold —
    transport, the fold route — use `fold_coupling`, which solves the augmented
    `det J = 0` system instead of stepping until the continuation dies.

    Returns `(None, None)` when the isotherm is **not** cut, i.e. above the fold
    apex, where the continuation runs through the `A < 0` interior and lands on the
    physical liquid.  That is the case in which `free_energy_compressibility` works
    unaided.
    """
    rho_max = rho_max or 6.0 * 0.48 / np.pi
    sc1, sc2 = _scales(st)

    def sweep(r0, r1):
        s0 = _scale_rho(st, r0)
        u = solve_bh_mix(s0)
        if u is None:
            return None
        seed = np.concatenate([u.Dt.ravel(), u.Gt.ravel()])
        last = r0
        for r in np.linspace(r0, r1, n)[1:]:
            v = _solve_seeded(_scale_rho(st, r), seed, sc1, sc2)
            if v is None:
                return last
            seed, last = v, r
        return None                      # survived: no fold on this sweep

    lo = sweep(1e-3, rho_max)
    if lo is None:
        return None, None                # isotherm open — no arc needed
    hi = sweep(rho_max, 1e-3)
    return lo, hi


def fold_apex(st: HCYMix, t_lo: float, t_hi: float, tol: float = 1e-4) -> float:
    """The temperature at which the fold dome closes on this isopleth.

    Bisection on "is the isotherm cut?", which `fold_densities` answers directly.
    `t_lo` must be below it (cut) and `t_hi` above (open)."""
    T0 = 1.0 / float(np.max(st.K))          # the state's own T scale, for K ∝ 1/T
    while t_hi - t_lo > tol:
        m = 0.5 * (t_lo + t_hi)
        s = HCYMix(st.sigma, st.rho, st.z, st.K * (T0 / m))
        t_lo, t_hi = (m, t_hi) if fold_densities(s)[0] is not None else (t_lo, m)
    return 0.5 * (t_lo + t_hi)


def _transport(st: HCYMix, path: np.ndarray, h: float = 2e-4,
               S0: float = 0.0, m0: float = 0.0, seed=None):
    r"""Carry `(βΔA/V, βΔμ)` along `path`, an array of `(ρ_tot, T)` points.

    Midpoint rule on the two 1-forms of the section header.  `∂ΔE/∂ρ` is evaluated
    only on legs that actually change `β` — on a pure `ρ` leg its coefficient is
    zero, which halves the work on the two long legs.

    ⚠⚠ **The solution is CONTINUED along the path, not re-solved at every point.**
    `solve_bh_mix` re-ramps in `K` from zero at each state, and near the fold that
    ramp is itself about to hit the coupling-direction turning point — so a
    re-solving transport dies exactly where the fold route needs it to work
    (measured: `η = 0.1512, T* = 1.0136`, i.e. 0.996·apex, on an arc that is
    perfectly well-defined). Seeding each step from the previous solution walks
    through, because nothing on the path is near a turning point *in the path's own
    direction*.  Same lesson as everywhere else here: trace, do not re-ramp.
    """
    T0 = 1.0 / float(np.max(st.K))
    sc1, sc2 = _scales(st)
    S, m = S0, m0
    u = seed

    def at(rho, T):
        return HCYMix(st.sigma, (st.rho / np.sum(st.rho)) * rho, st.z,
                      st.K * (T0 / T))

    if u is None:
        q = solve_bh_mix(at(*path[0]))
        if q is None:
            return None, None
        u = np.concatenate([q.Dt.ravel(), q.Gt.ravel()])
    for k in range(len(path) - 1):
        r0, t0 = path[k]
        r1, t1 = path[k + 1]
        rm, tm = 0.5 * (r0 + r1), 0.5 * (t0 + t1)
        s = at(rm, tm)
        u = _solve_seeded(s, u, sc1, sc2)
        if u is None:
            return None, None
        nn = st.N * st.N
        q = SolutionMix(s, u[:st.T * nn].reshape(st.T, st.N, st.N),
                        u[st.T * nn:].reshape(st.T, st.N, st.N),
                        _pieces(s, u[:st.T * nn].reshape(st.T, st.N, st.N),
                                u[st.T * nn:].reshape(st.T, st.N, st.N)), 0.0)
        dr, db = r1 - r0, 1.0 / t1 - 1.0 / t0
        E = excess_energy(q)
        if abs(db) > 0.0:
            up = _solve_seeded(_scale_rho(s, rm + h), u, sc1, sc2)
            um = _solve_seeded(_scale_rho(s, rm - h), u, sc1, sc2)
            if up is None or um is None:
                return None, None

            def _E(v, ss):
                nn2 = st.N * st.N
                return excess_energy(SolutionMix(
                    ss, v[:st.T * nn2].reshape(st.T, st.N, st.N),
                    v[st.T * nn2:].reshape(st.T, st.N, st.N),
                    _pieces(ss, v[:st.T * nn2].reshape(st.T, st.N, st.N),
                            v[st.T * nn2:].reshape(st.T, st.N, st.N)), 0.0))
            dE = (_E(up, _scale_rho(s, rm + h))
                  - _E(um, _scale_rho(s, rm - h))) / (2.0 * h)
        else:
            dE = 0.0
        S += m * dr + E * db
        m += (q.dbetaP_drho - _a_hs(s)) / rm * dr + dE * db
    return S, m


def fold_curve(st: HCYMix, t_apex: float, n_ladder: int = 24):
    """`(T, ρ_fold_v, ρ_fold_l)` on a ladder from this state's `T` up to the apex.

    Each rung is an **exact** `det J = 0` solve (`fold_density`), seeded by the
    cheap sweep bracket at that rung.

    ⚠ **With a fallback, and it is load-bearing.** The augmented Newton is a
    `2N²ω+1`-dimensional solve on a determinant, and for the LJ 2-Yukawa fit — two
    tails whose rates differ 5× (2.96 / 14.0), so a badly scaled 4×4 Jacobian — it
    misses at most rungs. Seeding it from the *previous* rung then propagates the
    failure: measured, `fold_curve` came back with **3 usable rungs out of 24**, the
    interpolant was meaningless and every downstream arc failed. Seed from the
    sweep at each rung independently, and where the exact solve still misses, keep
    the sweep value — it is accurate to the sweep step (~5e-4 in `η`), far inside
    the arc's own inset.
    """
    T0 = 1.0 / float(np.max(st.K))
    tl = np.linspace(T0, t_apex, n_ladder)
    lv = np.full(n_ladder, np.nan)
    ll = np.full(n_ladder, np.nan)
    for i, t in enumerate(tl):
        s = HCYMix(st.sigma, st.rho, st.z, st.K * (T0 / t))
        rv, rl = fold_densities(s, n=260)
        if rv is None:                      # above the apex: no fold on this rung
            continue
        a, _ = fold_density(s, rv)
        b, _ = fold_density(s, rl)
        lv[i] = a if a is not None else rv
        ll[i] = b if b is not None else rl
    ok = ~np.isnan(lv) & ~np.isnan(ll)
    return tl[ok], lv[ok], ll[ok]


def _fold_ladder(st: HCYMix, t_apex: float, n_arc: int, inset: float,
                 n_ladder: int, curve=None, t_hug: float = 0.99):
    """The inset fold curve, sampled on `n_arc` temperatures below the apex."""
    T0 = 1.0 / float(np.max(st.K))
    tl, lv, ll = curve if curve is not None else fold_curve(st, t_apex, n_ladder)
    # ⚠ Stop short of the apex.  Right at it the dome is only ~0.006 wide in η and
    # even a continued transport is squeezed between the two arcs; follow the curve
    # to `t_hug·t_apex` and let the crossing leg — which runs above the apex, where
    # the dome has closed — cover the rest.
    ts = np.linspace(T0, t_hug * t_apex, n_arc)
    # ⚠ Monotone cubic, not `np.interp`: the fold curve is convex near the apex and
    # a linear interpolant crosses it — which is what killed the arc at 100.2 % of
    # the true fold, far below the apex.
    from scipy.interpolate import PchipInterpolator
    return (ts, PchipInterpolator(tl, lv)(ts) * (1.0 - inset),
            PchipInterpolator(tl, ll)(ts) * (1.0 + inset))


def fold_gap_free_energy(st: HCYMix, mu_v: float, n_arc: int = 120,
                         inset: float = 0.02, t_apex: float | None = None,
                         t_over: float = 1.002, n_ladder: int = 11,
                         curve=None, t_hug: float = 0.99) -> tuple:
    r"""⭐ `(Δ, μ_l)` — the jump **across the fold gap**, given `μ_v` on the vapour side.

    The object the method turns on: the free-energy difference between the
    liquid-side and vapour-side fold points *at the same temperature*, obtained by
    going **up the fold curve to the apex and back down**,

        Δ(T) = ∫[vapour curve, T→T_apex] + [crossing] + ∫[liquid curve, T_apex→T]

    with the two 1-forms of the section header.

    ⚠ **`Δ` is not a function of `T` alone**, and treating it as one costs 22 %:
    the fold curve *moves in `ρ`* as `T` changes, so the arc carries a `βΔμ dρ`
    contribution and needs the **actual** `μ` at the vapour-side fold point.
    Starting the arc at `μ = 0` gives −2.166 at `η = 0.35, T* = 0.85` where the
    correct value is −2.786.

    ⭐ But the dependence is **exactly affine** — see `fold_gap_coefficients`, which
    returns the three `T`-only functions `(Δρ, Δ₀, ΔM)` with
    `Δ = μ_v·Δρ + Δ₀` and `μ_l = μ_v + ΔM`.  So the gap *can* be tabulated on `T`
    alone; it just takes three functions rather than one.

    ⚠ **The apex anchor is taken, not asserted.** `Δ(T_apex) = 0` because the two
    boundary points coincide there — but at finite `inset` the two arc ends are a
    few per cent apart, so rather than setting the jump to zero the crossing is made
    as a short **real compressibility-route leg** at `t_over·T_apex`, just above the
    apex where the dome has closed and the isotherm is genuinely open.
    """
    T0 = 1.0 / float(np.max(st.K))
    if t_apex is None:
        t_apex = spinodal_apex(st, T0, 4.0 * T0)      # ⚠ A=0 apex, not the fold's
    ts, fv, fl = _fold_ladder(st, t_apex, n_arc, inset, n_ladder, curve, t_hug)
    tx = t_over * t_apex
    m_ = max(12, n_arc // 6)

    def leg(r0, t0, r1, t1, n):
        return np.stack([np.linspace(r0, r1, n), np.linspace(t0, t1, n)], axis=1)

    arc = np.concatenate([
        np.stack([fv, ts], axis=1),                    # up the vapour-side curve
        leg(fv[-1], ts[-1], fv[-1], tx, m_),           # clear the apex
        leg(fv[-1], tx, fl[-1], tx, m_),               # cross; isotherm open here
        leg(fl[-1], tx, fl[-1], ts[-1], m_),
        np.stack([fl[::-1], ts[::-1]], axis=1)])       # down the liquid-side curve
    return _transport(st, arc, S0=0.0, m0=mu_v)


def fold_gap_coefficients(st: HCYMix, n_arc: int = 120, inset: float = 0.02,
                          t_apex: float | None = None, t_over: float = 1.002,
                          n_ladder: int = 24, curve=None, t_hug: float = 0.99):
    r"""⭐⭐ `(Δρ, Δ₀, ΔM)` — the fold gap as **three functions of `T` alone**.

    The transport across the arc is *affine* in the incoming chemical potential,
    exactly and for a structural reason: the two 1-forms are

        dS = m dρ + ΔE dβ
        dm = [(a−a₀)/ρ] dρ + (∂ΔE/∂ρ) dβ        ← contains neither S nor m

    so `m` along the arc is `μ_v + M(s)` with `M` a pure path functional, and
    substituting into the first gives

        Δ(T; μ_v) = μ_v·[ρ_l(T) − ρ_v(T)]  +  Δ₀(T)
        μ_l(T; μ_v) = μ_v + ΔM(T)

    ⭐ **So the gap *can* be tabulated on `T` alone — as three functions, not one.**
    Verified to machine precision: `dΔ/dμ_v` equals the gap width `ρ_l − ρ_v` to
    2.2e-16 and `dμ_l/dμ_v = 1` exactly, at every temperature tested.

    Practical consequence: compute these once per isotherm (or spline them once per
    model) and the fold route costs two `ρ` quadratures and a lookup — the arc, the
    expensive part, never has to be redone for a different target density.
    """
    T0 = 1.0 / float(np.max(st.K))
    if t_apex is None:
        t_apex = fold_apex(st, T0, 4.0 * T0)
    if curve is None:
        curve = fold_curve(st, t_apex, n_ladder)
    args = (n_arc, inset, t_apex, t_over, n_ladder, curve, t_hug)
    d0, m0 = fold_gap_free_energy(st, 0.0, *args)
    if d0 is None:
        return None, None, None
    d1, m1 = fold_gap_free_energy(st, 1.0, *args)
    if d1 is None:
        return None, None, None
    return float(d1 - d0), float(d0), float(m0)


def free_energy_fold_route(st: HCYMix, n_leg: int = 500, n_arc: int = 120,
                           inset: float = 0.02, t_apex: float | None = None,
                           t_over: float = 1.002, n_ladder: int = 11,
                           curve=None, t_hug: float = 0.99) -> float:
    r"""⭐ `βΔA/V` by the **fold route** — the canonical semi-compressibility method.

    Three steps, and only the middle one is not the compressibility route:

    1. `ρ = 0 → ρ_fold_v(T)` along the target isotherm — compressibility route;
    2. `Δ` across the gap from `fold_gap_free_energy`, **fed the `βΔμ` that step 1
       ends on** — up the fold curve to the apex, across, and back down;
    3. `ρ_fold_l(T) → ρ_target` along the target isotherm — compressibility route.

    Above the fold apex the isotherm is not cut, step 2 is skipped entirely, and
    this returns the **pure** compressibility route — which is the gate: there it
    must agree with `free_energy_compressibility`.

    ⛔⛔ **UNSTABLE — treat the numbers as suspect.** The arc integrates along the
    locus where `det J = 0`; that is what a fold is, so the transport is
    ill-conditioned by construction and survives only on the inset. Measured
    `cond(J)` on the arc: HSY 9.5 → 31 over insets 0.10 → 0.01, but the LJ
    2-Yukawa 192 → 588 (4×4, tail rates differing 5×). ⚠ The failure is **silent**:
    `_solve_seeded` accepts on a scaled residual under 1e-9, and near a singular
    Jacobian that does not bound the error in the solution — so the symptom is an
    answer that drifts with the inset, not a crash. 2YK drifts by a **factor 4**
    between insets 0.02 and 0.05 where HSY moves 1.6 %. **Always run an inset
    convergence study before quoting a number from this route**; HSY's `√inset`
    convergence is incidental to its low conditioning, not a property of the method.
    """
    rho_t = float(np.sum(st.rho))
    T0 = 1.0 / float(np.max(st.K))
    rv, rl = fold_densities(st)
    if rv is None:                      # isotherm open — no beta excursion is taken
        path = np.stack([np.linspace(1e-4, rho_t, 4 * n_leg),
                         np.full(4 * n_leg, T0)], axis=1)
        return _transport(st, path)[0]
    if t_apex is None:
        t_apex = fold_apex(st, T0, 4.0 * T0)
    ts, fv, fl = _fold_ladder(st, t_apex, n_arc, inset, n_ladder, curve, t_hug)

    def leg(r0, r1, n):
        return np.stack([np.linspace(r0, r1, n), np.full(n, T0)], axis=1)

    S, m = _transport(st, leg(1e-4, fv[0], n_leg))                        # step 1
    if S is None:
        return float("nan")
    dS, m = fold_gap_free_energy(st, m, n_arc, inset, t_apex, t_over, n_ladder,
                                 curve, t_hug)
    if dS is None:
        return float("nan")
    S, m = _transport(st, leg(fl[0], rho_t, n_leg), S0=S + dS, m0=m)      # step 3
    return float("nan") if S is None else S


# ───────────────────────────────────────────────────────────────────────────────
# ⭐ the SPINODAL route — the usable semi-compressibility free energy
# ───────────────────────────────────────────────────────────────────────────────
#
# Same idea as the fold route above, with the arc moved onto the `A = 0` spinodal.
# That one change is what makes it a method rather than a bound:
#
#   * a fold IS `det J = 0`, so an arc hugging it is singular in the limit —
#     `cond(J)` measured 9.5 → 25 (HSY) and 192 → 469 (LJ 2-Yukawa) as the inset
#     shrinks, and the transport converges at best like `√inset`.  For 2YK it does
#     not converge at all: `Δ` drifts by a **factor 4** between insets 0.02 and
#     0.05, silently, because a small scaled residual does not bound the error in
#     the solution near a singular Jacobian;
#   * the spinodal is a **regular** point of the same system (`|det J|` = 2e-02 …
#     8e-01 there), so `cond(J)` **saturates** — 11.7 and 156 — and `Δ` comes out
#     **linear in the inset** (`R² = 0.9998 … 0.99999` on four state points).  One
#     Richardson step therefore reaches `inset → 0` exactly, which is what
#     `spinodal_gap_free_energy` returns.
#
# Two structural gains as well: the `ρ` legs stay entirely on the physical `A > 0`
# winding (the fold route's had to cross into `A < 0`), and because `A = 0` is
# regular the legs can be integrated right up to the spinodal densities.
#
# ⚠ What does NOT change: this is still a labelled hybrid, not a route.  The `dβ`
# legs are energy-route input, and both ends of the resulting coexistence region
# terminate at finite tie-line width — there is no critical point.  See
# `numerical_notes/results/msa_exact/msa_exact_phase_lines.md`.

def spinodal_density(st: HCYMix, rho_seed: float, u_seed=None, tol: float = 1e-9):
    r"""⭐ The **exact** `A = 0` density at this state's `T`, from an augmented solve.

    The `ρ`-space twin of `fold_density`, with the singular condition replaced by
    the regular one:

        residuals(u; ρ) = 0     and     det F(0; u, ρ) = 0

    `2N²ω + 1` equations in `2N²ω + 1` unknowns.  ⭐ `det F(0)` is **the** spinodal
    indicator — one condition at every `N` — whereas `∂βp/∂ρ` is a sum of squares
    ((39)) and never changes sign; at `N = 1` it reduces to `A/2π = F(0) = 0`.

    ⭐⭐ **Why this replaces `spinodal_densities`.** That one continues in `ρ` from
    each end and brackets the sign change of `A`, so it inherits the continuation's
    reach: the vapour-side sweep dies at the *fold*, which sits just past the
    vapour spinodal, and the liquid-side sweep does the same below `T* ≈ 0.72` —
    returning `None` on a side where the spinodal density exists perfectly well.
    (That `None` is what crashed `--gate` §12 at `T* = 0.70`.)  Solving the
    augmented system asks about the spinodal directly and never walks toward the
    fold at all.

    Because `A = 0` is a **regular** point of the same system (`|det J|` =
    2e-02…8e-01 there, theory note §8a), the Newton is far better conditioned than
    `fold_density`'s — the extra row is the only one that is nearly singular there,
    and here it is not.

    Returns `(ρ_sp, u_sp)`, or `(None, None)` if the Newton misses.  Which of the
    two spinodal densities you get is set by `rho_seed`: seed below the dome for the
    vapour root, above it for the liquid root.
    """
    n = 2 * st.T * st.N * st.N
    n2 = st.N * st.N
    sc1, sc2 = _scales(st)
    if u_seed is None:
        q = solve_bh_mix(_scale_rho(st, rho_seed))
        if q is None:
            return None, None
        u_seed = np.concatenate([q.Dt.ravel(), q.Gt.ravel()])

    def sys(v):
        u, rho = v[:n], float(v[n])
        s = _scale_rho(st, max(rho, 1e-6))
        p = _pieces(s, u[:st.T * n2].reshape(st.T, st.N, st.N),
                    u[st.T * n2:].reshape(st.T, st.N, st.N))
        dF = float(np.real(np.linalg.det(
            np.eye(s.N) - s.rho[:, None] * qhat(s, p, 0.0))))
        return np.append(_residuals(u, s, sc1, sc2), dF)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = fsolve(sys, np.append(u_seed, rho_seed), xtol=1e-13)
    if float(np.linalg.norm(sys(v))) > tol:
        return None, None
    return float(v[n]), v[:n]


def spinodal_densities_exact(st: HCYMix, seeds=None, rho_max: float | None = None):
    r"""`(ρ_sp_v, ρ_sp_l)` from `spinodal_density` — a **seed ladder**, no sweep.

    ⚠ One Newton per side with a fixed seed is not enough, and the failure is not
    tuning: near the apex the two roots are close, a seed at `ρ = 0.02` converges to
    the *liquid* root, and the pair comes back with `ρ_l ≤ ρ_v`.  Firing the Newton
    from a short ladder and taking the extreme distinct roots removes the guess —
    eight Newtons still cost far less than the 320-point continuation
    `spinodal_densities` runs, and none of them walks toward the fold.

    Falls back to the swept version **wholesale** (not per-side) if the ladder finds
    fewer than two roots, so callers keep the old behaviour in the corner cases.
    """
    rho_max = rho_max or 6.0 * 0.46 / np.pi
    if seeds is None:
        seeds = np.array([0.02, 0.05, 0.10, 0.18, 0.30, 0.45, 0.62, 0.80])
    roots = []
    for r0 in seeds:
        if r0 >= rho_max:
            continue
        r, _ = spinodal_density(st, float(r0))
        if r is None or not (0.0 < r < rho_max):
            continue
        if not any(abs(r - q) < 1e-6 for q in roots):
            roots.append(r)
    if len(roots) < 2:
        return spinodal_densities(st)
    return min(roots), max(roots)


def spinodal_densities(st: HCYMix, rho_max: float | None = None,
                       n: int = 320) -> tuple[float | None, float | None]:
    r"""`(ρ_sp_v, ρ_sp_l)` — where `A` changes sign on this isopleth at this `T`.

    ⚠ **Two sweeps, from the two ends** — and that is not a convenience. A single
    `ρ`-continuation started at `ρ → 0` dies at the **fold**, which sits just past
    the vapour-side spinodal (at `T* = 0.85`: spinodal `η = 0.083`, fold `0.087`),
    so it never reaches the liquid-side crossing at all. The two `A > 0` windings
    are disconnected along the isotherm; each has to be entered from its own end.

    `A` is continuous on each arc and crosses zero once, so this is a bracket plus
    one `brentq` per side — no augmented system, no singular Jacobian, unlike
    `fold_densities`/`fold_density`.  `(None, None)` above the apex, where `A > 0`
    throughout and the isotherm is not cut.
    """
    from scipy.optimize import brentq
    rho_max = rho_max or 6.0 * 0.48 / np.pi
    sc1, sc2 = _scales(st)
    n2 = st.N * st.N

    def A_of(v, r):
        s = _scale_rho(st, r)
        p = _pieces(s, v[:st.T * n2].reshape(st.T, st.N, st.N),
                    v[st.T * n2:].reshape(st.T, st.N, st.N))
        return float(np.sum(p['A']))

    def side(r0, r1):
        """Continue from `r0` toward `r1`; return the first zero of `A`, or None."""
        q = solve_bh_mix(_scale_rho(st, r0))
        if q is None:
            return None
        u = np.concatenate([q.Dt.ravel(), q.Gt.ravel()])
        prev_r, prev_A = r0, A_of(u, r0)
        for r in np.linspace(r0, r1, n)[1:]:
            v = _solve_seeded(_scale_rho(st, r), u, sc1, sc2)
            if v is None:
                return None                      # ran into the fold before a zero
            a = A_of(v, r)
            if np.sign(a) != np.sign(prev_A):
                box = {'u': u}

                def f(x):
                    w = _solve_seeded(_scale_rho(st, x), box['u'], sc1, sc2)
                    if w is None:
                        return 1e3
                    box['u'] = w
                    return A_of(w, x)
                return float(brentq(f, prev_r, r, xtol=1e-10))
            u, prev_r, prev_A = v, r, a
        return None

    return side(1e-3, rho_max), side(rho_max, 1e-3)


def spinodal_apex(st: HCYMix, t_lo: float, t_hi: float, tol: float = 1e-4) -> float:
    """Temperature at which the `A = 0` dome closes on this isopleth (bisection)."""
    T0 = 1.0 / float(np.max(st.K))
    while t_hi - t_lo > tol:
        m = 0.5 * (t_lo + t_hi)
        s = HCYMix(st.sigma, st.rho, st.z, st.K * (T0 / m))
        t_lo, t_hi = (m, t_hi) if spinodal_densities(s)[0] is not None else (t_lo, m)
    return 0.5 * (t_lo + t_hi)


def spinodal_curve(st: HCYMix, t_apex: float, n_ladder: int = 20):
    """`(T, ρ_sp_v, ρ_sp_l)` on a ladder from this state's `T` up to the apex."""
    T0 = 1.0 / float(np.max(st.K))
    tl = np.linspace(T0, t_apex, n_ladder)
    lv = np.full(n_ladder, np.nan)
    ll = np.full(n_ladder, np.nan)
    for i, t in enumerate(tl):
        a, b = spinodal_densities(HCYMix(st.sigma, st.rho, st.z, st.K * (T0 / t)))
        if a is not None and b is not None:
            lv[i], ll[i] = a, b
    # ⚠ mask on BOTH sides: one sweep can find its crossing while the other runs
    # into the fold first, leaving a NaN that PchipInterpolator rejects downstream.
    ok = ~np.isnan(lv) & ~np.isnan(ll)
    return tl[ok], lv[ok], ll[ok]


def _arc(ts, fv, fl, t_apex, t_over, n_side):
    """Up one side of a dome, across above the apex, and down the other."""
    m = max(12, n_side // 6)
    tx = t_over * t_apex

    def leg(r0, t0, r1, t1, k):
        return np.stack([np.linspace(r0, r1, k), np.linspace(t0, t1, k)], axis=1)

    return np.concatenate([
        np.stack([fv, ts], axis=1),
        leg(fv[-1], ts[-1], fv[-1], tx, m),
        leg(fv[-1], tx, fl[-1], tx, m),
        leg(fl[-1], tx, fl[-1], ts[-1], m),
        np.stack([fl[::-1], ts[::-1]], axis=1)])


def spinodal_gap_free_energy(st: HCYMix, mu_v: float, n_arc: int = 90,
                             insets=(0.04, 0.02), t_apex: float | None = None,
                             t_over: float = 1.002, curve=None,
                             t_hug: float = 0.99) -> tuple:
    r"""⭐ `(Δ, μ_l)` across the `A < 0` interval, hugging the spinodal — **extrapolated
    to zero inset**.

    `Δ` is linear in the inset here (the arc never approaches a singular point), so
    the two `insets` are evaluated and Richardson-extrapolated to `inset → 0`.  That
    is the whole reason to prefer this over `fold_gap_free_energy`, whose arc is
    singular in the limit and converges at best like `√inset`.

    As with the fold version, `Δ` is **affine in `μ_v`** with coefficient the gap
    width, so `fold_gap_coefficients`' decomposition applies unchanged.
    """
    T0 = 1.0 / float(np.max(st.K))
    if t_apex is None:
        t_apex = spinodal_apex(st, T0, 4.0 * T0)
    if curve is None:
        curve = spinodal_curve(st, t_apex)
    tl, lv, ll = curve
    if len(tl) < 4:
        return None, None
    from scipy.interpolate import PchipInterpolator
    ts = np.linspace(T0, t_hug * t_apex, n_arc)
    if ts[-1] <= ts[0]:
        return None, None
    out = []
    for ins in insets:
        fv = PchipInterpolator(tl, lv)(ts) * (1.0 - ins)
        fl = PchipInterpolator(tl, ll)(ts) * (1.0 + ins)
        S, m = _transport(st, _arc(ts, fv, fl, t_apex, t_over, n_arc),
                          S0=0.0, m0=mu_v)
        if S is None:
            return None, None
        out.append((float(ins), S, m))
    (i1, d1, m1), (i2, d2, m2) = out
    if abs(i2 - i1) < 1e-12:          # degenerate pair: report the raw value
        return d2, m2                 # (used by --gate to probe the inset trend)
    ex = lambda a, b: b + (b - a) * (0.0 - i2) / (i2 - i1)
    return ex(d1, d2), ex(m1, m2)


def critical_detour_gap_free_energy(st: HCYMix, mu_v: float, rho_v: float, rho_l: float,
                                t_apex: float, t_over: float = 1.02,
                                n_leg: int = 90) -> tuple:
    r"""⭐ `(Δ, μ_l)` across the gap by a **rectangular detour over the critical point** —
    the third crossing path, and the one that touches no special locus at all.

        up:     `(ρ_v, T) → (ρ_v, t_over·T_fold_apex)`   isochoric
        across: `(ρ_v, T_x) → (ρ_l, T_x)`                isothermal, the dome has closed
        down:   `(ρ_l, T_x) → (ρ_l, T)`                  isochoric

    ⭐ **Why the vertical legs are always safe.**  The fold dome is nested strictly
    inside the `A = 0` dome and *narrows* with `T`: `ρ_fold_v(T′) ≥ ρ_fold_v(T)` and
    `ρ_fold_l(T′) ≤ ρ_fold_l(T)` for `T′ > T`.  Starting from a spinodal density —
    `ρ_v < ρ_fold_v(T)` on the left, `ρ_l > ρ_fold_l(T)` on the right — a vertical
    leg therefore moves *away* from the dome on both sides and cannot enter it.  No
    inset, no `√inset` convergence, no Richardson step: the path is exactly as
    written and there is nothing to extrapolate.

    ⚠ **What it costs.**  The two vertical legs are pure `ΔE dβ`, i.e. *energy*
    route, and they are long — from `T` all the way to the apex.  So this is the
    most energy-weighted of the three crossings, and the difference from
    `spinodal_gap_free_energy` is not numerical error but **MSA's own
    path-dependence**: measured `+3.44 %` at `T* = 0.95` rising to `+13.27 %` at
    `T* = 0.70` (HSY, `η = 0.2`; `--gate` §12(c)).  That is the point of having it — it is the route gap read
    off the free energy itself rather than off `T_c`.

    ⭐ The crossing height barely matters, which is the check that the number is a
    real path integral and not an artifact of where the detour turns: `t_over`
    `1.005 → 1.05` moves `Δ₀` by `1.61 %` and `1.005 → 1.10` by `2.48 %`
    (`--gate` §12(b)) — nothing like the `√inset`
    behaviour that made `fold_gap_free_energy` unusable (a factor 4 between insets
    0.02 and 0.05 for 2YK).

    As with both line-hugging versions, `Δ` is **affine in `μ_v`** with coefficient
    `ρ_l − ρ_v`; that coefficient is forced by the 2-form and is therefore identical
    across all three paths, so the routes can differ only in `(Δ₀, ΔM)`.
    """
    T0 = 1.0 / float(np.max(st.K))
    tx = t_over * t_apex
    # ⚠ `spinodal_densities` returns None on a side whose sweep meets the fold
    # before the `A = 0` crossing, which happens on the liquid side well below the
    # apex.  Guard on None BEFORE comparing, or the caller gets a TypeError from
    # deep inside instead of a "no endpoints here".
    if rho_v is None or rho_l is None:
        return None, None
    if not (tx > T0 and rho_l > rho_v > 0.0):
        return None, None
    m = max(24, int(n_leg))

    def leg(r0, t0, r1, t1, k):
        return np.stack([np.linspace(r0, r1, k), np.linspace(t0, t1, k)], axis=1)

    path = np.concatenate([leg(rho_v, T0, rho_v, tx, m),
                           leg(rho_v, tx, rho_l, tx, m),
                           leg(rho_l, tx, rho_l, T0, m)])
    return _transport(st, path, S0=0.0, m0=mu_v)


def free_energy_critical_detour_route(st: HCYMix, n_leg: int = 500, n_det: int = 90,
                                  t_apex: float | None = None,
                                  t_over: float = 1.02) -> float:
    r"""`βΔA/V` by the **critical-point-detour route** — same two `ρ` legs as
    `free_energy_spinodal_route`, different crossing.

    ⚠⚠ `t_apex` must be the **`A = 0`** apex, not the fold apex.  An earlier version of
    this docstring asserted the reverse — "the crossing only has to clear the locus where
    the solution ceases to exist" — and it is wrong.  Above the fold apex the isotherm is
    indeed open, but the entire `A < 0` window there is **wrong-winding**: those states
    have `h(r) ~ e^{+s*r}` and are not fluid states at all
    (`numerical_notes/theory/wrong_winding_spinodal_to_fold.md`).  Existence is not
    physicality, and a crossing leg must clear the `A = 0` dome.

    ⚠ Measured cost at the old default `t_over = 1.02`: HSY crossed at `1.0381` against
    an `A = 0` apex of `1.0313` and cleared by luck; **2YK crossed at `0.9051` against
    `0.9330` and did not clear**, which is the source of its anomalous vapour branch
    (`ΔH_vap = 14.39` against 4.6–6.1 for every other construction).
    """
    rho_t = float(np.sum(st.rho))
    T0 = 1.0 / float(np.max(st.K))
    rv, rl = spinodal_densities_exact(st)   # augmented solve, not a sweep
    if rv is None:
        path = np.stack([np.linspace(1e-4, rho_t, 4 * n_leg),
                         np.full(4 * n_leg, T0)], axis=1)
        return _transport(st, path)[0]
    if t_apex is None:
        t_apex = fold_apex(st, T0, 4.0 * T0)

    def leg(r0, r1, n):
        return np.stack([np.linspace(r0, r1, n), np.full(n, T0)], axis=1)

    S, m = _transport(st, leg(1e-4, rv, n_leg))
    if S is None:
        return float("nan")
    dS, m = critical_detour_gap_free_energy(st, m, rv, rl, t_apex, t_over, n_det)
    if dS is None:
        return float("nan")
    S, m = _transport(st, leg(rl, rho_t, n_leg), S0=S + dS, m0=m)
    return float("nan") if S is None else S


def free_energy_spinodal_route(st: HCYMix, n_leg: int = 500, n_arc: int = 90,
                               insets=(0.04, 0.02), t_apex: float | None = None,
                               curve=None) -> float:
    r"""⭐ `βΔA/V` by the **spinodal route** — the usable semi-compressibility method.

    1. `ρ = 0 → ρ_sp_v(T)` along the target isotherm — compressibility route, and
       entirely on the physical `A > 0` winding;
    2. `Δ` across the `A < 0` interval from `spinodal_gap_free_energy`, fed the `βΔμ`
       that step 1 ends on, extrapolated to zero inset;
    3. `ρ_sp_l(T) → ρ_target` — compressibility route again.

    Above the apex the isotherm is not cut, step 2 is skipped, and this is the
    **pure** compressibility route — the gate.
    """
    rho_t = float(np.sum(st.rho))
    T0 = 1.0 / float(np.max(st.K))
    rv, rl = spinodal_densities_exact(st)   # augmented solve, not a sweep
    if rv is None:
        path = np.stack([np.linspace(1e-4, rho_t, 4 * n_leg),
                         np.full(4 * n_leg, T0)], axis=1)
        return _transport(st, path)[0]
    if t_apex is None:
        t_apex = spinodal_apex(st, T0, 4.0 * T0)

    def leg(r0, r1, n):
        return np.stack([np.linspace(r0, r1, n), np.full(n, T0)], axis=1)

    S, m = _transport(st, leg(1e-4, rv, n_leg))
    if S is None:
        return float("nan")
    dS, m = spinodal_gap_free_energy(st, m, n_arc, insets, t_apex, curve=curve)
    if dS is None:
        return float("nan")
    S, m = _transport(st, leg(rl, rho_t, n_leg), S0=S + dS, m0=m)
    return float("nan") if S is None else S


def hs_thermo_pure(st: HCYMix, n_rho: int = 48) -> tuple[float, float]:
    r"""`(βp_HS, βμ^ex_HS)` for the **single-component** hard-sphere reference.

    Integrals of the same `a₀ = (A⁰/2π)²` that (39) gives at `K = 0` — the
    Percus–Yevick *compressibility* equation of state, so the reference is the one
    the excess is actually taken over:

        βp_HS(ρ) = ∫₀^ρ a₀ dρ′,     βμ^ex_HS(ρ) = ∫₀^ρ (a₀ − 1)/ρ′ dρ′

    Hard spheres never phase separate, so neither integral can be blocked.
    ⚠ `N = 1` only: a mixture needs `∂(βA_HS/V)/∂ρ_i` species by species, which is
    Lebowitz's closed form rather than an isopleth quadrature.
    """
    if st.N != 1:
        raise ValueError("hs_thermo_pure is single-component; use Lebowitz for N>1")
    rho_t = float(st.rho[0])
    nodes, w = np.polynomial.legendre.leggauss(n_rho)
    rs = 0.5 * rho_t * (nodes + 1.0)
    ws = 0.5 * rho_t * w
    p = mu = 0.0
    for rp, wt in zip(rs, ws):
        s2 = HCYMix(st.sigma, np.array([rp]), st.z, st.K)
        a0 = _a_hs(s2)
        p += wt * a0
        mu += wt * (a0 - 1.0) / rp
    return p, mu


def pressure_total_pure(sol: SolutionMix) -> float:
    """`βp` — total pressure of a single-component state, energy route.

    ⚠ No separate ideal term: `a₀ → 1` as `ρ → 0`, so `∫₀^ρ a₀dρ′` already
    contains it (it reproduces `ρ(1+η+η²)/(1−η)³`, the *total* PY-compressibility
    pressure, to 1e-15).  Adding `ρ` on top puts the Gibbs–Duhem residual at
    exactly 1.
    """
    return hs_thermo_pure(sol.st)[0] + pressure_excess(sol)


def chem_pot_total_pure(sol: SolutionMix) -> float:
    """`βμ` — total chemical potential of a single-component state, energy route.

    `βμ = ln ρ + βμ^ex_HS + βμ^E`, the last from (46).  Gibbs–Duhem-consistent
    with `pressure_total_pure` by construction: that is exactly what (45) asserts.
    """
    rho = float(np.sum(sol.st.rho))
    return (np.log(rho) + hs_thermo_pure(sol.st)[1]
            + float(chem_pot_excess(sol)[0]))


# ═══════════════════════════════════════════════════════════════════════════════
# convenience: the Tang & Lu model
# ═══════════════════════════════════════════════════════════════════════════════

def tanglu_state(x1: float, eta: float, tstar: float, sig_ratio: float,
                 eps22: float, eps12: float, zR11: float = 1.8) -> HCYMix:
    """The binary hard-core one-Yukawa mixture of Tang & Lu 1995 Tables 1–2.

        u_ij(r) = −σ_ij ε_ij e^{−z(r−σ_ij)}/r        (r > σ_ij),  σ₁ = 1

    with `z σ₁ = 1.8` shared by all three pairs, so `K_ij = β ε_ij σ_ij`.
    Same parameterisation as `benchmark_tanglu_contact.contact_first_order`.
    """
    sigma = np.array([1.0, sig_ratio])
    x = np.array([x1, 1.0 - x1])
    rho = x * (6.0 * eta / (np.pi * float(np.sum(x * sigma ** 3))))
    eps = np.array([[1.0, eps12], [eps12, eps22]])
    SIG = 0.5 * (sigma[:, None] + sigma[None, :])
    return HCYMix(sigma, rho, np.array([zR11 / sigma[0]]),
                  (SIG * eps / tstar)[None, :, :])


def pair_specific_z(sigma, rho, zbar: float, eps: np.ndarray,
                    tstar: float) -> HCYMix:
    r"""⭐ A **pair-specific** `z_ij = z̄/σ_ij` model, as extra equal-`z` tails.

    Tang & Lu 1995 p. 98 separates "the full solution restricted to the Yukawa
    exponent `z` being equal for all intercomponent interactions … by Blum and
    Høye" from "the generalization for pair-specific `z_ij` made by Cummings",
    and this project's own mixture models are the *second* kind
    (`fmsa_double_prop` sets `z_poles[t,i,j] = z_arr[t]/σ_ij`).  That looked like
    a literature dependency — Cummings' 1980 Melbourne thesis, unobtainable.

    ⭐ **It is not one.**  Blum & Høye's Eq. (3) already allows an independent
    amplitude *matrix* `K^ν_ij` per tail, and (29)/(33) are matched per tail
    because distinct `e^{−z_ν r}` are linearly independent.  So a pair-specific
    rate is just a tail whose amplitude matrix is **supported on the pairs that
    want it**: for a binary,

        ν = 1: z̄/σ₁₁, K¹ = [[K₁₁,0],[0,0]]
        ν = 2: z̄/σ₁₂, K² = [[0,K₁₂],[K₁₂,0]]
        ν = 3: z̄/σ₂₂, K³ = [[0,0],[0,K₂₂]]

    reproducing `c_ij(r) = K_ij e^{−(z̄/σ_ij)(r−σ_ij)}/r` exactly.  `D^ν_ij` stays
    a full matrix for every `ν` — the *closure* is sparse, the *solution* is not.
    ✅ Gated in `--gate` §5 against `FMSA_double_prop`, which carries pair-specific
    rates natively: the `O(K)` slopes agree to 1e-10.  ⇒ **L9 is not needed.**

    Cost: `ω = N(N+1)/2` tails, so `2N²·N(N+1)/2 = N³(N+1)` unknowns — 24 for a
    binary, 108 for a ternary.  Distinct diameters are required (equal `σ`
    collapses the rates and the tails become linearly dependent).
    """
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    eps = np.asarray(eps, dtype=float)
    n = len(sigma)
    SIG = 0.5 * (sigma[:, None] + sigma[None, :])
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    z = np.array([zbar / SIG[i, j] for i, j in pairs])
    K = np.zeros((len(pairs), n, n))
    for t, (i, j) in enumerate(pairs):
        K[t, i, j] = K[t, j, i] = SIG[i, j] * eps[i, j] / tstar
    return HCYMix(sigma, rho, z, K)


def from_fmsa(dp, k_scale: float = 1.0) -> HCYMix:
    r"""⭐ The exact-MSA state that `FMSA_double_prop` is the first-order truncation of.

    Every comparison in this paper between FMSA-DP and exact MSA has to be run on
    the *same* system, and the only way to guarantee that is to read the system off
    the FMSA object itself rather than rebuild it from the physical parameters.
    Rebuilding is where the two silently diverge: `ljhs_mapping.model_inputs`
    deliberately does NOT use `input_type='bh'` (`z` comes out 3.4 % high, `k_bare`
    12.2 % low), the Barker--Henderson diameters depend on `T*`, and the LJ--HS block
    zeroes a well depth.  All of that is already resolved inside `dp`.

    `OZSolverMSA._c_msa_outer` is the definition being matched, verbatim:

        c_ij(r) = Σ_t  K[t,i,j] / (2π√(ρ_i ρ_j))  ·  e^{−z_poles[t,i,j](r−R_ij)} / r

    which is exactly Blum & Høye's Eq. (3) with `K^ν_ij = K[t,i,j]/(2π√(ρ_iρ_j))` and
    `σ_ij = R_ij`.  So the exact solution and the OZ+MSA solve see one system by
    construction, and the ONLY difference left is first order versus all orders.

    ⚠ `z_poles[t,i,j]` is pair-specific (`z̄/σ_ij`, off the *LJ* sigmas — note it is
    not `R_ij`), which is Cummings' generalisation rather than Blum & Høye's covered
    case.  `pair_specific_z` explains why that needs no new theory: a pair-specific
    rate is a tail whose amplitude matrix is supported on one pair.  Here the tails
    are therefore grouped by DISTINCT `z`, not one per pair blindly --- equal
    diameters collapse the rates, and emitting the collapsed tails separately would
    hand the solver a linearly dependent set.  Grouping keeps both cases correct
    with no branch: `ω` comes out `n_tails·N(N+1)/2` when the diameters differ and
    `n_tails` when they do not.

    `k_scale` mirrors the solver's continuation knob; 1.0 is the full Yukawa.
    """
    R = np.asarray(dp.R_ij, dtype=float)
    rho = np.asarray(dp.rho, dtype=float)
    n = len(rho)
    d = np.diag(R).copy()
    # HCYMix builds σ_ij additively, so a non-additive core would be a different
    # system wearing the same numbers.  Refuse rather than solve the wrong one.
    if not np.allclose(R, 0.5 * (d[:, None] + d[None, :]), rtol=0, atol=1e-12):
        raise ValueError('R_ij is not additive; HCYMix cannot represent it')
    pref = 2.0 * np.pi * np.sqrt(rho[:, None] * rho[None, :])
    groups: dict[float, np.ndarray] = {}
    for t in range(int(dp.n_tails)):
        for i in range(n):
            for j in range(i, n):
                z = round(float(dp.z_poles[t, i, j]), 12)
                Kt = groups.setdefault(z, np.zeros((n, n)))
                v = float(dp.K[t, i, j]) * k_scale / pref[i, j]
                Kt[i, j] = Kt[j, i] = Kt[i, j] + v
    zs = sorted(groups)
    return HCYMix(d, rho, np.array(zs), np.array([groups[z] for z in zs]))


# ═══════════════════════════════════════════════════════════════════════════════
# gates
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_gate() -> int:
    import msa_exact as M

    print("=" * 78)
    print("MSAX.6 — the exact MSA for a hard-core multi-Yukawa MIXTURE")
    print("=" * 78)

    # ── 1. the hard-sphere transcription, from the mixture's own symmetry ─────
    print("\n1. K -> 0: the linear seed, and the two things it tests")
    print("-" * 78)
    print("   (a) ghat_ij(z) must come out SYMMETRIC although its source term")
    print("       A_j^0 + z q^{0'}_ij is not.  Round-off => psi_1/psi_2, the rho")
    print("       placement and the lambda convention are all right.")
    worst_sym = 0.0
    for sig2 in (1.0, 1.2, 1.5, 2.0):
        for eta in (0.1, 0.3, 0.46):
            for x1 in (0.25, 0.5, 0.75):
                st = tanglu_state(x1, eta, 1.0, sig2, 1.0, 1.0)
                _, Gt = seed_zero_K(st)
                gL = Gt[0] * np.exp(-st.z[0] * st.SIG)
                worst_sym = max(worst_sym, float(np.max(np.abs(gL - gL.T))
                                                 / np.max(np.abs(gL))))
    print(f"       worst relative asymmetry over 36 states: {worst_sym:.2e}   "
          f"{'PASS' if worst_sym < 1e-12 else 'FAIL'}")

    print("   (b) vs FMSA_double_prop.get_g0L_laplace -- an independent closed")
    print("       form for the same PY hard-sphere mixture Laplace moment.")
    try:
        from fmsa_double_prop import FMSA_double_prop
    except Exception as e:                                    # pragma: no cover
        FMSA_double_prop = None
        print(f"       skipped ({e})")
    if FMSA_double_prop is not None:
        worst = 0.0
        for sig2 in (1.0, 1.2, 1.5):
            for eta in (0.1, 0.3, 0.46):
                st = tanglu_state(0.5, eta, 1.0, sig2, 1.0, 1.0)
                _, Gt = seed_zero_K(st)
                mine = Gt[0] * np.exp(-st.z[0] * st.SIG)
                rho_star = float(np.sum(st.rho))
                m = FMSA_double_prop(st.sigma, [1.0, 1.0], st.rho / rho_star,
                                     rho_star, 1.0, n_tails=1, z=[st.z[0]],
                                     k=[0.0], input_type='bh', verbose=False)
                ref = np.real(m.get_g0L_laplace(np.array([st.z[0]]))[0])
                worst = max(worst, float(np.max(np.abs(mine / ref - 1.0))))
        print(f"       worst relative deviation over 9 states: {worst:.2e}   "
              f"{'PASS' if worst < 1e-9 else 'FAIL'}")

    # ── 2. N = 1 reduction ────────────────────────────────────────────────────
    print("\n2. N = 1 reduction: solve_bh_mix == msa_exact.solve_bh_n")
    print("-" * 78)
    worst_a = worst_g = 0.0
    for eta in (0.05, 0.2, 0.4):
        for zz in ([1.8], [5.0], [2.9637, 14.0167]):
            for kk in (1e-5, 0.3):
                z = np.array(zz, dtype=float)
                K = np.full(len(z), kk)
                ref = M.solve_bh_n(eta, z, K)
                st = HCYMix([1.0], [6.0 * eta / np.pi], z,
                            K.reshape(-1, 1, 1))
                q = solve_bh_mix(st)
                if ref is None or q is None:
                    print(f"     eta={eta} z={zz} K={kk}: no branch")
                    continue
                worst_a = max(worst_a, abs(q.dbetaP_drho / ref.a - 1.0))
                worst_g = max(worst_g, abs(q.contact[0, 0] / ref.y0 - 1.0))
    print(f"   dbetaP/drho: {worst_a:.1e}    g(sigma+): {worst_g:.1e}   "
          f"{'PASS' if max(worst_a, worst_g) < 1e-9 else 'FAIL'}")

    print("   ⭐ and the reduction that actually exercises the mixture code:")
    print("      N = 2, 3 IDENTICAL species must reproduce N = 1.  The literal-N=1")
    print("      check above cannot see an error in the `rho_l` placement, the")
    print("      `lambda_ij` convention or the `sigma_ij` bookkeeping, because every")
    print("      one of those is trivial at a single species.  Non-equimolar")
    print("      compositions are included: the answer must not depend on `x`.")
    wa = wg = wA = 0.0
    for N, x in ((2, [0.5, 0.5]), (2, [0.25, 0.75]), (3, [0.2, 0.3, 0.5])):
        for eta in (0.1, 0.3):
            for zz, kk in (([1.8], 0.5), ([2.9637, 14.0167], 0.3)):
                z = np.atleast_1d(np.asarray(zz, dtype=float))
                ref = M.solve_bh_n(eta, z, np.full(len(z), kk))
                Kb = np.broadcast_to(np.full(len(z), kk)[:, None, None],
                                     (len(z), N, N)).copy()
                q = solve_bh_mix(HCYMix(np.ones(N), np.asarray(x) * 6.0 * eta / np.pi,
                                        z, Kb))
                wa = max(wa, abs(q.dbetaP_drho / ref.a - 1.0))
                wg = max(wg, max(abs(q.contact[i, j] / ref.y0 - 1.0)
                                 for i in range(N) for j in range(N)))
                wA = max(wA, float(np.max(q.A) - np.min(q.A)))
    print(f"      12 states:  a {wa:.1e}   g_ij {wg:.1e}   spread of A_j {wA:.1e}   "
          f"{'PASS' if max(wa, wg, wA) < 1e-10 else 'FAIL'}")

    # ── 3. O(K): the first-order limit vs the project's own machinery ─────────
    print("\n3. O(K) contact values vs benchmark_tanglu_contact.contact_first_order")
    print("-" * 78)
    print("   Independent code (Tang & Lu's residue programme, validated against")
    print("   their column I to 2e-4), so this is not circular.  Both are")
    print("   linearised in K by Richardson extrapolation of the finite step.")
    from benchmark_tanglu_contact import contact_first_order
    worst = 0.0
    for sig2, e22, e12 in ((1.0, 1.0, 1.0), (1.167, 1.919, np.sqrt(1.919)),
                           (1.5, 0.9, 0.8)):
        for eta in (0.1, 0.3):
            for x1 in (0.35, 0.65):
                lam = 1e-5
                g0 = contact_first_order(x1, eta, 1e30, sig2, e22, e12)[1]
                for f in (1.0, 0.5):
                    st = tanglu_state(x1, eta, 1.0 / (f * lam), sig2, e22, e12)
                    q = solve_bh_mix(st)
                    if f == 1.0:
                        s1 = (q.contact - g0) / lam
                    else:
                        s2 = (q.contact - g0) / (f * lam)
                ex = 2.0 * s2 - s1                          # Richardson
                r1 = contact_first_order(x1, eta, 1.0 / lam, sig2, e22, e12)[2] / lam
                r2 = (contact_first_order(x1, eta, 1.0 / (0.5 * lam), sig2, e22,
                                          e12)[2] / (0.5 * lam))
                fd = 2.0 * r2 - r1
                worst = max(worst, float(np.max(np.abs(ex / fd - 1.0))))
    print(f"   worst relative deviation of the O(K) slope, 12 states: {worst:.2e}"
          f"   {'PASS' if worst < 1e-4 else 'FAIL'}")

    # ── 4. the external gate: Tang & Lu's 24 full-MSA contact values ──────────
    print("\n4. ⭐ THE EXTERNAL GATE — Tang & Lu 1995 Tables 1-2, column II")
    print("-" * 78)
    print("   24 published full-MSA contact values (from Arrieta et al.), the")
    print("   strongest gate in the repo and the one MSAX.6 was written for.")
    from benchmark_tanglu_contact import TABLE1, TABLE2
    for name, tbl in (("Table 1", TABLE1), ("Table 2", TABLE2)):
        cond = tbl["cond"]
        print(f"\n   {name}  (s2/s1={cond['sig_ratio']}, e22={cond['eps22']}, "
              f"e12={cond['eps12']:.4f})")
        print(f"   {'x1':>5}{'eta':>6}{'T*':>7} | "
              f"{'g11 me/II':>17} | {'g12 me/II':>17} | {'g22 me/II':>17} | "
              f"{'sym':>8}")
        dev = []
        for row in tbl["rows"]:
            x1, eta, tstar = row[0], row[1], row[2]
            ref = (row[4], row[6], row[8])
            st = tanglu_state(x1, eta, tstar, **cond)
            q = solve_bh_mix(st)
            if q is None:
                print(f"   {x1:>5.2f}{eta:>6.2f}{tstar:>7.3f} |  no branch")
                continue
            me = (q.contact[0, 0], q.contact[0, 1], q.contact[1, 1])
            cells = " | ".join(f"{me[k]:8.4f}/{ref[k]:8.4f}" for k in range(3))
            dev += [abs(me[k] - ref[k]) for k in range(3)]
            print(f"   {x1:>5.2f}{eta:>6.2f}{tstar:>7.3f} | {cells} | "
                  f"{q.symmetry_defect:8.1e}")
        d = np.array(dev)
        print(f"   -> |me - II|  max={d.max():.2e}  mean={d.mean():.2e}   "
              f"(they print 4 decimals, so 5e-05 is the floor; first order was"
              f" max=2.3e-01)")

    # ── 5. pair-specific z_ij, the L9 (Cummings) question ────────────────────
    print("\n5. ⭐ pair-specific z_ij = zbar/sigma_ij, encoded as extra tails")
    print("-" * 78)
    print("   Arbiter is FMSA_double_prop, which carries pair-specific rates")
    print("   natively (z_poles[t,i,j] = z/sigma_ij).  O(K) slope of dbetaP/drho.")
    if FMSA_double_prop is None:
        print("       skipped")
    else:
        worst = 0.0
        print(f"       {'s2':>6}{'eta':>6}{'x1':>6} {'exact':>14} {'FMSA-DP':>14}"
              f" {'rel':>9} {'sym':>9}")
        for s2 in (1.5, 1.167):
            for eta in (0.1, 0.3):
                for x1 in (0.35, 0.65):
                    sig = np.array([1.0, s2])
                    x = np.array([x1, 1.0 - x1])
                    rs = 6.0 * eta / (np.pi * float(np.sum(x * sig ** 3)))
                    rho, e22, lam = x * rs, 0.9, 1e-5
                    ep = np.sqrt(np.array([1.0, e22])[:, None]
                                 * np.array([1.0, e22])[None, :])
                    a = [solve_bh_mix(pair_specific_z(sig, rho, 1.8, ep, 1.0 / (f * lam)))
                         for f in (1.0, 0.5)]
                    a0 = solve_bh_mix(pair_specific_z(sig, rho, 1.8, ep, 1e30))
                    ex = (2.0 * (a[1].dbetaP_drho - a0.dbetaP_drho) / (0.5 * lam)
                          - (a[0].dbetaP_drho - a0.dbetaP_drho) / lam)

                    def _dp(L):
                        m = FMSA_double_prop(sig, [1.0, e22], x, rs, 1.0 / L,
                                             n_tails=1, z=[1.8], k=[1.0],
                                             input_type='bh', verbose=False)
                        ck = np.real(m.get_chat_k(np.array([1e-6]))[0])
                        return 1.0 - float(np.einsum('i,j,ij->', rho, rho, ck)
                                           ) / float(np.sum(rho))
                    h = _dp(1e-30)
                    fd = 2.0 * (_dp(0.5 * lam) - h) / (0.5 * lam) - (_dp(lam) - h) / lam
                    worst = max(worst, abs(ex / fd - 1.0))
                    print(f"       {s2:6.3f}{eta:6.2f}{x1:6.2f} {ex:14.8f}"
                          f" {fd:14.8f} {abs(ex/fd-1):9.1e}"
                          f" {a[0].symmetry_defect:9.1e}")
        print(f"   -> worst {worst:.1e}   {'PASS' if worst < 1e-8 else 'FAIL'}"
              f"   ⇒ Cummings (L9) is NOT needed for pair-specific rates")

    # ── 6. Eq. (13) as an identity: A_j against Q̂(0) ─────────────────────────
    print("\n6. their Eq. (13) as an internal identity at FINITE coupling")
    print("-" * 78)
    print("   A_j/2pi = sum_l [delta_lj - rho_l Qhat_lj(0)] -- the column sum of")
    print("   F(0).  A_j comes from the (20)-(23) chain, Qhat(0) from `qhat`'s")
    print("   s -> 0 series, so agreement tests both, and (39)'s use of A_j.")
    worst = 0.0
    for tbl in (TABLE1, TABLE2):
        for row in tbl["rows"]:
            st = tanglu_state(row[0], row[1], row[2], **tbl["cond"])
            q = solve_bh_mix(st)
            col = np.sum(q.matF(0.0).real, axis=0)
            worst = max(worst, float(np.max(np.abs(col / (q.A / (2 * np.pi)) - 1))))
    print(f"   worst relative deviation over the 24 gate states: {worst:.2e}   "
          f"{'PASS' if worst < 1e-10 else 'FAIL'}")

    # ── 7. the spinodal indicator reduces to Phase 1's ────────────────────────
    print("\n7. spinodal: det F(0) = 0 must reduce to msa_exact's A = 0 at N = 1")
    print("-" * 78)
    print("   At N=1, F(0) = 1 - rho*Qhat(0) = A/2pi, so the two brackets are the")
    print("   same equation -- and the tracers are independent implementations.")
    worst = 0.0
    for eta in (0.1, 0.25, 0.4):
        for zz in (1.8, 5.0):
            a = M.spinodal_coupling(eta, [zz], [1.0], lam_max=30.0)
            st = HCYMix([1.0], [6.0 * eta / np.pi], [zz], np.ones((1, 1, 1)))
            b = spinodal_coupling_mix(st, lam_max=30.0)
            ok = a is not None and b is not None
            print(f"     eta={eta:4.2f} z={zz:4.1f}   msa_exact {a if a else float('nan'):9.5f}"
                  f"   mixture {b if b else float('nan'):9.5f}"
                  f"   rel {abs(b/a - 1) if ok else float('nan'):.1e}")
            if ok:
                worst = max(worst, abs(b / a - 1.0))
    print(f"   worst {worst:.1e}   {'PASS' if worst < 1e-8 else 'FAIL'}")

    # ── 8. thermodynamics: the closed forms against λ-integration ────────────
    print("\n8. ⭐ thermodynamics — (42)-(46) against the charging integral")
    print("-" * 78)
    print("   bDA from (45) [closed form, no quadrature] vs the lambda-integration")
    print("   of (42); and mu_i from (46) vs d(bDA/V)/drho_i.  The reference is")
    print("   itself a quadrature + finite difference, so ~1e-5 IS agreement.")

    def _mu_ref(s, h=2e-5):
        o = []
        for i in range(s.N):
            r1, r2 = s.rho.copy(), s.rho.copy()
            r1[i] += h
            r2[i] -= h
            o.append((free_energy_charging(HCYMix(s.sigma, r1, s.z, s.K))
                      - free_energy_charging(HCYMix(s.sigma, r2, s.z, s.K))) / (2 * h))
        return np.array(o)

    wA = wM = 0.0
    cases = [HCYMix([1.0], [6.0 * e / np.pi], [zz], np.array([[[kk]]]))
             for e in (0.1, 0.25, 0.4) for kk, zz in ((0.1, 1.8), (0.5, 1.8), (0.5, 5.0))]
    cases += [tanglu_state(x1, e, 0.9, sig_ratio=1.5, eps22=0.9, eps12=0.8)
              for x1 in (0.35, 0.65) for e in (0.20, 0.35)]
    for s in cases:
        q = solve_bh_mix(s)
        wA = max(wA, abs(free_energy_excess(q) / free_energy_charging(s) - 1.0))
        wM = max(wM, float(np.max(np.abs(chem_pot_excess(q) / _mu_ref(s) - 1.0))))
    print(f"   bDA (45) vs charging, 13 states (9 pure + 4 binary): {wA:.1e}   "
          f"{'PASS' if wA < 1e-3 else 'FAIL'}")
    print(f"   mu_i (46) vs d(bDA)/drho_i:                          {wM:.1e}   "
          f"{'PASS' if wM < 1e-3 else 'FAIL'}")
    print("   ⚠ Both need (43) read with its two print defects repaired — see")
    print("     `pressure_excess`.  Taking (43) literally gives 14-94 % errors.")

    # ── 9. the pole finder, for the mixture Fisher–Widom study ───────────────
    print("\n9. leading_poles_mix must reduce to msa_exact.leading_poles at N = 1")
    print("-" * 78)
    print("   Zeros of det F(s) vs zeros of the scalar F(s).  Entry point for the")
    print("   mixture FW line (FWL.8 cleared GO); nothing consumes it yet.")
    worst = 0.0
    for eta in (0.1, 0.25, 0.4):
        for zz, kk in ((1.8, 0.3), (1.8, 0.8), (5.0, 0.5)):
            ref = M.leading_poles(M.solve_bh_n(eta, [zz], [kk]))
            s = HCYMix([1.0], [6.0 * eta / np.pi], [zz], np.array([[[kk]]]))
            me = leading_poles_mix(solve_bh_mix(s))
            for a, b in zip(me, ref):
                if np.isfinite(a) and np.isfinite(b):
                    worst = max(worst, abs(a / b - 1.0))
    print(f"   worst relative deviation over 9 states x 3 poles: {worst:.1e}   "
          f"{'PASS' if worst < 1e-9 else 'FAIL'}")

    # ── 10. the fold route ───────────────────────────────────────────────────
    print("\n10. ⭐ the fold route — the canonical semi-compressibility free energy")
    print("-" * 78)

    def _hcy(eta, T):
        return HCYMix([1.0], [6.0 * eta / np.pi], [1.8], np.array([[[1.0 / T]]]))

    print("   (a) the fold dome: where the isotherm is cut, and where it closes")
    for T in (0.85, 0.98, 1.02):
        rv, rl = fold_densities(_hcy(0.35, T))
        print(f"       T*={T:5.2f}  " + ("OPEN — no fold, the isotherm gets through"
              if rv is None else
              f"cut over eta = {np.pi*rv/6:.4f} .. {np.pi*rl/6:.4f}"))
    ap = fold_apex(_hcy(0.35, 0.85), 0.85, 1.20)
    print(f"       fold apex T*_fold = {ap:.4f}   (spinodal apex T*_c = 1.0313)")

    print("   (b) ⭐ THE GATE: above the apex the fold route must reduce to the")
    print("       PURE compressibility route — no beta excursion is taken there.")
    worst = 0.0
    for T in (1.10, 1.05, 1.02):
        s = _hcy(0.35, T)
        worst = max(worst, abs(free_energy_fold_route(s, n_leg=300)
                               / free_energy_compressibility(s) - 1.0))
    print(f"       worst relative deviation, 3 states: {worst:.1e}   "
          f"{'PASS' if worst < 5e-3 else 'FAIL'}   (midpoint-rule step error)")

    print("   (c) below the apex it delivers a liquid free energy where the")
    print("       isothermal compressibility route has none.")
    print(f"       {'eta':>6}{'T*':>7} | {'fold route':>12}{'energy route':>14}"
          f"{'dev %':>8} | {'compr':>10}")
    for eta, T in ((0.35, 0.95), (0.35, 0.85), (0.40, 0.85)):
        s = _hcy(eta, T)
        f = free_energy_fold_route(s, n_leg=300, n_arc=80, t_apex=ap)
        e = free_energy_excess(solve_bh_mix(s))
        c = free_energy_compressibility(s)
        print(f"       {eta:6.2f}{T:7.2f} | {f:12.6f}{e:14.6f}{100*(f/e-1):8.2f} | "
              + (f"{c:10.6f}" if np.isfinite(c) else f"{'nan (cut)':>10}"))
    print("   ⚠ It is a LABELLED HYBRID: the dbeta legs are energy-route input and")
    print("     cannot be removed.  What the fold curve buys is that the path is no")
    print("     longer chosen — it is the theory's own boundary — and that the apex")
    print("     anchors it exactly (the two boundary points coincide there).")

    # ── 11. the SPINODAL route — the usable one ──────────────────────────────
    print("\n11. ⭐⭐ the SPINODAL route — the usable semi-compressibility method")
    print("-" * 78)
    print("   (a) `A` really vanishes at the densities `spinodal_densities` returns")
    worst = 0.0
    for T in (0.85, 0.95, 1.00):
        rv, rl = spinodal_densities(_hcy(0.2, T))
        vals = []
        for r in (rv, rl):
            q = solve_bh_mix(_scale_rho(_hcy(0.2, T), r))
            vals.append(abs(float(np.sum(q.p['A']))) if q else float('inf'))
        worst = max(worst, max(vals))
        print(f"       T*={T:5.2f}  eta_sp = {np.pi*rv/6:.4f} .. {np.pi*rl/6:.4f}"
              f"   |A| there = {vals[0]:.1e} / {vals[1]:.1e}")
    ap_s = spinodal_apex(_hcy(0.2, 0.85), 0.85, 1.20)
    print(f"       apex T*_c(A=0) = {ap_s:.4f}   {'PASS' if worst < 1e-6 else 'FAIL'}")

    print("   (a') ⭐ the SAME densities from an AUGMENTED SOLVE, not a sweep")
    print("        `spinodal_density`: residuals(u;rho)=0 AND det F(0)=0 together.")
    print("        The sweep inherits its continuation's reach -- below T* ~ 0.72 the")
    print("        liquid-side sweep meets the FOLD before the crossing and returns")
    print("        None on a side where the root exists.  The Newton never walks")
    print("        toward the fold at all.")
    print(f"        {'T*':>6}{'swept rv':>11}{'rl':>11}{'exact rv':>11}{'rl':>11}"
          f"{'rel dev':>10}")
    worst = 0.0
    rescued = 0
    for T in (1.02, 0.95, 0.85, 0.75, 0.70, 0.60):
        a, b = spinodal_densities(_hcy(0.2, T))
        c, d = spinodal_densities_exact(_hcy(0.2, T))
        fm = lambda x: (f"{x:11.6f}" if x is not None else f"{'None':>11}")
        if None not in (a, b, c, d):
            dv = max(abs(c / a - 1.0), abs(d / b - 1.0))
            worst = max(worst, dv)
            tail = f"{dv:10.1e}"
        else:
            rescued += (b is None and d is not None) + (a is None and c is not None)
            tail = f"{'rescued':>10}"
        print(f"        {T:6.2f}{fm(a)}{fm(b)}{fm(c)}{fm(d)}{tail}")
    print(f"        worst deviation where both work: {worst:.1e}   "
          f"{'PASS' if worst < 1e-8 else 'FAIL'}   "
          f"({rescued} side(s) the sweep lost and the solve recovered)")

    print("   (b) ⭐ THE GATE: above the apex it must reduce to the PURE")
    print("       compressibility route — no beta excursion is taken there.")
    worst = 0.0
    for T in (1.10, 1.05):
        s = _hcy(0.35, T)
        worst = max(worst, abs(free_energy_spinodal_route(s, n_leg=300)
                               / free_energy_compressibility(s) - 1.0))
    print(f"       worst relative deviation, 2 states: {worst:.1e}   "
          f"{'PASS' if worst < 5e-3 else 'FAIL'}   (midpoint-rule step error)")

    print("   (c) ⭐ why the spinodal and not the fold: `Delta` is LINEAR in the inset")
    print("       there (the arc never nears a singular point), so one Richardson")
    print("       step reaches inset -> 0.  On the fold arc it is sqrt(inset) at best.")
    cur = spinodal_curve(_hcy(0.2, 0.6 * ap_s), ap_s)
    st = _hcy(0.2, 0.85)
    ins = np.array([0.05, 0.04, 0.03, 0.02])
    dv = []
    for i in ins:
        dS, _ = spinodal_gap_free_energy(st, -1.0, n_arc=80, insets=(float(i),) * 2,
                                         t_apex=ap_s, curve=cur)
        dv.append(dS if dS is not None else np.nan)
    # ⚠ Filter before fitting: an inset that fails returns None -> NaN, and
    # `lstsq` answers that with `SVD did not converge`, which reads as a numerical
    # blow-up rather than as "one point is missing".
    dv = np.array(dv, dtype=float)
    ok = np.isfinite(dv)
    print("       Delta at insets " + " ".join(f"{i:.2f}" for i in ins) + ":  "
          + " ".join((f"{v:.4f}" if np.isfinite(v) else "  --  ") for v in dv))
    if ok.sum() < 3:
        print(f"       only {ok.sum()} of {len(ins)} insets returned — FAIL")
    else:
        A = np.vstack([ins[ok], np.ones(ok.sum())]).T
        c, *_ = np.linalg.lstsq(A, dv[ok], rcond=None)
        r2 = 1 - ((dv[ok] - A @ c) ** 2).sum() / ((dv[ok] - dv[ok].mean()) ** 2).sum()
        print(f"       linear fit R^2 = {r2:.6f} on {ok.sum()} points   "
              f"extrapolated to inset 0: {c[1]:.4f}   "
              f"{'PASS' if r2 > 0.999 else 'FAIL'}")

    print("   (d) cond(J) on the arc: the fold DIVERGES, the spinodal SATURATES")
    for nm, dens in (("fold    ", fold_densities), ("spinodal", spinodal_densities)):
        rv, _ = dens(_hcy(0.2, 0.85))
        row = []
        for i in (0.10, 0.05, 0.02, 0.01):
            s2 = _scale_rho(_hcy(0.2, 0.85), (1 - i) * rv)
            q = solve_bh_mix(s2)
            sc1, sc2 = _scales(s2)
            u = np.concatenate([q.Dt.ravel(), q.Gt.ravel()])
            row.append(np.linalg.cond(_jacobian(u, s2, sc1, sc2)))
        print(f"       {nm}  " + "  ".join(f"{v:8.2f}" for v in row)
              + f"   (inset 0.10 -> 0.01;  ratio {row[-1]/row[0]:.2f}x)")

    print("   (e) below the apex it delivers a liquid free energy where the")
    print("       isothermal compressibility route has none.")
    print(f"       {'eta':>5}{'T*':>7} | {'spinodal route':>15}{'energy route':>14}"
          f"{'dev %':>8} | {'compr':>10}")
    for eta, T in ((0.35, 0.95), (0.35, 0.85)):
        s = _hcy(eta, T)
        fs = free_energy_spinodal_route(s, n_leg=350, n_arc=80, t_apex=ap_s, curve=cur)
        fe = free_energy_excess(solve_bh_mix(s))
        fc = free_energy_compressibility(s)
        print(f"       {eta:5.2f}{T:7.2f} | {fs:15.6f}{fe:14.6f}{100*(fs/fe-1):8.2f} | "
              + (f"{fc:10.6f}" if np.isfinite(fc) else f"{'nan (cut)':>10}"))
    print("   ⇒ use THIS route; `free_energy_fold_route` is the extremal bound only.")

    # ── 12. the APEX-DETOUR route — the third crossing, and what it measures ──
    print("\n12. ⭐ the APEX-DETOUR route — a rectangle over the FOLD apex")
    print("-" * 78)
    ap_f = fold_apex(_hcy(0.2, 0.85), 0.85, 1.20)
    print(f"       fold apex T* = {ap_f:.4f}   vs   A=0 apex T* = {ap_s:.4f}"
          f"   (the fold dome is nested INSIDE)")
    print("   (a) THE GATE: above the apex it must reduce to the PURE")
    print("       compressibility route -- no beta excursion is taken there.")
    worst = 0.0
    for T in (1.10, 1.05):
        s = _hcy(0.35, T)
        worst = max(worst, abs(free_energy_critical_detour_route(s, n_leg=300)
                               / free_energy_compressibility(s) - 1.0))
    print(f"       worst relative deviation, 2 states: {worst:.1e}   "
          f"{'PASS' if worst < 5e-3 else 'FAIL'}   (midpoint-rule step error)")

    print("   (b) ⭐ the crossing HEIGHT barely matters -- which is the check that")
    print("       this is a path integral and not an artifact of where it turns.")
    print(f"       {'t_over':>8}{'T_cross':>10}{'Delta_0':>12}{'vs 1.005':>10}")
    # ⚠ Endpoints from the TRACED `A=0` curve, not from `spinodal_densities` at the
    # target `T`: the latter's liquid-side sweep meets the fold before the crossing
    # below ~0.72, so it returns None there and the comparison would silently lose
    # the coldest -- and most interesting -- rows.  This is also exactly how
    # `spinodal_gap_free_energy` gets its own endpoints, so the two stay identical.
    tl_, lv_, ll_ = cur

    def sp_of(T):
        if not (tl_[0] <= T <= tl_[-1]):
            return None, None
        return float(np.interp(T, tl_, lv_)), float(np.interp(T, tl_, ll_))

    base = None
    for tov in (1.005, 1.02, 1.05, 1.10):
        rv, rl = sp_of(0.85)
        d0, _ = critical_detour_gap_free_energy(_hcy(0.2, 0.85), 0.0, rv, rl, ap_f, tov)
        if d0 is None:
            continue
        base = d0 if base is None else base
        print(f"       {tov:8.3f}{tov*ap_f:10.4f}{d0:12.5f}"
              f"{100*(d0/base-1):9.2f}%")
    print("       (compare: the FOLD arc drifted by a factor 4 between insets")
    print("        0.02 and 0.05 for 2YK -- that is the difference between a path")
    print("        that avoids the singular locus and one that hugs it.)")

    print("   (c) ⭐⭐ the two usable crossings side by side.  IDENTICAL endpoints,")
    print("       identical rho legs, identical mu handed across -- only the path")
    print("       between the two spinodal densities differs.  So the spread is")
    print("       MSA's OWN path-dependence, not numerical error.")
    print(f"       {'T*':>6} | {'hug Delta_0':>12}{'detour':>11}{'diff %':>9}"
          f" | {'hug M':>10}{'detour M':>11}{'diff %':>9}")
    for T in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
        st = _hcy(0.2, T)
        rv, rl = sp_of(T)
        if rv is None:
            print(f"       {T:6.2f} |  outside the traced A=0 curve")
            continue
        dh, mh = spinodal_gap_free_energy(st, 0.0, n_arc=80, t_apex=ap_s, curve=cur)
        dd, md = critical_detour_gap_free_energy(st, 0.0, rv, rl, ap_f)
        if dh is None or dd is None:
            print(f"       {T:6.2f} |  {'hug' if dh is None else 'detour'} crossing "
                  "did not transport")
            continue
        print(f"       {T:6.2f} | {dh:12.5f}{dd:11.5f}{100*(dd/dh-1):8.2f}%"
              f" | {mh:10.4f}{md:11.4f}{100*(md/mh-1):8.2f}%")
    print("   ⇒ the gap between the two grows from ~3 % just below the apex to")
    print("     ~13 % at T* = 0.70: the further the detour has to climb, the more")
    print("     energy-route leg it carries, and MSA does not close the 2-form.")
    print("     ⚠ Neither is 'the' answer -- the pair BRACKETS it, and the bracket")
    print("     width is the honest error bar on any semi-compressibility liquid")
    print("     free energy below the apex.")

    # ── 13. the VIRIAL route — the third one ─────────────────────────────────
    print("\n13. ⭐⭐ the VIRIAL route, and the binodal that needs NO detour")
    print("-" * 78)
    print("   (a) K -> 0 gate: it must return the PY VIRIAL equation of state,")
    print("       where (43)+ref returns the PY COMPRESSIBILITY one.  The classic")
    print("       PY inconsistency is therefore visible at ZERO coupling -- the")
    print("       cheapest possible proof that the two are different objects.")
    print(f"       {'eta':>6}{'p_virial':>12}{'PY virial':>12}{'dev':>9}"
          f"{'p_(43)+ref':>13}{'PY compr':>11}{'dev':>9}")
    worst = 0.0
    for eta in (0.05, 0.15, 0.25, 0.35, 0.45):
        s0 = HCYMix([1.0], [6.0 * eta / np.pi], [1.8], np.array([[[1e-9]]]))
        q = solve_bh_mix(s0)
        rho = float(s0.rho[0])
        pv, pyv = pressure_virial(q), rho * (1 + 2*eta + 3*eta**2) / (1 - eta)**2
        p43 = hs_thermo_pure(s0)[0] + pressure_excess(q)
        pyc = rho * (1 + eta + eta**2) / (1 - eta)**3
        worst = max(worst, abs(pv/pyv - 1), abs(p43/pyc - 1))
        print(f"       {eta:6.2f}{pv:12.6f}{pyv:12.6f}{abs(pv/pyv-1):9.1e}"
              f"{p43:13.6f}{pyc:11.6f}{abs(p43/pyc-1):9.1e}")
    print(f"       worst: {worst:.1e}   {'PASS' if worst < 1e-7 else 'FAIL'}")

    print("   (b) ⭐ a THIRD critical point.  `dbetaP/drho` on the compressibility")
    print("       route is (A/2pi)^2, a PERFECT SQUARE, so that route has no loop")
    print("       and no binodal at all.  The virial route carries no such")
    print("       constraint and has its own.")
    print("                        compressibility    energy        virial")
    print("       HSY  z=1.8   T*_c    1.0331         1.2191        1.4132")
    print("                    eta_c   0.1617         0.1650        0.1616")
    print("       2YK LJ->2Y   T*_c    0.9330         1.1849        1.1041")
    print("                    eta_c   0.1680         0.1590        0.1289")
    print("       ⚠ The ORDERING is not universal: HSY is compr < energy < virial,")
    print("         2YK is compr < VIRIAL < energy.  Spread +36.8 % (HSY) vs")
    print("         +27.0 % (2YK), and on 2YK it is the ENERGY route that tops out.")
    print("       ⚠ And the eta_c do NOT generally agree: they fall within 2 % on")
    print("         HSY but the virial eta_c is 23 % BELOW the compressibility one")
    print("         on 2YK -- so 'the inconsistency is purely in T' is false.")

    print("   (c) ⭐⭐ the binodal, by ordinary Maxwell construction -- no crossing")
    print("       step, no inset, no bracket.  `min A` says whether the whole loop")
    print("       sits on the physical winding.")
    print(f"       {'T*':>7}{'T/Tc':>7}{'eta_v':>9}{'eta_l':>9}{'betaP':>10}"
          f"{'min A on loop':>15}{'cut?':>7}")
    VIR_TC = 1.4132
    for T in (1.35, 1.20, 1.05, 1.02, 1.00):
        s0 = HCYMix([1.0], [6.0 * 0.2 / np.pi], [1.8], np.array([[[1.0 / T]]]))
        res = free_energy_virial(s0)
        fv, _ = fold_densities(s0)
        cut = "no" if fv is None else "YES"
        b = binodal_virial(s0)
        if res is None or b is None:
            print(f"       {T:7.3f}{T/VIR_TC:7.3f}{'  no coexistence bracketed':>28}"
                  f"{'':17}{cut:>7}")
            continue
        r, p, mu, A = res
        d = np.diff(p) / np.diff(r)
        neg = np.where(d < 0)[0]
        m = (r >= r[neg[0]]) & (r <= r[neg[-1] + 1])
        print(f"       {T:7.3f}{T/VIR_TC:7.3f}{np.pi*b[0]/6:9.4f}{np.pi*b[1]/6:9.4f}"
              f"{b[2]:10.5f}{float(np.min(A[m])):15.4f}{cut:>7}")
    print("   => direct binodal over T/Tc = 0.955 .. 0.743 with the ENTIRE loop on")
    print("      the physical A>0 sheet; 0.743 .. 0.720 still uncut but the loop")
    print("      dips onto the wrong winding; below the fold apex (0.720) the")
    print("      isotherm is cut and the detour is needed again.")
    print("   ⚠ The fold is common to EVERY route -- there the solution does not")
    print("     exist -- so no route removes it.  What the virial route removes is")
    print("     the PERFECT-SQUARE obstruction, which was the compressibility")
    print("     route's own.  ⚠ And the energy route was never blocked at all.")

    # ── 14. the bridge: from_fmsa must reproduce the OZ solver's own MSA tail ──
    print("\n14. ⭐ from_fmsa -- the exact state FMSA-DP is a truncation of")
    print("-" * 78)
    print("   The paper compares FMSA-DP, OZ+MSA and exact MSA at the same state,")
    print("   so they must see ONE system.  `from_fmsa` reads it off the FMSA")
    print("   object; the definition it has to match is OZSolverMSA._c_msa_outer,")
    print("   which is what the OZ+MSA leg actually solves against.")
    import contextlib as _cl, io as _io, warnings as _w

    from fmsa_double_prop import FMSA_double_prop as _DP
    from fmsa_ga_matrix_mix import FMSA_GA_matrix_mix as _GA
    from oz_numerical import OZSolverMSA as _OZ
    print(f"   {'system':<34}{'omega':>7}{'worst rel err':>16}")
    worst_all = 0.0
    for lbl, sg, ep, xx, rr, TT in (
            ('binary, unequal sigma', [1.0, 2.0], [1.0, 0.5], [0.25, 0.75], 0.139, 1.0),
            ('binary, EQUAL sigma', [1.0, 1.0], [1.0, 0.5], [0.5, 0.5], 0.800, 1.5),
            ('ternary (two sigma_ij coincide)', [1.0, 1.5, 2.0], [1.0] * 3,
             [1 / 3] * 3, 0.150, 1.0)):
        with _w.catch_warnings(), _cl.redirect_stdout(_io.StringIO()):
            _w.simplefilter('ignore')
            dpm = _DP(np.array(sg), np.array(ep), np.array(xx), rr, TT,
                      n_tails=2, verbose=False)
            ozs = _OZ(_GA(np.array(sg), np.array(ep), np.array(xx), rr, TT,
                          n_tails=2, verbose=False), n_r=256)
        stb = from_fmsa(dpm)
        e = 0.0
        for i in range(len(sg)):
            for j in range(len(sg)):
                R = float(dpm.R_ij[i, j])
                for r in (R * 1.001, R * 1.3, R * 2.5):
                    a = ozs._c_msa_outer(r, i, j)
                    b = float(np.sum(stb.K[:, i, j]
                                     * np.exp(-stb.z * (r - stb.SIG[i, j])) / r))
                    e = max(e, abs(a - b) / max(abs(a), 1e-30))
        worst_all = max(worst_all, e)
        print(f"   {lbl:<34}{stb.T:7d}{e:16.2e}")
    print(f"   => worst over all three: {worst_all:.2e}   "
          f"{'PASS' if worst_all < 1e-11 else 'FAIL'}")
    print("   ⚠ The ternary is the case that matters: sigma_13 = sigma_22 = 1.5, so")
    print("     two PAIRS share one rate and omega is 10, not 2*6 = 12.  Emitting")
    print("     one tail per pair would have handed the solver a linearly")
    print("     DEPENDENT set; grouping by distinct z is what keeps it honest, and")
    print("     it degrades to Blum-Hoye's own covered case at equal diameters")
    print("     (omega = 2) with no branch.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", action="store_true", help="run the MSAX.6 gates")
    a = p.parse_args()
    if a.gate:
        return cmd_gate()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
