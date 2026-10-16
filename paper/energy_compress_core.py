"""energy_compress_core.py — shared machinery for the energy vs compressibility
route comparison over the three models that survived benchmarking:

    FMSA-DP   `FMSA_double_prop`      analytic, first order in the tail
    OZ-MSA    `OZSolverMSA(closure='msa')`   converged MSA, all orders
    OZ-HNCB   `OZSolverMSA(closure='hncb')`  converged HNC + hard-sphere bridge

Both routes are written as ∂βP/∂ρ* with the **same** hard-sphere block, so the
hard-sphere reference cancels between them exactly, for every model:

    compressibility   C(ρ) = [1 − xᵀĈ_HS(0)ρ] − xᵀĈ₁(0)ρ
    energy            E(ρ) = [1 − xᵀĈ_HS(0)ρ] + ρ·d²A/dρ²
    route gap         C − E = −xᵀĈ₁(0)ρ − ρ·A″(ρ)          ← tail only

where, per model,

    Ĉ_HS(0) : the model's own hard-sphere limit (λ=0)
    Ĉ₁(0)   : Ĉ_tot(0) − Ĉ_HS(0), the tail's contribution to the DCF
    A(ρ)    : the tail's contribution to βf_ex, by coupling-constant integration

        A(ρ) = ∫₀¹ dλ · 2π Σ_ij ρᵢρⱼ ∫ r²·βu_ij(r)·g_ij^{(λ)}(r) dr

λ (not β) is ramped, so the hard core — and with it d_BH and βf_HS — is held
fixed; only the Yukawa tail is switched on.  This is what makes A model-uniform,
and it reproduces FMSA-DP's analytic βf₁+βf₂ exactly: with g^{(λ)} = g⁰ + λg¹,
∫₀¹(g⁰+λg¹)dλ = g⁰ + g¹/2, which is the 2π…g⁰ + π…g¹ pair of
`FMSAFreeEnergy.betaf_outcore`.

Numerics (all validated in-session against FMSA-DP's closed form):
  · A″ by a 3-point second difference, δ_ρ = 0.3 % of the ρ-range.  OZ: δ 0.004
    → 0.001 moves A″ by 0.04 %.  DP: matches its analytic value.
  · λ-quadrature: 3-point Gauss–Legendre.  GL3 vs GL5 differ by 2e-6 — W(λ) is
    nearly linear.
  · OZ solves are λ-continued from the pure hard-sphere solution
    (`_init_c_hs_only`), never from the polynomial guess, which is known to
    select a wrong OZ branch.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from fmsa_double_prop import FMSA_double_prop
from fmsa_free_energy import FMSAFreeEnergy
from oz_numerical import OZSolverMSA

# 3-point Gauss–Legendre on [0,1]
GL_X = np.array([0.1127016653792583, 0.5, 0.8872983346207417])
GL_W = np.array([5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0])

OZ_NR = 2048          # radial points; 4096 shifts absolute ĉ(0) by ~1 %, gaps far less
OZ_TOL = 1e-11
OZ_MAXIT = 800

MODELS = ("FMSA-DP", "OZ-MSA", "OZ-HNCB")
STYLE = {                       # colour, linestyle, linewidth
    "FMSA-DP": ("#2a78d6", "-",  2.4),
    "OZ-MSA":  ("#e34948", "--", 2.0),
    "OZ-HNCB": ("#4a3aa7", "-.", 2.0),
}


# ─── systems ──────────────────────────────────────────────────────────────────

def make_model(sysd: dict, rhostar: float, tstar: float) -> FMSA_double_prop:
    """The FMSA-DP object.  It also carries the potential (R_ij, k_bare, z_poles)
    that the OZ solver reads, so all three models see the identical system."""
    # z/k default to None, which is exactly FMSA_double_prop's own default, so
    # every existing caller is bit-for-bit unaffected.  They are needed for the
    # hard-core Yukawa system of Figure 11, whose tail is not the 2YK LJ fit.
    return FMSA_double_prop(sysd["sig"], sysd["eps"], sysd["x"], rhostar, tstar,
                            n_tails=sysd.get("nt", 2),
                            z=sysd.get("z"), k=sysd.get("k"),
                            eps_ij=sysd.get("eps_ij"),
                            input_type=sysd.get("input_type", "lj"),
                            verbose=False)


_MODEL_CACHE: dict = {}


def cached_model(sysd: dict, rhostar: float, tstar: float) -> FMSA_double_prop:
    key = (sysd["key"], round(float(rhostar), 12), round(float(tstar), 12))
    m = _MODEL_CACHE.get(key)
    if m is None:
        if len(_MODEL_CACHE) > 4000:
            _MODEL_CACHE.clear()
        m = _MODEL_CACHE[key] = make_model(sysd, rhostar, tstar)
    return m


def beta_u_tail(m: FMSA_double_prop, r: np.ndarray, i: int, j: int) -> np.ndarray:
    """βu_ij(r) for r ≥ R_ij (0 inside — g vanishes there anyway).

    The MSA outer closure of every model here is c = −βu with
    βu_ij(r) = −Σ_t β·k̄_{t,ij}·e^{−z_t(r−R_ij)}/r.
    """
    R = float(m.R_ij[i, j])
    out = np.zeros_like(r)
    sel = r >= R
    for t in range(m.n_tails):
        z = float(m.z_poles[t, i, j])
        out[sel] += -m.beta * float(m.k_bare[t, i, j]) * np.exp(-z * (r[sel] - R)) / r[sel]
    return out


# ─── hard-sphere block, shared by both routes ─────────────────────────────────

def _betaf_hs_raw(rho: np.ndarray, diam: np.ndarray) -> float:
    """White Bear I (BMCSL) FMT β·F_HS/V — the same expression as
    `FMSAFreeEnergy.betaf_hs`, as a plain function of (ρ, diameters)."""
    R = diam / 2.0
    n0 = float(np.sum(rho))
    n1 = float(np.sum(rho * R))
    n2 = 4.0 * np.pi * float(np.sum(rho * R**2))
    n3 = (4.0 / 3.0) * np.pi * float(np.sum(rho * R**3))
    dl = 1.0 - n3
    return (-n0 * np.log(dl) + n1 * n2 / dl
            + (n2**3 / (36.0 * np.pi)) * (np.log(dl) / n3**2 + 1.0 / (n3 * dl**2)))


def chat0_hs_fmt(m: FMSA_double_prop, h: float = 1e-4) -> np.ndarray:
    """ĉ^FMT_HS,ij(0) = −∂²(βf_HS)/∂ρᵢ∂ρⱼ — exact for White Bear I FMT, which is
    FMSA-DP's own hard-sphere reference (`hs_ref='fmt'`)."""
    diam, rho = m.diag_r, m.rho
    N = len(rho)
    out = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            def f(si: float, sj: float) -> float:
                r = rho.copy()
                r[i] += si * h
                r[j] += sj * h
                return _betaf_hs_raw(r, diam)
            out[i, j] = -(f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4.0 * h * h)
    return out


# ─── FMSA-DP: everything closed form ──────────────────────────────────────────

def dp_chat0(m: FMSA_double_prop, k: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """(Ĉ_HS(0), Ĉ₁(0)) for FMSA-DP.  ĉ₁ from the closed-form transform at k→0
    (flat to 1e-8 below k=1e-4); the hard-sphere block from the FMT identity."""
    c1 = np.real(m.get_c1hat_k(np.array([k]))[0])
    return chat0_hs_fmt(m), c1


def dp_A(m: FMSA_double_prop) -> float:
    """A(ρ) for FMSA-DP = βf₁ + βf₂ = ∫₀¹ dλ ⟨…g^{(λ)}⟩ with g^{(λ)}=g⁰+λg¹."""
    return FMSAFreeEnergy(m).betaf_outcore()


# ─── OZ: λ-continued solves ───────────────────────────────────────────────────

def _oz_solver(m: FMSA_double_prop, closure: str, n_r: int = OZ_NR) -> OZSolverMSA:
    s = OZSolverMSA(m, n_r=n_r)
    if closure == "hncb":
        with contextlib.redirect_stdout(io.StringIO()):
            s._compute_hs_bridge()          # FMT-based hard-sphere bridge
    return s


def _raw_step(s: OZSolverMSA, lam: float, c0: np.ndarray, closure: str,
              method: str = "diis", alpha: float = 0.2,
              maxit: int = OZ_MAXIT, tol: float = OZ_TOL):
    s._k_scale = lam
    with contextlib.redirect_stdout(io.StringIO()):
        ok = s._run_iteration(c0, tol, maxit, alpha, closure=closure,
                              method=method, m_diis=5, iterate="perturbation")
    if not ok:
        return None, None
    return s._c_converged.copy(), s._h_converged


def _oz_step(s: OZSolverMSA, lam: float, c0: np.ndarray, closure: str,
             lam_prev: float = 0.0):
    """One converged solve at coupling λ, warm-started from c0.

    Retry ladder — HNCB in particular oscillates rather than stalls near its own
    spinodal, and plain DIIS then fails at λ→1:
      (a) DIIS from the warm start
      (b) sub-ramp λ_prev → λ in 5 steps, DIIS at each
      (c) damped Picard (α=0.05)
    Returns (c, h) or (None, None) when the closure genuinely has no solution
    reachable from the hard-sphere branch.
    """
    c, h = _raw_step(s, lam, c0, closure)
    if c is not None:
        return c, h
    cc = c0
    for sub in np.linspace(lam_prev, lam, 6)[1:]:
        cc, hh = _raw_step(s, float(sub), cc, closure)
        if cc is None:
            break
    if cc is not None:
        return cc, hh
    return _raw_step(s, lam, c0, closure, method="picard", alpha=0.05, maxit=900)


def _W(s: OZSolverMSA, h: np.ndarray) -> float:
    """W(λ) = 2π Σ_ij ρᵢρⱼ ∫ r² βu_ij g_ij^{(λ)} dr."""
    m, r = s.fmsa, s.r
    tot = 0.0
    for i in range(m.N):
        for j in range(m.N):
            bu = beta_u_tail(m, r, i, j)
            tot += float(m.rho[i]) * float(m.rho[j]) * np.trapz(
                r**2 * bu * (1.0 + h[i, j]), r)
    return 2.0 * np.pi * tot


def _chat0_from_c(s: OZSolverMSA, c: np.ndarray) -> np.ndarray:
    N, r = s.fmsa.N, s.r
    out = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            out[i, j] = 4.0 * np.pi * np.trapz(r**2 * c[i, j], r)
    return out


_LAM_PATH = tuple(sorted(set([0.0] + [float(v) for v in GL_X] + [1.0])))


def oz_point(sysd: dict, rhostar: float, tstar: float, closure: str,
             n_r: int = OZ_NR, seeds: dict | None = None) -> dict | None:
    """A(ρ), Ĉ_HS(0), Ĉ₁(0) at one state point, plus the converged c at each λ.

    `seeds` — {λ: c} from a nearby ρ.  The stencil points sit 0.2 % away, so
    seeding each λ from the centre's own solution at the same λ turns their
    solves into two or three DIIS iterations.  Falls back to the hard-sphere
    branch when a seed is not supplied or fails.
    """
    m = cached_model(sysd, rhostar, tstar)
    s = _oz_solver(m, closure, n_r)

    sols: dict[float, np.ndarray] = {}
    A = 0.0
    w_of = dict(zip((float(v) for v in GL_X), (float(v) for v in GL_W)))
    c_prev, lam_prev = None, 0.0
    for lam in _LAM_PATH:
        seed = (seeds or {}).get(lam)
        start = seed if seed is not None else (
            c_prev if c_prev is not None else s._init_c_hs_only())
        c, h = _oz_step(s, lam, start, closure, lam_prev)
        if c is None and seed is not None:          # bad seed → hard-sphere branch
            start = c_prev if c_prev is not None else s._init_c_hs_only()
            c, h = _oz_step(s, lam, start, closure, lam_prev)
        if c is None:
            return None
        sols[lam] = c
        c_prev, lam_prev = c, lam
        if lam == 0.0:
            chat_hs = _chat0_from_c(s, c)
        if lam in w_of:
            A += w_of[lam] * _W(s, h)
        if lam == 1.0:
            chat_tot = _chat0_from_c(s, c)
    return {"A": A, "chat_hs": chat_hs, "chat1": chat_tot - chat_hs, "sols": sols}


# ─── the two route indicators ─────────────────────────────────────────────────

def indicators_full(sysd: dict, rhostar: float, tstar: float, model: str,
                    drho: float | None = None) -> tuple:
    """(C, E, det) at one state point, from a single set of solves.

    C, E — the two routes' ∂βP/∂ρ* (isopleth / mechanical), sharing the
           hard-sphere block so it cancels in C − E.
    det  — det(I − X̂Ĉ(0)), the full binary stability determinant, i.e. the
           correct mixture spinodal; C is its isopleth projection.  It costs
           nothing extra here because Ĉ_tot(0) is already in hand.
    """
    if drho is None:
        drho = sysd.get("drho", 2e-3)
    x = np.asarray(sysd["x"], dtype=float)
    m = cached_model(sysd, rhostar, tstar)

    if model == "FMSA-DP":
        chs, c1 = dp_chat0(m)
        hs_block = 1.0 - float(x @ chs @ m.rho)
        C = hs_block - float(x @ c1 @ m.rho)
        Ap = dp_A(cached_model(sysd, rhostar + drho, tstar))
        A0 = dp_A(m)
        Am = dp_A(cached_model(sysd, rhostar - drho, tstar))
        E = hs_block + rhostar * (Ap - 2.0 * A0 + Am) / drho**2
        Ctot = chs + c1
    else:
        closure = {"OZ-MSA": "msa", "OZ-HNCB": "hncb"}[model]
        p_0 = oz_point(sysd, rhostar, tstar, closure)
        if p_0 is None:
            return None, None, None
        seeds = p_0["sols"]
        p_hi = oz_point(sysd, rhostar + drho, tstar, closure, seeds=seeds)
        p_lo = oz_point(sysd, rhostar - drho, tstar, closure, seeds=seeds)
        if p_hi is None or p_lo is None:
            return None, None, None
        hs_block = 1.0 - float(x @ p_0["chat_hs"] @ m.rho)
        C = hs_block - float(x @ p_0["chat1"] @ m.rho)
        A2 = (p_hi["A"] - 2.0 * p_0["A"] + p_lo["A"]) / drho**2
        E = hs_block + rhostar * A2
        Ctot = p_0["chat_hs"] + p_0["chat1"]

    Xh = np.sqrt(np.outer(m.rho, m.rho))
    det = float(np.linalg.det(np.eye(len(m.rho)) - Xh * Ctot))
    return C, E, det


def indicators(sysd: dict, rhostar: float, tstar: float, model: str,
               drho: float | None = None) -> tuple[float, float] | tuple[None, None]:
    """(C, E) — see `indicators_full`."""
    C, E, _ = indicators_full(sysd, rhostar, tstar, model, drho)
    return C, E
