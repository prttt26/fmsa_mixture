r"""probe_truncation_cost_exact.py -- what the first-order truncation costs, measured
against the EXACT MSA, on the paper's reference binary.

Sec. I K of the paper ("What the truncation costs") used to quote the residual against
a numerically solved OZ+MSA, which brought a radial-grid error with it and had to be
shown converging under grid refinement.  Exact MSA (Blum--Hoye / Ginoza) solves the SAME
closure to all orders, and its real-space core is the Baxter convolution evaluated by
adaptive quadrature, so there is no grid anywhere in this comparison:

    exact      brk_completeness.machinery on X.from_fmsa(dp)   (real-space Baxter Q)
    exact c0   X.chat0(sol)                                     (analytic, Baxter at k=0)
    FMSA       dp.get_c_total(r, hs_ref="py")                   (closed form, PY HS)

FMSA is read on its PY hard sphere because the exact MSA IS PY at zero coupling, so
the difference is the truncation and nothing else.

System = the paper's reference binary, built exactly as probe_antisym.py builds it
(Fig. 2): sigma=[1,2], eps=[1,0.5], x=[0.25,0.75], rho*=0.139, T*=1, two Yukawa tails.

Gates (any failure raises):
  G1  exact c(r): 4 pi Int r^2 c dr (core, quad) + analytic tail  ==  chat0, per pair
  G2  outside the core FMSA and exact coincide (both are -beta u by the closure)
  G3  FMSA chat(0) from get_chat_k  ==  4 pi Int r^2 c_FMSA dr + tail, per pair

Run:  python3 probe_truncation_cost_exact.py
Writes numerical_notes/results/truncation_cost_exact.txt (header carries the command).
"""
from __future__ import annotations
import sys, time, pathlib
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import msa_exact_mix as X
from fmsa_double_prop import FMSA_double_prop
from brk_completeness import machinery

SIGS = np.array([1.0, 2.0]); EPS = np.array([1.0, 0.5]); XF = np.array([0.25, 0.75])
RHO, T = 0.139, 1.0
OUT = pathlib.Path("numerical_notes/results/truncation_cost_exact.txt")
PAIRS = ((0, 0), (0, 1), (1, 1))


def main() -> int:
    lines = [f"# python3 {' '.join(sys.argv)}   ({time.strftime('%Y-%m-%d %H:%M')})",
             f"# system: sigma={SIGS.tolist()} eps={EPS.tolist()} x={XF.tolist()} "
             f"rho*={RHO} T*={T} n_tails=2 (as probe_antisym.py)"]
    say = lambda s: (print(s, flush=True), lines.append(s))

    dp = FMSA_double_prop(SIGS, EPS, XF, RHO, T, n_tails=2, verbose=False)
    st = X.from_fmsa(dp)
    sol = X.solve_bh_mix(st)
    if sol is None:
        raise RuntimeError("exact MSA has no solution at the reference state")
    R = np.asarray(st.SIG, float); K = np.asarray(st.K, float); z = np.asarray(st.z, float)
    say(f"# exact MSA: omega={len(z)} tails  eta={np.pi/6*np.sum(st.rho*st.sigma**3):.4f}")

    cfun, *_ = machinery(sigma=list(st.sigma), rho=list(st.rho), K=K, z=list(z))
    c_ex = lambda i, j, r: cfun(i, j, float(r)) / (2.0 * np.pi * float(r))
    c_fm = lambda r: np.asarray(dp.get_c_total(np.atleast_1d(r), hs_ref="py"))
    tail0 = lambda i, j: 4*np.pi*sum(K[t, i, j]*(1.0 + z[t]*R[i, j])/z[t]**2
                                     for t in range(len(z)))
    lam = lambda i, j: abs(0.5*(st.sigma[i] - st.sigma[j]))

    # ---- G1: exact c(r) integrates to the analytic chat0 -------------------------
    ref = X.chat0(sol)
    g1 = 0.0
    for i, j in PAIRS:
        pts = [lam(i, j)] if lam(i, j) > 0 else None
        core = quad(lambda r: 4*np.pi*r*r*c_ex(i, j, r), 0.0, R[i, j],
                    points=pts, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
        e = abs((core + tail0(i, j)) / ref[i, j] - 1.0); g1 = max(g1, e)
    say(f"G1  exact: 4pi Int r^2 c + tail vs chat0        worst rel {g1:.2e}")
    if g1 > 1e-8:
        raise AssertionError("G1 failed")

    # ---- G2: outside the core the two coincide -----------------------------------
    g2 = 0.0
    for i, j in PAIRS:
        rs = np.linspace(R[i, j]*1.0005, 4.0*R[i, j], 200)
        cf = c_fm(rs)[i, j]
        ce = np.array([c_ex(i, j, r) for r in rs])
        g2 = max(g2, float(np.max(np.abs(cf - ce))))
    say(f"G2  outside R_ij: max|FMSA - exact|              {g2:.2e}")
    if g2 > 1e-9:
        raise AssertionError("G2 failed")

    # ---- G3 + chat(0): FMSA by two paths ----------------------------------------
    ck = np.asarray(dp.get_chat_k(np.array([1e-7])))
    ck = ck.reshape(-1, 2, 2)[0] if ck.size >= 4 else None
    c0f = np.zeros((2, 2)); c0hs = np.zeros((2, 2)); g3 = 0.0
    for i, j in PAIRS:
        pts = [lam(i, j)] if lam(i, j) > 0 else None
        core = quad(lambda r: 4*np.pi*r*r*float(c_fm(r)[i, j]), 0.0, R[i, j],
                    points=pts, limit=400, epsabs=1e-12, epsrel=1e-11)[0]
        c0f[i, j] = c0f[j, i] = core + tail0(i, j)
        hs = quad(lambda r: 4*np.pi*r*r*dp.get_HS_PY(r, i, j), 0.0, R[i, j],
                  points=pts, limit=400, epsabs=1e-12, epsrel=1e-11)[0]
        c0hs[i, j] = c0hs[j, i] = hs
        if ck is not None:
            g3 = max(g3, abs(c0f[i, j]/ck[i, j] - 1.0))
    say(f"G3  FMSA chat(0): quad vs get_chat_k            worst rel {g3:.2e}"
        + ("" if ck is not None else "   (get_chat_k shape unexpected: G3 skipped)"))
    if ck is not None and g3 > 1e-5:
        raise AssertionError("G3 failed")

    # ---- the measurement ---------------------------------------------------------
    say("")
    say("pair  R_ij   in-core max|resid|  max|c1|   resid/c1   L2 resid/c1"
        "   chat0 exact   chat0 FMSA   rel err   resid / first-order part")
    for i, j in PAIRS:
        n = 1200
        # ⚠ Never sample r == |lambda_ij| exactly: fmsa_double_prop.get_HS_PY returns BOTH
        # pieces' terms at the knot itself (-12.28 against -7.26 on either side at the
        # reference state), a measure-zero defect that a grid starting AT the knot turns
        # into a spurious 1600 % "residual".  Both segments therefore exclude the knot.
        L = lam(i, j)
        rs = (np.linspace(1e-4, R[i, j]*0.9999, n) if L == 0 else
              np.concatenate([np.linspace(1e-4, L, n//3)[:-1], np.linspace(L, R[i, j]*0.9999, n)[1:]]))
        ce = np.array([c_ex(i, j, r) for r in rs])
        cf = c_fm(rs)[i, j]
        c1 = np.array([dp.get_c1_inner(float(r), i, j) for r in rs])
        d = cf - ce
        m_res, m_c1 = float(np.max(np.abs(d))), float(np.max(np.abs(c1)))
        l2 = float(np.sqrt(np.trapz(rs*rs*d*d, rs) / np.trapz(rs*rs*c1*c1, rs)))
        rel0 = (c0f[i, j] - ref[i, j]) / ref[i, j]
        # residual as a fraction of FMSA's own first-order part of chat0 (core + tail)
        frac0 = (c0f[i, j] - ref[i, j]) / (c0f[i, j] - c0hs[i, j])
        say(f"{i+1}{j+1}   {R[i,j]:.3f}   {m_res:12.4e}  {m_c1:9.4e}   {m_res/m_c1:7.1%}"
            f"     {l2:7.1%}     {ref[i,j]:11.5f}  {c0f[i,j]:11.5f}  {rel0:+8.2%}   {frac0:+8.2%}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
