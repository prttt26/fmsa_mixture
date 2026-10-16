r"""msa_exact_phase_lines.py — binodal / spinodal / fold for the *analytic* exact MSA.

Single component, two potentials:

    HSY   hard-core Yukawa, one tail        z = 1.8,   kv = [+1]
    HC2Y  hard core + two Yukawas          z = [2.9637, 14.0167], kv = [+2.1714, -2.1714]
          ⚠ NOT the soft-core `2YK` of HSY_vs_2YK.md — same name, but that one
            has no excluded volume.  The hard-core YK model IS the object of
            study; mapping LJ onto it would need a T-dependent BH diameter and is
            out of scope (that is the cDFT engine's problem).  ⭐ sigma is FIXED
            at 1 here and in fmsa_phase_lines.py alike — verified identical to the
            last digit at T* = 0.7/1.0/1.4 — which is what makes the two figures
            share an axis.  The JSON key stays '2YK' for continuity.

The coupling multiplier is the inverse temperature, `λ = 1/T*` (`K = λ·kv`), so a point on
`trace_coupling`'s branch at multiplier `λ` is the physical state at `T* = 1/λ`.

Three lines, each from the *same* analytic branch object (`msa_exact`):

  * **spinodal**  `A = 0`  (the perfect square `a = (A/2π)²` touches zero) — `spinodal_coupling`.
  * **fold**      `det J = 0`  — the branch terminates; `trace_coupling`'s last point.
  * **two compressibility hybrids** — same `ρ` legs and endpoints, different bridge across the
                  `A < 0` gap: hug the `A = 0` spinodal, or a rectangle over the *fold* apex.  Their
                  spacing is MSA's path-dependence, not error (3.4–13.3 % in `Δ₀`).
  * **binodal**   coexistence from the energy-route free energy.  The exact-MSA internal energy is
                  closed form in Waisman's amplitude, `βU_ex/N = −½ Σ_t v_t` with
                  `v_t = 24ξ K_t ĝ(z_t)e^{z_t}` (= `SolutionN.v`), so the tail free energy is the
                  coupling-constant (charging) integral

                      βf_tail(η,T*)/N = −½ ∫₀^{1/T*} (dλ/λ) Σ_t v_t(λ),

                  added to the ideal term and the PY-compressibility hard-sphere reference
                  `βf_HS/N = 3/(2(1−η)²) − 3/2 − ln(1−η)` (whose ∂βp/∂ρ is a_PY, so the K=0 edge is
                  consistent with the spinodal's perfect square — gated below).  A **Carnahan–
                  Starling** reference is available via `hs="cs"` / `binodal_on`, for the one job
                  that needs it: comparison against a model built on a CS core, where an
                  unmatched reference would be mistaken for a closure or truncation difference.
                  It is NOT the default — it breaks the `a_PY` consistency above.  See `bf_hs_ex`.

⚠ Route note: the binodal is an *energy*-route object; the spinodal (`a=0`) and fold are
*compressibility*/branch objects.  MSA is not route-consistent, so the binodal need not osculate the
`a=0` spinodal exactly at the critical point — that residual is the energy-vs-compressibility gap this
project quantifies elsewhere (X.9).  Both spinodals are reported so the gap is visible.

    python msa_exact_phase_lines.py            # compute + save JSON + render figure
    python msa_exact_phase_lines.py --render   # re-render from saved JSON
"""
from __future__ import annotations
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import msa_exact as M
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, fsolve

MODELS = {
    "HSY":  dict(z=[1.8],            kv=[1.0],            label=r"HSY  ($z=1.8$)"),
    "2YK":  dict(z=[2.9637, 14.0167], kv=[2.1714, -2.1714], label=r"HC2Y  (hard core + 2 Yukawa)"),
}
OUTDIR = "numerical_notes/results/msa_exact"
JSON = OUTDIR + "/msa_exact_phase_lines.json"
PNG = OUTDIR + "/msa_exact_phase_lines.png"
LAM_MAX = 12.0          # λ = 1/T*; the spinodal/fold sit at λ ~ 1–4, binodal reaches T* ~ 0.35


def rho_of(eta: float) -> float:
    return 6.0 * eta / np.pi


def bf_hs_ex(eta, hs: str = "py"):
    """Hard-sphere excess free energy per particle, in one of TWO references.

    `hs="py"` (default, and the only one the compressibility objects may use)
        PY-compressibility, `3/(2(1−η)²) − 3/2 − ln(1−η)`.  Its `∂βp/∂ρ` is exactly
        `a_PY`, which is what makes the `K=0` edge of the binodal consistent with the
        `A = 0` spinodal's perfect square.  Changing this would break that gate.

    `hs="cs"`  Carnahan–Starling, `η(4−3η)/(1−η)²`.
        ⚠ Used ONLY to put this leg on the same hard-sphere reference as FMSA-DP, whose
        `FMSAFreeEnergy` is built on BMCSL (= CS at `N=1`).  The two `Z`s differ by
        0.1 % at `η=0.10` but **4.3 % at `η=0.40`**, and the densities that matter are
        the liquid-branch ones: measured end to end, swapping the reference moves the
        exact-MSA `ρ*_L` by up to **7 %**, which is LARGER than the first-order
        truncation the comparison exists to measure.  The resulting curve is a HYBRID
        (MSA tail + CS core) and is no longer consistent with its own `A=0` spinodal —
        exactly the same kind of hybrid FMSA-DP already is, which is the point.
        See `vle_hcy_shukla.exact_msa`.
    """
    if hs == "cs":
        return eta * (4.0 - 3.0 * eta) / (1.0 - eta) ** 2
    return 1.5 / (1.0 - eta) ** 2 - 1.5 - np.log(1.0 - eta)


def bmu_hs_ex(eta, hs: str = "py"):
    """`βμ^ex_HS = d(ρ·bf_hs_ex)/dρ = bf_hs_ex + η·d(bf_hs_ex)/dη` — in closed form.

    Needed because the coexistence construction must not obtain `μ` by
    differentiating a spline: see `fold_tangent`.
    """
    if hs == "cs":
        return bf_hs_ex(eta, "cs") + eta * (4.0 - 2.0 * eta) / (1.0 - eta) ** 3
    return (bf_hs_ex(eta)
            + eta * (3.0 / (1.0 - eta) ** 3 + 1.0 / (1.0 - eta)))


# ── per-η branch: trace once to the fold, extract spinodal / fold / energy integral ──────────────

def branch(eta: float, z, kv, lam_max: float = LAM_MAX):
    """Returns dict with lam_sp (A=0), lam_fold (branch end), and I(λ)=∫₀^λ (Σv/λ')dλ' spline.

    `lam_fold` is None if the branch ran to `lam_max` without folding (no fold at this η)."""
    tr = M.trace_coupling(eta, z, kv, lam_max=lam_max)
    if len(tr) < 4:
        return None
    lam = np.array([t[0] for t in tr])
    A = np.array([t[1].A for t in tr])
    vsum = np.array([float(np.sum(t[1].v)) for t in tr])

    # spinodal: first A = 0
    lam_sp = None
    s = np.where(np.sign(A[:-1]) != np.sign(A[1:]))[0]
    if len(s):
        j = s[0]
        f = (0.0 - A[j]) / (A[j + 1] - A[j])
        lam_sp = float(lam[j] + f * (lam[j + 1] - lam[j]))

    # a genuine fold stops the trace early; hitting lam_max means no fold at this η
    lam_fold = float(lam[-1]) if lam[-1] < 0.98 * lam_max else None

    # energy integral over the PHYSICAL part (up to the spinodal), integrand Σv/λ, finite at 0
    hi = (np.searchsorted(lam, lam_sp) if lam_sp else len(lam))
    lp, vp = lam[:hi], vsum[:hi]
    if len(lp) < 3:
        return dict(lam_sp=lam_sp, lam_fold=lam_fold, I=None, lam_phys=lp[-1] if len(lp) else 0.0)
    h = np.empty_like(lp)
    h[0] = vp[1] / lp[1] if lp[1] > 0 else 0.0          # lim Σv/λ = (Σv)'(0)
    h[1:] = vp[1:] / lp[1:]
    Icum = np.concatenate([[0.0], cumulative_trapezoid(h, lp)])
    I = CubicSpline(lp, Icum, extrapolate=False)
    return dict(lam_sp=lam_sp, lam_fold=lam_fold, I=I, lam_phys=float(lp[-1]))


def collect(model: dict, etas: np.ndarray):
    z, kv = model["z"], model["kv"]
    rows = {}
    for e in etas:
        b = branch(float(e), z, kv)
        if b is not None:
            rows[float(e)] = b
    return rows


# ── free energy per particle at (η, T*) on the physical branch ───────────────────────────────────

def bfN(eta: float, tstar: float, b: dict, hs: str = "py") -> float | None:
    lam = 1.0 / tstar
    if b["I"] is None or lam > b["lam_phys"]:
        return None                                    # inside the dome / past physical reach
    f_tail = -0.5 * float(b["I"](lam))
    rho = rho_of(eta)
    return float(np.log(rho) - 1.0 + bf_hs_ex(eta, hs) + f_tail)


def _lower_hull(rho, fv):
    """Lower convex hull of (ρ, f) points, sorted by ρ (Andrew's monotone chain)."""
    idx = np.argsort(rho)
    rho, fv = rho[idx], fv[idx]
    hull = []
    for x, y in zip(rho, fv):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) <= 0:   # not a lower-convex turn
                hull.pop()
            else:
                break
        hull.append((x, y))
    return np.array(hull)


def coexistence(etas, rows, tstar, hs: str = "py"):
    """Common tangent as the lower-convex-hull edge that bridges the dome (no seeds).

    The exact-MSA free energy exists only where `a ≥ 0`, so the two branches end exactly at the
    `A=0` spinodal densities; the bridging tangent therefore always lands *outside* the spinodal.
    Hull endpoints are refined by a local tangent solve restricted to each branch."""
    rho, fv = [], []
    for e in etas:
        b = rows.get(float(e))
        if b is None:
            continue
        val = bfN(float(e), tstar, b, hs)
        if val is None or not np.isfinite(val):
            continue
        r = rho_of(float(e))
        rho.append(r); fv.append(r * val)               # free energy per volume
    if len(rho) < 8:
        return None
    rho = np.array(rho); fv = np.array(fv)
    rho_lo, rho_hi = rho.min(), rho.max()
    hull = _lower_hull(rho, fv)
    if len(hull) < 3:
        return None
    gaps = np.diff(hull[:, 0])
    k = int(np.argmax(gaps))
    if gaps[k] < 2.0 * np.median(np.diff(np.sort(rho))):  # no dome ⇒ single phase at this T*
        return None
    rg0, rl0 = float(hull[k, 0]), float(hull[k + 1, 0])
    # a tie-line endpoint sitting on the η-grid edge means the branch ran off the grid
    # (dilute gas below η_min, or dense liquid above η_max) — unresolved, drop it
    if rg0 <= rho_lo + 1e-9 or rl0 >= rho_hi - 1e-9:
        return None
    mid = 0.5 * (rg0 + rl0)
    gas = rho <= rg0 + 1e-12
    liq = rho >= rl0 - 1e-12
    if gas.sum() < 3 or liq.sum() < 3:
        return rg0, rl0                                   # hull endpoints are already outside
    sg = CubicSpline(rho[gas], fv[gas], extrapolate=False)
    sl = CubicSpline(rho[liq], fv[liq], extrapolate=False)

    def res(u):
        rg, rl = u
        mg, ml = sg(rg, 1), sl(rl, 1)
        chord = (sl(rl) - sg(rg)) / (rl - rg)
        return [mg - ml, mg - chord]
    try:
        sol, _, ok, _ = fsolve(res, [rg0, rl0], full_output=True, xtol=1e-12)
        rg, rl = float(sol[0]), float(sol[1])
        if ok == 1 and rho[gas].min() <= rg <= rg0 + 1e-6 and rl0 - 1e-6 <= rl <= rho[liq].max():
            return rg, rl
    except Exception:
        pass
    return rg0, rl0


def energy_spinodal(etas, rows, tstar):
    """(η_gas, η_liq) bounding `a_energy = ρ·d²f_vol/dρ² < 0` — the **energy-route** spinodal.

    Nests with the energy-route binodal (same route), unlike the `A=0` compressibility spinodal.
    Works both where the `A=0` dome leaves a gap (`T* < Tc_compress`) and where the branch is
    contiguous (`Tc_compress < T* < Tc_binodal`, the whole upper dome) by taking a **local**
    non-uniform second difference and skipping any stencil that straddles the gap."""
    pts = []
    for e in etas:
        b = rows.get(float(e))
        if b is None:
            continue
        val = bfN(float(e), tstar, b)
        if val is None or not np.isfinite(val):
            continue
        r = rho_of(float(e)); pts.append((r, r * val))
    if len(pts) < 8:
        return None
    pts.sort()
    rho = np.array([p[0] for p in pts]); fv = np.array([p[1] for p in pts])
    med = np.median(np.diff(rho))
    ae = np.full(len(rho), np.nan)
    for i in range(1, len(rho) - 1):
        h1, h2 = rho[i] - rho[i - 1], rho[i + 1] - rho[i]
        if h1 > 3 * med or h2 > 3 * med:                  # stencil straddles the A=0 dome gap
            continue
        fpp = 2.0 / (h1 + h2) * ((fv[i + 1] - fv[i]) / h2 - (fv[i] - fv[i - 1]) / h1)
        ae[i] = rho[i] * fpp
    negs = np.where((~np.isnan(ae)) & (ae < 0))[0]
    if len(negs) == 0:                                    # a_energy ≥ 0 everywhere ⇒ above Tc
        return None
    ig, il = int(negs[0]), int(negs[-1])

    def cross(a, b):                                      # a_energy=0 between opposite-sign nodes
        return rho[a] + (0.0 - ae[a]) / (ae[b] - ae[a]) * (rho[b] - rho[a])
    eta_g = (cross(ig - 1, ig) if ig > 0 and not np.isnan(ae[ig - 1]) and ae[ig - 1] > 0
             else rho[ig]) * np.pi / 6
    eta_l = (cross(il, il + 1) if il < len(rho) - 1 and not np.isnan(ae[il + 1]) and ae[il + 1] > 0
             else rho[il]) * np.pi / 6
    return float(eta_g), float(eta_l)


def binodal(rows: dict, etas: np.ndarray):
    """Trace, in one downward T* scan, both the coexistence curve and the energy-route spinodal.

    The *energy*-route binodal has its own (higher) critical temperature than the compressibility
    `a=0` spinodal — MSA's route inconsistency — so the scan starts well above the spinodal Tc and
    lets `coexistence` return None until it enters the binodal's own dome.  The energy-route
    spinodal (`a_energy=0`) nests with the binodal (same route)."""
    sp = [1.0 / rows[e]["lam_sp"] for e in sorted(rows) if rows[e]["lam_sp"]]
    if not sp:
        return [], []
    Tc_sp = max(sp)
    bino, spin_e = [], []
    T = 1.6 * Tc_sp
    step = 0.01 * Tc_sp
    while T > 0.30 * Tc_sp:
        ct = coexistence(etas, rows, T)
        if ct is not None:
            rg, rl = ct
            if rl - rg > 1e-3:
                bino.append(dict(T=float(T), eta_g=float(rg * np.pi / 6),
                                 eta_l=float(rl * np.pi / 6)))
        es = energy_spinodal(etas, rows, T)
        if es is not None:
            spin_e.append(dict(T=float(T), eta_g=es[0], eta_l=es[1]))
        T -= step
    return bino, spin_e


# ── the same binodal on a CHOSEN hard-sphere reference, for cross-model comparison ───────────────

ETAS_DEFAULT = np.round(np.concatenate([np.arange(0.003, 0.03, 0.003),
                                        np.arange(0.03, 0.492, 0.006)]), 4)


def binodal_on(model_key: str, temps, hs: str = "cs", etas=None, verbose: bool = True):
    """Energy-route coexistence for one `MODELS` entry, at GIVEN `T*`, on HS reference `hs`.

    Exists so the exact-MSA leg of the FMSA-DP comparison can be evaluated at the same
    temperatures and on the same hard-sphere core as FMSA-DP; see `bf_hs_ex` for why the
    reference has to be matched before any difference is called "the truncation".

    The expensive part -- tracing one MSA branch per `η` -- is reference-INDEPENDENT
    (`branch` returns only the tail's charging integral), so both references reuse one
    `collect`.  Returns `[{T, rho_v, rho_l}, ...]`, ascending in `T*`.
    """
    etas = ETAS_DEFAULT if etas is None else etas
    rows = collect(MODELS[model_key], etas)
    out = []
    for T in sorted(float(t) for t in temps):
        ct = coexistence(etas, rows, T, hs)
        if ct is None or ct[1] - ct[0] <= 1e-3:
            if verbose:
                print(f"  T* = {T:.3f}   no coexistence")
            continue
        out.append(dict(T=T, rho_v=float(ct[0]), rho_l=float(ct[1])))
        if verbose:
            print(f"  T* = {T:.3f}   rho_v = {ct[0]:.4f}   rho_l = {ct[1]:.4f}")
    return out



# ── the compressibility–fold route binodal ───────────────────────────────────────────────────────
#
# The fourth line on the figure.  `binodal` above is the ENERGY route; this one uses the
# compressibility route everywhere it exists and pays the energy route only on the minimal β
# excursion needed to clear the fold dome, along the theory's own fold curve
# (`msa_exact_mix.free_energy_fold_route`, results/msa_exact/msa_exact.md §8).
#
# ⚠ Why it is worth a separate line: the *pure* compressibility route has no binodal at all —
# `a = F(0)²` is a perfect square, so `p` and `μ` are monotone along an isotherm and no two
# densities can share both.  This line is the closest thing to a compressibility-route coexistence
# curve that exists, and its distance from the blue one is the route gap read on the phase diagram.

def spinodal_curve_of(d_model, kmax, n=24, t_lo=0.50):
    """`((T_internal, ρ_v, ρ_l), T_apex_internal, T_apex_phys)` from the stored `A=0` line."""
    sp = sorted(d_model["spinodal"], key=lambda x: x["eta"])
    ipk = max(range(len(sp)), key=lambda i: sp[i]["T"])
    apS = sp[ipk]["T"]
    gv, gl = sp[:ipk + 1], sp[ipk:]
    ts = np.linspace(t_lo * apS, apS, n)
    rv = np.array([rho_of(np.interp(t, [q["T"] for q in gv], [q["eta"] for q in gv]))
                   for t in ts])
    rl = np.array([rho_of(np.interp(t, [q["T"] for q in gl][::-1],
                                    [q["eta"] for q in gl][::-1])) for t in ts])
    return (ts / kmax, rv, rl), apS / kmax, apS


def fold_apex_of(d_model, kmax):
    """`(T_fold_apex_internal, T_fold_apex_phys)` from the stored `det J = 0` line.

    ⚠⚠ **A crossing must clear the `A = 0` apex, not this one** — an earlier version of
    this docstring said the opposite ("a crossing only has to clear the former") and the
    detour was built on it.  Clearing the fold guarantees only that the leg finds a
    SOLUTION; between the two apexes the isotherm is whole and the entire window is
    wrong-winding (`h(r) ~ e^{+s*r}`, see
    `numerical_notes/theory/wrong_winding_spinodal_to_fold.md`).

    Measured cost of the error at the old default `t_over = 1.02`: HSY crossed at
    `1.0381 > 1.0313` and cleared by luck; **2YK crossed at `0.9051` against an `A = 0`
    apex of `0.9330` and did not clear at all** — which is where its anomalous vapour
    branch came from (`ΔH_vap = 14.39` against 4.6–6.1 for every other construction).
    Kept because the fold apex is still the right object for "is the isotherm cut".
    """
    f = d_model.get("fold", [])
    if not f:
        return None, None
    ap = max(p["T"] for p in f)
    return ap / kmax, ap


def _hybrid_f_per_volume(z, kv, tstar, etas, curve, ap_int,
                         insets=(0.04, 0.02), gap="hug", cross_ap_int=None):
    r"""`(ρ, βA/V, nsplit)` on both branches at `T*` — the compressibility hybrids.

    Two crossings are available, and they share **everything else**: the same
    `A = 0` endpoint densities, the same two `ρ` quadratures, the same
    `βΔμ` handed across.  `gap` selects only how the `A < 0` interval is bridged, so
    the distance between the two resulting binodals is *purely* path-dependence.

      * `"hug"`    — the arc follows the `A = 0` spinodal up to its apex and back
                     down, extrapolated to zero inset (the default, described below);
      * `"detour"` — a rectangular detour **over the critical point**
                     (`msa_exact_mix.critical_detour_gap_free_energy`): straight up at
                     `ρ_v`, across above the fold dome, straight down at `ρ_l`.  It
                     touches no special locus, needs no inset and no extrapolation,
                     but its two vertical legs are pure *energy* route and long.

    ⭐ The arc hugs the **`A = 0` spinodal**, not the fold, and that is the whole
    reason this works.  A fold is by definition `det J = 0`, so an arc following it
    is singular in the limit: `cond(J)` measured 9.5 → 25 (HSY) and 192 → 469 (2YK)
    as the inset shrinks, i.e. **diverging**, and the answer never converges — for
    2YK it drifted by a factor 4 between insets 0.02 and 0.05 and was unusable.
    The spinodal is a **regular** point of the same system: `cond(J)` **saturates**
    (HSY → 11.7, 2YK → 156), and `Δ` comes out **linear in the inset**
    (`R² = 0.9998 … 0.99999` on four state points), so one Richardson step gives the
    `inset → 0` value.

    Two further benefits, both structural: the `ρ` legs stay entirely on the
    physical `A > 0` winding (the fold route's legs had to cross into `A < 0`), and
    because `A = 0` is regular the legs can be integrated right up to the spinodal
    densities themselves rather than stopping short of them.
    """
    import msa_exact_mix as X
    K = np.array(kv, float)[:, None, None] / tstar
    zz = np.array(z, float)
    state = lambda r: X.HCYMix([1.0], [r], zz, K)
    kmax = float(np.max(np.array(kv, float)))
    t_int = tstar / kmax
    ts, cv, cl = curve
    if not (ts[0] <= t_int <= ts[-1]):
        return None
    rv = float(np.interp(t_int, ts, cv))
    rl = float(np.interp(t_int, ts, cl))
    if not (rl > rv > 0):
        return None

    rho_all = np.array([rho_of(float(e)) for e in etas])
    # ⚠ The vapour branch must be tabulated FAR below the `eta` grid.  Coexistence
    # is the crossing of `mu_gas(p)` and `mu_liq(p)`, so it can only be found inside
    # the tabulated `p` window — and the window's lower end is the *gas* branch's
    # lowest pressure.  With the grid's own floor (eta = 0.006, p = 1.1e-02) the
    # crossing runs off the bottom below `T* ~ 0.73`: measured `d(p_lo)` flips from
    # −0.26 at 0.7484 to +0.06 at 0.7205, i.e. the root moves just outside, not away.
    # That reads as "no coexistence" and is nothing but truncation of the vapour.
    # A geometric ladder down to 1e-6 costs a few transport steps (the gas is nearly
    # ideal there) and pushes the floor to p ~ 1e-06.
    gas_lo = 1e-6
    gas = np.unique(np.concatenate([
        np.geomspace(gas_lo, rv, 26), rho_all[(rho_all > gas_lo) & (rho_all <= rv)]]))
    liq = rho_all[rho_all >= rl]
    if len(gas) < 5 or len(liq) < 5:
        return None

    def sweep(grid, S, m):
        """Tabulate `(βΔA/V, βΔμ)` on `grid`.  ⭐ `m` is carried, not reconstructed."""
        oS, oM, prev = [], [], grid[0]
        for r in grid:
            if r > prev:
                path = np.stack([np.linspace(prev, r, 12), np.full(12, tstar)], axis=1)
                S, m = X._transport(state(prev), path, S0=S, m0=m)
                if S is None:
                    return None, None, None, None
            oS.append(S)
            oM.append(m)
            prev = r
        return np.array(oS), np.array(oM), S, m

    # Geometric pre-leg: the reference point is `rho -> 0`, where the excess is zero,
    # and a linear leg from 1e-4 cannot even reach the new floor.
    p0 = np.stack([np.geomspace(1e-8, gas[0], 60), np.full(60, tstar)], axis=1)
    S, m = X._transport(state(1e-8), p0)
    if S is None:
        return None
    fg, mug, S, m = sweep(gas, S, m)
    if fg is None:
        return None
    p1 = np.stack([np.linspace(gas[-1], rv, 20), np.full(20, tstar)], axis=1)
    S, m = X._transport(state(gas[-1]), p1, S0=S, m0=m)
    if S is None:
        return None

    # ⭐ the one hybrid step — the two routes differ HERE and nowhere else
    if gap == "detour":
        if cross_ap_int is None:
            return None
        # ⚠⚠ The `A = 0` apex, NOT the fold apex.  Clearing the fold only guarantees the
        # crossing leg finds a SOLUTION; it does not guarantee the solution is physical.
        # Between the two apexes the isotherm is whole but the whole window is
        # wrong-winding, so a leg that clears the fold and not the `A = 0` dome walks
        # through states with `h(r) ~ e^{+s*r}`.
        dS, ml = X.critical_detour_gap_free_energy(state(rv), m, rv, rl, cross_ap_int)
        if dS is None:
            return None
    else:
        # Richardson-extrapolated to inset → 0 (Δ is linear in it on a regular locus)
        vals = []
        for ins in insets:
            dS, ml = X.fold_gap_free_energy(state(rho_of(0.2)), m, n_arc=90, inset=float(ins),
                                            t_apex=ap_int, curve=curve, t_hug=0.99)
            if dS is None:
                return None
            vals.append((float(ins), dS, ml))
        (i1, d1, m1), (i2, d2, m2) = vals        # i2 < i1, both small
        ex = lambda a, b: b + (b - a) * (0.0 - i2) / (i2 - i1)  # linear → inset 0
        dS, ml = ex(d1, d2), ex(m1, m2)
        # ⚠⚠ CONVERGENCE GATE, and it bites exactly where the arc is shortest.
        # The hug arc runs from `T` up to `t_hug·T_apex`, so approaching the apex its
        # length goes to zero while the inset does not — the extrapolation baseline
        # collapses.  Measured spread between insets 0.04 and 0.02 (HSY, eta = 0.2):
        # 8.1 % at T* = 0.85, 10.1 % at 0.9534, 13.6 % at 0.9982, 15.8 % at 1.0094,
        # monotonically worse as the span shrinks 0.173 -> 0.013.  Left ungated this
        # showed up as a NON-MONOTONE green branch over T* in [0.97, 1.01] — an
        # inset artefact the size of the whole hug-vs-detour signal.  Require the
        # Richardson step to be a *correction*, not the leading term.
        # ⚠ Test `Δ₀`, NOT `Δ`.  `Δ = μ_v·(ρ_l − ρ_v) + Δ₀` is affine, and the first
        # term is large (`μ_v ~ −4`) and inset-converged, so it dilutes the ratio:
        # gating on `Δ` passed `T* = 1.0094` at 3 % where `Δ₀` itself moves 23 %.
        # ⚠⚠ And the slope is the INSET-WIDENED gap `ρ_l(1+i) − ρ_v(1−i)`, not
        # `ρ_l − ρ_v`: those are the arc's actual endpoints.  Verified directly —
        # `dΔ/dμ_v` = 0.262480 / 0.249632 at insets 0.04 / 0.02, against
        # `ρ_l(1+i) − ρ_v(1−i)` = 0.261894 / 0.249047 and a fixed `ρ_l − ρ_v` =
        # 0.236199.  Using the fixed width leaves `m·i·(ρ_l+ρ_v)` behind, which is
        # itself inset-dependent and of the same size as the real spread — it
        # roughly doubled the measured movement (22 % where the truth is 13 %) and
        # rejected the hug over most of its range.
        w1 = rl * (1.0 + i1) - rv * (1.0 - i1)
        w2 = rl * (1.0 + i2) - rv * (1.0 - i2)
        d0_1, d0_2 = d1 - m * w1, d2 - m * w2
        d0 = ex(d0_1, d0_2)
        if abs(d0) < 1e-12 or abs(d0 - d0_2) / abs(d0) > 0.15:
            print(f"    hug  T*={tstar:.4f}  Richardson moves Delta_0 by "
                  f"{100*abs(d0 - d0_2)/max(abs(d0),1e-12):.0f}% — arc too short "
                  "near the apex, not converged", flush=True)
            return None

    p2 = np.stack([np.linspace(rl, liq[0], 20), np.full(20, tstar)], axis=1)
    S, m = X._transport(state(rl), p2, S0=S + dS, m0=ml)
    if S is None:
        return None
    fl_, mul, _, _ = sweep(liq, S, m)
    if fl_ is None:
        return None
    rho = np.concatenate([gas, liq])
    dA = np.concatenate([fg, fl_])
    dmu = np.concatenate([mug, mul])
    eta = rho * np.pi / 6.0
    return (rho, rho * (np.log(rho) - 1.0 + bf_hs_ex(eta)) + dA,
            np.log(rho) + bmu_hs_ex(eta) + dmu, len(gas))


def fold_tangent(rho, fv, mu, nsplit):
    r"""Coexistence between two free-energy branches, by the `μ`–`p` crossing.

    ⚠ A convex hull over the `η` grid is not good enough: its endpoints *are* grid
    points, so the tie line jitters by the grid spacing and the liquid branch comes
    out non-monotone — which is exactly how the first version of this line looked.

    ⚠⚠ **And neither is splining `βA/V` and differentiating it**, which was the
    second version.  Once the vapour branch is tabulated on a *geometric* ladder
    down to `ρ = 1e-6`, a cubic spline over that spacing puts wiggles into
    `μ = d(βA/V)/dρ` — measured: `p = ρμ − βA/V` came out **non-monotone on the gas
    branch**, though `dp/dρ = a ≥ 0` forbids it.  Near the apex the `p` window is
    only `1.3e-03` wide and the wiggles are the same size, so the crossing was lost
    at `T* = 1.0228, 1.0180` for a purely numerical reason (`d` at both ends
    `−0.0009` and `−0.0028`: same sign, both ≈ 0).

    ⭐ The repair is not a better spline: `_transport` **already carries** `βΔμ`
    alongside `βΔA/V` — it is the second component of the 1-form pair, integrated on
    the same path.  So `μ` is *transported*, never reconstructed, and the hard-sphere
    part is `bmu_hs_ex` in closed form.  Then

        p = ρμ − βA/V

    is monotone on each branch by construction, and coexistence is the single
    crossing of `μ_gas(p)` and `μ_liq(p)` — one scalar bisection.
    """
    out = []
    for lo, hi in ((0, nsplit), (nsplit, len(rho))):
        r, f, m = rho[lo:hi], fv[lo:hi], mu[lo:hi]
        if len(r) < 5:
            return None
        out.append((r, m, r * m - f))
    (rg, mg, pg), (rl, ml, pl) = out
    lo, hi = max(pg.min(), pl.min()), min(pg.max(), pl.max())
    if not (hi > lo):
        return None
    d = lambda q: np.interp(q, pg, mg) - np.interp(q, pl, ml)
    if d(lo) * d(hi) > 0:
        return None
    q = brentq(d, lo, hi, xtol=1e-14)
    return float(np.interp(q, pg, rg)), float(np.interp(q, pl, rl))


def fold_binodal(model: dict, d_model: dict, n_T: int = 22, t_lo: float = 0.55,
                 gap: str = "hug"):
    """Coexistence from a compressibility hybrid free energy — `gap` picks the crossing.

    Name kept for the JSON key; the `"hug"` arc follows the `A = 0` spinodal rather
    than the fold — see `_hybrid_f_per_volume` for why (conditioning, and 2YK) — and
    `"detour"` is the fold-apex rectangle.
    """
    kmax = float(np.max(np.array(model["kv"], float)))
    curve, ap_int, apS = spinodal_curve_of(d_model, kmax, t_lo=min(0.50, t_lo))
    if gap == "detour" and ap_int is None:
        return [], []
    etas = np.array(sorted({q["eta"] for q in d_model["spinodal"]}
                           | {q["eta"] for q in d_model["fold"]}))
    u = np.linspace(0.0, 1.0, n_T) ** 1.5
    Ts = apS * (0.99 - (0.99 - t_lo) * u)
    tag = "critical-point detour" if gap == "detour" else "spinodal-hug"
    out, raw = [], []
    for T in Ts:
        r = _hybrid_f_per_volume(model["z"], model["kv"], float(T), etas, curve, ap_int,
                                 gap=gap, cross_ap_int=ap_int)
        if r is None:
            continue
        rho, fv, mu, ns = r
        raw.append(dict(T=float(T), rho=list(map(float, rho)),
                        f=list(map(float, fv)), mu=list(map(float, mu)),
                        nsplit=int(ns)))
        ct = fold_tangent(rho, fv, mu, ns)
        if ct is None:
            print(f"    {tag}  T*={T:.4f}  curve ok, no tie line", flush=True)
            continue
        rg, rl = ct
        # ⚠ The liquid branch is tabulated only to the eta grid's ceiling (0.486,
        # i.e. essentially hard-sphere freezing).  A root that lands on the last
        # point is the ceiling, not a tangency — report it as truncation.
        if rl >= 0.995 * rho[-1]:
            print(f"    {tag}  T*={T:.4f}  liquid root at the grid ceiling "
                  f"(eta {rl*np.pi/6:.4f}) — truncated, dropped", flush=True)
            continue
        if rl - rg > 1e-3:
            out.append(dict(T=float(T), eta_g=rg * np.pi / 6, eta_l=rl * np.pi / 6))
            print(f"    {tag} coexistence  T*={T:.4f}  eta = {rg*np.pi/6:.4f}"
                  f" .. {rl*np.pi/6:.4f}", flush=True)
    return out, raw


def _energy_isotherm(z, kv, tstar, rho_hi=None, n=110):
    r"""`[(ρ, βp, βμ), …]` — the energy-route isotherm in **closed form**, by segment.

    ⭐ `βp` is Blum & Høye (43)+(44) and `βμ` is their (46); both are algebraic in the
    converged solution, so nothing here is integrated and nothing is differentiated.
    The hard-sphere reference is the Percus–Yevick *compressibility* one,
    `βp_HS = ρ(1+η+η²)/(1−η)³` and `βμ^ex_HS = bmu_hs_ex`, which is the reference the
    excesses are taken over (`--gate` §8 pins the pair to the λ-charging integral).

    Two segments are returned when the fold cuts the isotherm, one when it does not;
    each is walked with the solution **continued**, never re-ramped.
    """
    import msa_exact_mix as X
    rho_hi = rho_hi or 6.0 * 0.48 / np.pi
    zz, kk = np.array(z, float), np.array(kv, float)[:, None, None] / float(tstar)
    mk = lambda r: X.HCYMix([1.0], [float(r)], zz, kk)

    def walk(grid):
        q = X.solve_bh_mix(mk(grid[0]))
        if q is None:
            return None
        u = np.concatenate([q.Dt.ravel(), q.Gt.ravel()])
        out = []
        for r in grid:
            st2 = mk(r)
            v = X._solve_seeded(st2, u, *X._scales(st2))
            if v is None:
                break                                   # the fold
            u = v
            nn = st2.N * st2.N
            sol = X.SolutionMix(st2, u[:st2.T*nn].reshape(st2.T, st2.N, st2.N),
                                u[st2.T*nn:].reshape(st2.T, st2.N, st2.N),
                                X._pieces(st2, u[:st2.T*nn].reshape(st2.T, st2.N, st2.N),
                                          u[st2.T*nn:].reshape(st2.T, st2.N, st2.N)), 0.0)
            e = float(r) * np.pi / 6.0
            out.append((float(r),
                        float(r) * (1 + e + e**2) / (1 - e)**3 + X.pressure_excess(sol),
                        float(np.log(r) + bmu_hs_ex(e)
                              + float(X.chem_pot_excess(sol)[0]))))
        return np.array(out) if len(out) >= 8 else None

    up = np.concatenate([np.geomspace(1e-5, 0.02, 18), np.linspace(0.025, rho_hi, n)])
    a = walk(up)
    if a is not None and a[-1, 0] >= 0.999 * rho_hi:
        return [a]                                       # isotherm not cut
    b = walk(up[::-1])
    segs = [x for x in (a, b) if x is not None]
    return [x[np.argsort(x[:, 0])] for x in segs] if segs else []


def energy_binodal(model: dict, t_hi: float, n_T: int = 30, t_lo_frac: float = 0.30):
    r"""Coexistence on the **energy route**, from closed-form `p` and `μ`.

    ⚠⚠ **This replaces a lower-convex-hull construction, and the reason is visible on
    the figure.** The hull's vertices *are* grid points, so the tie-line endpoints
    could only ever move in steps of the `η` grid (0.006) — which is exactly the
    staircase the blue branch showed. It is the same defect that was already fixed
    for the hybrid lines in `fold_tangent`, and the fix is the same: work with `p`
    and `μ` rather than with `βA/V`, and bisect.

    ⭐ Here it is better still, because `p` and `μ` are *closed form* — (43) and (46)
    — so unlike `fold_tangent` there is no transport and no quadrature at all: the
    charging integral that the hull version needed is gone. Coexistence is the single
    crossing of `μ_gas(p)` and `μ_liq(p)`, resolved far below the grid.
    """
    from scipy.interpolate import CubicSpline
    out = []
    for T in np.linspace(t_hi, t_lo_frac * t_hi, n_T):
        segs = _energy_isotherm(model["z"], model["kv"], float(T))
        if not segs:
            continue
        if len(segs) == 1:                       # connected: split at the loop
            r, p, mu = segs[0][:, 0], segs[0][:, 1], segs[0][:, 2]
            d = np.diff(p) / np.diff(r)
            neg = np.where(d < 0)[0]
            if len(neg) == 0:
                continue                          # above this route's T_c
            br = [(r[:neg[0] + 1], p[:neg[0] + 1], mu[:neg[0] + 1]),
                  (r[neg[-1] + 1:], p[neg[-1] + 1:], mu[neg[-1] + 1:])]
        else:                                     # cut by the fold: two branches
            br = [(x[:, 0], x[:, 1], x[:, 2]) for x in segs[:2]]
        # ⚠⚠ Trim each branch to the stretch where `p` is MONOTONE, and this is not
        # tidying.  When the fold cuts the isotherm the two segments run all the way
        # to the fold densities, i.e. **past the `A = 0` spinodal and onto the
        # wrong-winding sheet**, where `p` turns over.  `np.interp` requires an
        # increasing `x` and returns silent nonsense otherwise: it put `η_g` at
        # 0.129 instead of 0.023 for HSY at `T* = 0.9864`, and the corruption began
        # exactly at each model's fold apex (HSY 1.0177, 2YK 0.8874) — which is what
        # identified it.
        def _mono(b, gas):
            r, pp, mm = b
            d = np.diff(pp)
            bad = np.where(d <= 0)[0]
            if len(bad) == 0:
                return b
            k = int(bad[0]) + 1 if gas else int(bad[-1]) + 1
            return (r[:k], pp[:k], mm[:k]) if gas else (r[k:], pp[k:], mm[k:])
        br = sorted(br, key=lambda b: b[0][0])
        br = [_mono(br[0], True), _mono(br[1], False)]
        if min(len(b[0]) for b in br) < 5:
            continue
        (rg, pg, mg), (rl, pl, ml) = br
        lo, hi = max(pg.min(), pl.min()), min(pg.max(), pl.max())
        if not (hi > lo):
            continue
        f = lambda q: np.interp(q, pg, mg) - np.interp(q, pl, ml)
        if f(lo) * f(hi) > 0:
            continue
        q = brentq(f, lo, hi, xtol=1e-14)
        rv, rr = float(np.interp(q, pg, rg)), float(np.interp(q, pl, rl))
        if rr - rv < 1e-3:
            continue
        out.append(dict(T=float(T), eta_g=rv * np.pi / 6, eta_l=rr * np.pi / 6))
        print(f"    energy coexistence  T*={T:.4f}  eta = {rv*np.pi/6:.5f}"
              f" .. {rr*np.pi/6:.5f}", flush=True)
    return out


def virial_critical(model: dict, t_lo: float, t_hi: float):
    """`(T*_c, η_c)` on the **virial** route — where `min_η ∂βp_virial/∂ρ` reaches 0.

    ⚠ Needed separately, and not just for tidiness: the coexistence sweep loses the
    tie line well before `T_c` (the loop becomes too shallow to bracket — measured,
    it fails at `T/T_c = 0.977`), so the top of the drawn line is **not** the
    critical point.  Reading it as one overstates nothing about the physics but
    misstates the route gap by ~5 %.
    """
    import msa_exact_mix as X
    z, kv = np.array(model["z"], float), np.array(model["kv"], float)

    def dpv(eta, T):
        st = X.HCYMix([1.0], [rho_of(eta)], z, kv[:, None, None] / float(T))
        q = X.solve_bh_mix(st)
        return np.nan if q is None else X.dbetaP_drho_virial(q)

    es = np.linspace(0.10, 0.24, 15)

    def mn(T):
        v = np.array([dpv(float(e), T) for e in es])
        ok = np.isfinite(v)
        if ok.sum() < 4:
            return np.nan, np.nan
        e2, v2 = es[ok], v[ok]
        i = int(np.argmin(v2))
        if 0 < i < len(v2) - 1:      # parabola through the three lowest, not the node
            x0, x1, x2 = e2[i - 1], e2[i], e2[i + 1]
            y0, y1, y2 = v2[i - 1], v2[i], v2[i + 1]
            dd = y0 - 2 * y1 + y2
            ec = x1 - 0.5 * (x2 - x0) * (y2 - y0) / (2 * dd) if abs(dd) > 1e-14 else x1
        else:
            ec = e2[i]
        return float(v2[i]), float(ec)

    try:
        Tc = brentq(lambda T: mn(T)[0], t_lo, t_hi, xtol=1e-4)
    except Exception:
        return None, None
    return float(Tc), mn(Tc)[1]


def virial_spinodal(model: dict, Tc: float, etac: float, n_T: int = 14,
                    t_lo_frac: float = 0.62):
    r"""`∂βp_virial/∂ρ = 0` — the **virial route's own spinodal**, both branches.

    ⭐ This is where the virial critical point actually comes from.  The binodal
    cannot deliver it: near `T_c` the loop depth falls off like `(T_c−T)²`, the two
    coexisting roots merge, and the `μ`–`p` crossing goes degenerate — measured, the
    coexistence sweep stops at `0.971·T_c` on **both** models.  That is generic to
    solving coexistence, not a defect of this route.  The spinodal is a *regular*
    root of a smooth function, so bisection walks right up to the apex.

    Same lesson as the compressibility route, where the critical point is read off
    `A = 0` and never from a coexistence construction.

    ⚠ Walks down in `T` reusing the previous roots as brackets — a fresh `η` scan at
    every temperature costs three solves per point and there is no need for it, the
    dome being smooth.  Falls back to a widened bracket when a step overshoots.
    """
    import msa_exact_mix as X
    z, kv = np.array(model["z"], float), np.array(model["kv"], float)

    def dpv(eta, T):
        st = X.HCYMix([1.0], [rho_of(float(eta))], z, kv[:, None, None] / float(T))
        q = X.solve_bh_mix(st)
        return np.nan if q is None else X.dbetaP_drho_virial(q)

    def dpv_safe(eta, T):
        """`dpv` with the fold gap mapped to a negative sentinel.

        ⚠ `brentq` refuses a NaN outright (`ValueError: function value is NaN`), and
        a NaN here means only one thing — no real solution, i.e. the fold gap, which
        lies **inside** the virial spinodal dome at every `T` tested.  So the sign
        there is known even though the value is not, and the sentinel carries it.
        Returning NaN instead cost the 2YK branch after four temperatures.
        """
        v = dpv(eta, T)
        return -1e3 if not np.isfinite(v) else v

    def root(T, lo, hi):
        """A sign change of `dpv` in `[lo, hi]`, widening once if the bracket missed."""
        for span in (1.0, 2.0, 4.0):
            a = max(0.004, etac - span * (etac - lo))
            b = min(0.47, etac + span * (hi - etac))
            xs = np.linspace(a, b, 9)
            vs = np.array([dpv(x, T) for x in xs])
            ok = np.isfinite(vs)
            if ok.sum() < 3:
                continue
            xs, vs = xs[ok], vs[ok]
            sc = np.where(np.sign(vs[:-1]) != np.sign(vs[1:]))[0]
            if len(sc) >= 1:
                if len(sc) < 2:
                    continue
                try:
                    return [float(brentq(lambda x: dpv_safe(x, T),
                                         xs[i], xs[i + 1], xtol=1e-5))
                            for i in (sc[0], sc[-1])]
                except Exception:
                    continue
        return None

    out = []
    lo, hi = etac * 0.85, etac * 1.15
    for T in np.linspace(0.995 * Tc, t_lo_frac * Tc, n_T):
        try:
            r = root(T, lo, hi)
        except Exception:
            r = None
        if r is None or r[1] - r[0] < 1e-4:
            continue
        lo, hi = r[0], r[1]
        out.append(dict(T=float(T), eta_g=r[0], eta_l=r[1]))
        print(f"    virial spinodal  T*={T:.4f}  eta = {r[0]:.4f} .. {r[1]:.4f}",
              flush=True)
    return out


def virial_binodal(model: dict, d_model: dict, n_T: int = 24):
    r"""Coexistence from the **virial** route — ordinary Maxwell, no crossing step.

    ⭐ The one route with *both* a van der Waals loop *and* an uncut isotherm over
    most of its dome.  The compressibility route has neither: `∂βp/∂ρ = (A/2π)²` is
    a perfect square, so `βA(ρ)` is convex and there is nothing to construct on even
    where the isotherm is whole.  `msa_exact_mix.binodal_virial` does the work.

    ⚠⚠ **Corrects an earlier version of this line.** It used to stop below the
    critical point and the gap was attributed to the loop being "too shallow to
    bracket".  That was `msa_exact_mix.binodal_virial`'s 2-D `fsolve` seed, not the
    physics: with the bracketed construction the binodal tracks to **`0.999·T_c` on
    both models**, so the dome is now closed at the critical point and there is no
    untracked cap.  Refining the `ρ` grid alone changed nothing, which is what
    identified the seed as the cause.

    ⚠ The **cold** end is still a real stop: below the fold apex the isotherm is
    genuinely cut and no construction can bridge it.  That end is left open.
    """
    import msa_exact_mix as X
    z, kv = np.array(model["z"], float), np.array(model["kv"], float)
    b = d_model.get("binodal", [])
    tc_e = max((p["T"] for p in b), default=None)
    if tc_e is None:
        return [], None, None
    Tc, ec = virial_critical(model, 0.85 * tc_e, 1.45 * tc_e)
    print(f"    virial critical point: T*_c = {Tc}  eta_c = {ec}", flush=True)
    hi = (0.999 * Tc) if Tc else 1.20 * tc_e
    lo = (0.66 * Tc) if Tc else 0.55 * tc_e
    out = []
    for T in np.linspace(hi, lo, n_T):
        st = X.HCYMix([1.0], [rho_of(0.2)], z, kv[:, None, None] / float(T))
        try:
            r = X.binodal_virial(st, n=200)   # dense enough for the near-Tc loop
        except Exception:
            r = None
        if r is None or r[1] - r[0] < 1e-3:
            continue
        rv, rl, p_co = r
        out.append(dict(T=float(T), eta_g=rv * np.pi / 6, eta_l=rl * np.pi / 6,
                        p=float(p_co)))
        print(f"    virial coexistence  T*={T:.4f}  eta = {rv*np.pi/6:.4f}"
              f" .. {rl*np.pi/6:.4f}   betaP = {p_co:.5f}", flush=True)
    if out and Tc:                      # close the dome AT the critical point
        out.append(dict(T=float(Tc), eta_g=float(ec), eta_l=float(ec), p=float("nan")))
    return out, Tc, ec


# ── driver ───────────────────────────────────────────────────────────────────────────────────────

def compute():
    out = {}
    etas = np.round(np.concatenate([np.arange(0.003, 0.03, 0.003),
                                    np.arange(0.03, 0.492, 0.006)]), 4)
    for key, model in MODELS.items():
        print(f"[{key}] tracing {len(etas)} isochores …", flush=True)
        rows = collect(model, etas)
        spin = [dict(eta=e, T=1.0 / rows[e]["lam_sp"]) for e in sorted(rows) if rows[e]["lam_sp"]]
        fold = [dict(eta=e, T=1.0 / rows[e]["lam_fold"])
                for e in sorted(rows) if rows[e].get("lam_fold")]
        print(f"[{key}] energy spinodal …", flush=True)
        _, spin_e = binodal(rows, etas)
        print(f"[{key}] energy-route coexistence (closed form) …", flush=True)
        bino = energy_binodal(model, 1.30 * max(q["T"] for q in spin))
        fold_ap = max((1.0 / rows[e]["lam_fold"] for e in rows if rows[e].get("lam_fold")),
                      default=None)
        bino_f, raw_f = [], []
        bino_d, raw_d = [], []
        if fold_ap:
            dm = dict(spinodal=spin, fold=fold)
            print(f"[{key}] compressibility-spinodal hybrid coexistence …", flush=True)
            bino_f, raw_f = fold_binodal(model, dm, gap="hug")
            print(f"[{key}] compressibility-critical-point detour hybrid coexistence …", flush=True)
            bino_d, raw_d = fold_binodal(model, dm, gap="detour")
        print(f"[{key}] virial-route coexistence …", flush=True)
        bino_v, tcv, ecv = virial_binodal(model, dict(binodal=bino))
        spin_v = virial_spinodal(model, tcv, ecv) if tcv else []
        Tc_sp = max((s["T"] for s in spin), default=None)
        Tc_bi = max((b["T"] for b in bino), default=None)
        Tc_se = max((s["T"] for s in spin_e), default=None)
        print(f"[{key}] Tc(spinodal a=0)~{Tc_sp}  Tc(spinodal a_e=0)~{Tc_se}  "
              f"Tc(binodal)~{Tc_bi}   n_bino={len(bino)}", flush=True)
        out[key] = dict(label=model["label"], spinodal=spin, spinodal_energy=spin_e,
                        fold=fold, binodal=bino, binodal_fold=bino_f,
                        fold_curves=raw_f, binodal_apex=bino_d, apex_curves=raw_d,
                        binodal_virial=bino_v, virial_Tc=tcv, virial_etac=ecv,
                        spinodal_virial=spin_v)
    with open(JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[saved] {JSON}")
    return out


def _mono_branch(pts, field, decreasing):
    """Sort a spinodal branch by descending T* and keep it monotone in `field` (dome envelope)."""
    out, prev = [], None
    for p in sorted(pts, key=lambda x: -x["T"]):
        v = p[field]
        if prev is None or (v <= prev + 1e-3 if decreasing else v >= prev - 1e-3):
            out.append((p["T"], v)); prev = v
    return out


# ⭐ Figure convention, applied uniformly below:
#   colour = ROUTE      energy #2a78d6 · compressibility #e34948/#e67e22 · virial #6d4c41
#   style  = OBJECT     binodal/coexistence SOLID · spinodal DASHED · untracked DOTTED
# The two compressibility detour constructions are deliberately near-identical hues:
# they are one route computed two ways, and the figure should say so before the
# legend does.  `fold` is grey because it is not a route at all — it is where the
# solution ceases to exist, and it is common to all three.
C_ENERGY, C_COMPR, C_HUG, C_DETOUR, C_VIRIAL = ("#2a78d6", "#e34948",
                                                "#e67e22", "#f5b041", "#6d4c41")


def render(data=None):
    """Draw the eight lines.

    ⚠ Layout note: every label placement below is deliberate.  The six curves fill
    the middle of each panel, so the legend is lifted **out of the axes** to a
    figure-level block, and the annotations are pinned to the free wedges —
    upper-right above the binodal, and the low-`T` strip under the fold line.
    Anything placed "loc=best" inside the axes lands on a curve or on the legend.

    ⚠ The two hybrids get **separate** dotted terminating tie lines: they stop at
    different temperatures (HSY 1.0228/0.7484 vs 1.0064/0.6860), so one pair cannot
    stand for both.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if data is None:
        data = json.load(open(JSON))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.2), sharey=True)
    ymax = 1.0
    for key in ("HSY", "2YK"):
        # ⚠ The virial dome tops ~15 % above the energy one, so it must be in the
        # y-range calculation or it is silently clipped by the axes.
        for k2 in ("binodal", "binodal_virial"):
            b = data[key].get(k2, [])
            if b:
                ymax = max(ymax, max(p["T"] for p in b))
    handles = {}
    for ax, key in zip(axes, ("HSY", "2YK")):
        d = data[key]

        def keep(line, label):
            handles.setdefault(label, line)

        b = sorted(d["binodal"], key=lambda x: x["T"])
        Tc_bi = eta_c_bi = None
        if b:
            eg = [p["eta_g"] for p in b]; el = [p["eta_l"] for p in b]
            T = [p["T"] for p in b]
            ln, = ax.plot(eg + el[::-1], T + T[::-1], "-", color=C_ENERGY, lw=2.2)
            keep(ln, "binodal (energy route)")
            top = max(b, key=lambda p: p["T"])
            Tc_bi = top["T"]; eta_c_bi = 0.5 * (top["eta_g"] + top["eta_l"])
            ax.plot([eta_c_bi], [Tc_bi], "o", color=C_ENERGY, ms=7, zorder=5)

        bv = sorted(d.get("binodal_virial", []), key=lambda x: x["T"])
        Tc_v = d.get("virial_Tc")
        if bv:
            # ⚠ Drawn OPEN, not as a closed dome.  The sweep loses the tie line
            # before T_c (the loop gets too shallow to bracket), so joining the two
            # branches at the top point would assert a critical point that is not
            # there — which is exactly what the first version of this line did.
            # The real one is located separately by `virial_critical` and marked.
            T = [p["T"] for p in bv]
            eg = [p["eta_g"] for p in bv]
            el = [p["eta_l"] for p in bv]
            ln, = ax.plot(eg + el[::-1], T + T[::-1], "-", color=C_VIRIAL, lw=2.2)
            keep(ln, "binodal (virial route)")
            sv = sorted(d.get("spinodal_virial", []), key=lambda x: x["T"])
            if sv:
                # ⭐ The route's own spinodal — and the open circle sits on ITS apex,
                # which is the point of drawing it: the critical point comes from
                # here, never from the coexistence sweep.
                Tv = [p["T"] for p in sv]
                ln2, = ax.plot([p["eta_g"] for p in sv], Tv, "--",
                               color=C_VIRIAL, lw=1.6)
                keep(ln2, "spinodal  ($\\partial\\beta p_{\\mathrm{vir}}"
                          "/\\partial\\rho=0$, virial route)")
                ax.plot([p["eta_l"] for p in sv], Tv, "--", color=C_VIRIAL, lw=1.6)
            if Tc_v and d.get("virial_etac"):
                ax.plot([d["virial_etac"]], [Tc_v], "o", color=C_VIRIAL, ms=7,
                        zorder=6)

        bf = sorted(d.get("binodal_fold", []), key=lambda x: x["T"])
        if bf:
            # ⚠ NOT "the compressibility route's binodal" — that route has none:
            # `a = (A/2π)²` is a perfect square, so `βA(ρ)` is convex on every
            # isotherm.  All of this line's coexistence comes from the OFFSET the
            # hybrid carries across the `A<0` interval, so its ends are set by gates
            # (p-ranges overlapping, then a μ-crossing existing) rather than by a
            # critical point — see msa_exact_phase_lines.md.
            T = [p["T"] for p in bf]
            ln, = ax.plot([p["eta_g"] for p in bf], T, "-", color=C_HUG, lw=2.2)
            keep(ln, "coexistence (compressibility–spinodal hybrid)")
            ax.plot([p["eta_l"] for p in bf], T, "-", color=C_HUG, lw=2.2)
            # ⚠ Only the UPPER terminator is drawn.  The lower one used to mark
            # "crossing lost, then p ranges stop overlapping" — but that limit was
            # the *vapour branch's tabulation floor* (eta = 0.006), not physics:
            # extending the gas ladder to rho = 1e-6 moved it from 0.686 to 0.568
            # and the remaining stop is just where the T scan ends.  A horizontal
            # line drawn there would assert a boundary that is not there.
            top = max(bf, key=lambda p: p["T"])
            ax.plot([top["eta_g"], top["eta_l"]], [top["T"], top["T"]], ":",
                    color=C_HUG, lw=1.6)
            # ⚠ xytext in AXES fractions.  In data coordinates it was pinned to a
            # linear eta axis and jumped onto another curve the moment the scale
            # changed to asinh.
        ba = sorted(d.get("binodal_apex", []), key=lambda x: x["T"])
        if ba:
            # Same endpoints, same rho legs, same mu handed across as the teal line —
            # only the CROSSING differs, so the spacing between them is nothing but
            # MSA's own path-dependence.
            T = [p["T"] for p in ba]
            ln, = ax.plot([p["eta_g"] for p in ba], T, "-", color=C_DETOUR, lw=2.2)
            keep(ln, "coexistence (compressibility–critical-point detour hybrid)")
            ax.plot([p["eta_l"] for p in ba], T, "-", color=C_DETOUR, lw=2.2)
            # its own open ends: the two hybrids stop at DIFFERENT temperatures, so
            # one pair of terminating tie lines cannot stand for both.
            q = max(ba, key=lambda p: p["T"])
            ax.plot([q["eta_g"], q["eta_l"]], [q["T"], q["T"]], ":",
                    color=C_DETOUR, lw=1.6)
        if not (bf or ba):
            ax.text(0.985, 0.03, "⚠ hybrid coexistence not resolved in the scanned "
                    "window", transform=ax.transAxes, fontsize=7.0,
                    color="#c0392b", ha="right", va="bottom")
        # ⚠ No method note on the plot.  What separates the two hybrids (path
        # dependence), and what sets each one's ends, are several sentences each —
        # they belong in msa_exact_phase_lines.md, not in a grey block competing
        # with the curves.  Only the Tc gap stays, because it is one number.

        se = d.get("spinodal_energy", [])
        if se:
            gas = _mono_branch(se, "eta_g", decreasing=True)
            liq = _mono_branch(se, "eta_l", decreasing=False)
            ln, = ax.plot([e for _, e in gas], [t for t, _ in gas], "--",
                          color=C_ENERGY, lw=1.6)
            keep(ln, "spinodal  ($a_{\\mathrm{energy}}=0$, energy route)")
            ax.plot([e for _, e in liq], [t for t, _ in liq], "--",
                    color=C_ENERGY, lw=1.6)

        sp = sorted(d["spinodal"], key=lambda x: x["eta"])
        ln, = ax.plot([p["eta"] for p in sp], [p["T"] for p in sp], "--",
                      color=C_COMPR, lw=1.8)
        keep(ln, "spinodal  ($A=0$, compressibility)")
        Tc_sp = max(sp, key=lambda p: p["T"])
        ax.plot([Tc_sp["eta"]], [Tc_sp["T"]], "o", color=C_COMPR, ms=7, zorder=5)

        f = sorted(d["fold"], key=lambda x: x["eta"])
        # ⚠ dash-DOT, not dash: the legend now encodes the object in the line style,
        # and `--` is already spoken for by "spinodal".  Two different objects
        # sharing a style is exactly what the split legend exists to prevent.
        ax.plot([p["eta"] for p in f], [p["T"] for p in f], "-.",
                color="#7a7a7a", lw=1.6)

        if Tc_bi:
            gap = 100.0 * (Tc_bi - Tc_sp["T"]) / Tc_sp["T"]
            txt = (f"$T_c$: {Tc_sp['T']:.3f} compr. / {Tc_bi:.3f} energy"
                   + (f" / {Tc_v:.3f} virial" if Tc_v else "")
                   + f"\nroute spread "
                   + (f"{100*(Tc_v/Tc_sp['T']-1):+.0f}%" if Tc_v
                      else f"{gap:+.0f}%"))
            # upper-left: the only corner both panels leave empty once the vapour
            # branches are expanded onto the asinh axis
            ax.text(0.02, 0.975, txt, transform=ax.transAxes,
                    fontsize=7.6, color="#333333", ha="left", va="top")
        ax.set_title(d["label"], fontsize=11)
        # ⚠ `asinh`, not linear.  The vapour branches now reach eta ~ 1e-4 (the gas
        # ladder runs to rho = 1e-6), and on a linear 0–0.5 axis everything below
        # eta ~ 0.01 collapses into the single pixel column at the axis — the whole
        # vapour side of all four coexistence lines becomes invisible below T_c.
        # `asinh` is linear near 0 (so eta = 0 stays representable, which a log axis
        # cannot do) and logarithmic above `linear_width`.  ⚠ The cost is real and is
        # stated on the label: above 0.02 the axis is logarithmic too, so the dome's
        # liquid flank is compressed and the figure must NOT be read as linear in
        # eta.  `symlog` does the same job with a kink at the threshold; `asinh` is
        # the smooth version.
        # ⚠⚠ A custom LOG-BELOW / LINEAR-ABOVE scale, not `asinh`.
        # `asinh(linear_width=w)` is linear below `w` and logarithmic *everywhere
        # above* it, so the two requirements fight: `w = 0.02` buried the coldest
        # vapour points (eta ~ 2.5e-05) at the origin, and `w = 0.002` fixed that
        # but left the liquid flank 0.2–0.5 with only ~17 % of the axis.
        # This scale is logarithmic below `b` and **exactly linear above** it, glued
        # C¹ (slope 1 on both sides at `b`), so the liquid side keeps a linear
        # reading — which is how a phase diagram is meant to be read — while the
        # vapour side gets four decades.  eta = 0 is not representable; the axis
        # starts below the smallest point on any curve.
        _b = 0.03

        def _fwd(x):
            x = np.asarray(x, float)
            with np.errstate(divide="ignore", invalid="ignore"):
                lo = _b * (1.0 + np.log(np.maximum(x, 1e-300) / _b))
            return np.where(x >= _b, x, lo)

        def _inv(f):
            f = np.asarray(f, float)
            with np.errstate(over="ignore"):
                lo = _b * np.exp(np.clip(f / _b - 1.0, -700, 700))
            return np.where(f >= _b, f, lo)

        ax.set_xscale("function", functions=(_fwd, _inv))
        # ⚠ 0.03 is dropped from the ticks though it is the break point: on the
        # log side it sits ~2 mm from 0.01 and the two labels overprinted.  The
        # break is named on the axis label instead.
        ax.set_xticks([1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xticklabels(["$10^{-5}$", "$10^{-4}$", "0.001", "0.01", "0.05",
                            "0.1", "0.2", "0.3", "0.4", "0.5"], fontsize=8)
        ax.set_xlabel(r"packing fraction  $\eta$"
                      "\n(log below 0.03, linear above — glued $C^1$)")
        ax.set_xlim(8e-6, 0.5)
        ax.set_ylim(0.0, 1.34 * ymax / 1.22)
    axes[0].set_ylabel(r"reduced temperature  $T^*$")
    # ⭐ Split legend: style says WHAT the curve is, colour says WHICH ROUTE it came
    # from, and the two compressibility constructions get their own block because
    # they are one route computed two ways.  A flat eight-entry legend forced the
    # reader to decode "solid teal" and "dashed brown" one at a time; this states
    # the vocabulary once.
    from matplotlib.lines import Line2D
    L = lambda **kw: Line2D([0], [0], **kw)
    g = "#555555"
    style = [(L(color=g, lw=2.2, ls="-"), "binodal / coexistence"),
             (L(color=g, lw=1.6, ls="--"), "spinodal"),
             (L(color="#7a7a7a", lw=1.6, ls="-."), "fold  ($\\det J=0$)"),
             (L(color=g, lw=1.6, ls=":"), "hybrid high-$T$ limit (untracked)"),
             (L(color=g, lw=0, marker="o", ms=6.5), "critical point")]
    route = [(L(color=C_ENERGY, lw=2.4), "energy route"),
             (L(color=C_COMPR, lw=2.4), "compressibility route"),
             (L(color=C_VIRIAL, lw=2.4), "virial route")]
    hyb = [(L(color=C_HUG, lw=2.4), "compressibility–spinodal hybrid"),
           (L(color=C_DETOUR, lw=2.4), "compressibility–critical-point detour hybrid")]
    for blk, x, ttl in ((style, 0.185, "line style = object"),
                        (route, 0.535, "colour = route"),
                        (hyb, 0.835, "compressibility, two crossings")):
        lg = fig.legend([h for h, _ in blk], [t for _, t in blk],
                        loc="upper center", bbox_to_anchor=(x, 0.165),
                        fontsize=8.0, frameon=False, title=ttl,
                        title_fontsize=8.4, handlelength=2.6, labelspacing=0.45)
        lg._legend_box.align = "left"
        fig.add_artist(lg)
    fig.suptitle("Exact MSA, single component — three routes: binodals, spinodals, "
                 "fold, and two compressibility hybrids", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.185, 1, 0.95))
    fig.savefig(PNG, dpi=150)
    print(f"[fig] {PNG}")
    render_vapour(data)


PNG_VAP = OUTDIR + "/msa_exact_vapour_branch.png"


def render_vapour(data=None):
    r"""Companion figure: `ln η_v` against `1/T*` — the vapour side, made legible.

    ⭐ The reason the vapour branch is invisible on the phase diagram is not the
    axis, it is the physics: `ρ_v` falls off **exponentially** in `1/T`.  Squeezing
    that into the corner of a linear plot discards it; plotting its logarithm
    against `β` **straightens it out** instead.

    For a nearly ideal vapour `βp ≈ ρ_v`, and Clausius–Clapeyron gives
    `ln p ≈ −ΔH_vap·β + const`, so each coexistence line should come out close to a
    straight line whose slope is minus the enthalpy of vaporisation.  ⭐ That makes
    the companion do work the main figure cannot: it turns an invisible corner into
    a quantitative comparison of `ΔH_vap` across the routes, and any **curvature**
    marks where the vapour stops being ideal.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if data is None:
        data = json.load(open(JSON))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    import msa_exact_hybrids_compare as HC
    series = (("binodal", "energy route", C_ENERGY, "-"),
              ("binodal_virial", "virial route", C_VIRIAL, "-"),
              ("binodal_fold", "compr.–spinodal hybrid", C_HUG, "-"),
              ("binodal_apex", "compr.–critical-point detour", C_DETOUR, "-"),
              ("__ftail", r"compr.–$\beta f_{\rm tail}$ bridge", "#17a589", "-"),
              ("__splice", r"compr.–$C^0$ FMSA splice", "#111111", "-"))
    handles = {}
    for ax, key in zip(axes, ("HSY", "2YK")):
        d = data[key]
        for _lab_i, (jkey, lab, col, st) in enumerate(series):
            if jkey == "__splice":
                # ⭐ Solved here too.  Worth its own line because it is the one
                # completion that mixes two THEORIES on the SAME route (both
                # compressibility) rather than two routes — so the mixing conjecture
                # predicts it should be STRAIGHTER than the others.
                Ts = [p["T"] for p in sorted(d.get("binodal_fold", []),
                                             key=lambda p: p["T"])]
                pts = []
                for t in Ts:
                    v = HC.fmsa_splice_binodal(key, float(t))
                    if v is not None:
                        pts.append(dict(T=float(t), eta_g=v[0], eta_l=v[1]))
            elif jkey == "__ftail":
                # ⭐ SOLVED here, not pasted from another figure's JSON.  The ladder is
                # the hybrid's own `T` grid so the two lines are compared at matched
                # temperatures; `bridge_binodal` caches, so a re-render is free.
                Ts = [p["T"] for p in sorted(d.get("binodal_fold", []),
                                             key=lambda p: p["T"])]
                pts = [dict(T=t, eta_g=g, eta_l=l)
                       for t, g, l in HC.bridge_binodal_curve(key, Ts)]
            else:
                pts = sorted(d.get(jkey, []), key=lambda p: p["T"])
            pts = [p for p in pts if p["eta_g"] > 0]
            if len(pts) < 3:
                continue
            b = np.array([1.0 / p["T"] for p in pts])
            y = np.log(np.array([p["eta_g"] for p in pts]))
            ln, = ax.plot(b, y, st, color=col, lw=2.0, marker="o", ms=3)
            handles.setdefault(lab, ln)
            # slope of the coldest half = -dH_vap, the quantity the plot exists for
            m = b > np.median(b)
            if m.sum() >= 3:
                sl = np.polyfit(b[m], y[m], 1)[0]
                # ⚠ Staggered: on 2YK the three routes' ΔH_vap agree to 5.7 %, so
                # fixed-offset labels land on top of each other and read as one.
                ax.annotate(f"{-sl:.2f}", xy=(b[m][-1], y[m][-1]),
                            xytext=(5, -3 - 11*_lab_i), textcoords="offset points",
                            fontsize=7.5, color=col, ha="left", va="top")
        ax.set_title(d["label"], fontsize=11)
        ax.set_xlabel(r"$1/T^*$")
    axes[0].set_ylabel(r"$\ln \eta_{\mathrm{vapour}}$")
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center",
               ncol=5, fontsize=8.0, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(r"Exact MSA — the vapour branch, $\ln\eta_v$ vs $1/T^*$;  "
                 r"slope $=-\Delta H_{\mathrm{vap}}$ (annotated, cold half)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(PNG_VAP, dpi=150)
    print(f"[fig] {PNG_VAP}")


def fold_only():
    """Recompute just the hybrid coexistence line and re-render, from the saved JSON.

    The 86-isochore trace and the energy-route lines are untouched by anything the
    hybrid does, and they dominate the runtime."""
    data = json.load(open(JSON))
    tlo = float(sys.argv[sys.argv.index("--tlo") + 1]) if "--tlo" in sys.argv else 0.55
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    def _save():
        """⚠ Save after EVERY model, not once at the end.

        A crash in the second model used to discard the first model's completed
        work: the 2YK `brentq` NaN killed the process before `json.dump`, and HSY's
        virial spinodal — 14 points, already computed and printed — was lost with
        it.  The next `--only 2YK` run then reloaded a JSON that no longer had it
        and wrote back only its own half, so the loss was silent and permanent.
        """
        with open(JSON, "w") as fh:
            json.dump(data, fh, indent=1)

    for key, model in MODELS.items():
        if only and key != only:
            continue
        which = sys.argv[sys.argv.index("--gap") + 1] if "--gap" in sys.argv else "both"
        if which in ("both", "energy"):
            print(f"[{key}] energy-route coexistence (closed form) …", flush=True)
            data[key]["binodal"] = energy_binodal(
                model, 1.30 * max(q["T"] for q in data[key]["spinodal"]))
            _save()
        if which in ("both", "hug"):
            print(f"[{key}] compressibility-spinodal hybrid coexistence …", flush=True)
            data[key]["binodal_fold"], data[key]["fold_curves"] = fold_binodal(
                model, data[key], t_lo=tlo, gap="hug")
        if which in ("both", "virial"):
            print(f"[{key}] virial-route coexistence …", flush=True)
            (data[key]["binodal_virial"], data[key]["virial_Tc"],
             data[key]["virial_etac"]) = virial_binodal(model, data[key])
            _save()
            if data[key]["virial_Tc"]:
                data[key]["spinodal_virial"] = virial_spinodal(
                    model, data[key]["virial_Tc"], data[key]["virial_etac"])
            _save()
        if which in ("both", "detour"):
            print(f"[{key}] compressibility-critical-point detour hybrid coexistence …", flush=True)
            data[key]["binodal_apex"], data[key]["apex_curves"] = fold_binodal(
                model, data[key], t_lo=tlo, gap="detour")
    with open(JSON, "w") as fh:
        json.dump(data, fh, indent=1)
    print(f"[saved] {JSON}")
    render(data)


def retangent():
    """Redo the coexistence construction from the stored free-energy curves."""
    data = json.load(open(JSON))
    for key in MODELS:
        for rk, bk in (("fold_curves", "binodal_fold"), ("apex_curves", "binodal_apex")):
            raw = data[key].get(rk, [])
            out = []
            for c in raw:
                if "mu" not in c:      # pre-transported-mu JSON: regenerate, do not guess
                    print(f"[{key}] {rk}: stored without mu — rerun --fold-only")
                    break
                ct = fold_tangent(np.array(c["rho"]), np.array(c["f"]),
                                  np.array(c["mu"]), c["nsplit"])
                if ct is None:
                    continue
                rg, rl = ct
                if rl - rg > 1e-3:
                    out.append(dict(T=c["T"], eta_g=rg * np.pi / 6, eta_l=rl * np.pi / 6))
            data[key][bk] = out
            print(f"[{key}] {bk}: {len(out)}/{len(raw)} temperatures gave a tie line")
    with open(JSON, "w") as fh:
        json.dump(data, fh, indent=1)
    render(data)


if __name__ == "__main__":
    if "--vapour" in sys.argv:
        render_vapour()
    elif "--retangent" in sys.argv:
        retangent()
    elif "--fold-only" in sys.argv:
        fold_only()
    elif "--render" in sys.argv:
        render()
    else:
        data = compute()
        render(data)
