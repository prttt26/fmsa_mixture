r"""fmsa_phase_lines.py — the phase diagram of **FMSA-DP alone**, by three routes.

Companion to `msa_exact_phase_lines.py`, which does the same for the *exact* MSA.
Same two potentials, same axis, same legend vocabulary, so the two figures can be
read side by side.  ⚠ Deliberately **one method**: this is not a model comparison,
it is about how far one method's own routes disagree.

    HSY   hard-core Yukawa, one tail     z = 1.8,  k = [+1]
    HC2Y  hard core + two Yukawas        z = [2.9637, 14.0167], k = [+2.1714, -2.1714]
          ⚠ `input_type='bh'` IS the hard-core reading (the class says so), and
            it leaves sigma FIXED at 1 — verified against msa_exact_mix.HCYMix,
            sigma = 1.0000000000 and eta identical to the last digit at
            T* = 0.7/1.0/1.4.  Same potential, same axis, so the two figures are
            readable side by side.  It is NOT the soft-core 2YK of HSY_vs_2YK.md;
            LJ is out of scope here.

⭐ **Why this figure exists.**  In the exact MSA the compressibility route provably
has **no binodal at all**: `dbetap/drho = (A/2pi)^2` is a perfect square, so `betaA(rho)`
is convex on every isotherm and there is nothing to Maxwell-construct.  FMSA is that
same theory truncated at first order in the tail, and the truncation is under no
obligation to preserve the square.  Measured (`--probe`), it does not:

    HSY, T* = 1.00:  C = +0.366, -0.203, -0.594, -0.638, +1.476  at eta = 0.05 ... 0.30

⇒ **first order manufactures a compressibility-route binodal that the exact theory
forbids.**  That is the sharpest available statement of what the truncation costs,
and it is a statement about phase *lines*, not about a critical point.

⚠ **The critical point is deliberately not a subject here.**  It is treated in
`critical_point/` with machinery built for it; the spinodal apexes are drawn as
dots for orientation only and no number is claimed from them.

⚠ FMSA-DP has **no fold**: it is an analytic first-order expression, its branch does
not terminate, and every isotherm is connected.  So none of the crossing/detour
machinery the exact-MSA figure needs appears here — which is itself part of the
comparison.

    python fmsa_phase_lines.py            # compute + save JSON + render
    python fmsa_phase_lines.py --render   # re-render from saved JSON
    python fmsa_phase_lines.py --probe    # the C-goes-negative table above
"""
from __future__ import annotations
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from fmsa_double_prop import FMSA_double_prop
import energy_compress_core as EC

SYS = {
    "HSY": dict(key="HSY_z18", sig=[1.], eps=[1.], x=[1.], nt=1, input_type="bh",
                z=[1.8], k=[1.0], label=r"HSY  ($z=1.8$)"),
    "2YK": dict(key="2YK_lj", sig=[1.], eps=[1.], x=[1.], nt=2, input_type="bh",
                z=[2.9637, 14.0167], k=[2.1714, -2.1714],
                label=r"HC2Y  (hard core + 2 Yukawa)"),
}
OUT = "numerical_notes/results/compare_energy_compress/phase_lines"
JSON = OUT + "/fmsa_phase_lines.json"
PNG = OUT + "/fmsa_phase_lines.png"
C_ENERGY, C_COMPR, C_VIRIAL = "#2a78d6", "#e34948", "#6d4c41"

rho_of = lambda eta: 6.0 * eta / np.pi


def mk(sd, rho, T):
    return FMSA_double_prop(sd["sig"], sd["eps"], sd["x"], float(rho), float(T),
                            n_tails=sd["nt"], input_type=sd["input_type"],
                            z=sd["z"], k=sd["k"], verbose=False)


def p_virial(sd, rho, T, hs: float = 1e-3) -> float:
    r"""`βp` by the **virial route**, from FMSA-DP's own `g`.

        βp = ρ + (2π/3)ρ²σ³g(σ⁺) − (2π/3)ρ²Σ_t K_t e^{z_tσ}[ĝ(z_t) − z_t ĝ′(z_t)]

    identical in form to `msa_exact_mix.pressure_virial`; `ĝ` is `get_gL_laplace`
    and `ĝ′` a central difference in the Laplace variable.

    ⚠⚠ **The `K → 0` gate is Carnahan–Starling, not Percus–Yevick**, and getting
    that wrong is easy: FMSA-DP's hard-sphere reference is **FMT**, whose contact
    value is the CS one, whereas the exact-MSA Baxter route returns PY.  Measured at
    `K = 1e-9`: agreement with the CS virial EOS is **1e-9**, with the PY one it
    drifts to **13 %** by `η = 0.45`.  The first version of this gate compared
    against PY and reported FAIL — the failure was in the reference, not the code.

    ⭐ Consequence worth carrying: **FMSA-DP and the exact MSA do not share a
    hard-sphere reference**, so the two phase-diagram figures cannot be overlaid.
    Only the *route spread within each method* is comparable.
    """
    m = mk(sd, rho, T)
    g = float(m.get_g_contact(0, 0))
    tail = 0.0
    for zt, kt in zip(sd["z"], sd["k"]):
        gl = lambda s: float(np.real(np.ravel(m.get_gL_laplace(np.array([s])))[0]))
        dg = (gl(zt + hs) - gl(zt - hs)) / (2.0 * hs)
        tail += (kt / T) * np.exp(zt) * (gl(zt) - zt * dg)
    return rho + (2 * np.pi / 3) * rho ** 2 * g - (2 * np.pi / 3) * rho ** 2 * tail


def derivs(sd, rho, T, drho: float = 2e-3, routes=("C", "E", "V")) -> dict:
    """`∂βp/∂ρ` on each route at one state — the object all three lines come from."""
    out = {}
    m = mk(sd, rho, T)
    x = np.asarray(sd["x"], float)
    hs = 1.0 - float(x @ EC.chat0_hs_fmt(m) @ m.rho)
    if "C" in routes:
        c1 = np.real(m.get_c1hat_k(np.array([1e-5]))[0])
        out["C"] = hs - float(x @ c1 @ m.rho)
    if "E" in routes:
        A = lambda r: EC.dp_A(mk(sd, r, T))
        out["E"] = hs + rho * (A(rho + drho) - 2 * A(rho) + A(rho - drho)) / drho ** 2
    if "V" in routes:
        out["V"] = (p_virial(sd, rho + drho, T) - p_virial(sd, rho - drho, T)) / (2 * drho)
    return out


def isotherm(sd, T, n_lo: int = 14, n_hi: int = 260, rho_max: float = 0.90,
             routes=("C", "E", "V")):
    r"""`(ρ, {route: ∂βp/∂ρ})` along one isotherm.

    ⚠ `n_hi = 260` (`Δρ ≈ 0.0034`), not the 74 this started with.  Near `T_c` the
    loop is only a few hundredths wide, and a coarse grid loses the coexistence
    crossing for a purely numerical reason: measured on all three routes,
    `n_hi = 74` fails at `T/T_c = 0.999` and `n_hi = 260` resolves it, with
    `(η_v+η_l)/2` landing on `η_c` to four decimals.  FMSA-DP costs 15 ms a point,
    so the refinement is free and there is no excuse for the coarse version.
    """
    rs = np.concatenate([np.geomspace(1e-4, 0.02, n_lo),
                         np.linspace(0.025, rho_max, n_hi)])
    got = {k: [] for k in routes}
    keep = []
    for r in rs:
        try:
            d = derivs(sd, float(r), T, routes=routes)
        except Exception:
            continue
        if not all(np.isfinite(v) for v in d.values()):
            continue
        keep.append(float(r))
        for k in got:
            got[k].append(d[k])
    if len(keep) < 30:
        return None, None
    return np.array(keep), {k: np.array(v) for k, v in got.items()}


def spinodal_of(rho, r):
    """The two densities where `∂βp/∂ρ` changes sign, or `None` above `T_c`."""
    sc = np.where(np.sign(r[:-1]) != np.sign(r[1:]))[0]
    if len(sc) < 2:
        return None
    f = lambda x: np.interp(x, rho, r)
    return (float(brentq(f, rho[sc[0]], rho[sc[0] + 1])),
            float(brentq(f, rho[sc[-1]], rho[sc[-1] + 1])))


def p_mu(rho, r):
    r"""`(βp, βμ)` by integrating this route's own `∂βp/∂ρ` along the isotherm.

    ⭐ Both from the same integrand, so the pair is thermodynamically consistent
    *within* the route by construction — which is the only consistency any single
    route can offer here.  ⚠ `μ` is **not** obtained by differentiating a spline of
    `βA/V`: on a geometric ρ ladder that puts wiggles into `p = ρμ − βA/V` large
    enough to lose the coexistence crossing (the lesson from
    `msa_exact_phase_lines.fold_tangent`).
    """
    p = np.concatenate([[rho[0]], rho[0] + cumulative_trapezoid(r, rho)])
    integ = (r - 1.0) / rho
    mu = np.log(rho) + np.concatenate([[0.0], cumulative_trapezoid(integ, rho)])
    return p, mu


def binodal_of(rho, r):
    """Coexistence by the `μ`–`p` crossing on the two mechanically stable branches."""
    p, mu = p_mu(rho, r)
    neg = np.where(r < 0)[0]
    if len(neg) == 0:
        return None
    i, j = int(neg[0]), int(neg[-1])
    if i < 5 or j > len(rho) - 6:
        return None
    pg, mg, rg = p[:i], mu[:i], rho[:i]
    pl, ml, rl = p[j + 1:], mu[j + 1:], rho[j + 1:]
    lo, hi = max(pg.min(), pl.min()), min(pg.max(), pl.max())
    if not (hi > lo):
        return None
    d = lambda q: np.interp(q, pg, mg) - np.interp(q, pl, ml)
    if d(lo) * d(hi) > 0:
        return None
    q = brentq(d, lo, hi, xtol=1e-14)
    a, b = float(np.interp(q, pg, rg)), float(np.interp(q, pl, rl))
    return (a, b, float(q)) if b - a > 1e-3 else None


def critical_of(sd, route: str, t_lo: float = 0.6, t_hi: float = 2.2):
    r"""`(T*_c, η_c)` for one route — **from the spinodal apex**, by bisection.

    ⭐ The critical point is where `min_ρ ∂βp/∂ρ` reaches zero.  That is a *regular*
    root of a smooth function, so bisection walks straight to it; the coexistence
    solve cannot, because the loop depth falls off like `(T_c−T)²` and the two roots
    merge.  Same division of labour as everywhere else here: **binodal from
    coexistence, critical point from the spinodal.**

    ⚠ FMSA-DP admits this on all three routes and over the whole plane — it is
    analytic, has no fold, and no isotherm is cut — so unlike the exact-MSA figure
    nothing needs to be left untracked.
    """
    rs = np.concatenate([np.geomspace(0.02, 0.05, 4), np.linspace(0.06, 0.62, 22)])

    def worst(T):
        v = []
        for r in rs:
            try:
                v.append(derivs(sd, float(r), float(T), routes=(route,))[route])
            except Exception:
                v.append(np.nan)
        v = np.array(v, float)
        ok = np.isfinite(v)
        if ok.sum() < 5:
            return np.nan, np.nan
        rr, vv = rs[ok], v[ok]
        i = int(np.argmin(vv))
        if 0 < i < len(vv) - 1:          # parabola through the three lowest
            x0, x1, x2 = rr[i-1], rr[i], rr[i+1]
            y0, y1, y2 = vv[i-1], vv[i], vv[i+1]
            dd = y0 - 2*y1 + y2
            rc = x1 - 0.5*(x2-x0)*(y2-y0)/(2*dd) if abs(dd) > 1e-14 else x1
        else:
            rc = rr[i]
        return float(vv[i]), float(rc)

    try:
        Tc = brentq(lambda T: worst(T)[0], t_lo, t_hi, xtol=1e-4)
    except Exception:
        return None, None
    return float(Tc), float(worst(Tc)[1] * np.pi / 6)


def compute(n_T: int = 24, t_lo: float = 0.55):
    r"""Both lines for all three routes, each on **its own** `T` grid up to its own `T_c`.

    ⭐ Every line is closed at the critical point, and nothing is left untracked:
    `critical_of` locates `T_c` from the spinodal apex (a regular bisection), the
    scan runs from `0.999·T_c` down, and the critical point itself is appended as
    the closing entry of **both** the spinodal and the binodal — which is not an
    interpolation but the definition: the two curves meet there.

    ⚠ This is possible because FMSA-DP is analytic, has no fold, and cuts no
    isotherm.  The exact-MSA figure cannot do it and has to draw an untracked cap.
    """
    out = {}
    for key, sd in SYS.items():
        rec = {}
        for rt in ("C", "E", "V"):
            Tc, ec = critical_of(sd, rt)
            if Tc is None:
                rec[rt] = {"spinodal": [], "binodal": [], "Tc": None, "etac": None}
                continue
            print(f"[{key}] {rt}: T_c = {Tc:.4f}  eta_c = {ec:.4f}", flush=True)
            Ts = np.concatenate([Tc * np.array([0.999, 0.996, 0.992, 0.986]),
                                 np.linspace(0.975 * Tc, t_lo, n_T)])
            sp, bi = [], []
            for T in Ts:
                rho, rr = isotherm(sd, float(T), routes=(rt,))
                if rho is None:
                    continue
                s2 = spinodal_of(rho, rr[rt])
                if s2:
                    sp.append(dict(T=float(T), eta_g=s2[0] * np.pi / 6,
                                   eta_l=s2[1] * np.pi / 6))
                b2 = binodal_of(rho, rr[rt])
                if b2:
                    bi.append(dict(T=float(T), eta_g=b2[0] * np.pi / 6,
                                   eta_l=b2[1] * np.pi / 6, p=b2[2]))
            # close both curves exactly at the critical point
            for L in (sp, bi):
                if L:
                    L.append(dict(T=float(Tc), eta_g=float(ec), eta_l=float(ec)))
            print(f"    spinodal {len(sp)} pts, binodal {len(bi)} pts "
                  f"(highest tracked {max((p['T'] for p in bi[:-1]), default=float('nan'))/Tc:.3f} T_c)",
                  flush=True)
            rec[rt] = {"spinodal": sp, "binodal": bi, "Tc": float(Tc), "etac": float(ec)}
        out[key] = dict(label=sd["label"], **rec)
        with open(JSON, "w") as fh:      # save per model, never lose a finished one
            json.dump(out, fh, indent=1)
    print(f"[saved] {JSON}")
    return out


def render(data=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if data is None:
        data = json.load(open(JSON))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.0), sharey=True)
    ymax = max([data[k][r]["Tc"] for k in data for r in ("C", "E", "V")
                if data[k][r].get("Tc")] + [1.0])
    for ax, key in zip(axes, ("HSY", "2YK")):
        d = data[key]
        for rk, col in (("E", C_ENERGY), ("C", C_COMPR), ("V", C_VIRIAL)):
            for jk, ls, lw in (("binodal", "-", 2.2), ("spinodal", "--", 1.6)):
                pts = sorted(d[rk][jk], key=lambda p: p["T"])
                if len(pts) < 3:
                    continue
                T = [p["T"] for p in pts]
                # both curves already close AT the critical point, so they are
                # drawn as closed domes and there is no untracked cap to mark
                eg = [p["eta_g"] for p in pts]
                el = [p["eta_l"] for p in pts]
                ax.plot(eg + el[::-1], T + T[::-1], ls, color=col, lw=lw)
            if d[rk].get("Tc"):
                ax.plot([d[rk]["etac"]], [d[rk]["Tc"]], "o", color=col, ms=6.5,
                        zorder=6)
        _b = 0.03
        f = lambda x: np.where(np.asarray(x, float) >= _b, x,
                               _b * (1 + np.log(np.maximum(np.asarray(x, float), 1e-300) / _b)))
        g = lambda y: np.where(np.asarray(y, float) >= _b, y,
                               _b * np.exp(np.clip(np.asarray(y, float) / _b - 1, -700, 700)))
        ax.set_xscale("function", functions=(f, g))
        ax.set_xticks([1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xticklabels(["$10^{-4}$", "0.001", "0.01", "0.05", "0.1", "0.2",
                            "0.3", "0.4", "0.5"], fontsize=8)
        ax.set_xlim(5e-5, 0.5)
        ax.set_ylim(0.0, 1.18 * ymax)
        ax.set_xlabel(r"packing fraction  $\eta$"
                      "\n(log below 0.03, linear above — glued $C^1$)")
        ax.set_title(d["label"], fontsize=11)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"reduced temperature  $T^*$")
    L = lambda **kw: Line2D([0], [0], **kw)
    blocks = (([(L(color="#555", lw=2.2), "binodal"),
                (L(color="#555", lw=1.6, ls="--"), "spinodal"),
                (L(color="#555", lw=0, marker="o", ms=6.5), "critical point")],
               0.22, "line style = object"),
              ([(L(color=C_ENERGY, lw=2.4), "energy route"),
                (L(color=C_COMPR, lw=2.4), "compressibility route"),
                (L(color=C_VIRIAL, lw=2.4), "virial route")], 0.62, "colour = route"))
    for blk, x, ttl in blocks:
        lg = fig.legend([h for h, _ in blk], [t for _, t in blk], loc="upper center",
                        bbox_to_anchor=(x, 0.155), fontsize=8.4, frameon=False,
                        title=ttl, title_fontsize=8.8, handlelength=2.6)
        lg._legend_box.align = "left"
        fig.add_artist(lg)
    fig.suptitle("FMSA-DP alone — binodals and spinodals by three routes "
                 "(the truncation's own route spread)", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.175, 1, 0.94))
    fig.savefig(PNG, dpi=150)
    print(f"[fig] {PNG}")


PNG_VAP = OUT + "/fmsa_vapour_branch.png"


def render_vapour(data=None):
    r"""`ln η_v` against `1/T*` — the vapour side, made legible, for all three routes.

    ⭐ Same construction as `msa_exact_phase_lines.render_vapour`, and it does the same
    work: `ρ_v` falls off exponentially in `1/T`, so a linear phase diagram throws the
    vapour branch into a corner.  Plotting its logarithm against `β` straightens it,
    and for a near-ideal vapour Clausius–Clapeyron gives `ln p ≈ −ΔH_vap·β + const`
    ⇒ the slope of the cold half **is** minus the enthalpy of vaporisation.

    ⭐ For FMSA that turns the route gap into a single number per route, and one that
    the critical point cannot show: `T_c` compares the tops of the domes, `ΔH_vap`
    compares their flanks.  ⚠ Curvature is the diagnostic — a route whose line bends
    is one whose vapour branch has stopped being trustworthy, whatever its `T_c`.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if data is None:
        data = json.load(open(JSON))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    ser = (("C", "compressibility route", C_COMPR),
           ("E", "energy route", C_ENERGY),
           ("V", "virial route", C_VIRIAL))
    handles = {}
    for ax, key in zip(axes, [k for k in ("HSY", "2YK") if k in data]):
        d = data[key]
        for _lab_i, (rt, lab, col) in enumerate(ser):
            pts = sorted(d.get(rt, {}).get("binodal", []), key=lambda p: p["T"])
            pts = [p for p in pts if p.get("eta_g", 0) > 0]
            if len(pts) < 3:
                continue
            b = np.array([1.0/p["T"] for p in pts])
            y = np.log(np.array([p["eta_g"] for p in pts]))
            ln, = ax.plot(b, y, "-", color=col, lw=2.0, marker="o", ms=3)
            handles.setdefault(lab, ln)
            m = b > np.median(b)
            if m.sum() >= 3:
                sl = np.polyfit(b[m], y[m], 1)[0]
                # ⚠ Staggered: on 2YK the three routes' ΔH_vap agree to 5.7 %, so
                # fixed-offset labels land on top of each other and read as one.
                ax.annotate(f"{-sl:.2f}", xy=(b[m][-1], y[m][-1]),
                            xytext=(5, -3 - 11*_lab_i), textcoords="offset points",
                            fontsize=7.5, color=col, ha="left", va="top")
        ax.set_title(d.get("label", key), fontsize=11)
        ax.set_xlabel(r"$1/T^*$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$\ln \eta_{\mathrm{vapour}}$")
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center",
               ncol=3, fontsize=8.6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(r"FMSA-DP — the vapour branch, $\ln\eta_v$ vs $1/T^*$;  "
                 r"slope $=-\Delta H_{\mathrm{vap}}$ (annotated, cold half)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(PNG_VAP, dpi=150)
    print(f"[fig] {PNG_VAP}")


def cmd_convention():
    r"""⭐⭐ Gate: does FMSA-DP's tail mean the same thing as the exact MSA's?

    Both files carry `2YK` as `z = [2.9637, 14.0167]`, `k = [2.1714, −2.1714]`, and the
    figures were labelled "LJ → 2 Yukawa" — which would require an LJ→Yukawa conversion
    on both sides, with a temperature-dependent Barker–Henderson diameter.  ⚠ **Neither
    side does one.**  `input_type='bh'` with `sig=[1.0]` means `sig_mix` IS the diameter
    matrix, i.e. a fixed hard core of 1.0 (see `FMSA_double_prop`: "'bh' is for a
    hard-core Yukawa model"), and `msa_exact_mix.HCYMix([1.0], …)` is the same thing.
    ⇒ both are a hard-core two-Yukawa fluid at `σ = 1`, the label was stale, and nothing
    needs converting.

    ⭐ What DOES need checking is that the two share the tail *convention* — the same `K`
    normalisation and sign.  FMSA-DP is exact to first order in `K`, so the `K`-linear
    slope of any observable must agree with the exact MSA's as `K → 0`.  Measured on
    `∂βp/∂ρ` at `ρ = 0.35`, `T* = 1.2`:

        lambda   exact MSA     FMSA-DP    ratio
         0.050   −3.936981   −3.947313   1.002624
         0.020   −3.943227   −3.947313   1.001036
         0.010   −3.945277   −3.947313   1.000516
         0.005   −3.946297   −3.947313   1.000257

    The ratio halves with `λ` ⇒ the residual is the `O(K²)` truncation and nothing else.

    ⚠ The `K = 0` references still differ — `4.196577` (Baxter/PY) vs `4.143787`
    (FMT/CS), 1.3 % apart — which is why the two phase diagrams may not be overlaid.
    Same tail, different hard sphere.
    """
    import msa_exact_mix as X
    z = np.array([2.9637, 14.0167]); kv = np.array([2.1714, -2.1714])
    T, rho = 1.2, 0.35

    def a_msa(lam):
        q = X.solve_bh_mix(X.HCYMix([1.0], [rho], z, lam*kv[:, None, None]/T))
        return (float(np.sum(q.p['A']))/(2*np.pi))**2

    def a_fmsa(lam):
        sd = dict(SYS['2YK']); sd['k'] = list(lam*kv)
        return derivs(sd, rho, T, routes=('C',))['C']

    print(f"{'lambda':>9}{'exact MSA':>13}{'FMSA-DP':>13}{'ratio':>11}")
    ok = True
    for lam in (0.05, 0.02, 0.01, 0.005):
        sm = (a_msa(lam) - a_msa(0.0))/lam
        sf = (a_fmsa(lam) - a_fmsa(0.0))/lam
        print(f"{lam:9.3f}{sm:13.6f}{sf:13.6f}{sf/sm:11.6f}")
        ok &= abs(sf/sm - 1.0) < 4.0*lam        # O(K) convergence to the same limit
    print(f"  HS reference at K=0:  PY {a_msa(0.0):.6f}   FMT/CS {a_fmsa(0.0):.6f}"
          f"   ({100*abs(a_fmsa(0.0)/a_msa(0.0)-1):.1f} % apart — NOT overlayable)")
    print("  GATE " + ("PASS — same tail convention" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_probe():
    print(__doc__)
    print("=" * 78)
    print("K -> 0 gate for the virial route: it must return the CARNAHAN-STARLING")
    print("virial EOS -- FMSA-DP's hard-sphere reference is FMT, not PY.")
    print("=" * 78)
    sd = dict(SYS["HSY"]); sd["k"] = [1e-9]
    print(f"   {'eta':>6}{'p_virial':>12}{'CS virial':>12}{'rel dev':>10}")
    worst = 0.0
    for eta in (0.05, 0.15, 0.25, 0.35):
        r = rho_of(eta)
        pv = p_virial(sd, r, 1.0)
        py = r * (1 + eta + eta ** 2 - eta ** 3) / (1 - eta) ** 3
        worst = max(worst, abs(pv / py - 1))
        print(f"   {eta:6.2f}{pv:12.6f}{py:12.6f}{abs(pv/py-1):10.1e}")
    print(f"   worst {worst:.1e}   {'PASS' if worst < 1e-6 else 'FAIL'}")
    print("\n" + "=" * 78)
    print("C = compressibility dbetaP/drho, E = energy, V = virial")
    print("=" * 78)
    for nm, s2 in SYS.items():
        print(f"\n  {nm}")
        print(f"  {'T*':>7} | " + "".join(f"{e:>22.2f}" for e in (0.05, 0.10, 0.15, 0.20, 0.30)))
        for T in (1.40, 1.20, 1.00, 0.90):
            row = []
            for e in (0.05, 0.10, 0.15, 0.20, 0.30):
                try:
                    dd = derivs(s2, rho_of(e), T)
                    row.append(f"{dd['C']:6.3f}/{dd['E']:6.3f}/{dd['V']:6.3f}")
                except Exception:
                    row.append("err")
            print(f"  {T:7.2f} | " + "".join(f"{v:>22}" for v in row))
    return 0


if __name__ == "__main__":
    if "--convention" in sys.argv:
        raise SystemExit(cmd_convention())
    if "--probe" in sys.argv:
        raise SystemExit(cmd_probe())
    if "--vapour" in sys.argv:
        render_vapour()
        raise SystemExit(0)
    if "--render" in sys.argv:
        render()
    else:
        render(compute())
