r"""fmsa_vdw_loop.py — the FMSA-DP van der Waals loop, the one the exact MSA has to
manufacture.

The companion of `msa_exact_vdw_loop.py`, and the point is the contrast.  In the exact
MSA the compressibility route gives `∂βp/∂ρ = (A/2π)²`, a perfect square, so it is
**non-negative on every isotherm**: no loop, no common tangent, and a binodal only after
a completion is bolted on across the unreachable interval.  FMSA-DP truncates at first
order in `K`, which **destroys the square**, and the loop is simply there:

* `a` goes genuinely negative — no bridge, no ansatz;
* the isotherm is never cut — no fold, so no missing interval;
* `A` has no sign to get wrong — no wrong-winding stretch to discard.

⇒ Everything the exact-MSA figure needed four different completions to supply, the
truncated theory produces for free. ⚠ That is not a point in its favour. The same
truncation puts the spinodal **12–30 % too high** (`msax3_n_tails`), and here it is the
sole reason a compressibility-route loop exists at all.

⚠⚠ **Do not overlay this on the exact-MSA figure.** FMSA-DP's hard-sphere reference is
FMT (contact value = Carnahan–Starling); the exact-MSA Baxter route's is Percus–Yevick.
The two do not share a reference, so only the *route spread within each method* is
comparable — see `fmsa_phase_lines.p_virial`.

All three routes are drawn, each integrated from its **own** `∂βp/∂ρ`, so each is
internally consistent and each has its own spinodal, its own coexistence and its own
tilt.  Their disagreement is the route gap, read off one isotherm instead of off `T_c`.

Run:  python fmsa_vdw_loop.py [--render]
"""
from __future__ import annotations
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
import fmsa_phase_lines as FP

OUT = "numerical_notes/results/compare_energy_compress/phase_lines"
JSON = OUT + "/fmsa_vdw_loop.json"
PNG = OUT + "/fmsa_vdw_loop.png"
ROUTES = (("C", "compressibility", FP.C_COMPR),
          ("E", "energy", FP.C_ENERGY),
          ("V", "virial", FP.C_VIRIAL))
T_FRAC = 0.88            # of each system's own compressibility-route `T_c`


def coexist_exact(rho, a):
    r"""Coexistence by a 2-D Newton on `(ρ_g, ρ_l)` with the exact Jacobian.

    Same construction as `msa_exact_vdw_loop.coexist_exact`, and for the same reason:
    `Δβf = βf − (βμ_c ρ − βp_c)` vanishes at both densities as an **identity**, so the
    residual there is a free gate on the solve and nothing else.  `dβp/dρ = a`,
    `dβμ/dρ = a/ρ`, `det J = a_g a_l (1/ρ_g − 1/ρ_l) > 0`.
    """
    seed = FP.binodal_of(rho, a)
    if seed is None:
        return None
    p, mu = FP.p_mu(rho, a)
    P, M = PchipInterpolator(rho, p), PchipInterpolator(rho, mu)
    dP, dM = P.derivative(), M.derivative()
    x = np.array(seed[:2], float)
    for _ in range(60):
        F = np.array([P(x[1]) - P(x[0]), M(x[1]) - M(x[0])])
        J = np.array([[-dP(x[0]), dP(x[1])], [-dM(x[0]), dM(x[1])]])
        if abs(np.linalg.det(J)) < 1e-14:
            break
        x = x + np.clip(np.linalg.solve(J, -F), -0.05, 0.05)
        if np.max(np.abs(F)) < 1e-14:
            break
    rg, rl = float(x[0]), float(x[1])
    pc, mc = float(0.5*(P(rg) + P(rl))), float(0.5*(M(rg) + M(rl)))
    gate = max(abs(float(rg*(M(rg) - mc) - (P(rg) - pc))),
               abs(float(rl*(M(rl) - mc) - (P(rl) - pc))))
    return dict(rho_g=rg, rho_l=rl, p_coex=pc, mu_coex=mc, gate=gate)


def compute():
    main = json.load(open(FP.JSON))
    out = {}
    for key, sd in FP.SYS.items():
        Tc = main[key]["C"]["Tc"]
        T = T_FRAC * Tc
        print(f"[{key}] T* = {T:.4f}  = {T_FRAC} x the compressibility "
              f"T_c = {Tc:.4f}", flush=True)
        rho, rr = FP.isotherm(sd, T, routes=("C", "E", "V"))
        if rho is None:
            print("    isotherm failed", flush=True); continue
        rec = dict(T=float(T), Tc_C=float(Tc), label=sd["label"],
                   rho=rho.tolist(), routes={})
        for rt, name, _ in ROUTES:
            a = rr[rt]
            p, mu = FP.p_mu(rho, a)
            cx = coexist_exact(rho, a)
            sp = FP.spinodal_of(rho, a)
            if cx is None:
                print(f"    {name:16s} no coexistence at this T", flush=True)
                continue
            f = rho*mu - p
            tilt = f - (cx["mu_coex"]*rho - cx["p_coex"])
            ins = (rho > sp[0]) & (rho < sp[1]) if sp else np.zeros_like(rho, bool)
            print(f"    {name:16s} eta {cx['rho_g']*np.pi/6:.5f} .. "
                  f"{cx['rho_l']*np.pi/6:.5f}   beta p_coex = {cx['p_coex']:.7f}"
                  f"   min a = {a.min():+.4f}   barrier = {tilt[ins].max():.4f}"
                  f"   gate = {cx['gate']:.1e}", flush=True)
            rec["routes"][rt] = dict(a=a.tolist(), p=p.tolist(), mu=mu.tolist(),
                                     f=f.tolist(), spinodal=list(sp) if sp else None,
                                     barrier=float(tilt[ins].max()), **cx)
        out[key] = rec
    with open(JSON, "w") as fh:
        json.dump(out, fh)
    print(f"[saved] {JSON}")
    return out


def render(data=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if data is None:
        data = json.load(open(JSON))
    keys = [k for k in ("HSY", "2YK") if k in data]
    fig, axes = plt.subplots(len(keys), 4, figsize=(15.6, 4.1*len(keys) + 0.6),
                             squeeze=False)

    for row, key in enumerate(keys):
        d = data[key]
        rho = np.array(d["rho"])
        lo = min(v["rho_g"] for v in d["routes"].values())
        hi = max(v["rho_l"] for v in d["routes"].values())
        x0, x1 = max(0.0, lo - 0.05), hi + 0.05
        m = (rho >= x0) & (rho <= x1)

        for col, (qty, lab) in enumerate(
                (("a", r"$a=\partial\beta p/\partial\rho$"),
                 ("p", r"$\beta p\sigma^3$"), ("mu", r"$\beta\mu$"),
                 ("tilt", r"tilted  $\Delta\beta f$"))):
            ax = axes[row][col]
            ys = []
            for rt, name, c in ROUTES:
                v = d["routes"].get(rt)
                if v is None:
                    continue
                y = (np.array(v["f"]) - (v["mu_coex"]*rho - v["p_coex"])
                     if qty == "tilt" else np.array(v[qty]))
                ax.plot(rho[m], y[m], "-", color=c, lw=2.0)
                ys.append(y[m])
                # its own spinodal (open) and its own coexistence (filled)
                if v["spinodal"]:
                    for r in v["spinodal"]:
                        ax.plot([r], [np.interp(r, rho, y)], "o", mfc="none",
                                mec=c, mew=1.6, ms=6, zorder=6)
                for r in (v["rho_g"], v["rho_l"]):
                    ax.plot([r], [np.interp(r, rho, y)], "o", color=c, ms=6,
                            zorder=7)
            if qty in ("a", "tilt"):
                ax.axhline(0.0, color="0.45", lw=0.9, ls=":")
            # ⚠ Scale on the COEXISTENCE interval, not the plotted range.  Past the
            # liquid density `a` climbs to ~12 and the tilted `f` shoots up with it,
            # so autoscaling on everything drawn flattens the loop — which is the
            # only thing the panel is for.  Same failure as in `msa_exact_vdw_loop`,
            # one level up: there it was the whole isotherm, here the last 0.05 in ρ.
            ms = (rho >= x0) & (rho <= hi)
            v = np.concatenate([y[ms] for y in
                                [np.array(d["routes"][rt][qty]) if qty != "tilt"
                                 else np.array(d["routes"][rt]["f"])
                                 - (d["routes"][rt]["mu_coex"]*rho
                                    - d["routes"][rt]["p_coex"])
                                 for rt, _, _ in ROUTES if rt in d["routes"]]])
            span = max(float(v.max() - v.min()), 1e-6)
            if qty == "tilt":
                bar = max(w["barrier"] for w in d["routes"].values())
                ax.set_ylim(-0.09*bar, 1.30*bar)
            else:
                ax.set_ylim(float(v.min()) - 0.10*span,
                            float(v.max()) + 0.12*span)
            ax.set_xlim(x0, x1)
            ax.set_ylabel(lab)
            if row == len(keys) - 1:
                ax.set_xlabel(r"density  $\rho\sigma^3$")
        axes[row][0].set_title(
            "%s   $T^*=%.4f$  $=%.2f\\,T_c$ (compressibility route)"
            % (d["label"], d["T"], T_FRAC), fontsize=10, loc="left")

    L = lambda **kw: Line2D([0], [0], **kw)
    h = [L(color=c, lw=2.0) for _, _, c in ROUTES]
    h += [L(color="0.3", lw=0, marker="o", mfc="none", mec="0.3", mew=1.6, ms=6),
          L(color="0.3", lw=0, marker="o", ms=6)]
    fig.legend(h, ["%s route" % n for _, n, _ in ROUTES] +
               [r"spinodal ($a=0$), each route's own",
                "coexistence, each route's own"],
               loc="lower center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.004))
    fig.suptitle("FMSA-DP — the van der Waals loop is simply there: "
                 r"$a$ goes negative, no fold cuts the isotherm, nothing to bridge."
                 "\n⚠ The exact MSA's compressibility route can do none of this "
                 r"($a=(A/2\pi)^2\geq 0$) — the loop here is made by the "
                 "first-order truncation.", fontsize=11)
    fig.tight_layout(rect=(0, 0.075, 1, 0.90))
    fig.savefig(PNG, dpi=150)
    print(f"[fig] {PNG}")


if __name__ == "__main__":
    render() if "--render" in sys.argv else render(compute())
