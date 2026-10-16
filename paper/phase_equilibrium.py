r"""phase_equilibrium.py — vapour–liquid equilibrium for the three benchmarked
models, against Potoff & Panagiotopoulos 1998 Table I.

The free energy is already complete in `energy_compress_core`: for every model

    βf(ρ₁,ρ₂) = Σᵢρᵢ(lnρᵢ−1)  +  βf_HS(ρ;d(T))  +  A(ρ₁,ρ₂)

with A the tail contribution from coupling-constant integration.  Only the
*isopleth* second derivative of A was needed before; a phase equilibrium needs
the gradient in the full (ρ₁,ρ₂) plane, which is the same object evaluated off
the isopleth:

    βμᵢ = lnρᵢ + ∂(βf_HS + A)/∂ρᵢ
    βp  = Σᵢρᵢ + Σᵢρᵢ·∂(βf_HS + A)/∂ρᵢ − (βf_HS + A)

Bubble-point calculation.  T\* and the liquid composition x₁ᴸ are taken from the
table; the three unknowns (ρᴸ, ρ₁ⱽ, ρ₂ⱽ) are solved from

    βμ₁ᴸ = βμ₁ⱽ ,  βμ₂ᴸ = βμ₂ⱽ ,  βpᴸ = βpⱽ

The model then *predicts* p\*, y₁, ρ\*_liq and ρ\*_gas, all four of which the table
measures.  Starting from the measured state is deliberate — Newton leaves it
immediately and converges to the model's own solution; it only keeps the solver
in the right basin.

    python phase_equilibrium.py --model FMSA-DP
    python phase_equilibrium.py --render
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import root

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

from energy_compress_core import (MODELS, STYLE, _betaf_hs_raw, dp_A, oz_point,
                                  make_model)
from fmsa_double_prop import FMSA_double_prop

OUTDIR = "numerical_notes/results/compare_energy_compress/critical_point"
REFJSON = os.path.join(OUTDIR, "reference_potoff1998.json")
CACHE = os.path.join(OUTDIR, "cache_vle")

SIG = np.array([1.0, 1.0])
EPS = np.array([1.0, 0.5])
DRHO = 2e-3            # step for the ∂A/∂ρᵢ central differences


def _sysd(rho_vec: np.ndarray) -> dict:
    """System dict at an arbitrary partial-density vector."""
    rt = float(np.sum(rho_vec))
    x = np.asarray(rho_vec, dtype=float) / rt
    return dict(key=f"vle|{x[0]:.9f}", sig=SIG, eps=EPS, x=x, nt=2), rt


_A_CACHE: dict = {}


def A_of(rho_vec: np.ndarray, tstar: float, model: str) -> float | None:
    """Tail free energy A at an arbitrary (ρ₁,ρ₂)."""
    key = (model, round(float(rho_vec[0]), 10), round(float(rho_vec[1]), 10),
           round(float(tstar), 10))
    if key in _A_CACHE:
        return _A_CACHE[key]
    sysd, rt = _sysd(rho_vec)
    if model == "FMSA-DP":
        val = dp_A(make_model(sysd, rt, tstar))
    elif model == "MSA-exact":
        # The SAME tail, solved to all orders.  `from_fmsa` reads the Yukawa
        # system off the FMSA object, so this differs from the FMSA-DP branch
        # above in the solution order and in nothing else.
        #
        # ⚠ `free_energy_excess` is βΔA/V over hard spheres, which is the same
        # quantity `dp_A` returns, and it is added to the same White-Bear
        # `_betaf_hs_raw` in `f_ex`.  Blum & Høye's Δ is over the PY hard sphere
        # rather than White Bear, exactly as it is for the other three legs --
        # the reference is shared, so it cancels from every comparison here even
        # though none of the four is internally pure.
        import msa_exact_mix as X
        try:
            sol = X.solve_bh_mix(X.from_fmsa(make_model(sysd, rt, tstar)))
            val = None if sol is None else float(X.free_energy_excess(sol))
            if val is not None and not np.isfinite(val):
                val = None
        except Exception:                                        # noqa: BLE001
            val = None                       # no real root: not a failure to report
    else:
        closure = {"OZ-MSA": "msa", "OZ-HNCB": "hncb"}[model]
        p = oz_point(sysd, rt, tstar, closure)
        val = None if p is None else p["A"]
    if len(_A_CACHE) > 20000:
        _A_CACHE.clear()
    _A_CACHE[key] = val
    return val


def _diam(tstar: float) -> np.ndarray:
    """Barker–Henderson diameters — composition-independent at fixed T."""
    m = FMSA_double_prop(SIG, EPS, np.array([0.5, 0.5]), 0.1, tstar,
                         n_tails=2, verbose=False)
    return m.diag_r.copy()


def f_ex(rho_vec: np.ndarray, tstar: float, model: str,
         diam: np.ndarray) -> float | None:
    A = A_of(rho_vec, tstar, model)
    if A is None:
        return None
    return _betaf_hs_raw(np.asarray(rho_vec, dtype=float), diam) + A


def mu_p(rho_vec: np.ndarray, tstar: float, model: str,
         diam: np.ndarray, d: float = DRHO):
    """(βμ₁, βμ₂, βp) including the ideal terms."""
    rho = np.asarray(rho_vec, dtype=float)
    f0 = f_ex(rho, tstar, model, diam)
    if f0 is None:
        return None
    grad = np.zeros(2)
    for i in range(2):
        hp, hm = rho.copy(), rho.copy()
        hp[i] += d
        hm[i] -= d
        if hm[i] <= 0.0:
            hm[i] = rho[i] * 1e-3
        fp = f_ex(hp, tstar, model, diam)
        fm = f_ex(hm, tstar, model, diam)
        if fp is None or fm is None:
            return None
        grad[i] = (fp - fm) / (hp[i] - hm[i])
    mu = np.log(rho) + grad
    p = float(np.sum(rho)) + float(rho @ grad) - f0
    return mu[0], mu[1], p


def bubble_point(tstar: float, x1L: float, model: str,
                 guess: tuple[float, float, float],
                 diam: np.ndarray | None = None,
                 mu_p_fn=None):
    """Solve βμᵢᴸ=βμᵢⱽ, βpᴸ=βpⱽ at fixed T and liquid composition x₁ᴸ.

    Unknowns (log-parameterised to keep densities positive):
        u = [ln ρᴸ, ln ρ₁ⱽ, ln ρ₂ⱽ]
    Returns dict with p, y1, rho_liq, rho_gas, or None if it does not converge.
    """
    if diam is None:
        diam = _diam(tstar)
    if mu_p_fn is None:
        mu_p_fn = mu_p                       # energy route (βf_HS + A)
    xL = np.array([x1L, 1.0 - x1L])

    def resid(u):
        rhoL = np.exp(u[0]) * xL
        rhoV = np.exp(u[1:])
        a = mu_p_fn(rhoL, tstar, model, diam)
        b = mu_p_fn(rhoV, tstar, model, diam)
        if a is None or b is None:
            return np.array([1e3, 1e3, 1e3])
        return np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])

    rL0, r1V0, r2V0 = guess
    u0 = np.log([max(rL0, 1e-8), max(r1V0, 1e-10), max(r2V0, 1e-10)])
    sol = root(resid, u0, method="hybr", options={"xtol": 1e-10})
    if not sol.success or np.max(np.abs(resid(sol.x))) > 1e-6:
        return None
    rhoL = float(np.exp(sol.x[0]))
    rhoV = np.exp(sol.x[1:])
    tot = float(np.sum(rhoV))
    # Reject the trivial root ρᴸ=ρⱽ: it is what the solver returns when the state
    # is above the model's own critical line, i.e. no two-phase state exists.
    if abs(rhoL - tot) / max(rhoL, tot) < 1e-3:
        return {"trivial": True}
    a = mu_p_fn(rhoL * xL, tstar, model, diam)
    return {"p": a[2], "y1": float(rhoV[0] / tot),
            "rho_liq": rhoL, "rho_gas": tot,
            "resid": float(np.max(np.abs(resid(sol.x))))}


# ─── driver ───────────────────────────────────────────────────────────────────

QTY = ("p", "y1", "rho_liq", "rho_gas")


def _table_I() -> list[dict]:
    """Reference rows.  Measured values are prefixed `ref_` so the model's own
    predictions can sit beside them without colliding."""
    with open(REFJSON) as fh:
        ref = json.load(fh)
    tab = ref["coexistence_table_I"]
    out = []
    for key, T in (("T0.90", 0.90), ("T1.00", 1.00), ("T1.10", 1.10)):
        for row in tab[key]:
            p, x1, y1, rl, rg = row
            out.append(dict(T=T, x1=x1, ref_p=p, ref_y1=y1,
                            ref_rho_liq=rl, ref_rho_gas=rg))
    return out


def compute(model: str, rows_wanted: set[int] | None = None) -> None:
    """One JSON per (model, tie line) so tie lines can be run in parallel."""
    os.makedirs(CACHE, exist_ok=True)
    rows = _table_I()
    for k, r in enumerate(rows):
        if rows_wanted is not None and k not in rows_wanted:
            continue
        t0 = time.time()
        guess = (r["ref_rho_liq"], r["ref_rho_gas"] * r["ref_y1"],
                 r["ref_rho_gas"] * (1.0 - r["ref_y1"]))
        out = bubble_point(r["T"], r["x1"], model, guess)
        triv = bool(out and out.get("trivial"))
        rec = {**r, "model": model, "row": k,
               **({} if out is None or triv else out),
               "ok": out is not None and not triv, "trivial": triv}
        with open(os.path.join(CACHE, f"{model}__{k:02d}.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
        s = ("FAIL" if out is None else
             "supercritical for this model (trivial root)" if triv else
             f"p={out['p']:.5f} y1={out['y1']:.4f} "
             f"rl={out['rho_liq']:.4f} rg={out['rho_gas']:.5f}")
        print(f"  [{k+1:2d}/{len(rows)}] T*={r['T']:.2f} x1={r['x1']:.4f}  {s}"
              f"  ({time.time()-t0:.1f}s)", flush=True)


def _load(model: str) -> list[dict]:
    """Merge the per-tie-line caches, in table order."""
    out = []
    for k in range(len(_table_I())):
        p = os.path.join(CACHE, f"{model}__{k:02d}.json")
        if os.path.exists(p):
            out.append(json.load(open(p)))
    return out


def render() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "legend.fontsize": 12, "axes.titlesize": 16})

    data = {}
    for m in MODELS:
        rows = _load(m)
        if rows:
            data[m] = rows
    if not data:
        print("no VLE cache yet")
        return

    lines = ["# Vapour–liquid equilibrium vs Potoff & Panagiotopoulos 1998 Table I\n",
             "Bubble points: T\\* and the liquid composition x₁ᴸ are taken from the "
             "table; the model\nsolves βμᵢᴸ=βμᵢⱽ, βpᴸ=βpⱽ and predicts p\\*, y₁, "
             "ρ\\*_liq, ρ\\*_gas.\n"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    Ts = [0.90, 1.00, 1.10]
    summary = {m: {k: [] for k in ("p", "y1", "rho_liq", "rho_gas")} for m in data}

    for ax, T in zip(axes, Ts):
        ref_rows = [r for r in next(iter(data.values())) if r["T"] == T]
        ax.plot([r["x1"] for r in ref_rows], [r["ref_p"] for r in ref_rows],
                "ks", ms=9, label="P&P 1998 (liquid)")
        ax.plot([r["ref_y1"] for r in ref_rows], [r["ref_p"] for r in ref_rows],
                "kD", ms=8, mfc="none", label="P&P 1998 (vapour)")
        for m, rows in data.items():
            col, ls, lw = STYLE[m]
            sel = [r for r in rows if r["T"] == T and r.get("ok")]
            if not sel:
                continue
            ax.plot([r["x1"] for r in sel], [r["p"] for r in sel],
                    "-o", color=col, lw=lw, ms=5, label=f"{m} (liquid)")
            ax.plot([r["y1"] for r in sel], [r["p"] for r in sel],
                    ":s", color=col, lw=lw, ms=5, mfc="none",
                    label=f"{m} (vapour)")
        ax.set_xlabel("x₁, y₁"); ax.set_ylabel("p*")
        ax.set_title(f"T* = {T:.2f}")
        ax.grid(alpha=0.3)
        if T == 0.90:
            ax.legend(fontsize=10)

    for m, rows in data.items():
        for r in rows:
            if not r.get("ok"):
                continue
            for k in QTY:
                ref = float(r["ref_" + k])
                summary[m][k].append((r[k] - ref) / ref)

    lines.append("| model | converged | mean \\|Δp\\|/p | mean \\|Δy₁\\|/y₁ | "
                 "mean \\|Δρ_liq\\|/ρ | mean \\|Δρ_gas\\|/ρ |")
    lines.append("|---|---|---|---|---|---|")
    for m, rows in data.items():
        n_ok = sum(1 for r in rows if r.get("ok"))
        cells = ["—" if not summary[m][k] else
                 f"{np.mean(np.abs(summary[m][k])):.1%}" for k in QTY]
        lines.append(f"| {m} | {n_ok}/{len(rows)} | " + " | ".join(cells) + " |")

    # Same tie lines for every model — the table above averages each model over
    # its own converged subset, and OZ-HNCB's is the easiest one.
    common = [i for i in range(len(next(iter(data.values()))))
              if all(data[m][i].get("ok") for m in data)]
    if common and any(len(common) != sum(1 for r in rows if r.get("ok"))
                      for rows in data.values()):
        lines.append(f"\n**Restricted to the {len(common)} tie lines every model "
                     "converged on:**\n")
        lines.append("| model | mean \\|Δp\\|/p | mean \\|Δy₁\\|/y₁ | "
                     "mean \\|Δρ_liq\\|/ρ | mean \\|Δρ_gas\\|/ρ |")
        lines.append("|---|---|---|---|---|")
        for m in data:
            cells = []
            for k in QTY:
                v = [abs((data[m][i][k] - data[m][i]["ref_" + k])
                         / data[m][i]["ref_" + k]) for i in common]
                cells.append(f"{np.mean(v):.1%}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines.append("\n## Per tie line\n")
    lines.append("| T\\* | x₁ᴸ | quantity | P&P | " + " | ".join(data) + " |")
    lines.append("|---|---|---|---|" + "---|" * len(data))
    ref_rows = next(iter(data.values()))
    for i, rr in enumerate(ref_rows):
        for k, lbl in (("p", "p\\*"), ("y1", "y₁"),
                       ("rho_liq", "ρ\\*_liq"), ("rho_gas", "ρ\\*_gas")):
            cells = ["—" if not data[m][i].get("ok") else f"{data[m][i][k]:.4f}"
                     for m in data]
            lines.append(f"| {rr['T']:.2f} | {rr['x1']:.4f} | {lbl} | "
                         f"{float(rr['ref_' + k]):.4f} | " + " | ".join(cells) + " |")

    fig.suptitle("Vapour–liquid equilibrium, LJ mixture σ₁=σ₂, ε₁=2ε₂ — "
                 "model bubble points vs Potoff & Panagiotopoulos 1998 Table I",
                 fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "vle_table1.png"), dpi=150,
                bbox_inches="tight")
    with open(os.path.join(OUTDIR, "vle_table1_tables.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", os.path.join(OUTDIR, "vle_table1.png"))


if __name__ == "__main__":
    if "--model" in sys.argv:
        want = None
        if "--rows" in sys.argv:
            want = {int(v) for v in sys.argv[sys.argv.index("--rows") + 1].split(",")}
        compute(sys.argv[sys.argv.index("--model") + 1], want)
    if "--render" in sys.argv:
        render()
