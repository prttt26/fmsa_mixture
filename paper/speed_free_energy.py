"""The free-energy speed gap, at two granularities.

The existing benchmark (`compare_speed_dp_hncb.py`) prices ONE task: build and
evaluate c_ij(r) on a grid.  The thermodynamic sections need a different one --
the free energy and its gradient -- and the ratio is NOT the same, so it has to
be measured rather than carried over.

Two granularities, because they answer different questions and differ by ~2x:

  A_of      one cold free-energy evaluation      -> the inner-loop ratio
  mu_p      (mu_1, mu_2, p): 5 evaluations       -> what a phase-equilibrium
                                                    residual actually costs

⚠ `A_of` memoises.  Every timing below clears `_A_CACHE` first; without that the
second model to run inherits the first's warm cache and the ratio is fiction.

⚠ Run on a quiet machine.  Competing load corrupts wall-clock ratios -- this is
why it is not run while the binodal sweep is in flight.
"""
import json
import os
import sys
import time
import importlib.util

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("pe", "phase_equilibrium.py")
pe = importlib.util.module_from_spec(spec)
sys.modules["pe"] = pe
spec.loader.exec_module(pe)

SCRATCH = ("/tmp/claude-1002/-home-rpan-YK-tail-FMSA/"
           "26f8ac35-3cf1-4b01-8052-afbc0c9f5d44/scratchpad")
BINODAL = os.path.join(SCRATCH, "binodal_x085.json")
OUT = os.path.join(SCRATCH, "speed_free_energy.json")
REPEATS = 3


def states():
    """State points spanning the regime the thermodynamic sections actually use.

    The eight points of the existing speed benchmark are all ordinary fluid
    states; NONE sits near a phase boundary, which is exactly where an iterative
    solver is slowest.  So the coexisting densities traced by the binodal sweep
    are included here -- they are on the boundary by construction.
    """
    out = [("dilute", np.array([0.010 * 0.85, 0.010 * 0.15]), 1.0),
           ("moderate", np.array([0.30 * 0.85, 0.30 * 0.15]), 1.0),
           ("dense", np.array([0.70 * 0.85, 0.70 * 0.15]), 1.0),
           ("supercritical", np.array([0.40 * 0.85, 0.40 * 0.15]), 2.0)]
    if os.path.exists(BINODAL):
        b = json.load(open(BINODAL))
        rows = b.get("FMSA-DP", [])
        for r in rows[::max(1, len(rows) // 4)][:4]:
            t, rl = r["T"], r["rho_liq"]
            out.append((f"coex liquid T*={t:.2f}",
                        np.array([rl * 0.85, rl * 0.15]), t))
            rv, y = r["rho_gas"], r["y1"]
            out.append((f"coex vapour T*={t:.2f}",
                        np.array([rv * y, rv * (1.0 - y)]), t))
    return out


def timeit(fn):
    best = float("inf")
    for _ in range(REPEATS):
        pe._A_CACHE.clear()                 # cold every time -- see header
        t0 = time.perf_counter()
        v = fn()
        dt = time.perf_counter() - t0
        if v is None:
            return None, None
        best = min(best, dt)                # min, not mean: noise is one-sided
    return best, v


if __name__ == "__main__":
    res = []
    for label, rho, T in states():
        d = pe._diam(T)
        row = {"state": label, "T": T, "rho": float(np.sum(rho))}
        for m in ("FMSA-DP", "OZ-MSA", "OZ-HNCB"):
            ta, _ = timeit(lambda: pe.A_of(rho, T, m))
            tm, _ = timeit(lambda: pe.mu_p(rho, T, m, d))
            row[m] = {"A": ta, "mu_p": tm}
            print(f"  {label:24s} {m:8s} A={ta if ta is None else f'{ta:.4f}'}"
                  f"  mu_p={tm if tm is None else f'{tm:.4f}'}", flush=True)
        res.append(row)

    print("\n=== ratios against FMSA-DP ===", flush=True)
    for key in ("A", "mu_p"):
        for m in ("OZ-MSA", "OZ-HNCB"):
            r = [x[m][key] / x["FMSA-DP"][key] for x in res
                 if x[m][key] and x["FMSA-DP"][key]]
            if r:
                print(f"  {key:5s} {m:8s} median {np.median(r):6.1f}x  "
                      f"range {min(r):.1f}-{max(r):.1f}x  (n={len(r)})",
                      flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)
