r"""fmsa_vdw_loop_family.py — the FMSA-DP loop across temperature, and the two
lines it generates.

`fmsa_vdw_loop.py` prices ONE isotherm, at `0.88 T_c`, which is what Fig. 12(d)
of the paper draws.  One isotherm cannot answer the question that panel raises:
is the loop a systematic property of the truncation, or one lucky temperature?

So this sweeps `T/T_c` and, at each temperature, records

  * the isotherm `βp(ρ)` -- the loop itself,
  * its two turning points `∂βp/∂ρ = 0` -- the **spinodal**, taken from the
    extrema of the very curves being plotted rather than computed separately,
  * the Maxwell/common-tangent pair `(ρ_g, ρ_l)` -- the **binodal**.

⭐ THE SPINODAL IS DERIVED FROM THE LOOP, NOT ALONGSIDE IT.  That is the whole
design: a spinodal drawn as its own curve is an unfalsifiable line (no
simulation measures one, and FMSA-DP's is known to sit 12--30 % high).  Read off
the turning points of the plotted isotherms it is instead a *construction line*
-- manifestly consistent with the panel above it, and free.

⚠ Compressibility route, matching Fig. 12(b)-(d).  `T_c` here is the
compressibility-route `T_c` of `fmsa_phase_lines.json`, the same normalisation
`fmsa_vdw_loop.py` uses, so `T_FRAC = 0.88` reproduces that file exactly.

⚠ The exact-MSA companion needs no sweep: by the perfect-square identity its
`∂βp/∂ρ = (A/2π)^2` is non-negative at EVERY temperature, so there is no family
to draw -- which is the asymmetry the figure exists to show.

Run:  python fmsa_vdw_loop_family.py
"""
from __future__ import annotations

import json
import warnings

warnings.filterwarnings('ignore')

import numpy as np

import fmsa_phase_lines as FP
from fmsa_vdw_loop import coexist_exact

OUT = 'numerical_notes/results/compare_energy_compress/phase_lines'
JSON = OUT + '/fmsa_vdw_loop_family.json'

# Fractions of the system's own compressibility-route T_c.  0.88 is the
# isotherm Fig. 12(d) already shows, kept so the family contains the panel it
# generalises; the top of the range sits close enough to T_c that the loop is
# visibly closing rather than absent.
T_FRAC = (0.80, 0.84, 0.88, 0.92, 0.96, 0.99)
ROUTE = 'C'


def compute(system: str = 'HSY') -> dict:
    main = json.load(open(FP.JSON))
    sd = FP.SYS[system]
    Tc = main[system][ROUTE]['Tc']
    print(f'[{system}] compressibility-route T_c = {Tc:.4f}')

    rec = dict(system=system, label=sd['label'], route=ROUTE, Tc=float(Tc),
               isotherms=[])
    for frac in T_FRAC:
        T = frac * Tc
        rho, rr = FP.isotherm(sd, T, routes=(ROUTE,))
        if rho is None:
            print(f'  T/Tc={frac:.2f}  isotherm failed')
            continue
        a = rr[ROUTE]
        p, mu = FP.p_mu(rho, a)
        sp = FP.spinodal_of(rho, a)
        cx = coexist_exact(rho, a)
        row = dict(T_frac=float(frac), T=float(T),
                   rho=rho.tolist(), p=p.tolist(), a=a.tolist(),
                   spinodal=list(sp) if sp else None,
                   min_a=float(a.min()))
        if cx is not None:
            row.update(rho_g=cx['rho_g'], rho_l=cx['rho_l'],
                       p_coex=cx['p_coex'], gate=cx['gate'])
        rec['isotherms'].append(row)
        print(f'  T/Tc={frac:.2f}  T*={T:.4f}  min dbp/drho={a.min():+.4f}'
              f"  spinodal={('%.4f..%.4f' % sp) if sp else 'none':>17}"
              f"  coex={('%.4f..%.4f' % (cx['rho_g'], cx['rho_l'])) if cx else 'none':>17}"
              f"  gate={cx['gate']:.1e}" if cx else '')
    with open(JSON, 'w') as fh:
        json.dump(rec, fh)
    print(f'[saved] {JSON}  ({len(rec["isotherms"])} isotherms)')
    return rec


if __name__ == '__main__':
    compute()
