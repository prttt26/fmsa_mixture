#!/usr/bin/env python3
"""Assemble every benchmark source into one DCF pool with an honest fidelity grade.

The model comparison wants one thing from every source: c_ij(r) on an absolute-r
grid, together with the state point that produced it and a statement of how much
of that curve can be believed.  The sources do not agree on any of this --
TN 1331 tabulates c directly but on a per-pair reduced abscissa, its RDF tables
give g and need an OZ inversion that its own truncation spoils, our MD writes
absolute r, and Yu & Jin label the LARGE species 1 where everybody else labels
it 2.  This script normalises all of that and writes one index.

What goes in
------------
tn1331/dcf/*.csv        39 tables, already DCF.  fidelity `direct`.
tn1331/rdf/*.csv        54 tables of g; OZ-inverted here via convert.py.
                        fidelity `inverted_truncated` -- the published g stops at
                        R* = 2.5, well short of the h the report itself inverted,
                        and the inverse transform amplifies what is missing
                        (~450x onto chat22(0) at rho*=0.139).  Usable in the core,
                        not in the tail.
converted/md_*/c.csv    our own LAMMPS MD, g written to ~half the box, so the
                        inversion is not truncation-limited.  fidelity `inverted`.
                        The only source at size ratios other than 2.
literature_direct/rdf/  Yu & Jin Fig. 13, colour-separated from the figure.
                        fidelity `digitized`, and see the warning below.

The outer region is graded, not assumed
---------------------------------------
numerical_notes/rootcause/c22_outer_inversion_artifact.md showed the TN 1331
outer c22 excess is an artifact of that report's own OZ inversion, density-driven
and absent below eta ~ 0.25, and measured it with the tail scale
s_ij = fit of c_ij to -beta*phi_ij over r in [1.6, 2.5] sigma_ij, which theory
puts at 1.

That statistic is computed here for EVERY source, and `outer_trust` is simply
s_ij in [0.5, 2.0] on all three pairs -- the tail is right to within a factor of
two.  No source is trusted a priori, and that matters: our own MD, with its own
g and its own inversion, reproduces the same defect above eta ~ 0.43
(s22 = 2.1-2.6 at eta = 0.455, and s running to +45 / -25 at eta = 0.50).  The
amplification of far-field error in h is a property of inverting OZ at high
packing, not of Huber and Ely's pipeline.

For the same reason every source is scored only out to R_SCORE_MAX = 2.5
sigma_ij.  TN 1331 stops there anyway; our MD reaches 6.7-13.5 sigma_11, but that
far field is exactly where the finite-size error lives, and under the r^2 weight
of chat(0) it swamps the physics (chat22(0) for md_ratio3.0 runs from -679 at a
cutoff of 8 sigma_11 to +1207 at 13.5).  One common cutoff makes the sources
comparable and keeps the noise out.

Yu & Jin Fig. 13 carries a further caveat that this script records rather than
repairs: the greyscale figure defeats colour separation in places, so the traced
g11 has values inside its own hard core and all three pairs pick up spurious
excursions past r ~ 1.8.  Only the rigorous in-core mask (g = 0 for r < sigma_ij,
exact for a hard-core model) is applied.  The entry is kept, flagged
`digitized`, with outer_trust False.

Usage
-----
    python build_dcf_pool.py                 # build, reusing existing conversions
    python build_dcf_pool.py --force         # redo every conversion
    python build_dcf_pool.py --skip-convert  # index only what is already there
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from convert import read_table
from structure_transforms import PAIRS, PAIR_TAG, Correlations, make_grid

HERE = Path(__file__).parent
TN = HERE / 'tn1331'
CONVERTED = HERE / 'converted'
LIT = HERE / 'literature_direct'
MD_RUNS = HERE / 'md' / 'runs'
POOL = HERE / 'dcf_pool'

FIELDS = [
    'label', 'source', 'potential', 'sigma_ratio', 'eps_ratio', 'T_star',
    'x1', 'rho_star', 'eta', 'c_path', 'r_convention',
    'r_max_11', 'r_max_12', 'r_max_22',
    'fidelity', 'quality', 'n_suspect', 'suspect_key',
    's_tail_11', 's_tail_12', 's_tail_22', 'outer_trust',
    'usable', 'skip_reason',
]

# Scored range, in units of each pair's own sigma_ij.  TN 1331 stops at 2.5 and
# our MD reaches 6.7-13.5, but the far field of ANY g -> c inversion is where
# finite-size error in h gets amplified; capping every source at the same 2.5
# both removes that and makes the sources directly comparable.
R_SCORE_MAX = 2.5
TAIL_WINDOW = (1.6, 2.5)        # same window as c22_outer_diagnosis.py
TAIL_BAND = (0.5, 2.0)          # 'the tail is right to within a factor of two'


def sig_mix_of(ratio: float) -> np.ndarray:
    """Lorentz-Berthelot pair diameters for sigma11 = 1, sigma22 = ratio."""
    s = np.array([1.0, float(ratio)])
    return 0.5 * (s[:, None] + s[None, :])


def eta_of(rho: float, x1: float, ratio: float) -> float:
    """Packing fraction (pi/6) sum_i rho_i sigma_ii^3, general in the size ratio.

    Uses the bare sigma_ii (not a Barker-Henderson diameter) so the bins here
    match those of c22_outer_diagnosis.py and the rootcause note.
    """
    return (np.pi / 6.0) * rho * (x1 * 1.0 + (1.0 - x1) * float(ratio) ** 3)


def _row(**kw) -> dict:
    row = {f: '' for f in FIELDS}
    row.update(kw)
    return row


def _grade_outer(cpath: Path, ratio: float, eps_ratio: float, T: float,
                 per_pair: bool, r_hi_pair: dict) -> dict:
    """Tail scales, the outer_trust verdict, and the capped per-pair r range."""
    s = tail_scales(cpath, ratio, eps_ratio, T, per_pair)
    lo, hi = TAIL_BAND
    ok = all(np.isfinite(s[t]) and lo <= s[t] <= hi for t in ('11', '12', '22'))
    sm = sig_mix_of(ratio)
    caps = {'11': (0, 0), '12': (0, 1), '22': (1, 1)}
    return dict(
        s_tail_11=f"{s['11']:.3f}", s_tail_12=f"{s['12']:.3f}",
        s_tail_22=f"{s['22']:.3f}", outer_trust=int(ok),
        **{f'r_max_{t}': f'{min(r_hi_pair[t], R_SCORE_MAX * float(sm[p])):.4f}'
           for t, p in caps.items()})


def tail_scales(cpath: Path, ratio: float, eps_ratio: float, T: float,
                per_pair: bool) -> dict:
    """s_ij = best fit of c_ij to -beta*phi_ij over r in [1.6, 2.5] sigma_ij.

    The statistic c22_outer_diagnosis.py used to show the TN 1331 outer excess
    is density-driven.  Theory says s -> 1 there; how far it strays measures how
    much of the tail is inversion artifact rather than physics.  Measured the
    same way for every source, so no source is trusted a priori.
    """
    tab = read_table(cpath)
    sm = sig_mix_of(ratio)
    E = {'11': 1.0, '12': float(np.sqrt(eps_ratio)), '22': float(eps_ratio)}
    rkey = 'R_star' if per_pair else 'r'
    out = {}
    for (i, j) in PAIRS:
        tag = PAIR_TAG[(i, j)]
        s = float(sm[i, j])
        r = tab[rkey] * (s if per_pair else 1.0)
        c = tab[f'c{tag}']
        m = (r >= TAIL_WINDOW[0] * s) & (r <= TAIL_WINDOW[1] * s) & np.isfinite(c)
        if m.sum() < 4:
            out[tag] = float('nan')
            continue
        x = s / r[m]
        ref = -4.0 * E[tag] * (x ** 12 - x ** 6) / T            # -beta*phi_ij
        out[tag] = float(np.sum(c[m] * ref) / np.sum(ref * ref))
    return out


def core_is_physical(cpath: Path) -> str:
    """'' if the inverted c has a repulsive core, else why it is rejected.

    Any fluid whose particles cannot overlap has c_ii(r -> 0) large and
    NEGATIVE.  An inversion that returns a positive like-pair core has been
    destroyed by whatever went into it -- a truncated or contaminated h -- and
    no amount of downstream care recovers it.  Applied to every inverted entry
    so the gate is uniform, not a judgement about one source.
    """
    try:
        tab = read_table(cpath)
    except Exception as exc:                                # unreadable / partial
        return f'unreadable c.csv: {exc}'
    r = tab['r']
    inner = r <= max(r[4], 0.05)                            # a few points at the origin
    for tag in ('11', '22'):
        v = tab[f'c{tag}'][inner]
        if v.size == 0 or not np.all(np.isfinite(v)):
            return f'c{tag} core is empty or non-finite'
        if float(np.mean(v)) > 0.0:
            return (f'unphysical inversion: c{tag}(r->0) = {float(np.mean(v)):+.1f} > 0 '
                    f'(a repulsive core requires c < 0)')
    return ''


# --------------------------------------------------------------------------
# TN 1331 -- the 39 tables that are already DCF
# --------------------------------------------------------------------------
def tn1331_dcf_entries() -> list[dict]:
    out = []
    for e in csv.DictReader((TN / 'index.csv').open()):
        if e['kind'] != 'dcf':
            continue
        stem = Path(e['file']).stem
        path = TN / e['file']
        rho = e['rho_star']
        if not rho:
            out.append(_row(label=f'tn1331dcf_{stem}', source='tn1331_dcf',
                            potential='lj', sigma_ratio=2.0, eps_ratio=e['eps'],
                            T_star=e['tstar'], x1=e['x1'], fidelity='direct',
                            quality=e['quality'], n_suspect=e['n_suspect'],
                            usable=0, skip_reason='no rho_star in index.csv'))
            continue
        rho, x1 = float(rho), float(e['x1'])
        eta = eta_of(rho, x1, 2.0)
        tab = read_table(path)
        rstar_max = float(tab['R_star'][-1])
        sm = sig_mix_of(2.0)
        r_hi = {t: rstar_max * float(sm[p]) for t, p in
                (('11', (0, 0)), ('12', (0, 1)), ('22', (1, 1)))}
        out.append(_row(
            label=f'tn1331dcf_{stem}', source='tn1331_dcf', potential='lj',
            sigma_ratio=2.0, eps_ratio=e['eps'], T_star=e['tstar'], x1=x1,
            rho_star=rho, eta=f'{eta:.4f}', c_path=str(path.relative_to(HERE)),
            r_convention='perpair',
            fidelity='direct', quality=e['quality'], n_suspect=e['n_suspect'],
            suspect_key=e['file'], usable=1,
            **_grade_outer(path, 2.0, float(e['eps']), float(e['tstar']),
                           True, r_hi)))
    return out


# --------------------------------------------------------------------------
# TN 1331 -- the 54 RDF tables, OZ-inverted through convert.py
# --------------------------------------------------------------------------
def tn1331_rdf_entries(force: bool, skip_convert: bool) -> list[dict]:
    out = []
    for e in csv.DictReader((TN / 'index.csv').open()):
        if e['kind'] != 'rdf':
            continue
        stem = Path(e['file']).stem
        label = f'tn1331rdf_{stem}'
        rho = e['rho_star']
        if not rho:
            out.append(_row(label=label, source='tn1331_rdf', potential='lj',
                            sigma_ratio=2.0, eps_ratio=e['eps'], T_star=e['tstar'],
                            x1=e['x1'], fidelity='inverted_truncated',
                            quality=e['quality'], n_suspect=e['n_suspect'],
                            usable=0,
                            skip_reason='no rho_star; g->c needs densities'))
            continue
        rho, x1 = float(rho), float(e['x1'])
        cpath = CONVERTED / label / 'c.csv'
        if (force or not cpath.exists()) and not skip_convert:
            cmd = [sys.executable, str(HERE / 'convert.py'), '--state', stem,
                   '--from', 'g', '--to', 'c', '--label', label,
                   '--rho', repr(rho), '--x1', repr(x1)]
            res = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
            if res.returncode != 0:
                out.append(_row(label=label, source='tn1331_rdf', potential='lj',
                                sigma_ratio=2.0, eps_ratio=e['eps'],
                                T_star=e['tstar'], x1=x1, rho_star=rho,
                                fidelity='inverted_truncated',
                                quality=e['quality'], n_suspect=e['n_suspect'],
                                usable=0,
                                skip_reason=f'convert.py failed: '
                                            f'{res.stderr.strip().splitlines()[-1:]}'))
                continue
        if not cpath.exists():
            out.append(_row(label=label, source='tn1331_rdf', potential='lj',
                            sigma_ratio=2.0, eps_ratio=e['eps'], T_star=e['tstar'],
                            x1=x1, rho_star=rho, fidelity='inverted_truncated',
                            quality=e['quality'], n_suspect=e['n_suspect'],
                            usable=0, skip_reason='no c.csv (conversion skipped)'))
            continue
        eta = eta_of(rho, x1, 2.0)
        # the inversion output spans the whole grid, but only r <= R*_max*sigma_ij
        # is supported by the source g; past that it is inversion artifact.
        rstar_max = float(read_table(TN / e['file'])['R_star'][-1])
        sm = sig_mix_of(2.0)
        bad = core_is_physical(cpath)
        if bad:
            out.append(_row(label=label, source='tn1331_rdf', potential='lj',
                            sigma_ratio=2.0, eps_ratio=e['eps'], T_star=e['tstar'],
                            x1=x1, rho_star=rho, eta=f'{eta:.4f}',
                            fidelity='inverted_truncated', quality=e['quality'],
                            n_suspect=e['n_suspect'], usable=0, skip_reason=bad))
            continue
        r_hi = {t: rstar_max * float(sm[p]) for t, p in
                (('11', (0, 0)), ('12', (0, 1)), ('22', (1, 1)))}
        grade = _grade_outer(cpath, 2.0, float(e['eps']), float(e['tstar']),
                             False, r_hi)
        grade['outer_trust'] = 0          # truncated source g; never trusted
        out.append(_row(
            label=label, source='tn1331_rdf', potential='lj', sigma_ratio=2.0,
            eps_ratio=e['eps'], T_star=e['tstar'], x1=x1, rho_star=rho,
            eta=f'{eta:.4f}', c_path=str(cpath.relative_to(HERE)),
            r_convention='absolute',
            fidelity='inverted_truncated', quality=e['quality'],
            n_suspect=e['n_suspect'], suspect_key=e['file'],
            usable=1, **grade))
    return out


# --------------------------------------------------------------------------
# our own MD
# --------------------------------------------------------------------------
def _two_phase(run: str) -> str:
    """'' if the run's own g_ij comes back toward 1, else why it does not.

    Read from g.csv rather than from the inverted c, because demixing is a
    STATEMENT ABOUT LONG RANGE and the inversion is least trustworthy exactly
    there.  A mean outside [0.5, 1.5] over the outer third is not the
    oscillation of a dense liquid, it is a two-phase state -- and a two-phase
    state cannot score a single-phase theory whatever its c looks like.  Same
    test, same numbers, as the RDF pool's `_grade`.
    """
    g = MD_RUNS / run / 'g.csv'
    if not g.exists():
        return ''
    try:
        tab = read_table(g)
    except Exception:
        return ''
    r = tab['r']
    m = r > 0.6 * float(r[-1])
    if not m.any():
        return ''
    plat = [float(np.mean(tab[f'g{t}'][m])) for t in ('11', '12', '22')]
    if max(plat) > 1.5 or min(plat) < 0.5:
        return ('two-phase: g plateau = '
                + ';'.join(f'{v:.2f}' for v in plat) + ', not ~1')
    return ''


def md_entries() -> list[dict]:
    out = []
    for d in sorted(CONVERTED.glob('md_*')):
        cpath = d / 'c.csv'
        man = d / 'manifest.json'
        if not (cpath.exists() and man.exists()):
            continue
        m = json.loads(man.read_text())
        run = Path(m['input']).parent.name
        pj = MD_RUNS / run / 'params.json'
        if not pj.exists():
            continue
        p = json.loads(pj.read_text())
        rho, x1, ratio = float(p['rho']), float(p['x1']), float(p['ratio'])
        pot = 'ljhs' if p.get('kind') == 'ljhs' else 'lj'
        r_out = float(m['grid']['r_out'])
        # a run still being written has a short or empty c.csv
        try:
            n_rows = len(read_table(cpath)['r'])
        except Exception:
            continue
        if n_rows < 50:
            continue
        bad = core_is_physical(cpath) or _two_phase(run)
        if bad:
            out.append(_row(label=f'md_{run}', source='md', potential=pot,
                            sigma_ratio=ratio, eps_ratio=p['eps_ratio'],
                            T_star=p['T'], x1=x1, rho_star=rho,
                            eta=f'{eta_of(rho, x1, ratio):.4f}',
                            fidelity='inverted', quality='MD', n_suspect=0,
                            usable=0, skip_reason=bad))
            continue
        grade = _grade_outer(cpath, ratio, float(p['eps_ratio']), float(p['T']),
                             False, {t: r_out for t in ('11', '12', '22')})
        if pot == 'ljhs':
            # The gate is s_ij = fit of c_ij to -beta*phi_ij, and for a hard
            # sphere phi_12 = phi_22 = 0 identically -- the ratio is 0/0, and
            # scoring those pairs outside the core would only be measuring the
            # inversion's noise against a reference that every mean-field model
            # reproduces exactly by construction.  So the LJ-HS block is scored
            # in the CORE on c, and over the full range on g (the RDF study,
            # which has no such gate).  Deliberate, not a 0/0 accident.
            grade['outer_trust'] = 0
        out.append(_row(
            label=f'md_{run}', source='md', potential=pot, sigma_ratio=ratio,
            eps_ratio=p['eps_ratio'], T_star=p['T'], x1=x1, rho_star=rho,
            eta=f'{eta_of(rho, x1, ratio):.4f}',
            c_path=str(cpath.relative_to(HERE)), r_convention='absolute',
            fidelity='inverted', quality='MD', n_suspect=0, usable=1, **grade))
    return out


# --------------------------------------------------------------------------
# Yu & Jin Fig. 13 -- HCTY, and the species labels are the other way round
# --------------------------------------------------------------------------
YUJIN_FIG13 = dict(
    # Yu & Jin, JCP 128, 014901 (2008), Fig. 13 = Table II case 2 at
    # T* = 6.0, rho* = 1.3, x_large = 0.5.  Their species 1 is the LARGE one
    # (sigma2/sigma1 = 0.5); we relabel so species 2 is large, as everywhere else.
    T_star=6.0, rho_star=1.3, x_large=0.5, ratio=2.0, eps_ratio=0.5,
    lam=(1.8, 8.0), kset=(-6.0, -6.0),
)


def yujin_entry(force: bool, skip_convert: bool) -> list[dict]:
    src = LIT / 'rdf' / 'yu_jin_2008_fig13.csv'
    label = 'yujin_fig13'
    p = YUJIN_FIG13
    if not src.exists():
        return []
    # our labelling: species 1 small (sigma11 = 1), species 2 large (sigma22 = ratio),
    # in units of the SMALL diameter.  Yu & Jin quote r in units of their large one.
    ratio = p['ratio']
    sm = sig_mix_of(ratio)
    x1 = 1.0 - p['x_large']                     # our x1 = their x_small
    rho = p['rho_star'] * ratio ** 3            # rho sigma_small^3 from rho sigma_large^3
    cpath = CONVERTED / label / 'c.csv'
    if (force or not cpath.exists()) and not skip_convert:
        tab = read_table(src)
        r_src = tab['r'] * ratio                # their sigma_large -> our sigma_small
        swapped = {                             # their 1 (large) is our 2
            (0, 0): tab['g22'], (0, 1): tab['g12'], (1, 1): tab['g11'],
        }
        r, k = make_grid(24.0, 4096)
        g = np.zeros((2, 2, r.size))
        for (i, j), col in swapped.items():
            ok = np.isfinite(col)
            v = np.zeros(r.size)
            inside = (r >= r_src[ok][0]) & (r <= r_src[ok][-1])
            v[inside] = np.interp(r[inside], r_src[ok], col[ok])
            v[r > r_src[ok][-1]] = 1.0
            v[r < sm[i, j]] = 0.0               # rigorous: hard core
            g[i, j] = g[j, i] = v
        rho_vec = np.array([x1, 1.0 - x1]) * rho
        corr = Correlations.from_g(g, r, k, rho_vec)
        c = corr.c
        cpath.parent.mkdir(parents=True, exist_ok=True)
        with cpath.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['r'] + [f'c{PAIR_TAG[q]}' for q in PAIRS])
            for n in np.nonzero(r <= float(r_src[-1]))[0]:
                w.writerow([f'{r[n]:.4f}'] + [f'{c[i, j, n]:.6f}' for i, j in PAIRS])
        (cpath.parent / 'manifest.json').write_text(json.dumps({
            'input': str(src), 'from': 'g', 'r_convention': 'absolute',
            'rho_total': rho, 'x1': x1,
            'note': 'Yu & Jin Fig. 13, species relabelled (their 1 = large -> our 2) '
                    'and r rescaled from their sigma_large to our sigma_small. '
                    'Only the rigorous in-core mask g=0 for r<sigma_ij is applied; '
                    'the trace still carries cross-curve contamination past r~1.8.',
        }, indent=2))
    if not cpath.exists():
        return []
    r_hi = float(read_table(src)["r"][-1]) * ratio
    bad = core_is_physical(cpath)
    if bad:
        return [_row(label=label, source='yujin_fig13', potential='hcty',
                     sigma_ratio=ratio, eps_ratio=p['eps_ratio'],
                     T_star=p['T_star'], x1=x1, rho_star=rho,
                     eta=f'{eta_of(rho, x1, ratio):.4f}', fidelity='digitized',
                     quality='digitized-approx', n_suspect=0, usable=0,
                     skip_reason=bad)]
    grade = _grade_outer(cpath, ratio, float(p['eps_ratio']), float(p['T_star']),
                         False, {t: r_hi for t in ('11', '12', '22')})
    grade['outer_trust'] = 0
    return [_row(
        label=label, source='yujin_fig13', potential='hcty', sigma_ratio=ratio,
        eps_ratio=p['eps_ratio'], T_star=p['T_star'], x1=x1, rho_star=rho,
        eta=f'{eta_of(rho, x1, ratio):.4f}',
        c_path=str(cpath.relative_to(HERE)), r_convention='absolute',
        fidelity='digitized', quality='digitized-approx', n_suspect=0,
        usable=1, **grade)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--force', action='store_true', help='redo every conversion')
    ap.add_argument('--skip-convert', action='store_true',
                    help='index only conversions that already exist')
    args = ap.parse_args()

    POOL.mkdir(parents=True, exist_ok=True)
    rows = (tn1331_dcf_entries()
            + tn1331_rdf_entries(args.force, args.skip_convert)
            + md_entries()
            + yujin_entry(args.force, args.skip_convert))

    out = POOL / 'pool_index.csv'
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    usable = [r for r in rows if r['usable'] == 1]
    print(f'{out}: {len(rows)} entries, {len(usable)} usable')
    for src in ('tn1331_dcf', 'tn1331_rdf', 'md', 'yujin_fig13'):
        sel = [r for r in rows if r['source'] == src]
        u = [r for r in sel if r['usable'] == 1]
        ot = [r for r in u if r['outer_trust'] == 1]
        print(f'  {src:<12} {len(sel):>3} entries  {len(u):>3} usable  '
              f'{len(ot):>3} outer-trusted')
    skipped = [r for r in rows if r['usable'] == 0]
    if skipped:
        print(f'\n  skipped ({len(skipped)}):')
        for r in skipped:
            print(f'    {r["label"]:<38} {r["skip_reason"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
