#!/usr/bin/env python3
"""Audit the NIST TN 1331 Appendix A tables against our own MD, point by point.

Everything else in this repo grades those tables INDIRECTLY -- by whether the
extractor's structural checks pass, or by how much a model's error inflates on
the flagged ones.  This measures them directly: re-simulate the same state point
and difference the two g_ij(R*).  Our MD is validated against TN 1331 itself
(md/README.md), so where the two disagree by much more than the run-to-run
noise, the table is wrong.

The failure mode the scan produces is loss of the leading integer (`1.8442`
printed as `0.8442`), so a corrupted point shows up as |TN - MD| ~ an integer.
That is a signature, not a threshold: a statistical discrepancy wanders, an OCR
one sits on 1.000.

    python audit_tn1331_rdf.py              # every tnaudit_* run present
    python audit_tn1331_rdf.py --repair     # also write repaired tables
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = HERE / 'md' / 'runs'
RDF = HERE / 'tn1331' / 'rdf'
SUSPECT = HERE / 'tn1331' / 'suspect.csv'
REPAIRED = HERE / 'tn1331' / 'rdf_repaired'
SIGMA = {'11': 1.0, '12': 1.5, '22': 2.0}      # size ratio 2, additive
INT_TOL = 0.15        # |dev - nearest integer| below this = a lost leading digit
NOISE = 0.30          # |dev| above this is not statistics at this sampling
MAX_INT = 9           # a lost LEADING digit shifts g by 1-9, no more
CATASTROPHIC = 20.0   # g this far off is not a fluid: a lost decimal point etc.

# Two failure modes, and they need different treatment:
#   lost_int      |dev| sits on an integer 1-9.  The value is well formed, lands
#                 on the report's own grid, and passes every structural check --
#                 this is the dangerous one, and it is repairable (add the digit
#                 back).
#   catastrophic  |dev| in the hundreds or thousands: a lost decimal point or
#                 digit soup.  Obvious on sight, and already rejected upstream by
#                 the pool's g_max > 20 gate.  Recorded, never repaired.


def read_csv(path: Path) -> dict:
    rows = [l for l in path.open() if not l.startswith('#')]
    rdr = csv.DictReader(rows)
    cols = {n: [] for n in rdr.fieldnames}
    for row in rdr:
        for n, v in row.items():
            cols[n].append(float(v) if v not in ('', None) else np.nan)
    return {n: np.asarray(v) for n, v in cols.items()}


def suspects() -> dict:
    out: dict = {}
    if not SUSPECT.exists():
        return out
    for r in csv.DictReader(SUSPECT.open()):
        out.setdefault((r['file'], r['column']), set()).add(round(float(r['R_star']), 4))
    return out


def audit_one(run: Path, sus: dict) -> dict:
    """One state point: MD vs its Appendix A table, per point."""
    stem = run.name.replace('tnaudit_', '')
    table = RDF / f'{stem}.csv'
    if not table.exists() or not (run / 'g.csv').exists():
        return {}
    return _audit(table, run / 'g.csv', stem, sus)


def repair(run: Path, rec: dict) -> Path | None:
    """Write a repaired copy: corrupted points get their lost integer back."""
    stem = rec['state']
    tn = read_csv(RDF / f'{stem}.csv')
    md = read_csv(run / 'g.csv')
    n_fix = 0
    for tag in ('11', '12', '22'):
        p = rec['pairs'][tag]
        if not p['n_corrupt']:
            continue
        gm = np.interp(tn['R_star'] * SIGMA[tag], md['r'], md[f'g{tag}'])
        dev = tn[f'g{tag}'] - gm
        for R, shift in zip(p['corrupt_R'], p['corrupt_shift']):
            i = int(np.argmin(np.abs(tn['R_star'] - R)))
            tn[f'g{tag}'][i] += shift * np.sign(-dev[i])
            n_fix += 1
    if not n_fix:
        return None
    REPAIRED.mkdir(parents=True, exist_ok=True)
    out = REPAIRED / f'{stem}.csv'
    names = [n for n in ('R_star', 'g11', 'g12', 'g22', 'xl11', 'xl12', 'xl22')
             if n in tn]
    with out.open('w', newline='') as fh:
        fh.write(f'# repaired against MD run tnaudit_{stem}: {n_fix} points had a '
                 'lost leading integer restored\n')
        w = csv.writer(fh)
        w.writerow(names)
        for k in range(len(tn['R_star'])):
            w.writerow([f'{tn[n][k]:.4f}' for n in names])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repair', action='store_true')
    args = ap.parse_args()

    sus = suspects()
    runs = sorted(RUNS.glob('tnaudit_*')) + [RUNS / 'validation']
    recs = []
    for run in runs:
        stem = run.name.replace('tnaudit_', '')
        if run.name == 'validation':
            stem = 'T1_eps0.50_x0.2500'
            if not (RDF / f'{stem}.csv').exists():
                continue
            rec = _audit(RDF / f'{stem}.csv', run / 'g.csv', stem, sus)
        else:
            rec = audit_one(run, sus)
        if rec:
            recs.append((run, rec))

    print('=' * 96)
    print('  TN 1331 Appendix A vs our own MD at the SAME state point')
    print('=' * 96)
    print(f"  {'state':<26} {'pair':>5} {'mean|d|':>8} {'clean':>7} {'max|d|':>7} "
          f"{'lost_int':>8} {'catas':>6} {'flagged':>8} {'caught':>7} {'missed':>7} {'false':>6}")
    tot = dict(corrupt=0, flagged=0, caught=0, missed=0, false=0, catas=0)
    for run, rec in recs:
        for tag in ('11', '12', '22'):
            p = rec['pairs'][tag]
            print(f"  {rec['state']:<26} g{tag:>4} {p['mean_abs']:8.4f} "
                  f"{p['mean_abs_clean']:7.4f} {p['max_abs']:7.3f} "
                  f"{p['n_corrupt']:8d} {p['n_catastrophic']:6d} {p['n_flagged']:8d} "
                  f"{p['n_flag_and_corrupt']:7d} {p['n_corrupt_unflagged']:7d} "
                  f"{p['n_flag_not_corrupt']:6d}")
            tot['corrupt'] += p['n_corrupt']; tot['flagged'] += p['n_flagged']
            tot['catas'] += p['n_catastrophic']
            tot['caught'] += p['n_flag_and_corrupt']
            tot['missed'] += p['n_corrupt_unflagged']
            tot['false'] += p['n_flag_not_corrupt']
    print(f"\n  TOTAL lost-leading-digit points found by MD: {tot['corrupt']}"
          f"   (plus {tot['catas']} catastrophic, already rejected upstream)")
    print(f"  suspect.csv flags: {tot['flagged']}")
    if tot['corrupt']:
        print(f"    caught by a flag   : {tot['caught']:3d}  "
              f"({tot['caught'] / tot['corrupt']:.0%} of the damage)")
        print(f"    missed by the flags: {tot['missed']:3d}")
    print(f"    flags on clean points: {tot['false']:3d}")

    print('\n  corrupted runs, with the integer that was lost:')
    for run, rec in recs:
        for tag in ('11', '12', '22'):
            p = rec['pairs'][tag]
            if p['n_corrupt']:
                shifts = sorted(set(p['corrupt_shift']))
                print(f"    {rec['state']:<26} g{tag}: R* {p['r_range'][0]:.3f}-"
                      f"{p['r_range'][1]:.3f}, {p['n_corrupt']} points, "
                      f"lost integer {'/'.join(f'{s:.0f}' for s in shifts)}")

    if args.repair:
        print('\n  repaired tables:')
        for run, rec in recs:
            out = repair(run, rec)
            if out:
                print(f'    {out}')
    (HERE / 'tn1331' / 'audit_vs_md.json').write_text(
        json.dumps([r for _, r in recs], indent=2))
    print(f"\n  wrote {HERE / 'tn1331' / 'audit_vs_md.json'}")
    return 0


def _audit(table: Path, gcsv: Path, stem: str, sus: dict) -> dict:
    tn, md = read_csv(table), read_csv(gcsv)
    key = f'rdf/{stem}.csv'
    rec = dict(state=stem, table=key, n_points=len(tn['R_star']), pairs={})
    for tag in ('11', '12', '22'):
        gt = tn[f'g{tag}']
        gm = np.interp(tn['R_star'] * SIGMA[tag], md['r'], md[f'g{tag}'])
        dev = gt - gm
        ok = np.isfinite(dev)
        flagged = sus.get((key, f'g{tag}'), set())
        is_flag = np.array([round(R, 4) in flagged for R in tn['R_star']])
        near_int = np.abs(np.abs(dev) - np.round(np.abs(dev))) < INT_TOL
        big = ok & (np.abs(dev) > NOISE)
        catas = big & (np.abs(dev) > CATASTROPHIC)
        corrupt = (big & ~catas & near_int
                   & (np.round(np.abs(dev)) >= 1) & (np.round(np.abs(dev)) <= MAX_INT))
        other = big & ~catas & ~corrupt
        rec['pairs'][tag] = dict(
            n_catastrophic=int(catas.sum()), n_other=int(other.sum()),
            mean_abs=float(np.mean(np.abs(dev[ok]))),
            mean_abs_clean=float(np.mean(np.abs(dev[ok & ~corrupt & ~catas & ~other]))),
            max_abs=float(np.max(np.abs(dev[ok]))) if ok.any() else float('nan'),
            n_corrupt=int(corrupt.sum()), n_flagged=int(is_flag.sum()),
            n_flag_and_corrupt=int((is_flag & corrupt).sum()),
            n_corrupt_unflagged=int((corrupt & ~is_flag).sum()),
            n_flag_not_corrupt=int((is_flag & ~corrupt).sum()),
            corrupt_R=[float(R) for R in tn['R_star'][corrupt]],
            corrupt_shift=[float(round(abs(x))) for x in dev[corrupt]],
            r_range=[float(tn['R_star'][corrupt].min()), float(tn['R_star'][corrupt].max())]
            if corrupt.any() else None)
    return rec


if __name__ == '__main__':
    raise SystemExit(main())
