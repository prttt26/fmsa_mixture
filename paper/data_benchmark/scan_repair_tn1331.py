#!/usr/bin/env python3
"""Scan every NIST TN 1331 table against our own MD and repair the scan damage.

The tables are a 1989 scan.  Its OCR layer fails in a small number of mechanical
ways -- it drops the leading digit (`1.8442` -> `0.8442`), loses the decimal
point (`0.8574` -> `8574`), or garbles one digit.  Each of those is a
TRANSFORMATION of the printed value, and each is testable: run our own MD at the
same state point and ask whether some such transformation reconciles the two.
If exactly one does, to within the column's own noise, the value is repaired.
If none does, the point is left alone and reported.

This replaces the extracted value in place.  `tn1331/repair_log.csv` records
every change (old value, new value, which transformation, the MD value and the
residual), so nothing is lost, but the tables themselves are clean afterwards.

Each appendix is repaired by the evidence it actually admits
------------------------------------------------------------
rdf   Appendix A tabulates g_ij and MD measures g_ij directly.  The quiet-point
      noise is 0.004-0.01, so a lost digit -- a shift of exactly 1 -- is a 100+
      sigma signal.  These are repaired AGAINST MD.  They cannot be repaired
      internally: g is not sign-constrained, has no contact bracket, and the
      damage clusters exactly at the first peak, where shape alone proves nothing.
dcf   Appendix C tabulates c_ij, which MD reaches only by OZ-inverting its own g.
      That inversion carries a SMOOTH bias of order 1 in the core (the DCF study
      measured c11 < 0.2 %, c12 ~ 2 %, c22 3-20 %, and c22 runs to -30), and a
      smooth bias of order 1 is exactly what a "lost digit of 1" looks like.
      Tried anyway, the search absorbs the bias one integer at a time, inferring
      +5, +5, +4, +3, +2 ... along a single column -- which is not how a scan
      fails.  **Appendix C is therefore never repaired against MD.**  It does not
      need to be: c IS sign-constrained inside the core and IS bracketed at
      contact, so it is repaired FROM ITSELF -- see internal_pass() and the
      write-up at numerical_notes/results/benchmark/repair_literature/.

    python scan_repair_tn1331.py --dry-run          # report only (vs MD)
    python scan_repair_tn1331.py --apply            # repair Appendix A in place
    python scan_repair_tn1331.py --internal         # report only (Appendix C)
    python scan_repair_tn1331.py --internal --apply # repair Appendix C in place
"""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = HERE / 'md' / 'runs'
CONVERTED = HERE / 'converted'
TN = HERE / 'tn1331'
LOG = TN / 'repair_log.csv'

SIGMA = {'11': 1.0, '12': 1.5, '22': 2.0}      # size ratio 2, additive
PAIRS = ('11', '12', '22')

QUIET = 0.15          # |dev| below this is the column's own noise, by definition
SIGMA_FLOOR = 0.004   # do not claim a tolerance tighter than this
TOL_SIGMA = 6.0       # accept a repair within this many column-sigmas
TOL_FLOOR = 0.05      # ... but never tighter than this in absolute terms
IMPROVE = 4.0         # and the repair must shrink the residual by this factor
RESID_MAX = 0.15      # a repaired value must land this close to MD, absolutely
SIGMA_MAX_RDF = 0.05  # our own MD is not good enough to certify a noisier column
REPAIRABLE = {'rdf'}  # against MD.  Appendix C is repaired INTERNALLY instead --
                      # see internal_pass() below.

# ---- internal-consistency repair of Appendix C (no external reference) -------
# R* = r/sigma_ij in these tables, running 0.025 .. 2.500 for every pair.
DEEP_CORE = 0.60      # c_ij < 0 here.  Calibrated against Appendix C itself, not
                      # assumed: over the 104 structurally-clean columns the earliest
                      # LEGITIMATE positive sits at R* = 0.675, and of 1261 values in
                      # R* = 0.60-0.90 only 17 are positive.  (Our own MD, OZ-inverted,
                      # puts a few c22 crossings as early as R* = 0.41 -- that is the
                      # known c22 inner-core inversion artefact, not physics: TN 1331's
                      # own c22 runs -25 to -62 through that region.  See
                      # numerical_notes/rootcause/c22_outer_inversion_artifact.md.)
FIT_MAX = 0.85        # trend stage stops here; the contact band is handled by
                      # bracketing instead (stage 3), which needs no extrapolation.
_OLD_FIT_NOTE = 1.00  # trend stage stops AT contact.  c_ij changes sign there (c ~ -beta*u
                      # and u_ij(sigma_ij) = 0), so past it the column turns steeply
                      # and a smooth-trend prediction is no longer valid -- run it to
                      # 1.6 and it "repairs" 65 legitimate positives into negatives.
                      # Extrapolating to contact costs accuracy: over 1360 clean steps
                      # the error is p50 0.003, p99 0.233, max 0.411, against p99 0.089
                      # below R* = 0.95.  Allowed that far it picks the wrong integer
                      # on values it has the sign of already (+0.435 -> -1.435 rather
                      # than -0.435) and makes four columns demonstrably worse.
CONTACT = (0.85, 1.15)  # stage 3 band.  c_ij rises monotonically through its zero here
                      # (c ~ -beta*u, u_ij(sigma_ij) = 0), and above contact it is
                      # positive in 188 of 189 tabulated values -- so the band can be
                      # bracketed between trusted values on BOTH sides and needs no
                      # extrapolation at all.
C_ABSURD = 100.0      # a clean gap: Appendix C has NO value in 100..1000, the
                      # largest real one being ~63, and the damaged ones are 1623+
FLAG = 0.5            # off-trend by more than half the candidate spacing => damaged.
                      # Leave-one-out prediction error on clean core points: p50 5e-4,
                      # p90 0.010, p99 0.14 -- and every genuine damage found is an
                      # exact integer offset (<= 7e-4 from an integer, n = 1..6).
ACCEPT = 0.25         # a reconstruction must land this close to the trend.  Candidates
                      # are spaced exactly 1.0 apart in magnitude, so < 0.5 makes the
                      # surviving candidate unique whenever one exists.
PRED_AGREE = 0.15     # two independent predictors (quadratic on the 4 nearest spine
                      # points, linear on the 2 nearest) must agree this closely, or
                      # the local trend is not determined well enough to certify
                      # anything and the point is reported instead.  This is the guard
                      # that matters: the candidates are spaced 1.0 apart, so a
                      # prediction that is wrong by ~1 will ALWAYS find a candidate
                      # sitting within ACCEPT of it and look certified.  Unguarded, a
                      # diverging extrapolation turned c22 = -3.275 into +93.275, each
                      # bad repair feeding the next.
MAG_SLACK = 1.5       # and no repair may exceed the spine's own magnitude range by
                      # more than this factor -- a second, cruder net under the same
                      # runaway.
SEED3_PROOF = 0.05    # a three-point seed is only as good as its first prediction:
                      # unless that reconstruction lands this close to the trend, the
                      # seed is abandoned and the column reported.
IMPROVE = 0.15        # ... and must beat the printed value's own residual by this
                      # factor.  This is what separates a real repair from a bad
                      # extrapolation: genuine damage improves by 10^2-10^3, while a
                      # quadratic mispredicting the steep turn near contact improves
                      # by ~3 and is rejected.

# A lost digit or a lost decimal point hits a RUN of consecutive cells -- that is
# how the scan fails.  A single-digit garble, by construction, hits ONE cell.  A
# smooth run of moderate deviations is therefore not a garble at all: it is a
# physical or statistical difference between the two simulations, and "repairing"
# it would be fitting our own MD error into their table.  So the garble family is
# accepted only for an ISOLATED point, and only on a strong deviation.
GARBLE_MIN_DEV = 0.30
GARBLE_TOL = 0.02


# --------------------------------------------------------------------------
def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    """Header comments, field names and raw string rows -- formatting preserved."""
    lines = path.read_text().splitlines()
    head = [l for l in lines if l.startswith('#')]
    body = [l for l in lines if not l.startswith('#') and l.strip()]
    names = body[0].split(',')
    rows = [l.split(',') for l in body[1:]]
    return head, names, rows


def write_table(path: Path, head: list[str], names: list[str], rows: list[list[str]]) -> None:
    with path.open('w') as fh:
        for l in head:
            fh.write(l + '\n')
        fh.write(','.join(names) + '\n')
        for r in rows:
            fh.write(','.join(r) + '\n')


def md_curve(run: Path, kind: str) -> dict | None:
    """g_ij(r) straight from the run, or c_ij(r) from its conversion."""
    if kind == 'rdf':
        src = run / 'g.csv'
    else:
        src = CONVERTED / f'md_{run.name}' / 'c.csv'
    if not src.exists():
        return None
    lines = [l for l in src.read_text().splitlines() if l and not l.startswith('#')]
    names = lines[0].split(',')
    data = np.array([[float(v) for v in l.split(',')] for l in lines[1:]])
    return {n: data[:, q] for q, n in enumerate(names)}


# --------------------------------------------------------------------------
def candidates(text: str, value: float) -> list[tuple[float, str]]:
    """Every value a known scan failure could have produced from `value`.

    Ordered by how well documented the failure is, and the search stops at the
    first family that reconciles -- so a lost digit is never re-explained as a
    coincidental single-digit garble.
    """
    out: list[tuple[float, str]] = []
    for n in range(1, 10):                       # lost leading digit
        out.append((value + n, f'lost_digit_+{n}'))
        out.append((value - n, f'lost_digit_-{n}'))
    for k in (-4, -3, -2, -1, 1, 2, 3, 4):       # decimal point moved
        out.append((value * 10.0 ** k, f'decimal_x10^{k}'))
    txt = text.strip()
    if '.' in txt and txt.replace('-', '').replace('.', '').isdigit():
        digits = list(txt)
        for i, ch in enumerate(digits):
            if not ch.isdigit():
                continue
            for d in '0123456789':
                if d == ch:
                    continue
                alt = digits.copy()
                alt[i] = d
                try:
                    out.append((float(''.join(alt)), f'digit{i}:{ch}->{d}'))
                except ValueError:
                    pass
    return out


def repair_column(rows, names, col, r_star, md_val, kind, tag):
    """-> (list of fixes, sigma, n_unreconciled).  Does not mutate `rows`."""
    j = names.index(col)
    printed = np.array([float(r[j]) for r in rows])
    dev = printed - md_val
    quiet = np.abs(dev) < QUIET
    sigma = max(float(np.median(np.abs(dev[quiet]))) if quiet.any() else QUIET,
                SIGMA_FLOOR)
    tol = max(TOL_SIGMA * sigma, TOL_FLOOR)
    fixes, unresolved = [], 0
    if kind not in REPAIRABLE:
        return fixes, sigma, int((~quiet).sum())        # report only
    if sigma > SIGMA_MAX_RDF:
        return fixes, sigma, int((~quiet).sum())        # our MD cannot certify this
    n = len(printed)
    for i in np.where(~quiet)[0]:
        isolated = ((i == 0 or quiet[i - 1]) and (i == n - 1 or quiet[i + 1]))
        best = None
        for cand, how in candidates(rows[i][j], printed[i]):
            garble = how.startswith('digit')
            if garble and not (isolated and abs(dev[i]) > GARBLE_MIN_DEV):
                continue
            lim = min(tol, RESID_MAX, GARBLE_TOL if garble else RESID_MAX)
            res = abs(cand - md_val[i])
            if res < lim and res * IMPROVE < abs(dev[i]):
                if best is None or res < best[1]:
                    best = (cand, res, how)
            if best is not None and how.startswith('decimal'):
                break                    # documented families first; stop early
        if best is None:
            unresolved += 1
            continue
        cand, res, how = best
        fixes.append(dict(row=int(i), col=col, R_star=float(r_star[i]),
                          old=printed[i], new=cand, md=md_val[i],
                          residual=res, mode=how, sigma=sigma))
    kept, dropped = _keep_consistent_runs(fixes)
    return kept, sigma, unresolved + dropped


def _keep_consistent_runs(fixes):
    """The scan loses the SAME digit through a run of cells.

    So a run of consecutive repaired points must share one transformation.  Where
    the inferred transformation drifts along the run (+5, +5, +4, +3, +2 ...) the
    algorithm is not identifying a lost digit at all -- it is absorbing a smooth
    offset between the two data sets, one integer at a time.  That is what
    happens when the reference is an OZ-inverted c rather than a measured g, and
    it is exactly the false repair this rule exists to refuse.

    An isolated single-point repair survives only if it is very tight (a garble),
    which `GARBLE_TOL` already enforces for that family; a lone lost-digit or
    decimal repair must meet the same bar.
    """
    if not fixes:
        return fixes, 0
    fixes = sorted(fixes, key=lambda f: f['row'])
    kept, dropped, run = [], 0, [fixes[0]]
    for f in fixes[1:]:
        if f['row'] == run[-1]['row'] + 1:
            run.append(f)
        else:
            k, d = _judge_run(run)
            kept.extend(k)
            dropped += d
            run = [f]
    k, d = _judge_run(run)
    kept.extend(k)
    dropped += d
    return kept, dropped


def _judge_run(run):
    if len(run) == 1:
        f = run[0]
        return ([f], 0) if f['residual'] < GARBLE_TOL else ([], 1)
    modes = {f['mode'] for f in run}
    if len(modes) == 1:
        return run, 0
    return [], len(run)


def audit_table(stem: str, kind: str, run: Path) -> dict | None:
    table = TN / kind / f'{stem}.csv'
    if not table.exists():
        return None
    md = md_curve(run, kind)
    if md is None:
        return None
    head, names, rows = read_table(table)
    rkey = 'R_star'
    r_star = np.array([float(r[names.index(rkey)]) for r in rows])
    letter = 'g' if kind == 'rdf' else 'c'
    rec = dict(stem=stem, kind=kind, run=run.name, n_points=len(rows), cols={})
    all_fixes = []
    for tag in PAIRS:
        col = f'{letter}{tag}'
        if col not in names:
            continue
        md_val = np.interp(r_star * SIGMA[tag], md['r'], md[col])
        fixes, sigma, unres = repair_column(rows, names, col, r_star, md_val, kind, tag)
        rec['cols'][col] = dict(sigma=sigma, n_fix=len(fixes), n_unresolved=unres)
        all_fixes.extend(fixes)
    rec['fixes'] = all_fixes
    rec['head'], rec['names'], rec['rows'], rec['path'] = head, names, rows, table
    return rec


def apply_fixes(rec: dict) -> int:
    names, rows = rec['names'], rec['rows']
    for f in rec['fixes']:
        j = names.index(f['col'])
        width = len(rows[f['row']][j].split('.')[-1]) if '.' in rows[f['row']][j] else 4
        rows[f['row']][j] = f'{f["new"]:.{width}f}'
    if rec['fixes']:
        note = (f'# repaired {date.today()} against MD run {rec["run"]}: '
                f'{len(rec["fixes"])} scan-damaged values replaced '
                f'(see tn1331/repair_log.csv)')
        head = [h for h in rec['head'] if 'repaired' not in h] + [note]
        write_table(rec['path'], head, names, rows)
    return len(rec['fixes'])


# --------------------------------------------------------------------------
# Appendix C: what the table can be checked against is ITSELF
# --------------------------------------------------------------------------
def _predict(R, v, spine, i, check=False):
    """Quadratic through the <=4 spine points nearest i, evaluated at R*[i].

    With check=True, also returns whether an independent linear fit through the two
    nearest spine points agrees to PRED_AGREE, and the spine's magnitude range.
    Extrapolation error is not uniform along a column -- it is tiny in the smooth
    core and large near contact, where c turns over -- so reliability is measured
    point by point here rather than assumed from a global calibration.

    The spine is the run of points already certified clean, so this is a one-step
    extrapolation off certified data -- which is the case that was calibrated:
    over 2919 clean steps its error is p50 0.002, p99 0.089, p99.9 0.201, max 0.319.
    Taking instead the nearest neighbours on BOTH sides looks more accurate and is
    useless here, because damage arrives in runs: the neighbours of the first bad
    point are the rest of the bad run, which drags the fit onto the damage and
    hides it (an 11-point run in T3_eps3.00_x0.9001 c22 goes completely undetected
    that way).
    """
    near = sorted(spine, key=lambda q: abs(R[q] - R[i]))
    o = sorted(near[:4])
    if len(o) < 3:
        return (None, False, 0.0) if check else None
    quad = float(np.polyval(np.polyfit(R[o], v[o], 2), R[i]))
    if not check:
        return quad
    p = sorted(near[:2])
    lin = float(np.polyval(np.polyfit(R[p], v[p], 1), R[i]))
    return quad, abs(quad - lin) <= PRED_AGREE, max(abs(v[q]) for q in o)


def _seed(R, v, fit):
    """The first self-consistent 4 points INSIDE the deep core.

    Not the globally most consistent window: three points determine a quadratic
    exactly, so a window of pure damage can reproduce its own fourth point by
    coincidence and score better than real data.  That is not hypothetical -- it put
    the seed for T3_eps3.00_x0.9001 c22 at R* = 0.75-0.825, in the middle of the
    damaged block, and the column was then "certified" against the damage.

    Starting deep is the principled choice instead: that is where c_ij is largest and
    varies most slowly, and -- decisively -- where stage 1 has already forced the sign.
    A seed outside the deep core carries no such guarantee, and a sign-flipped block
    out there is perfectly smooth, so it passes every self-consistency test while
    orienting the whole column backwards (c12 of T3_eps0.50_x0.9502 seeds on
    +1.388,+1.363,+1.339,+1.315 at R* = 0.675-0.750 and reconstructs the column
    positive).  The seed must also carry a printed integer somewhere: a window of
    values all below 1 in magnitude fixes the shape but not the LEVEL, and the column
    can come out shifted by a whole unit while looking perfectly smooth.
    """
    fit = [q for q in fit if R[q] < DEEP_CORE]
    for a in range(len(fit) - 3):
        w = fit[a:a + 4]
        if max(abs(v[q]) for q in w) >= C_ABSURD or max(abs(v[q]) for q in w) < 1.0:
            continue
        if abs(np.polyval(np.polyfit(R[w[:3]], v[w[:3]], 2), R[w[3]]) - v[w[3]]) < 0.05:
            return list(w)
    # No self-validating window of four.  A run of absurd values can cut the deep
    # core down to three points, and refusing there costs a whole column, so take
    # the deepest clean three that carry a printed integer -- a quadratic through
    # them cannot validate itself, so growth validates it instead: the first
    # reconstruction it certifies must land within 0.05 of the trend or the seed
    # is abandoned (see SEED3_PROOF).
    for a in range(len(fit) - 2):
        w = fit[a:a + 3]
        if max(abs(v[q]) for q in w) >= C_ABSURD or max(abs(v[q]) for q in w) < 1.0:
            continue
        return list(w)
    return None


def _candidates(printed):
    """Every value whose printed digits are what the scan actually shows.

    One mechanism explains all four damage families seen in Appendix C: the digits
    survive, the sign and the integer prefix do not.

        -3.964  printed  +3.964   the minus is gone
        -7.182  printed  -1.182   the leading 7 was read as a 1
       -10.449  printed  +0.449   the leading "-1" is gone
        -1.7565 printed    7565   the leading "-1." is gone, so it parsed as an int

    The first three keep the decimal point, so the fraction is read straight off.
    The fourth lost it, so the digit string is re-read as the fraction.  Either way
    the candidates are s*(m + f) -- spaced exactly 1.0 apart in magnitude, which is
    what makes the surviving one unique.
    """
    out = []
    a = abs(printed)
    fracs = [a - np.floor(a)]
    if a >= C_ABSURD:
        digits = f'{a:.0f}'.rstrip('0') or '0'
        fracs.append(float(digits) / 10.0 ** len(digits))
    for f in fracs:
        for s in (-1.0, 1.0):
            for m in range(0, 100):
                out.append(s * (m + f))
    return out


def internal_pass(stem: str) -> dict | None:
    """Repair one Appendix C table using only its own columns.

    Two stages, in this order:

    physics  Inside the core c_ij(r) < 0 is rigorous -- g -> 0 there, so c = -1 - gamma
             is large and negative.  A positive value at R* < DEEP_CORE is impossible
             and the minimal edit is to restore the lost minus.  This needs no trend,
             which is what lets it repair a column that lost its sign WHOLESALE: the
             trend of a uniformly sign-flipped column is perfectly smooth and tells
             you nothing.

    trend    With the deep core sign-correct, sweep for values off the column's own
             smooth trend by more than FLAG, worst first, refitting after each repair
             so a run of damage is unwound from its clean end inward.  A value is
             replaced only by a candidate that keeps its printed digits, is the unique
             one near the trend, and beats the printed residual by IMPROVE.

    Each cell is repaired at most once -- without that the refit chases its own tail
    (-0.4486 -> 8.4486 -> 6.4486 -> 5.4486 on one cell, and outright oscillation on
    another).  A flagged value with no unique reconstruction is dropped from the fit
    and reported, never guessed, and a merely noisy column is left alone.
    """
    path = TN / 'dcf' / f'{stem}.csv'
    if not path.exists():
        return None
    head, names, rows = read_table(path)
    R = np.array([float(r[names.index('R_star')]) for r in rows])
    rec = dict(stem=stem, kind='dcf', run='(internal)', n_points=len(rows),
               cols={}, fixes=[], head=head, names=names, rows=rows, path=path)
    for tag in PAIRS:
        col = f'c{tag}'
        if col not in names:
            continue
        j = names.index(col)
        v = np.array([float(r[j]) for r in rows])
        original = v.copy()
        mode = {}
        unresolved = []

        # -- stage 1: the inequality
        for i in np.where((R < DEEP_CORE) & (v > 0) & (np.abs(v) < C_ABSURD))[0]:
            v[i] = -v[i]
            mode[int(i)] = 'core_sign'

        # -- stage 2: grow a certified spine outward, repairing what it meets
        fit = list(np.where(R <= FIT_MAX)[0])
        spine = _seed(R, v, fit)
        # The spine fixes the column's LEVEL, and a level is only as good as the digits
        # it rests on.  If every seed value is below 1 in magnitude its integer part was
        # never printed, so the level is inferred, not observed -- and the whole column
        # can come out shifted by a whole unit while looking perfectly smooth.  That is
        # not hypothetical: it put c12 of T3_eps0.50_x0.9502 at -0.53..-0.61 where its
        # sibling table T3_eps0.50_x0.9001 has -1.70 at the same R*, and where c11 and
        # c22 of its own table both sit near -1.3.  Refuse instead.
        if spine is not None and max(abs(v[q]) for q in spine) < 1.0:
            unresolved.extend(fit)
            spine = None
        if spine is not None:
            lo, hi = spine[0], spine[-1]
            for direction, seq in ((+1, range(hi + 1, fit[-1] + 1)),
                                   (-1, range(lo - 1, fit[0] - 1, -1))):
                miss = 0
                unproven = len(spine) == 3
                for i in seq:
                    pred, agree, scale = _predict(R, v, spine, i, check=True)
                    if pred is None:
                        break
                    res = abs(v[i] - pred) if abs(v[i]) < C_ABSURD else float('inf')
                    if res <= FLAG:                       # clean: it joins the spine
                        spine.append(i)
                        miss = 0
                        continue
                    gate = max(0.02, IMPROVE * res) if np.isfinite(res) else ACCEPT
                    lim = MAG_SLACK * scale + 1.0
                    near = [c for c in _candidates(v[i])
                            if abs(c - pred) <= min(ACCEPT, gate) and abs(c) <= lim]
                    if not agree or len(near) != 1:
                        unresolved.append(i)
                        miss += 1
                        if miss > 3:      # extrapolating across a hole this wide is
                            break         # no longer the calibrated case -- stop
                        continue
                    if unproven and abs(near[0] - pred) > SEED3_PROOF:
                        unresolved.extend(q for q in fit if q not in spine)
                        break
                    unproven = False
                    v[i] = near[0]
                    mode[int(i)] = 'trend_digits' if int(i) not in mode else 'sign+digits'
                    rec.setdefault('_pred', {})[(col, int(i))] = pred
                    spine.append(i)
                    miss = 0
                spine.sort()

        # -- stage 3: the contact band, as a minimal-edit monotone chain
        band = [q for q in np.where((R >= CONTACT[0]) & (R <= 1.0))[0]
                if abs(v[q]) < C_ABSURD]
        above = [q for q in np.where((R > 1.0) & (R <= CONTACT[1]))[0]
                 if abs(v[q]) < C_ABSURD]
        below = [q for q in np.where(R < CONTACT[0])[0] if abs(v[q]) < C_ABSURD]
        if len(band) >= 3 and above and below:
            lower, upper = v[below[-1]], v[above[0]]
            opts = []
            for i in band:
                cs = {float(v[i])} | {c for c in _candidates(v[i])}
                cs = sorted(c for c in cs if lower <= c <= upper)
                opts.append(cs)
            if all(opts):
                # cheapest monotone chain through the candidate sets; ties are refused
                cost = [[(0 if c == v[i] else 1, 1) for c in opts[k]]
                        for k, i in enumerate(band)]
                back = [[None] * len(o) for o in opts]
                for k in range(1, len(band)):
                    for q, c in enumerate(opts[k]):
                        best, ways, arg = None, 0, None
                        for pq, pc in enumerate(opts[k - 1]):
                            if pc > c + 1e-12:
                                continue
                            sc = cost[k - 1][pq][0]
                            if best is None or sc < best:
                                best, ways, arg = sc, cost[k - 1][pq][1], pq
                            elif sc == best:
                                ways += cost[k - 1][pq][1]
                        add = 0 if c == v[band[k]] else 1
                        cost[k][q] = (float('inf'), 0) if best is None else (best + add, ways)
                        back[k][q] = arg
                fin = [(c, w, q) for q, (c, w) in enumerate(cost[-1]) if np.isfinite(c)]
                if fin:
                    mn = min(f[0] for f in fin)
                    ways = sum(f[1] for f in fin if f[0] == mn)
                    if ways == 1 and mn > 0:
                        q = next(f[2] for f in fin if f[0] == mn)
                        for k in range(len(band) - 1, -1, -1):
                            i, c = band[k], opts[k][q]
                            if c != v[i]:
                                v[i] = c
                                mode[int(i)] = ('contact_chain' if int(i) not in mode
                                                else 'sign+digits')
                                rec.setdefault('_pred', {})[(col, int(i))] = float('nan')
                            q = back[k][q] if k else 0
                    elif mn > 0:
                        unresolved.extend(band)

        # -- one record per cell that actually changed
        n_sign = n_trend = 0
        for i in np.where(original != v)[0]:
            m = mode.get(int(i), 'core_sign')
            pred = rec.get('_pred', {}).get((col, int(i)), float('nan'))
            rec['fixes'].append(dict(row=int(i), col=col, R_star=float(R[i]),
                                     old=float(original[i]), new=float(v[i]),
                                     md=float(pred),
                                     residual=float(abs(v[i] - pred)) if pred == pred else 0.0,
                                     mode=m, sigma=float(ACCEPT)))
            if m == 'core_sign':
                n_sign += 1
            else:
                n_trend += 1
        rec['cols'][col] = dict(sigma=0.0, n_fix=n_sign + n_trend,
                                n_unresolved=len(unresolved),
                                n_sign=n_sign, n_digits=n_trend)
    rec.pop('_pred', None)
    return rec


def run_internal(apply_: bool, backup: bool) -> list:
    stems = sorted(p.stem for p in (TN / 'dcf').glob('*.csv'))
    recs = [r for r in (internal_pass(s) for s in stems) if r]
    print('=' * 100)
    print(f'  DCF, INTERNAL CONSISTENCY  —  {len(recs)} Appendix C tables, no external '
          'reference used')
    print('=' * 100)
    print(f"  {'table':<30} " + ' '.join(f'{f"c{tg}: phys trend":>18}' for tg in PAIRS))
    n_sign = n_dig = n_unres = 0
    for rec in recs:
        if not rec['fixes']:
            continue
        cells = []
        for tag in PAIRS:
            c = rec['cols'].get(f'c{tag}')
            cells.append('              --' if not c else
                         f"{c['n_sign']:9d} {c['n_digits']:7d}")
            if c:
                n_sign += c['n_sign']
                n_dig += c['n_digits']
                n_unres += c['n_unresolved']
        print(f"  {rec['stem']:<30} " + ' '.join(f'{c:>18}' for c in cells))
    print(f'\n  {n_sign} lost minus signs restored by the core inequality, '
          f'{n_dig} values reconstructed from the trend, {n_unres} left alone')
    if apply_:
        if backup:
            dst = TN / 'dcf_asextracted'
            if not dst.exists():
                shutil.copytree(TN / 'dcf', dst)
                print(f'  as-extracted copy -> {dst}')
        for rec in recs:
            apply_fixes(rec)
        print(f"  applied in place to {sum(1 for r in recs if r['fixes'])} tables")
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--kind', nargs='*', default=['rdf', 'dcf'])
    ap.add_argument('--backup', action='store_true',
                    help='copy the as-extracted tables to tn1331/<kind>_asextracted/ first')
    ap.add_argument('--internal', action='store_true',
                    help='repair Appendix C from its own columns only (no MD)')
    args = ap.parse_args()

    if args.internal:
        recs = run_internal(args.apply, args.backup)
        if args.apply:
            fields = ['date', 'kind', 'table', 'md_run', 'col', 'R_star', 'old', 'new',
                      'md', 'residual', 'mode', 'sigma', 'row']
            exists = LOG.exists()
            with LOG.open('a', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
                if not exists:
                    w.writeheader()
                for rec in recs:
                    for f in rec['fixes']:
                        w.writerow(dict(date=str(date.today()), kind='dcf',
                                        table=f'dcf/{rec["stem"]}.csv',
                                        md_run='(internal)', **f))
            print(f'logged -> {LOG}')
        elif not args.apply:
            print('\n(dry run — nothing written; add --apply)')
        return 0

    runs = {p.name.replace('tnaudit_', ''): p for p in sorted(RUNS.glob('tnaudit_*'))}
    if (RUNS / 'validation' / 'g.csv').exists():
        runs.setdefault('T1_eps0.50_x0.2500', RUNS / 'validation')

    log_rows, totals = [], {}
    for kind in args.kind:
        recs = []
        for stem, run in sorted(runs.items()):
            rec = audit_table(stem, kind, run)
            if rec:
                recs.append(rec)
        if not recs:
            print(f'no {kind} tables have an MD counterpart yet')
            continue
        if args.apply and args.backup:
            dst = TN / f'{kind}_asextracted'
            if not dst.exists():
                shutil.copytree(TN / kind, dst)
                print(f'  as-extracted copy -> {dst}')

        print('=' * 100)
        print(f'  {kind.upper()}  —  {len(recs)} tables with an MD counterpart')
        print('=' * 100)
        print(f"  {'table':<26} " + ' '.join(f'{c:>21}' for c in
                                             (f'{"g" if kind == "rdf" else "c"}{t}' for t in PAIRS)))
        print(f"  {'':<26} " + ' '.join(f'{"sigma  fixed unres":>21}' for _ in PAIRS))
        n_fix = n_unres = 0
        for rec in recs:
            cells = []
            for tag in PAIRS:
                col = f'{"g" if kind == "rdf" else "c"}{tag}'
                c = rec['cols'].get(col)
                cells.append('        --' if c is None else
                             f"{c['sigma']:7.4f} {c['n_fix']:5d} {c['n_unresolved']:5d}")
                if c:
                    n_fix += c['n_fix']
                    n_unres += c['n_unresolved']
            print(f"  {rec['stem']:<26} " + ' '.join(f'{c:>21}' for c in cells))
        totals[kind] = (n_fix, n_unres)
        print(f'\n  {kind}: {n_fix} values repaired, {n_unres} deviating values that no '
              'known scan failure explains'
              + ('' if kind in REPAIRABLE else
                 '\n  (this kind is report-only: an OZ-inverted reference cannot '
                 'certify a lost digit — see the module docstring)'))
        modes: dict[str, int] = {}
        for rec in recs:
            for f in rec['fixes']:
                key = ('lost_digit' if f['mode'].startswith('lost') else
                       'decimal' if f['mode'].startswith('decimal') else 'digit_garble')
                modes[key] = modes.get(key, 0) + 1
        print('  by failure mode:', modes)

        if args.apply:
            for rec in recs:
                apply_fixes(rec)
                for f in rec['fixes']:
                    log_rows.append(dict(date=str(date.today()), kind=kind,
                                         table=f'{kind}/{rec["stem"]}.csv',
                                         md_run=rec['run'], **f))
            print(f'  applied in place to {sum(1 for r in recs if r["fixes"])} tables')

    if args.apply and log_rows:
        fields = ['date', 'kind', 'table', 'md_run', 'col', 'R_star', 'old', 'new',
                  'md', 'residual', 'mode', 'sigma', 'row']
        exists = LOG.exists()
        with LOG.open('a', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            if not exists:
                w.writeheader()
            w.writerows(log_rows)
        print(f'\nlogged {len(log_rows)} replacements -> {LOG}')
    elif not args.apply:
        print('\n(dry run — nothing written; pass --apply)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
