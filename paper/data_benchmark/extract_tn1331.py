#!/usr/bin/env python3
"""Extract the benchmark tables from NIST Technical Note 1331 into clean CSVs.

Source
------
M. L. Huber and J. F. Ely, "Properties of Lennard-Jones Mixtures at Various
Temperatures and Energy Ratios With a Size Ratio of Two," NIST Technical Note
1331, May 1989, 426 pp.  Public domain.
    https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1331.pdf

The report is an isothermal-MD study of a binary Lennard-Jones mixture with
sigma22/sigma11 = 2 (Lorentz-Berthelot mixing), tabulating, for each state point:

    Appendix A : g_ij(R*) partial radial distribution functions + local compositions
    Appendix B : plots of the g_ij (not machine readable, ignored here)
    Appendix C : c_ij(R*) direct correlation functions, obtained by the report
                 itself OZ-inverting its own simulated g_ij

Both tables use the PER-PAIR reduced separation R* = r / sigma_ij, so all three
pairs reach contact near R* ~ 1.  To get an absolute distance for pair (i,j),
multiply by sigma_ij = [[1.0, 1.5], [1.5, 2.0]].

Why this file exists
--------------------
The 1989 report is a SCAN.  pdftotext recovers most pages cleanly but some pages
are badly degraded (decimal points dropped entirely, e.g. "975  -1 2196" for
"0.975  -1.2196").  This parser therefore never trusts a page: it normalises the
known OCR damage patterns, then validates every block against the expected
R* grid and field count, and records a per-block quality flag.  Blocks that do
not validate are reported and skipped rather than silently emitted.

Usage
-----
    python extract_tn1331.py --pdf /path/to/tn1331.pdf --out tn1331
    python extract_tn1331.py --txt /path/to/tn1331.txt --out tn1331   # skip pdftotext

Outputs, under --out:
    index.csv                     one row per successfully extracted block
    dcf/T{T}_eps{E}_x{X}.csv      R_star,c11,c12,c22
    rdf/T{T}_eps{E}_x{X}.csv      R_star,g11,g12,g22,xl11,xl12,xl22
    rejected.csv                  blocks that failed validation, with the reason
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- fixed properties of the report's tables ------------------------------

STEP = 0.025
DCF_R_MAX = 2.500
DCF_N_ROWS = 50          # two-column layout: 50 lines x 2 halves = 100 R* points
DCF_FIELDS = 8           # R*, c11, c12, c22, R*, c11, c12, c22
RDF_FIELDS = 7           # R*, g11, g12, g22, xl11, xl12, xl22

# The report's composition and energy-ratio grids.  Parsed values are snapped to
# these, which both fixes OCR drift (0.25(DO -> 0.2500) and gives stable filenames.
CANON_X1 = [0.0498, 0.0995, 0.0999, 0.1001, 0.2500, 0.5000, 0.7500,
            0.9001, 0.9005, 0.9502]
CANON_EPS = [0.50, 1.00, 1.50, 2.00, 3.00]
CANON_T = [1.0, 2.0, 3.0]

# Digit-for-digit self-test.  These rows are the T*=1.0, eps=0.50, x1=0.2500
# block of Appendix C -- the table the project has been using as MC_DATA.
SELFTEST_KEY = (1.0, 0.50, 0.2500)
SELFTEST_ROWS = {
    0.025: (-8.9462, -7.2522, -27.7580),
    0.050: (-8.8774, -7.1518, -27.4832),
    0.100: (-8.6289, -7.1534, -26.4750),
    0.400: (-6.0626, -7.1781, -16.7880),
    1.650: (0.1642, 0.1998, 0.4495),
    2.500: (0.0300, 0.0121, 0.0194),
}


# --- OCR repair -----------------------------------------------------------

def normalise(line: str) -> str:
    """Repair the decimal damage this scan exhibits.

    Observed patterns, all fixed here:
        -12,,5168 / 0,.7083 / 0..8535   comma(s) used as, or doubled with, the point
        0. 025 / -8. 9462               space inserted after the decimal point
        1 .8894                         space inserted before the decimal point
        .0000                           leading zero dropped (handled at float())
    """
    s = line.replace(',', '.')
    # the scan frequently renders a leading minus as a bullet/middot glyph
    s = re.sub(r'[•·∙]\s*(?=\d)', '-', s)
    s = re.sub(r'\.{2,}', '.', s)
    # Only ever close up ONE or TWO spaces.  A wider gap is a column separator:
    # "0.0853       .1728" is two numbers (the second having lost its leading
    # zero), not one, and closing it up would silently fuse adjacent columns.
    s = re.sub(r'(\d)\.\s{1,2}(\d)', r'\1.\2', s)   # "0. 025"  -> "0.025"
    s = re.sub(r'(\d)\s{1,2}\.(\d)', r'\1.\2', s)   # "1 .8894" -> "1.8894"
    s = re.sub(r'\.{2,}', '.', s)
    return s


def numbers(line: str) -> list[float]:
    """Every token on the line that is cleanly a number, in order."""
    out = []
    for tok in normalise(line).split():
        if not re.fullmatch(r'-?\.?\d+\.?\d*', tok):
            continue
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


# --- block model ----------------------------------------------------------

@dataclass
class Block:
    kind: str                       # 'dcf' | 'rdf'
    line_no: int                    # 1-based line of the "R*" column header
    tstar: float | None = None
    eps: float | None = None
    x1: float | None = None
    rho: float | None = None
    rows: list[list[float]] = field(default_factory=list)
    quality: str = 'OK'
    note: str = ''

    @property
    def key(self):
        return (self.tstar, self.eps, self.x1)

    @property
    def stem(self):
        return f'T{self.tstar:.0f}_eps{self.eps:.2f}_x{self.x1:.4f}'


def _snap(value: float | None, grid: list[float], tol: float) -> float | None:
    if value is None:
        return None
    best = min(grid, key=lambda g: abs(g - value))
    return best if abs(best - value) <= tol else None


def parse_header(lines: list[str], idx: int) -> dict:
    """Scan backwards from the R* column header for T*, rho*, x1, eps.

    The scanned title/parameter lines are heavily OCR-damaged (the Greek sigma
    and epsilon glyphs are mangled into things like '^22/^11'), so only the
    numeric fields are trusted, and each is snapped to the report's known grid.
    """
    info: dict = {'tstar': None, 'eps': None, 'x1': None, 'rho': None}
    window = lines[max(0, idx - 16):idx]

    for raw in reversed(window):
        s = normalise(raw)

        if info['tstar'] is None:
            # "T* = 1.0", "T* - 2.0", "T*-l" (a mangled 1), "T* = 3,0"
            m = re.search(r'T\s*\*\s*[-=]?\s*([123l])', s)
            if m:
                tok = m.group(1)
                info['tstar'] = 1.0 if tok == 'l' else float(tok)

        if info['rho'] is None:
            m = re.search(r'[pP]\s*\*\s*[-="]*\s*(0\.\d+)', s)
            if m:
                info['rho'] = float(m.group(1))

        if info['x1'] is None:
            # composition token: 0.0498 ... 0.9502, sometimes as "0.25(DO"
            for m in re.finditer(r'0\.\d{2,4}', s):
                cand = _snap(float(m.group(0)), CANON_X1, 0.004)
                if cand is not None and cand != info['rho']:
                    info['x1'] = cand
                    break

        if info['eps'] is None:
            # trailing energy ratio, e.g. "e22/e11 = 0.50" -> a bare 0.50/1.00/...
            for m in re.finditer(r'(?<![\d.])([0-3]\.\d{1,2})(?![\d])', s):
                val = float(m.group(1))
                if val == 2.00:          # ambiguous with the sigma ratio 2.00
                    continue
                cand = _snap(val, CANON_EPS, 0.005)
                if cand is not None:
                    info['eps'] = cand
    return info


def _column_headers(lines: list[str]) -> list[int]:
    """Indices of the 'R*' column-header lines, one per table.

    The scan sometimes splits a single header across two or three consecutive
    lines ("R*   c" / "R*  ^11 ^12 ^22"), so headers closer together than a
    plausible table are merged and the last of the run is used as the anchor.
    """
    hits = [i for i, raw in enumerate(lines) if re.match(r'\s*R\s*\*', raw)]
    merged: list[int] = []
    for i in hits:
        if merged and i - merged[-1] <= 8:
            merged[-1] = i
        else:
            merged.append(i)
    return merged


def parse_blocks(text: str) -> list[Block]:
    # NB: split on '\n' only.  pdftotext emits a form feed at every page break,
    # and str.splitlines() breaks on those too, which shifts every line index.
    raw_lines = text.split('\n')
    # Each state point's table occupies one printed page, so the page breaks are
    # the most reliable block boundary available -- more reliable than the
    # headers, which the scan sometimes destroys outright.
    page_starts = sorted(i for i, ln in enumerate(raw_lines) if '\f' in ln)
    lines = [ln.replace('\f', ' ') for ln in raw_lines]
    heads = _column_headers(lines)
    blocks: list[Block] = []

    # Some column headers are too damaged to detect, so the appendix titles are
    # used as extra boundaries: without them a block would run on into the next
    # state point's rows and validation would reject both.
    titles = [i for i, raw in enumerate(lines)
              if re.search(r'DIRECT\s+CORREL|RADIAL\s+DISTRIB', raw, re.I)]

    for n, i in enumerate(heads):
        # A table runs from its header up to whichever comes first: the next
        # table's header, or the next appendix title.  Rows from the following
        # state point can therefore never leak into this one.
        stop = heads[n + 1] if n + 1 < len(heads) else len(lines)
        for boundary in (titles, page_starts):
            later = [t for t in boundary if t > i]
            if later:
                stop = min(stop, later[0])
        rows_dcf: list[list[float]] = []
        rows_rdf: list[list[float]] = []
        for raw2 in lines[i + 1:stop]:
            nums = numbers(raw2)
            if len(nums) == DCF_FIELDS:
                rows_dcf.append(nums)
            elif len(nums) == RDF_FIELDS:
                rows_rdf.append(nums)

        # Field count is the reliable discriminator: 8 -> Appendix C, 7 -> Appendix A.
        if len(rows_dcf) >= len(rows_rdf) and rows_dcf:
            kind, rows = 'dcf', rows_dcf
        elif rows_rdf:
            kind, rows = 'rdf', rows_rdf
        else:
            continue

        blk = Block(kind=kind, line_no=i + 1, rows=rows)
        for k, v in parse_header(lines, i).items():
            setattr(blk, k, v)
        blocks.append(blk)

    _infer_missing(blocks)
    return blocks


def _infer_missing(blocks: list[Block]) -> None:
    """Fill T*/eps that OCR destroyed, using the report's block ordering.

    Within one appendix the blocks run T* ascending, then energy ratio
    ascending, then composition ascending; a new energy-ratio group therefore
    begins exactly where x1 resets to its smallest value.
    """
    last_t = None
    eps_idx = 0
    last_x1 = None
    last_kind = None

    for blk in blocks:
        if blk.kind != last_kind:
            last_t, eps_idx, last_x1 = None, 0, None
            last_kind = blk.kind

        if blk.tstar is None:
            blk.tstar = last_t
            if blk.tstar is not None:
                blk.note += 'T*-inferred;'
        elif blk.tstar != last_t:
            eps_idx, last_x1 = 0, None
        last_t = blk.tstar

        if blk.x1 is not None:
            if last_x1 is not None and blk.x1 < last_x1 - 1e-9:
                eps_idx += 1                      # composition reset -> next eps group
            last_x1 = blk.x1

        if blk.eps is None:
            if eps_idx < len(CANON_EPS):
                blk.eps = CANON_EPS[eps_idx]
                blk.note += 'eps-inferred;'
        else:
            if blk.eps in CANON_EPS:
                eps_idx = CANON_EPS.index(blk.eps)


# --- validation -----------------------------------------------------------

def _on_grid(r: float, hi: float) -> bool:
    """Is R* a point of the report's 0.025 grid in (0, hi]?"""
    n = round(r / STEP)
    return 1 <= n <= round(hi / STEP) + 1e-9 and abs(n * STEP - r) < 1e-6


def suspect_points(rows: list[list[float]]) -> list[tuple[int, float, float]]:
    """Find values that lost a leading digit to the scan.

    This is the one corruption the R* grid check cannot see: the OCR sometimes
    drops the integer part of a whole run of values in one column, turning
    1.8442 into 0.8442.  The result is perfectly well-formed and lands on the
    grid, so it would otherwise pass silently.

    A dropped digit shifts a value by an exact integer while the underlying
    function stays smooth, so it is caught by predicting each point from the two
    before it and testing whether the residual is a non-zero integer.  Using the
    residual rather than the raw step is what keeps the direct correlation
    function's genuine jump at contact from being flagged: that jump is large
    but not an integer.

    Returns (column index, R*, residual) per suspect point.  Nothing is
    repaired -- the caller only records that the point is untrustworthy.

    Flags come in clusters: a corrupted RUN trips the test once where it starts
    and once where it ends, because the predictor is itself fed corrupted points
    in between.  Treat a flag as marking a suspect REGION of that column, not a
    single wrong number, and check it against the source page.
    """
    found = []
    n_cols = len(rows[0]) - 1 if rows else 0
    for col in range(1, n_cols + 1):
        v = [row[col] for row in rows]
        for n in range(2, len(v)):
            resid = v[n] - (2 * v[n - 1] - v[n - 2])
            if abs(resid) < 0.5:
                continue
            if round(resid) != 0 and abs(resid - round(resid)) < 0.2:
                found.append((col, rows[n][0], resid))
    return found


def validate(blk: Block) -> tuple[list[list[float]], str, str]:
    """Return (clean rows, quality, note).

    Judges the data ROW BY ROW and never repairs it.  Pages of this scan are
    damaged unevenly -- a table is typically intact except for a handful of
    lines whose digits the OCR dropped -- so rejecting a whole table for one bad
    line would throw away most of the report.  Instead every row must land on
    the report's own R* grid to be kept, and the coverage that survives is
    recorded so a consumer can see exactly what is missing.

    A row that contradicts an already-accepted value at the same R* means text
    from a neighbouring table bled in; that is unsafe, so the block is failed.
    """
    if blk.tstar is None or blk.eps is None or blk.x1 is None:
        return [], 'FAILED', 'unidentified state point'

    points: dict[float, list[float]] = {}
    conflicts = 0

    if blk.kind == 'dcf':
        # two-column layout: the right half is always the left half + 1.250
        for row in blk.rows:
            r_left, r_right = row[0], row[4]
            if not _on_grid(r_left, 1.250):
                continue
            if abs(r_right - (r_left + 1.250)) > 1e-6:
                continue
            for r, vals in ((r_left, row[1:4]), (r_right, row[5:8])):
                key = round(r, 3)
                if key in points and points[key] != list(vals):
                    conflicts += 1
                points[key] = list(vals)
        expected = round(DCF_R_MAX / STEP)            # 100 points
    else:
        for row in blk.rows:
            if not _on_grid(row[0], DCF_R_MAX):
                continue
            key = round(row[0], 3)
            if key in points and points[key] != list(row[1:]):
                conflicts += 1
            points[key] = list(row[1:])
        # g_ij is tabulated only from where it first departs zero, so the
        # expected length is the observed span rather than a fixed number.
        expected = (round((max(points) - min(points)) / STEP) + 1) if points else 0

    if conflicts:
        return [], 'FAILED', f'{conflicts} contradictory rows (neighbouring table bled in)'
    if not points:
        return [], 'FAILED', 'no rows on the R* grid'

    rows = [[r] + points[r] for r in sorted(points)]
    missing = expected - len(rows)
    if missing <= 0:
        return rows, 'OK', ''
    # Every surviving point has been checked against the report's grid, so a
    # gappy table is still sound -- just incomplete.  Grade it rather than
    # discard it, and always say how many points are absent.
    frac = len(rows) / expected
    note = f'{len(rows)}/{expected} R* points ({missing} lost to OCR damage)'
    if frac >= 0.90:
        return rows, 'PARTIAL', note
    if frac >= 0.70:
        return rows, 'SPARSE', note
    return [], 'FAILED', note


def selftest(kept: dict) -> tuple[bool, str]:
    blk = kept.get(('dcf',) + SELFTEST_KEY)
    if blk is None:
        return False, 'self-test block (T*=1, eps=0.50, x1=0.2500) not extracted'
    table = {round(r[0], 3): (r[1], r[2], r[3]) for r in blk}
    for r_star, expect in SELFTEST_ROWS.items():
        got = table.get(round(r_star, 3))
        if got is None:
            return False, f'self-test row R*={r_star} missing'
        if any(abs(a - b) > 5e-5 for a, b in zip(got, expect)):
            return False, f'self-test mismatch at R*={r_star}: {got} != {expect}'
    return True, f'{len(SELFTEST_ROWS)} reference rows match digit-for-digit'


# --- driver ---------------------------------------------------------------

def load_text(pdf: Path | None, txt: Path | None) -> str:
    if txt is not None:
        return txt.read_text(errors='replace')
    if pdf is None:
        sys.exit('give --pdf or --txt')
    if shutil.which('pdftotext') is None:
        sys.exit('pdftotext not found (install poppler-utils), or pass --txt')
    cached = pdf.with_suffix('.txt')
    if not cached.exists():
        subprocess.run(['pdftotext', '-layout', str(pdf), str(cached)], check=True)
    return cached.read_text(errors='replace')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pdf', type=Path, help='TN 1331 PDF (converted via pdftotext -layout)')
    ap.add_argument('--txt', type=Path, help='pre-converted text, skips pdftotext')
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'tn1331')
    args = ap.parse_args()

    blocks = parse_blocks(load_text(args.pdf, args.txt))
    print(f'located {len(blocks)} candidate tables '
          f'({sum(b.kind == "dcf" for b in blocks)} DCF / '
          f'{sum(b.kind == "rdf" for b in blocks)} RDF)')

    # Clear previous output first: a re-run with different settings extracts a
    # different set of state points, and a leftover CSV that no longer appears
    # in index.csv would look like valid data.
    for sub in ('dcf', 'rdf'):
        d = args.out / sub
        d.mkdir(parents=True, exist_ok=True)
        for stale in d.glob('*.csv'):
            stale.unlink()
    for stale_name in ('index.csv', 'rejected.csv', 'suspect.csv'):
        (args.out / stale_name).unlink(missing_ok=True)

    header = {'dcf': ['R_star', 'c11', 'c12', 'c22'],
              'rdf': ['R_star', 'g11', 'g12', 'g22', 'xl11', 'xl12', 'xl22']}
    index, rejected, kept, suspects = [], [], {}, []

    for blk in blocks:
        rows, quality, note = validate(blk)
        blk.quality, blk.note = quality, (blk.note + note).strip(';')
        if quality == 'FAILED':
            rejected.append({'kind': blk.kind, 'pdf_text_line': blk.line_no,
                             'tstar': blk.tstar, 'eps': blk.eps, 'x1': blk.x1,
                             'reason': note})
            continue

        key = (blk.kind,) + blk.key
        if key in kept:                       # duplicate identification -> keep neither silently
            rejected.append({'kind': blk.kind, 'pdf_text_line': blk.line_no,
                             'tstar': blk.tstar, 'eps': blk.eps, 'x1': blk.x1,
                             'reason': 'duplicate state point, ambiguous header'})
            continue
        kept[key] = rows

        path = args.out / blk.kind / f'{blk.stem}.csv'
        with path.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(header[blk.kind])
            for row in rows:
                w.writerow([f'{row[0]:.3f}'] + [f'{v:.4f}' for v in row[1:]])
        flagged = suspect_points(rows)
        for col, r_star, resid in flagged:
            suspects.append({'file': f'{blk.kind}/{blk.stem}.csv',
                             'column': header[blk.kind][col], 'R_star': r_star,
                             'residual': f'{resid:+.4f}',
                             'digits_lost': abs(round(resid))})
        index.append({'kind': blk.kind, 'tstar': blk.tstar, 'eps': blk.eps,
                      'x1': blk.x1, 'rho_star': blk.rho, 'n_points': len(rows),
                      'quality': quality, 'n_suspect': len(flagged),
                      'note': blk.note, 'file': f'{blk.kind}/{blk.stem}.csv'})

    index.sort(key=lambda d: (d['kind'], d['tstar'], d['eps'], d['x1']))
    with (args.out / 'index.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0].keys()) if index else ['kind'])
        w.writeheader()
        w.writerows(index)
    if rejected:
        with (args.out / 'rejected.csv').open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rejected[0].keys()))
            w.writeheader()
            w.writerows(rejected)

    if suspects:
        with (args.out / 'suspect.csv').open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(suspects[0].keys()))
            w.writeheader()
            w.writerows(suspects)

    ok, msg = selftest(kept)
    n_dcf = sum(d['kind'] == 'dcf' for d in index)
    n_rdf = sum(d['kind'] == 'rdf' for d in index)
    n_flagged = sum(1 for d in index if d['n_suspect'])
    print(f'wrote  {n_dcf} DCF + {n_rdf} RDF tables -> {args.out}')
    print(f'rejected {len(rejected)} blocks (see rejected.csv)')
    print(f'flagged {len(suspects)} values in {n_flagged} tables as having lost a '
          f'leading digit (see suspect.csv) -- these are NOT repaired')
    print(f'self-test: {"PASS" if ok else "FAIL"} -- {msg}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
