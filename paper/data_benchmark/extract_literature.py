#!/usr/bin/env python3
"""Extract directly-tabulated structure data from the paywalled literature.

Only data that a paper prints as an actual NUMERIC TABLE is taken here; figure
curves are handled separately by digitize_figure.py, and the honest negatives
(no tabulated full g(r)/c(r) beyond TN 1331, no tabulated S(k) anywhere) are
recorded in literature_direct/{dcf,sk}/NOTES.md.

What this pulls
---------------
1. Yu & Jin (2008), J. Chem. Phys. 128, 014901 -- Tables I-VIII: the radial
   distribution function AT CONTACT g_ij(sigma_ij), internal energy U/NkT, and
   compressibility factor Z, for a binary hard-core two-Yukawa (HCTY) mixture.
   This is our exact model class (two-Yukawa mixture) benchmarked against MC, at
   size ratio 2 and energy ratio 0.5 -- the closest published validation of an
   FMSA-type mixture solver.  -> literature_direct/rdf_contact/yu_jin_2008.csv

2. Ely (1986), Int. J. Thermophys. 7, 381 -- the LJ-mixture thermodynamic
   reference table (pressure / energy) for the same MD dataset that became
   TN 1331.  -> literature_direct/thermo/ely_1986.csv   (thermodynamics, not
   structure; kept for compressibility / energy cross-checks).

Convention note (Yu & Jin): species 1 is the LARGE particle (sigma2/sigma1 =
0.5), so their g11 is large-large and g22 small-small -- the opposite labelling
to TN 1331, where species 2 is large.  The size-ratio MAGNITUDE is 2 in both.

Source PDFs live in pdf/sources_paywall/ (clean text layer, no OCR needed).

Usage
-----
    python extract_literature.py                      # uses pdf/sources_paywall
    python extract_literature.py --yujin-txt a.txt --ely-txt b.txt
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
PAYWALL = HERE.parent / 'pdf' / 'sources_paywall'
YUJIN_PDF = (PAYWALL / 'Thermodynamic and structural properties of mixed colloids '
             'represented by a hard-core two-Yukawa mixture model fluid Monte Carlo '
             'simulations and an analytical theory.pdf')
ELY_PDF = (PAYWALL / 'A test of the mean density approximation for Lennard-Jones '
           'mixtures with large size ratios.pdf')

# Per-table metadata for Yu & Jin, read off the captions (the sigma/eps/lambda
# glyphs are OCR-mangled in the text layer, so the fixed parameters are recorded
# here rather than re-parsed; the varying column is 'var').  All tables:
# sigma2/sigma1 gives size-ratio 2, energy ratio 0.5.  lam/k describe the
# two-Yukawa potential exactly as printed.
YUJIN_TABLES = {
    'I':    dict(case=1, sigma2_over_sigma1=0.5, eps2_over_eps1=0.5, x1=None,
                 var='x1', lam='lam1=1.8, lam2,1=4.0', k='k1=k2=1.0'),
    'II':   dict(case=2, sigma2_over_sigma1=0.5, eps2_over_eps1=0.5, x1=None,
                 var='x1', lam='lam1=1.8, lam2,1=8.0', k='k1=k2=-6.0'),
    'III':  dict(case=3, sigma2_over_sigma1=0.5, eps2_over_eps1=0.5, x1=None,
                 var='x1', lam='lam1=1.8, lam2,1=8.0', k='k1=k2=-12.0'),
    'IV':   dict(case=1, sigma2_over_sigma1=None, eps2_over_eps1=0.5, x1=0.5,
                 var='sigma2_over_sigma1', lam='lam1=1.8, lam2,1=4.0', k='k1=k2=1.0'),
    'V':    dict(case=2, sigma2_over_sigma1=None, eps2_over_eps1=0.5, x1=0.5,
                 var='sigma2_over_sigma1', lam='lam1=1.8, lam2,1=8.0', k='k1=k2=-6.0'),
    'VI':   dict(case=1, sigma2_over_sigma1=0.5, eps2_over_eps1=None, x1=0.5,
                 var='eps2_over_eps1', lam='lam1=1.8, lam2,1=4.0', k='k1=k2=1.0'),
    'VII':  dict(case=2, sigma2_over_sigma1=0.5, eps2_over_eps1=None, x1=0.5,
                 var='eps2_over_eps1', lam='lam1=1.8, lam2,1=8.0', k='k1=k2=-6.0'),
    'VIII': dict(case=4, sigma2_over_sigma1=0.5, eps2_over_eps1=0.5, x1=None,
                 var='x1', lam='lam1=2.8647, lam2,1=13.5485', k='k1=k2=-1.4466'),
}

# One row known to be correct, to self-test the parse (Table I, first data row).
YUJIN_SELFTEST = dict(table='I', T_star=1.5, x1=0.5, rho_star=0.3,
                      g11=4.249, g22=2.113, g12=2.839, U_NkT=-1.058, Z=0.537)

FLOAT = re.compile(r'-?\d*\.\d+|-?\d+')


def pdf_to_text(pdf: Path, txt_override: Path | None) -> list[str]:
    if txt_override is not None:
        return txt_override.read_text(errors='replace').split('\n')
    if not pdf.exists():
        raise SystemExit(f'missing PDF: {pdf}')
    if shutil.which('pdftotext') is None:
        raise SystemExit('pdftotext not found (poppler-utils); pass a --*-txt override')
    cache = pdf.with_suffix('.txt')
    if not cache.exists():
        subprocess.run(['pdftotext', '-layout', str(pdf), str(cache)], check=True)
    return cache.read_text(errors='replace').split('\n')


def floats(line: str) -> list[float]:
    """Clean numeric tokens on a line, minus-sign glyphs and gaps normalised."""
    s = line.replace('−', '-').replace('–', '-')   # unicode minus / en-dash
    # the scan often prints a negative as "- 5.340" or "--4.766"; glue the sign
    # to its number so the value keeps its sign (else it reads as +5.340).
    s = re.sub(r'-\s+(?=\d)', '-', s)
    return [float(t) for t in FLOAT.findall(s)]


def plausible(t, var, rho, g11, g22, g12, u, z) -> bool:
    return (0.05 < t < 10 and 0 <= var <= 3 and 0 < rho < 1.5
            and all(0.1 < g < 40 for g in (g11, g22, g12))
            and -10 < u < 5 and -5 < z < 40)


def parse_yujin(lines: list[str]) -> list[dict]:
    """Walk the eight tables, forward-filling the sparse T* / varying columns."""
    # locate each "TABLE <roman>." caption line
    roman = re.compile(r'TABLE\s+(I|II|III|IV|V|VI|VII|VIII)\.')
    starts = [(m.group(1), i) for i, ln in enumerate(lines)
              for m in [roman.search(ln)] if m]
    starts.append((None, len(lines)))

    # A table's data ends where the figures / running text begin; captions there
    # ("... sigma2/sigma1 = 0.5, ... x1 = 0.5,") can otherwise parse as 8 floats.
    boundary = re.compile(r'FIG\.|Comparison of|as a function of|Helmholtz|'
                          r'reference and the|^\s*[A-Z]\.\s')

    rows: list[dict] = []
    for (name, i0), (_, i1) in zip(starts, starts[1:]):
        meta = YUJIN_TABLES[name]
        t_cur = var_cur = None
        started = False
        for ln in lines[i0:i1]:
            if started and boundary.search(ln):
                break
            f = floats(ln)
            if len(f) == 8:
                t_cur, var_cur, rho, *data = f
            elif len(f) == 6 and t_cur is not None:
                rho, *data = f
            else:
                continue
            g11, g22, g12, u, z = data
            if not plausible(t_cur, var_cur, rho, g11, g22, g12, u, z):
                continue
            row = dict(table=name, case=meta['case'],
                       sigma2_over_sigma1=meta['sigma2_over_sigma1'],
                       eps2_over_eps1=meta['eps2_over_eps1'], x1=meta['x1'],
                       T_star=t_cur, rho_star=rho,
                       g11=g11, g22=g22, g12=g12, U_NkT=u, Z=z,
                       lambda_set=meta['lam'], k_set=meta['k'])
            row[meta['var']] = var_cur          # resolve the varying column
            rows.append(row)
            started = True
    return rows


def selftest_yujin(rows: list[dict]) -> tuple[bool, str]:
    want = YUJIN_SELFTEST
    for r in rows:
        if (r['table'] == want['table'] and abs(r['T_star'] - want['T_star']) < 1e-9
                and abs(r['x1'] - want['x1']) < 1e-9
                and abs(r['rho_star'] - want['rho_star']) < 1e-9):
            for key in ('g11', 'g22', 'g12', 'U_NkT', 'Z'):
                if abs(r[key] - want[key]) > 5e-4:
                    return False, f'{key}: {r[key]} != {want[key]}'
            return True, 'Table I first row matches'
    return False, 'self-test row not found'


ELY_N_TOTAL = 864       # all runs used 864 particles; composition x = N1 / 864


def parse_ely(lines: list[str]) -> list[dict]:
    """Ely 1986 Table I: 864-LJ-particle thermodynamics, size ratio 2, T*=1.

    Columns (from the table's own footnote): N1, eps22/eps11, P* = P sigma11^3/eps11,
    rho* = rho sigma11^3, T* = kT/eps11, U* = U/(N eps11).  N1 is sparse (written
    once per block, carried down); N1/864 gives the mole fraction, which lands on
    exactly TN 1331's composition grid.  Space-separated minus signs on U* are
    repaired by floats().
    """
    grab = re.compile(r'Table\s+I\.', re.I)
    i0 = next((i for i, ln in enumerate(lines) if grab.search(ln)), None)
    if i0 is None:
        return []
    i1 = next((i for i, ln in enumerate(lines[i0 + 1:], i0 + 1)
               if re.search(r'Table\s+II\.', ln, re.I)), len(lines))

    rows, n1 = [], None
    for ln in lines[i0:i1]:
        f = floats(ln)
        if len(f) == 6:                     # N1, eps, P*, rho*, T*, U*
            n1, eps, p, rho, t, u = f
        elif len(f) == 5 and n1 is not None:  # eps, P*, rho*, T*, U* (N1 carried)
            eps, p, rho, t, u = f
        else:
            continue
        # sanity: N1 an integer 1..864, eps in {0.5,1,1.5}, T*=1, densities small
        if not (1 <= n1 <= ELY_N_TOTAL and 0.3 < eps < 2 and 0.9 < t < 1.1
                and 0.05 < rho < 0.8 and -20 < u < 0 and 0 < p < 3):
            continue
        rows.append(dict(N1=int(round(n1)), x=round(n1 / ELY_N_TOTAL, 4),
                         eps22_over_eps11=eps, P_star=p, rho_star=rho,
                         T_star=t, U_star=u))
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict],
              header_comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fh:
        for line in header_comment.strip('\n').split('\n'):
            fh.write(f'# {line}\n')
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--yujin-txt', type=Path, default=None)
    ap.add_argument('--ely-txt', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=HERE / 'literature_direct')
    args = ap.parse_args()

    # --- Yu & Jin contact RDF + thermodynamics ---
    yj_rows = parse_yujin(pdf_to_text(YUJIN_PDF, args.yujin_txt))
    ok, msg = selftest_yujin(yj_rows)
    fields = ['table', 'case', 'sigma2_over_sigma1', 'eps2_over_eps1', 'x1',
              'T_star', 'rho_star', 'g11', 'g22', 'g12', 'U_NkT', 'Z',
              'lambda_set', 'k_set']
    write_csv(args.out / 'rdf_contact' / 'yu_jin_2008.csv', fields, yj_rows,
              header_comment=(
                  'Yu & Jin, J. Chem. Phys. 128, 014901 (2008), Tables I-VIII.\n'
                  'Binary hard-core two-Yukawa (HCTY) mixture, Monte Carlo.\n'
                  'Contact RDF g_ij(sigma_ij), internal energy U/NkT, compressibility Z.\n'
                  'Species 1 = LARGE (sigma2/sigma1 = 0.5 => size-ratio magnitude 2).\n'
                  'fidelity: tabular-exact.  lambda_set/k_set define the two-Yukawa potential.'))
    print(f'Yu & Jin: {len(yj_rows)} rows -> rdf_contact/yu_jin_2008.csv')
    print(f'  self-test: {"PASS" if ok else "FAIL"} -- {msg}')

    # --- Ely 1986 thermodynamic reference (best-effort) ---
    try:
        ely = parse_ely(pdf_to_text(ELY_PDF, args.ely_txt))
    except SystemExit as e:
        ely, msg_e = [], str(e)
    else:
        msg_e = f'{len(ely)} candidate rows'
    if ely:
        write_csv(args.out / 'thermo' / 'ely_1986.csv',
                  ['N1', 'x', 'eps22_over_eps11', 'P_star', 'rho_star',
                   'T_star', 'U_star'], ely,
                  header_comment=(
                      'Ely, Int. J. Thermophys. 7, 381 (1986), Table I.\n'
                      'LJ mixture, size ratio 2, 864-particle isothermal MD -- the same\n'
                      'dataset that became NIST TN 1331.  Thermodynamics only.\n'
                      'N1 = number of species-1 particles; x = N1/864 (matches TN 1331 grid).\n'
                      'P* = P sigma11^3/eps11, rho* = rho sigma11^3, T* = kT/eps11,\n'
                      'U* = U/(N eps11).  fidelity: tabular-exact.'))
        msg_e = f'{len(ely)} rows'
    print(f'Ely 1986 thermo: {msg_e}'
          + (' -> thermo/ely_1986.csv' if ely else ' (skipped)'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
