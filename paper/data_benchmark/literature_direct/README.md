# `literature_direct/` — structure data taken directly from the literature

Benchmark structure data for size-asymmetric binary mixtures, extracted from
published papers. This complements `../tn1331/`, which holds the primary
directly-tabulated DCF+RDF source (NIST TN 1331, size ratio 2). Everything here
is data that some *other* paper prints directly; the honest finding of the
literature survey (`../../numerical_notes/sources/benchmark_data_source_TN1331.md`)
is that such data is scarce.

## Fidelity tags

Every file is one of two tiers, stated in its header comment:

- **`tabular-exact`** — copied from a numeric table in the paper. Trustworthy to
  the paper's own precision.
- **`digitized-approx`** — traced from a figure by `../digitize_figure.py`.
  Accurate to ~1–2% where colour separation is clean; inspect the `*.qa.png`
  overlay before using.

## Contents

### `rdf_contact/yu_jin_2008.csv` — tabular-exact ⭐
Yu & Jin, *J. Chem. Phys.* **128**, 014901 (2008), Tables I–VIII. Contact RDF
`g_ij(σ_ij)`, internal energy `U/NkT`, compressibility `Z` for a binary
**hard-core two-Yukawa (HCTY) mixture** by Monte Carlo — *our exact model class*,
benchmarking **FMSA** against MC. 145 rows spanning size ratio (Tables IV/V),
energy ratio (VI/VII), composition, temperature, density. Base cases at size
ratio 2, ε ratio 0.5. **Table VIII (case 4)** uses λ = 2.8647 / 13.5485, close to
this project's LJ→two-Yukawa mapping (`FMSA_MC.m`: 2.9637 / 14.0167).
Convention: **species 1 = LARGE** (σ2/σ1 = 0.5), opposite to TN 1331's labelling.
The most direct published check available for our solver's contact values.

### `rdf/yu_jin_2008_fig13.csv` (+ `.qa.png`) — digitized-approx
Yu & Jin Fig. 13, the full `g_ij(r)` curves (g11 solid, g22 dashed, g12 dotted;
colour-separated). HCTY, size ratio 2, ε ratio 0.5, **T\*=6.0, ρ\*=1.3 (case 2)**
— a dense, hot, off-target state, kept as a full-curve example. Digitized peaks
reproduce the figure to ~1–2% (g11 2.56@1.05, g22 2.00@0.55, g12 2.16@0.80; all
→1.0 at large r). Blank cells = inside a pair's hard core.

### `thermo/ely_1986.csv` — tabular-exact
Ely, *Int. J. Thermophys.* **7**, 381 (1986), Table I. LJ mixture, size ratio 2,
864-particle isothermal MD — the *same dataset that became TN 1331*. Pressure P\*,
energy U\*, density ρ\* over 7 compositions × 3 energy ratios. `x = N1/864`
reproduces TN 1331's composition grid exactly (cross-validation), and its ρ\*
values confirm TN 1331's densities (e.g. x=0.25, ε=0.5 → ρ\*=0.1386 ≈ 0.139).
Thermodynamics only — for compressibility / energy cross-checks, not structure.

### `dcf/NOTES.md`, `sk/NOTES.md` — honest negatives
No paper tabulates full `c_ij(r)` beyond TN 1331, and none tabulates `S_ij(k)` at
all. Both are produced by converting g or c with `../convert.py`. The NOTES record
the S(k) definitions each paper uses.

## Not extracted, and why

- **Ben-Naim & Mountain 2008** (JCP 128, 214504; open PDF at
  `../../pdf/open_access/`), g_ij(r) at **size ratio 1.5** — the most valuable
  new size ratio — and **You/Yu/Gao 2005** (JCP 123, 114705; open PDF there too),
  g_ij(r) at ratio 1.43. Both figures are **grayscale** (solid theory + dotted
  simulation, all black), so the colour-separated digitizer cannot split the
  three pairs. They need manual/semi-manual digitization; the PDFs are saved for
  that. This is a genuine tooling limit, not an oversight.
- Figs. 11/12 of Yu & Jin plot the *contact* values already in
  `rdf_contact/yu_jin_2008.csv`; Figs. 9/10 plot U and Z, also already tabulated.

## Regenerate

```bash
python ../extract_literature.py                 # rdf_contact + thermo (self-tested)
python ../digitize_figure.py --png fig13.png ... # rdf/*.csv  (see the script header)
```
