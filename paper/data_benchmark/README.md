# `data_benchmark/` — extracted benchmark data and the scripts that make it

Machine-readable structure data for binary Lennard-Jones mixtures, for scoring
FMSA / two-Yukawa mixture models. Provenance and the wider source survey live in
[`../numerical_notes/sources/benchmark_data_source_TN1331.md`](../numerical_notes/sources/benchmark_data_source_TN1331.md).

## Source

**M. L. Huber & J. F. Ely, "Properties of Lennard-Jones Mixtures at Various
Temperatures and Energy Ratios With a Size Ratio of Two," NIST Technical Note
1331, May 1989, 426 pp.** Public domain —
<https://nvlpubs.nist.gov/nistpubs/Legacy/TN/nbstechnicalnote1331.pdf>

Isothermal **molecular dynamics** (864 particles; *not* GCMC, despite the label
used elsewhere in this repo), binary LJ, σ22/σ11 = 2, Lorentz–Berthelot mixing.
Appendix A tabulates g_ij(R\*) and local compositions, Appendix C tabulates
c_ij(R\*) which the report obtained by OZ-inverting its own g_ij.

This is the source of the `MC_DATA` array in `FMSA_MC.m` — specifically the
Appendix C block `T*=1.0, ρ*=0.139, x1=0.2500, ε22/ε11=0.50`.

**R\* is the per-pair reduced separation** R\* = r/σ_ij, so all three pairs reach
contact near R\* ≈ 1. For an absolute distance multiply by
σ_ij = [[1.0, 1.5], [1.5, 2.0]]. Reading column 0 as an absolute r is wrong and
is the easiest mistake to make with these tables.

## Files

### Physics core & converters

| | |
|---|---|
| `structure_transforms.py` | the shared physics: 3-D radial transforms, matrix OZ, Ashcroft–Langreth & Bhatia–Thornton S(k), and the `Correlations` hub. No I/O; imported by the rest. |
| `convert.py` | general CLI to convert between g/h/c (real), ĥ/ĉ (Fourier) and S_AL/S_BT — from a TN 1331 state or any CSV → `converted/<label>/` |
| `oz_invert.py` | TN 1331 g↔c regression/validation tool (forward reproduces App. A to ~0.1%); thin wrapper over the core |
| `c22_outer_diagnosis.py` | (pre-existing) the c22 outer-region artifact analysis |
| `build_dcf_pool.py` | assemble every source into one DCF pool → `dcf_pool/pool_index.csv`, converting the RDF tables on the way and grading each entry's outer region by the tail scale `s_ij`. Feeds `../compare_models_dcf_multistate.py`. |

### Data extraction

| | |
|---|---|
| `extract_tn1331.py` | parses NIST TN 1331's PDF text into `tn1331/{dcf,rdf}/*.csv` |
| `extract_literature.py` | Yu & Jin contact RDF + Ely thermo → `literature_direct/` (self-tested) |
| `digitize_figure.py` | trace colour-separated g_ij(r) curves from a figure → `literature_direct/rdf/` |
| `md/` | **generate** our own g_ij(r) by LAMMPS MD (validated vs TN 1331 + Ely), for other size ratios the literature lacks; see `md/README.md` |

### Data directories

| | |
|---|---|
| `tn1331/` | primary tabulated DCF+RDF, size ratio 2 (39 DCF + 54 RDF; `index.csv` grades each; `rejected.csv`/`suspect.csv` flag OCR damage). DCF cols `R_star,c11,c12,c22`; RDF cols `R_star,g11,g12,g22,xl11,xl12,xl22`. |
| `literature_direct/` | data taken directly from other papers (Yu & Jin, Ely, digitized figures); see its own README + fidelity tags |
| `converted/` | outputs of `convert.py`, one subdir per run with a `manifest.json` |
| `dcf_pool/` | `pool_index.csv` — one row per benchmark state point, normalised to absolute r with its state parameters, fidelity grade, per-pair scored range and `outer_trust`. Built by `build_dcf_pool.py`. |

Regenerate with:

```bash
python extract_tn1331.py --pdf /path/to/tn1331.pdf --out tn1331
python extract_literature.py                              # literature_direct tables
python convert.py --state T1_eps0.50_x0.2500 --from g --to all   # -> converted/
```

## `convert.py` — interconverting structure quantities

Given one representation of the correlations, write out any of the others. The
physics is `structure_transforms.Correlations`: every input reduces to the
canonical node Ĥ(k) and every output derives from it, so all pairwise
conversions (RDF ↔ DCF ↔ S(k), real ↔ Fourier) are available.

- `--from {g,h,c,hk,ck,S_AL}` `--to {g,h,c,hk,ck,S_AL,S_BT,all}`
- input from a TN 1331 `--state`, or any `--input file.csv`
- `--r-convention perpair` (TN 1331: column 0 = R*=r/σ_ij) or `absolute` (own MD)
- densities from `index.csv` (for `--state`) or `--rho`/`--x1`; pure Fourier
  transforms (g↔ĥ, c↔ĉ) need none, OZ/S conversions do
- S(k) in two conventions: **Ashcroft–Langreth** (matches `c22_outer_diagnosis.py`)
  and **Bhatia–Thornton** (S_NN/S_CC/S_NC)

The same truncation caveat applies as for `oz_invert.py --direction inverse`:
converting a truncated real-space input to c or to k space loses long-range h and
the inverse transform amplifies it (~450× onto ĉ22(0) at ρ*=0.139). Feed g
extended to ~half the box length; `convert.py` records this in each `manifest.json`.

## Data quality — read this before using a table

TN 1331 is a **1989 scan**. The embedded OCR text layer is good on many pages and
badly damaged on others, so the extractor validates rather than trusts, and every
table carries a grade. Current yield: **39 DCF + 54 RDF tables** out of ~84 state
points; 75 blocks were rejected outright.

`quality` in `index.csv`:

- `OK` — every R\* point of the report's grid recovered
- `PARTIAL` — ≥90 % of points recovered; `note` says how many are missing
- `SPARSE` — ≥70 % recovered

Points that survive are trustworthy in the sense that each was checked to land on
the report's own 0.025 R\* grid, and any block where a neighbouring table's rows
bled in was failed outright. **Missing points are simply absent from the CSV** —
nothing is interpolated or invented.

### The failure mode that validation cannot see

The scan sometimes drops the *integer part* of a whole run of values in one
column, turning `1.8442` into `0.8442`. Such a value is well-formed and lands on
the grid, so it passes every structural check. `suspect.csv` flags these by
predicting each point from its two predecessors and testing whether the residual
is a non-zero integer — **358 values across 63 tables are flagged, and none are
repaired.** Flags cluster at the two ends of a corrupted run, so treat a flag as
marking a suspect *region* of that column and check it against the source page.

A worked example is `rdf/T1_eps0.50_x0.2500.csv`, column `g12`, R\* = 1.200–1.375:
the tabulated `0.8442 … 0.0519` should read `1.8442 … 1.0519`. This was not found
by inspection — it was caught by the forward OZ check described below, which is
the most reliable detector available for any state point having both tables.

**13 tables are both `OK` and unflagged**; `index.csv` lets you select them:

```python
import csv
clean = [r for r in csv.DictReader(open('tn1331/index.csv'))
         if r['quality'] == 'OK' and r['n_suspect'] == '0']
```

### Recovering the rejected blocks

The 75 rejected blocks are lost to the *embedded* OCR layer, not to the page
images. Re-OCRing the scan (`ocrmypdf --redo-ocr`, or tesseract on
`pdftoppm`-rendered pages) is the route to the remaining state points; neither
tool is installed here, so this has not been attempted.

## `oz_invert.py`

Solves the multicomponent OZ equation, `H = C + C·D·H` with `D = diag(ρ_i)`, in
Fourier space, mapping each pair from its own R\* grid onto a common absolute-r
grid first.

**`--direction forward` (c → g) is the regression test.** It is well conditioned
because c_ij is short ranged, and on TN 1331's own tables it reproduces Appendix A
from Appendix C to ~0.1 %:

```
mean |ours - TN|:  g11 = 0.0054,  g22 = 0.0019
```

That single number validates the extracted tables, the per-pair R\* → r
conversion, the density assignment, and the transforms together. Run it after any
change to the extractor.

**`--direction inverse` (g → c) is what we want for new simulation data, but it
cannot reproduce Appendix C from Appendix A** — for reasons that are physical,
not numerical. h_ij = g_ij − 1 is long ranged and the published tables stop at
R\* = 2.5, which for pair (2,2) is only r = 5 σ11, where h is still oscillating.
Huber and Ely inverted the h they had in the simulation, not the h they printed:
at 864 particles and ρ\* = 0.139 the box is ≈18 σ11 across, so their h reached
roughly 9 σ11. Zero-padding at 5 σ11 discards real structure and the inverse
transform amplifies the loss.

So: **use the inverse direction on our own MD output**, where g_ij can be written
out to half the box length — not on the published tables.

```bash
python oz_invert.py --state T1_eps0.50_x0.2500 --direction forward --plot fwd.png
python oz_invert.py --state T1_eps0.50_x0.2500                      # inverse
```

## Bearing on the c22 outer-region question

`../numerical_notes/results/single_point/oz_closures_comparison.md` §4 recorded that the
tabulated c_ij exceed the rigorous `c → −βu` limit beyond r ≈ 1.3 σ, worst for
(2,2). Two things are now settled:

1. The excess is **not** an inconsistency between the report's g and its c — the
   forward check shows Appendix A and Appendix C satisfy the OZ equation
   together to ~0.1 %.
2. It is therefore a property of the report's own inversion or of the long-range
   h that fed it, and cannot be re-derived or corrected from the published
   tables alone, because those stop well short of the h the inversion used.

`c22_outer_diagnosis.py` closes the question over the whole extracted dataset;
write-up in
[`../numerical_notes/rootcause/c22_outer_inversion_artifact.md`](../numerical_notes/rootcause/c22_outer_inversion_artifact.md).
Three results that change how these tables should be used:

- **The excess is density-driven and disappears below η ≈ 0.25.** The 11 tables
  at low packing satisfy `c = −βu` on all three pairs to 0.8–1.2. That makes them
  the only usable outer-region benchmark here — and raises the value of re-OCRing
  the rejected blocks, since most low-η tables are `PARTIAL`/`SPARSE`.
- **The reference `ĉ_ij(0) = +1.990 / −22.07 / −100.80` fails a compressibility
  check.** They give χ_T/χ_id = 0.10 where MSA, PY, HNC and HNCB all give
  0.056–0.080; correcting the c22 tail alone closes the gap.
- **Mechanism:** the k → 0 inversion amplifies by `M₂₂² = (1 − ρ₂ĉ₂₂(0))²`, which
  is 453 at ρ\* = 0.139 / T\* = 1 and only ~2 at low density. A δh₂₂ ≈ 10⁻⁴ error
  in the far tail is enough. **When inverting our own MD, write g_ij out to half
  the box length** — truncating h is not a small approximation at high η.

## `dcf_lowk.py` — the long-wavelength repair (supersedes the caveat above)

The paragraph above says the outer region "cannot be corrected from the
published tables alone". That is true of the *tables* and false of the
*problem*: what is missing at small k is not structural information but a
**bulk macroscopic** one, and the equation of state has it.

`dcf_lowk.py` takes each part of the k axis from wherever it is reliable —
simulation above `k_split = 1` (measured on the N = 512…8192 ladder), the
thermodynamic `ĉ_ij(0)` at k = 0, an even quartic between, and the exact
asymptote `c_ij → −βu_ij` outside a per-pair keep radius read off the artifact
level. Method after Zhao & Wu, *Mol. Phys.* **109**, 2553 (2011).

    python dcf_lowk.py --label validation                # one MD run
    python dcf_lowk.py --literature T1_eps0.50_x0.2500   # one TN 1331 table
    python dcf_lowk.py --all                             # 562 states, writes c_lowk.csv
    python dcf_lowk_figures.py

At the validation state the raw inversion gives χ_T/χ_id = **−0.025**
(impossible), the TN 1331 table 0.105, and the anchor 0.0552 — the measured EOS
value. Checked against things the repair never sees: the repaired `c` moves
**−80 % / −78 %** closer to TN 1331 Appendix C in the core on the (1,2) and
(2,2) pairs, and its `S₂₂(0)` agrees with the N → ∞ size-ladder extrapolation to
1.3 %.

The two halves are applied separately, because only one of them needs the
anchor: the exact asymptote runs on all 562 states, the `ĉ(0)` level correction
only where the pressure surface resolves it (231 / 253 / 227 of 562 by pair).
Scored as a third reference in `compare_models_dcf_multistate.py`, **the model
ranking is invariant** across `published` / `c22_mfphi` / `lowk`, while every
model moves closer to the benchmark on the compressibility path.

Outputs: `converted/md_<label>/c_lowk.csv`, `tn1331/dcf_lowk/<stem>.csv`,
index `dcf_lowk_index.csv`; write-up
[`../numerical_notes/results/benchmark/repair_lowk/dcf_lowk_repair.md`](../numerical_notes/results/benchmark/repair_lowk/dcf_lowk_repair.md).
