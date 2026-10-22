# Paper data generators

Everything that produces the figures and quoted numbers of

> R. Pan and J. Wu, *The First-Order Mean Spherical Approximation for Hard-Core
> Yukawa Mixtures: the Direct Correlation Function in Finite Closed Form*
> (in preparation),

with the input data it reads. The package one level up (`../fmsa_double_prop.py`
and friends) is the method on its own; this directory is the research code around
it, kept as it was when the paper's numbers were produced.

The directory mirrors the layout of the authors' working tree, because the scripts
locate their data relative to their own file: run every command **from this
directory** (`paper/`). Comments and docstrings refer to the authors' internal notes
(`numerical_notes/...`, lecture notes, Lean files); those references can be ignored.

## Requirements

Python 3 with NumPy, SciPy, pandas and matplotlib. The reference molecular dynamics
was run with LAMMPS (12 Jun 2025); it is only needed to regenerate the simulation
data, which is included. Set `MPLBACKEND=Agg` on a machine without a display.

## Layout

```
*.py                         figure scripts, benchmark drivers, solvers (flat; they import each other)
data_benchmark/              reference data and the scripts that built it
  md/runs/<run>/             one LAMMPS run each: in.lammps, params.json, g.csv, thermo.json
  md/run_md.py, run_grid.py  the MD drivers (state list in md/states.py)
  tn1331/                    DCF and RDF tables of NIST TN 1331, Appendix C, as extracted and repaired
  literature_direct/         digitised literature values
  reference_*.json           Caccamo et al. 1998, Shukla 2000 (and Mkanya et al. 2017, not used in the paper)
  dcf_pool/, rdf_pool/       the benchmark state pools
numerical_notes/results/     cached results the figure scripts read (JSON/CSV)
FMSA_dp_output/scripts/      construction check quoted in the paper (Sec. IV A)
```

Not included: `data_benchmark/converted/`, the DCFs obtained from the MD and TN 1331
RDFs by the low-k-repaired Ornstein-Zernike inversion (about 110 MB). Rebuild it with

```
python data_benchmark/dcf_lowk.py --all                                  # every MD run and anchorable TN 1331 table -> converted/*/c_lowk.csv
python data_benchmark/convert.py --state <TN1331 stem> --from g --to all # one TN 1331 RDF table -> converted/
```

(`data_benchmark/README.md` describes the whole pipeline.)

before rerunning the DCF benchmark from scratch. The figure scripts below redraw from
the cached metric tables and do not need it.

## Figures

Figure numbers are those of the current manuscript; `figure_split.py --pub` cuts a
script's composite figure into the per-panel PDFs used in the paper, and running the
script directly gives the composite.

| figure | redraw (from the cached results) | regenerate the results first |
|---|---|---|
| 1 | TikZ in the manuscript, no data | |
| 2 | `figure_split.py --pub figure_rdf_support_gap.py probe_antisym.py` | (computed on the fly) |
| 3 | `figure_split.py --pub probe_antisym.py`; `figure_antisym_explain.py` | (computed on the fly) |
| 4 | `figure_split.py --pub --pub-cols 2 figure_pure_limit_anchor.py` | (computed on the fly) |
| 5, 6, 8 | `figure_split.py --pub --pub-cols 2 --script-args='--ref lowk' dcf_multistate_figures.py`, `figure_split.py --pub --pub-cols 2 rdf_multistate_figures.py` | `compare_models_dcf_multistate.py`, `compare_models_rdf_multistate.py` |
| 7 | `figure_split.py --pub compare_speed_dp_hncb.py --script-args='--replot'` | `compare_speed_dp_hncb.py` (**re-times**: the speed-ups will drift) |
| 9 | `figure_vle_hcy.py` | `vle_hcy_compr.py`, `vle_hcy_hncb.py` |
| 10 | `figure_lle_caccamo.py` | `compare_caccamo_lambda_line.py`, `compare_caccamo_demixing_dome.py`, `compare_caccamo_domes_models.py` (~50 min), `gen_caccamo_ray_route.py` |
| S1 | `figure_split.py --pub --pub-cols 2 figure_msa_fold_vdw.py` | `fmsa_vdw_loop.py`, `msa_exact_phase_lines.py`, `mixture_fold_floor.py` |
| S2 | `figure_fmsa_vdw_family.py --panels` | `fmsa_vdw_loop_family.py` |
| S3 | `figure_split.py --pub probe_error_masking.py` | (computed on the fly) |
| S4 | `figure_incore_split.py` | |
| S5 | `figure_split.py --pub --pub-cols 2 figure_speed_tasks.py` | `speed_free_energy.py`, `solvability_exact_msa.py` (timings: run on a quiet machine) |
| S6, S7 | `figure_dcf_samples.py`, `figure_rdf_samples.py` | as for 5, 6, 8 |

## Quoted numbers

| quantity | script |
|---|---|
| construction check, max. relative discrepancy 1.1e-13 (Sec. IV A) | `FMSA_dp_output/scripts/check_dcf_mixture_construction.py` |
| truncation cost against exact MSA (Sec. IV B) | `probe_truncation_cost_exact.py` |
| low-k repaired DCF reference | `data_benchmark/dcf_lowk.py` |
| coupling-series radius $R_c$ and fold ratio (SI, Convergence) | `msa_series_radius.py` (`main`, `cmd_fold`) |
| fold vs. nearest singularity: monodromy, coefficient signs, competing-tail counterexample (SI, Convergence) | `series_radius_fold.py` -> `numerical_notes/results/series_radius_fold.json` |

The finite-order thermodynamics and the formal-verification checks quoted in the
Supplementary Material live with the Lean development,
<https://github.com/prttt26/FMSA_MSY_2YK_LEAN_code> (`scripts/`, `results/`).

## Simulation data

`data_benchmark/md/runs/` holds the LAMMPS input, the state parameters, the measured
partial RDFs and the thermodynamic averages of every run behind the benchmark pools.
The raw trajectories and logs are not included. `md/run_md.py` writes and runs one
state; `md/run_grid.py` the grid; `md/states.py` lists them.
