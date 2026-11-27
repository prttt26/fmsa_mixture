# Revision additions: scripts and inputs for the numbers added in the revision

Self-contained: `code/` holds the new scripts together with the module versions they run against, in the layout of
the authors' working tree, and the cached inputs they read are at the same relative paths. The files one directory
up (`paper/*.py`) are earlier snapshots and are not used here. Run every command **from this directory**:

```
python code/run.py <script>.py [options]
```

Python 3 with NumPy, SciPy, pandas and matplotlib.

| quantity in the paper / Supplementary Material | command | reads |
|---|---|---|
| uncertainty of the core comparison: intervals, paired differences, reference, pair and source breakdowns | `dcf_core_uncertainty_tables.py` | `.../all_samples/dcf_multistate_metrics.csv` |
| comparison on the states where OZ-HNCB converges: failure set, three bases | `hncb_converged_only.py` | same |
| hard-core two-Yukawa pool (Yu and Jin): contact and Z errors, intervals, paired differences, cases | `compare_models_hcty.py --workers 8` (rescore), then `hcty_uncertainty.py` | `data_benchmark/.../yu_jin_2008.csv`, `.../literature/hcty_metrics.csv` |
| truncation error against full MSA over state space | `truncation_state_dependence.py` (four grids, below), `truncation_critical_temperature.py`, `truncation_analysis.py` | computed |
| explicit closed form of the first-order DCF (main text) | `verify_explicit_closed_form.py` | computed |
| tau^2 scaling of the residual against exact MSA (about 10 minutes) | `verify_tau_scaling.py` | computed |

The four grids of the truncation scan:

```
S=truncation_state_dependence.py
python code/run.py $S --system binary
python code/run.py $S --system pure --Tmin 0.6 --Tmax 3.0 --nT 26 --nrho 26
python code/run.py $S --system binary --Tmin 0.42 --Tmax 0.75 --nT 34 --nrho 40 --rhomax 0.10 --tag _crit
python code/run.py $S --system pure --Tmin 0.85 --Tmax 1.35 --nT 30 --nrho 40 --rhomax 0.50 --tag _crit
```

`truncation_state_dependence.py` evaluates the exact MSA core by a discrete transform; only its L2-norm columns are
quoted. The maximum-norm values of the reference state come from `../probe_truncation_cost_exact.py`
(grid-free quadrature).
