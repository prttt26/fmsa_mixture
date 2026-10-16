# Directly-tabulated structure factor S(k) — none found anywhere

**No paper in the surveyed corpus tabulates a partial structure factor S_ij(k)
of a binary mixture as numbers.** The papers that mention S(k) only plot it or
define it. There is therefore no `literature_direct` S(k) data file; S(k) for any
state point is produced by converting from g_ij(r) or c_ij(r) with
`../../convert.py --to S_AL,S_BT`.

## S(k) conventions, recorded for later use

The converter emits two conventions (see `../../structure_transforms.py`):

- **Ashcroft–Langreth (symmetric)** — the internal / canonical form:
  `S_ij(k) = delta_ij + sqrt(rho_i rho_j) hhat_ij(k)`, i.e.
  `S = I + D^{1/2} Hhat D^{1/2} = (I - D^{1/2} Chat D^{1/2})^{-1}`.
  This matches the convention in `../../c22_outer_diagnosis.py`.
- **Bhatia–Thornton** — `S_NN`, `S_CC`, `S_NC`, the number/concentration form
  used by scattering experiments, with ideal limits `S_NN→1`, `S_CC→x1 x2`,
  `S_NC→0`.

## Definitions seen in the literature (for cross-referencing)

- **Zhou & Schweizer, JCP 150, 214902 (2019)** define, for a binary,
  `S_αβ(k) = x_α δ_αβ + ρ x_α x_β ĥ(k)` and use the
  *concentration–concentration* structure factor S_CC(k). This is a Kirkwood–
  Buff / number-fraction normalization; to compare against it, convert our
  Ashcroft–Langreth output to Bhatia–Thornton (`S_CC`) and mind the per-particle
  vs per-species normalization.
- **Enciso et al. (1987)** and **Caccamo, Phys. Rep. 274 (1996)** discuss
  structure factors of mixtures but tabulate none.

If a paper reporting numeric S_ij(k) at a useful size ratio is later found, drop
its table here (columns `k, S11, S12, S22` for Ashcroft–Langreth, or
`k, S_NN, S_CC, S_NC` for Bhatia–Thornton) and note the convention.
