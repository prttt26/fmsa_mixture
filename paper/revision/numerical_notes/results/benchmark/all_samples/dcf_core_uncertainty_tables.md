# Uncertainty of the core comparison (cluster bootstrap, 5000 resamples, seed 20261125)

## Core NRMSE, repaired reference (per state: mean of the three pairs; HNCB on the states where it converged)

| model | n | median | 95% interval | IQR |
|---|---|---|---|---|
| HNCB | 478 | 0.0869 | [0.0788, 0.1069] | 0.063--0.168 |
| MSA | 519 | 0.0967 | [0.0864, 0.1129] | 0.067--0.215 |
| PYHS | 519 | 0.1004 | [0.0908, 0.1185] | 0.066--0.221 |
| WB | 519 | 0.0973 | [0.0885, 0.1101] | 0.060--0.203 |

## Paired differences of the core NRMSE (positive: first model has the larger error), same states

| first - second | n | median | 95% interval | first larger |
|---|---|---|---|---|
| PYHS - WB | 519 | +0.0074 | [+0.0050, +0.0111] | 341/519 |
| MSA - WB | 519 | +0.0112 | [+0.0055, +0.0146] | 323/519 |
| MSA - PYHS | 519 | -0.0011 | [-0.0030, +0.0005] | 236/519 |
| HNCB - WB | 478 | +0.0057 | [+0.0026, +0.0141] | 281/478 |
| HNCB - PYHS | 478 | -0.0020 | [-0.0045, +0.0009] | 217/478 |
| HNCB - MSA | 478 | -0.0019 | [-0.0037, +0.0002] | 210/478 |

## Reference treatment, same 519 states: PYHS - WB

| reference | median | 95% interval |
|---|---|---|
| published | +0.0035 | [-0.0010, +0.0057] |
| lowk | +0.0074 | [+0.0050, +0.0111] |

## By pair (repaired reference), PYHS - WB, per-state NRMSE of that pair

| pair | median | 95% interval |
|---|---|---|
| (1,1) | -0.0105 | [-0.0143, -0.0075] |
| (1,2) | +0.0204 | [+0.0151, +0.0246] |
| (2,2) | +0.0139 | [+0.0115, +0.0261] |

## By source (repaired reference), PYHS - WB

| source | n states | median | 95% interval |
|---|---|---|---|
| md | 490 | +0.0076 | [+0.0053, +0.0121] |
| tn1331_dcf | 29 | +0.0012 | [-0.0276, +0.0104] |

## chat(0) relative error over the core, lowk reference (rows = state x pair)

| model | rows | median | 95% interval | IQR |
|---|---|---|---|---|
| HNCB | 1434 | 0.0875 | [0.0823, 0.0944] | 0.053--0.173 |
| MSA | 1557 | 0.0923 | [0.0865, 0.1001] | 0.051--0.202 |
| PYHS | 1557 | 0.0881 | [0.0805, 0.0966] | 0.048--0.204 |
| WB | 1557 | 0.0729 | [0.0682, 0.0791] | 0.037--0.179 |

| first - second | rows | median | 95% interval | first larger |
|---|---|---|---|---|
| PYHS - WB | 1557 | +0.0212 | [+0.0186, +0.0239] | 1053/1557 |
| MSA - WB | 1557 | +0.0310 | [+0.0270, +0.0339] | 1078/1557 |
| HNCB - WB | 1434 | +0.0262 | [+0.0215, +0.0295] | 984/1434 |
| HNCB - PYHS | 1434 | +0.0085 | [+0.0037, +0.0127] | 834/1434 |
| HNCB - MSA | 1434 | +0.0009 | [-0.0009, +0.0038] | 745/1434 |

## chat(0) relative error over the core, published reference (rows = state x pair)

| model | rows | median | 95% interval | IQR |
|---|---|---|---|---|
| HNCB | 1434 | 0.0796 | [0.0766, 0.0825] | 0.052--0.131 |
| MSA | 1557 | 0.0867 | [0.0814, 0.0922] | 0.053--0.165 |
| PYHS | 1557 | 0.0827 | [0.0769, 0.0904] | 0.048--0.166 |
| WB | 1557 | 0.0677 | [0.0621, 0.0732] | 0.036--0.161 |

| first - second | rows | median | 95% interval | first larger |
|---|---|---|---|---|
| PYHS - WB | 1557 | +0.0189 | [+0.0156, +0.0214] | 1007/1557 |
| MSA - WB | 1557 | +0.0271 | [+0.0228, +0.0306] | 1037/1557 |
| HNCB - WB | 1434 | +0.0197 | [+0.0155, +0.0248] | 929/1434 |
| HNCB - PYHS | 1434 | +0.0021 | [-0.0014, +0.0060] | 752/1434 |
| HNCB - MSA | 1434 | -0.0018 | [-0.0031, -0.0003] | 656/1434 |
