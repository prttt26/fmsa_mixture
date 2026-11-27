# States at which OZ-HNCB does not converge: 41 of 519

## by eta (pool state variable, Lennard-Jones diameters)

| eta | states | failed |
|---|---|---|
| (0.0, 0.2] | 79 | 2 |
| (0.2, 0.3] | 138 | 5 |
| (0.3, 0.4] | 128 | 8 |
| (0.4, 0.5] | 169 | 24 |
| (0.5, 0.7] | 5 | 2 |

## by source

| source | states | failed |
|---|---|---|
| md | 490 | 40 |
| tn1331_dcf | 29 | 1 |

## median core NRMSE of the other methods on the failed and on the converged states

| states | n | MSA | PYHS | WB |
|---|---|---|---|---|
| failed | 41 | 0.316 | 0.335 | 0.374 |
| converged | 478 | 0.090 | 0.093 | 0.092 |

## median core NRMSE on three bases

| basis | HNCB | MSA | PYHS | WB |
|---|---|---|---|---|
| own states | 0.0869 | 0.0967 | 0.1004 | 0.0973 |
| 478 common states | 0.0869 | 0.0896 | 0.0929 | 0.0918 |
| 519 states, failure as worst | 0.0982 | 0.0967 | 0.1004 | 0.0973 |
