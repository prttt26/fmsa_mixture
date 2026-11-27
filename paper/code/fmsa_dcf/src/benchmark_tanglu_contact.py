"""Numerical benchmark: first-order mixture RDF contact values vs Tang & Lu (1995).

Reproduces Tables 1 and 2 of

    Y. Tang and B. C.-Y. Lu, "Analytical solution of the Ornstein-Zernike
    equation for mixtures", Mol. Phys. 84, 89-103 (1995).

Model: additive hard-core one-Yukawa binary mixture,

    u_ij(r) = +inf                              (r < R_ij)
            = -R_ij eps_ij exp(-z (r - R_ij))/r (r > R_ij),   R_ij = (sig_i+sig_j)/2

with a single Yukawa decay z equal for all pairs (z R_11 = 1.8), matching the
Blum-Hoye full-MSA solution the paper quotes (column II, from Arrieta et al.).

Column I of each table is Tang & Lu's own FIRST-ORDER contact value, computed from
their eq (25): g_ij(R_ij) = g0_ij(R_ij) + g1_ij(R_ij), where g0 is the PY hard-sphere
mixture contact and g1 is the Yukawa first-order correction obtained from the residue
programme (eq 20-24).  That first-order correction is driven entirely by the
hard-sphere Baxter inverse [Q_hat_0(z)]^{-1} evaluated at the Yukawa decay z --
exactly the object formalised in the Lean development (Tang & Lu residue programme,
Task MML.17; AppendixWDet / AppendixBridgeN1 / Q0DetContourOrientation).

This script recomputes column I from scratch (standalone, no class dependency; the
Baxter conventions match fmsa_ga_matrix_mix.py's _build_Q0_Qpp / _build_Qhat /
_compute_A_prop) and reports:

  * my first-order  vs  paper column I   -> validation of the first-order machinery;
  * my first-order  vs  paper column II  -> the physics: first-order vs full MSA;
  * paper I         vs  paper column II  -> the paper's own "virtually identical" claim.
"""

import numpy as np


# ----------------------------------------------------------------------
# First-order hard-core Yukawa mixture contact values (standalone)
# ----------------------------------------------------------------------
def contact_first_order(x1, eta, tstar, sig_ratio, eps22, eps12, zR11=1.8):
    """Return (g, g0, g1): 2x2 arrays of contact values g_ij(R_ij).

    Parameters
    ----------
    x1        : mole fraction of species 1 (x2 = 1 - x1)
    eta       : total packing fraction (pi/6) sum_k rho_k sigma_k^3
    tstar     : reduced temperature kT / eps_11
    sig_ratio : diameter ratio sigma_2 / sigma_1 = R_22/R_11
    eps22     : eps_22 / eps_11
    eps12     : eps_12 / eps_11
    zR11      : Yukawa decay times R_11 (= 1.8 in both tables)
    """
    sigma = np.array([1.0, sig_ratio])              # diameters, R_11 = sigma_1 = 1
    x     = np.array([x1, 1.0 - x1])
    rho   = x * (6.0 * eta / (np.pi * float(np.sum(x * sigma**3))))
    N     = 2

    R_ij = 0.5 * (sigma[:, None] + sigma[None, :])
    lam  = 0.5 * (sigma[None, :] - sigma[:, None])  # lambda_ij = (sigma_j - sigma_i)/2
    xi2  = float(np.sum(rho * sigma**2))            # xi_2 = sum rho sigma^2 (no pi/6)
    vac  = 1.0 - (np.pi / 6.0) * float(np.sum(rho * sigma**3))
    z    = zR11 / sigma[0]
    sqrt_rr = np.sqrt(rho[:, None] * rho[None, :])

    # PY Baxter coefficients (matches fmsa_ga_matrix_mix._build_Q0_Qpp)
    Q0  = (2.0*np.pi/vac) * (R_ij + (np.pi*xi2/(4.0*vac)) * sigma[:, None]*sigma[None, :])
    Qpp = (2.0*np.pi/vac) * (1.0 + (np.pi*xi2/(2.0*vac)) * sigma)

    # Baxter factor Q_hat_0(z)  (matches _build_Qhat)
    zs = z * sigma
    p1 = (1.0 - zs - np.exp(-zs)) / z**2
    p2 = (1.0 - zs + 0.5*zs**2 - np.exp(-zs)) / z**3
    inner = p1[:, None]*Q0 + p2[:, None]*Qpp[None, :]
    Qhat  = np.eye(N) - sqrt_rr * np.exp(-z*lam) * inner
    Qinv  = np.linalg.inv(Qhat)

    # (I + A)_ij = Qinv_ij * exp(z lambda_ij), A = Tang&Lu propagator A_ij(z)
    # (matches _compute_A_prop: A[i,i]=Qinv[i,i]-1, A[i,j]=Qinv[i,j] exp(z lam))
    IpA = Qinv * np.exp(z * lam)

    # Yukawa amplitude K_ij = 2 pi sqrt(rho_i rho_j) R_ij beta eps_ij
    eps_ratio = np.array([[1.0, eps12], [eps12, eps22]])
    K = 2.0*np.pi * sqrt_rr * R_ij * (eps_ratio / tstar)

    # first-order correction: numerator N = (I+A) K (I+A)^T ; g1 = N / (2 pi sqrt(rr) R)
    Nmat = IpA @ K @ IpA.T
    g1 = Nmat / (2.0*np.pi * sqrt_rr * R_ij)

    # PY hard-sphere contact g0_ij(R_ij)
    g0 = (1.0/(R_ij*vac)) * (R_ij + (np.pi*xi2/(4.0*vac)) * sigma[:, None]*sigma[None, :])

    return g0 + g1, g0, g1


# ----------------------------------------------------------------------
# Reference data (Tang & Lu 1995, Tables 1 and 2)
# OCR: '-' '.' etc are decimal points.  Columns per row:
#   x1, eta, T*, [g11_I, g11_II, g12_I, g12_II, g22_I, g22_II]
# ----------------------------------------------------------------------
TABLE1 = dict(
    cond=dict(sig_ratio=1.500, eps22=0.900, eps12=0.800),
    rows=[
        (0.35, 0.30, 0.900, 2.6658, 2.7388, 2.4485, 2.4878, 2.6368, 2.6629),
        (0.35, 0.30, 0.717, 2.8145, 2.9456, 2.5085, 2.5794, 2.6947, 2.7411),
        (0.35, 0.46, 0.900, 3.9766, 4.0196, 3.9409, 3.9456, 4.4527, 4.4544),
        (0.35, 0.46, 0.717, 4.0886, 4.1618, 3.9577, 3.9656, 4.4710, 4.4738),
        (0.50, 0.30, 0.900, 2.6462, 2.7353, 2.4373, 2.4736, 2.6584, 2.6787),
        (0.50, 0.30, 0.717, 2.7828, 2.9511, 2.4857, 2.5544, 2.7111, 2.7479),
        (0.50, 0.46, 0.900, 3.9924, 4.0498, 3.9790, 3.9748, 4.5574, 4.5587),
        (0.50, 0.46, 0.717, 4.0901, 4.1944, 3.9835, 3.9754, 4.5749, 4.5770),
        (0.65, 0.30, 0.900, 2.6232, 2.7299, 2.4280, 2.4542, 2.6984, 2.7100),
        (0.65, 0.30, 0.717, 2.7441, 2.9764, 2.4623, 2.5169, 2.7466, 2.7670),
        (0.65, 0.46, 0.900, 4.0216, 4.0908, 4.0434, 4.0194, 4.7220, 4.7306),
        (0.65, 0.46, 0.717, 4.1013, 4.2564, 4.0339, 3.9766, 4.7434, 4.7638),
    ],
)
TABLE2 = dict(
    cond=dict(sig_ratio=1.167, eps22=1.919, eps12=float(np.sqrt(1.919))),
    rows=[
        (0.25, 0.30, 1.450, 2.3544, 2.3742, 2.4931, 2.5331, 2.6845, 2.7622),
        (0.25, 0.30, 1.200, 2.3785, 2.4185, 2.5332, 2.6165, 2.7493, 2.9160),
        (0.25, 0.46, 1.450, 3.9633, 3.9635, 4.1549, 4.1573, 4.3969, 4.4046),
        (0.25, 0.46, 1.200, 3.9685, 3.9689, 4.1666, 4.1702, 4.4199, 4.4317),
        (0.50, 0.30, 1.450, 2.3893, 2.4088, 2.5321, 2.5700, 2.7285, 2.7999),
        (0.50, 0.30, 1.200, 2.4149, 2.4499, 2.5741, 2.6431, 2.7957, 2.9280),
        (0.50, 0.46, 1.450, 4.0407, 4.0415, 4.2399, 4.2430, 4.4909, 4.4992),
        (0.50, 0.46, 1.200, 4.0471, 4.0483, 4.2531, 4.2578, 4.5160, 4.5287),
        (0.75, 0.30, 1.450, 2.4331, 2.4515, 2.5811, 2.6157, 2.7837, 2.8470),
        (0.75, 0.30, 1.200, 2.4605, 2.4915, 2.6254, 2.6839, 2.8538, 2.9619),
        (0.75, 0.46, 1.450, 4.1382, 4.1395, 4.3468, 4.3505, 4.6090, 4.6176),
        (0.75, 0.46, 1.200, 4.1460, 4.1479, 4.3620, 4.3674, 4.6366, 4.6498),
    ],
)


def run_table(name, tbl):
    cond = tbl["cond"]
    print(f"\n{'='*94}\n{name}  (R22/R11={cond['sig_ratio']}, e22/e11={cond['eps22']}, "
          f"e12/e11={cond['eps12']:.4f}, zR11=1.8)\n{'='*94}")
    print(f"{'x1':>5}{'eta':>6}{'T*':>7}  | "
          f"{'g11 me/I/II':>22} | {'g12 me/I/II':>22} | {'g22 me/I/II':>22}")
    d_me_I, d_me_II, d_I_II = [], [], []
    for row in tbl["rows"]:
        x1, eta, tstar = row[0], row[1], row[2]
        refI  = (row[3], row[5], row[7])   # g11_I, g12_I, g22_I
        refII = (row[4], row[6], row[8])   # g11_II, g12_II, g22_II
        g, _, _ = contact_first_order(x1, eta, tstar, **cond)
        me = (g[0, 0], g[0, 1], g[1, 1])
        cells = []
        for k in range(3):
            cells.append(f"{me[k]:.4f}/{refI[k]:.4f}/{refII[k]:.4f}")
            d_me_I.append(abs(me[k] - refI[k]))
            d_me_II.append(abs(me[k] - refII[k]))
            d_I_II.append(abs(refI[k] - refII[k]))
        print(f"{x1:>5.2f}{eta:>6.2f}{tstar:>7.3f}  | " + " | ".join(cells))
    return np.array(d_me_I), np.array(d_me_II), np.array(d_I_II)


def summarize(tag, d_me_I, d_me_II, d_I_II):
    print(f"\n{tag} summary  (n = {len(d_me_I)} contact values):")
    print(f"  |me - I |  (first-order machinery):  max={d_me_I.max():.4f}  "
          f"mean={d_me_I.mean():.4f}")
    print(f"  |me - II|  (my 1st-order vs full MSA): max={d_me_II.max():.4f}  "
          f"mean={d_me_II.mean():.4f}")
    print(f"  | I - II|  (paper 1st-order vs MSA):   max={d_I_II.max():.4f}  "
          f"mean={d_I_II.mean():.4f}")


if __name__ == "__main__":
    a1 = run_table("Table 1", TABLE1)
    a2 = run_table("Table 2", TABLE2)
    print("\n" + "#"*94)
    summarize("Table 1", *a1)
    summarize("Table 2", *a2)
    alld = [np.concatenate([a1[i], a2[i]]) for i in range(3)]
    summarize("Tables 1+2 combined", *alld)
