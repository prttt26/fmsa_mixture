"""example_binary.py -- one binary mixture state, every quantity the package provides.

System: two Lennard-Jones species mapped onto hard cores + two Yukawa tails
(the "2YK" fit), sigma = [1, 2], eps = [1, 0.5], x = [0.25, 0.75],
rho* = 0.139, T* = 1.0.  Run from the package directory:

    python examples/example_binary.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fmsa_double_prop import FMSA_double_prop            # noqa: E402
from fmsa_free_energy import FMSAFreeEnergy              # noqa: E402
import fmsa_compressibility as comp                      # noqa: E402

m = FMSA_double_prop(sig_pure=np.array([1.0, 2.0]), eps_pure=np.array([1.0, 0.5]),
                     x=np.array([0.25, 0.75]), rhostar=0.139, tstar=1.0,
                     n_tails=2, verbose=False)
N = m.N
pairs = [(i, j) for i in range(N) for j in range(i, N)]
print(f"state: rho* = {m.rhostar}, T* = {m.tstar}, x = {m.x}, "
      f"hard-core diameters d = {np.round(m.diag_r, 4)}")
print("contact distances R_ij:\n", np.round(m.radii, 4))

# ---- 1. direct correlation function -------------------------------------------
r = np.array([0.05, 0.3, 0.6, 0.9, 1.2, 1.6, 2.0, 3.0])
c_fmt = m.get_c_total(r, hs_ref='fmt')        # White-Bear FMT hard-sphere block
c_py = m.get_c_total(r, hs_ref='py')          # Percus-Yevick hard-sphere block
print("\n1. DCF c_ij(r)  [FMT reference | PY reference]")
for i, j in pairs:
    print(f"   c_{i+1}{j+1}:", "  ".join(f"r={rv:.2f}: {a:9.4f} | {b:9.4f}"
                                     for rv, a, b in zip(r[:4], c_fmt[i, j], c_py[i, j])))
print("   c_hat_ij(k) at k = 0, 2, 5 (closed form, PY reference):")
for k in (1e-5, 2.0, 5.0):
    print(f"     k={k:5.2f}:", np.round(np.real(m.get_chat_k(k)), 4).tolist())

# ---- 2. radial distribution function and structure factor --------------------
rr = np.linspace(0.5, 4.0, 8)
g = m.get_g_r(rr, route='fourier')            # (N, N, len(rr))
print("\n2. RDF g_ij(r) (Fourier route); contact values are algebraic:")
for i, j in pairs:
    print(f"   g_{i+1}{j+1}(R+) = {m.get_g_contact(i, j):.4f};  g(r) =",
          np.round(g[i, j], 4).tolist())
print("   S_ij(k) at k = 0.5:", np.round(m.get_S_k(0.5), 4).tolist())

# ---- 3. energy route ------------------------------------------------------------
# incore='lj' (default) adds the LJ shell between the hard core d and sigma_LJ --
# a correction to the POTENTIAL, appropriate for LJ-mapped input.  incore='none'
# keeps exactly the potential the DCF sees (hard core R_ij + Yukawa tail), which is
# the like-for-like partner of the compressibility route below.
fe = FMSAFreeEnergy(m)
print("\n3. Energy route                    incore='lj'     incore='none'")
print(f"   beta F_ex/V (vs ideal gas)     {fe.get_free_energy('lj'):12.6f}   {fe.get_free_energy('none'):12.6f}")
print(f"   beta P                         {fe.get_pressure(incore='lj'):12.6f}   {fe.get_pressure(incore='none'):12.6f}")
mu_lj, mu_no = fe.get_chemical_potentials(incore='lj'), fe.get_chemical_potentials(incore='none')
for i in range(N):
    print(f"   beta mu_{i+1}^ex                     {mu_lj[i]:12.6f}   {mu_no[i]:12.6f}")
print(f"   mixing excess per particle     {fe.get_excess_free_energy('lj'):12.6f}   "
      f"{fe.get_excess_free_energy('none'):12.6f}   (vs pure species at rho_i)")

# ---- 4. compressibility route ---------------------------------------------------
print("\n4. Compressibility route (FMT hard-sphere block, shared with the energy route)")
print(f"   (d beta P/d rho)_x   = {comp.dbetaP_drho(m):.6f}")
M = comp.stability_matrix(m)
print(f"   eig M (stability)    = {np.round(np.linalg.eigvalsh(M), 4).tolist()}  "
      "(all > 0: locally stable)")
print(f"   beta P   (ray)       = {comp.ray_pressure(m):.6f}")
print(f"   beta F_ex/V (ray)    = {comp.ray_free_energy(m):.6f}")
print("\nCompare with the energy route at incore='none': same potential, two routes.")
print("The difference is the thermodynamic inconsistency of the closure, not a bug.")
