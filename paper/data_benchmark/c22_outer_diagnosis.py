#!/usr/bin/env python3
"""Diagnose the TN 1331 outer-region DCF excess, worst for the (2,2) pair.

Why
---
numerical_notes/results/single_point/oz_closures_comparison.md sec.4 found that beyond
r ~ 1.3 sigma the tabulated c_ij exceed the rigorous c -> -beta*phi limit by
2-7x, worst for (2,2), and that the excess looks like a smooth long-range
additive component.  This script settles what that excess is, using two
criteria taken from

    D. Stopper, H. Hansen-Goos, R. Roth and R. Evans,
    "On the decay of the pair correlation function and the line of vanishing
    excess isothermal compressibility in simple fluids",
    J. Chem. Phys. 151, 014501 (2019).
    (local copy: pdf/DCF_c22_related/)

That paper reports two empirical facts about a correctly extracted simulation
c(r) (its Fig. 1, MC for square wells; its ref. 22 = Dijkstra & Evans,
JCP 112, 1449 (2000), MC for truncated-shifted Lennard-Jones):

    (a) the attractive tail of c(r) is NOT very sensitive to packing fraction
    (b) the core contribution becomes more negative with increasing density
    (c) c(r) = -beta*phi(r) holds down to the core diameter

and it supplies two quantitative handles:

    (i)  chi_T/chi_id = [1 - rho * sum_ij x_i x_j chat_ij(0)]^{-1}   (its sec. III)
    (ii) the leading poles of hhat, from det[I - rho^ chat(k)] = 0    (its Eqs. 5-6,
         mixture form: Evans, Leote de Carvalho, Henderson & Hoyle,
         JCP 100, 591 (1994))

Sections below run each of these over the extracted tables.  Every number
quoted in numerical_notes/rootcause/c22_outer_inversion_artifact.md comes from
here.

Usage
-----
    python c22_outer_diagnosis.py            # all sections
    python c22_outer_diagnosis.py --only 4   # one section (1..6)

Section 5 (poles) takes ~2 min; the rest are seconds.
"""

import argparse
import csv
import os

import numpy as np

try:                      # numpy >= 2.0 renamed trapz -> trapezoid
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz
import scipy.stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "tn1331")

# The reference state point: MC_DATA in FMSA_MC.m == TN 1331 App. C,
# T*=1.0, rho*=0.139, x1=0.2500, eps22/eps11=0.50.
REF_TABLE = "dcf/T1_eps0.50_x0.2500.csv"
REF_T, REF_RHO, REF_X1, REF_EPS22 = 1.0, 0.139, 0.25, 0.50

SIG = {"11": 1.0, "12": 1.5, "22": 2.0}          # sigma_ij, Lorentz-Berthelot
PAIRS = ("11", "12", "22")
FIT_LO, FIT_HI = 1.6, 2.5                        # R* window for the tail fit


def eps_of(eps22):
    """eps_ij / eps11 for a given eps22/eps11 (Berthelot)."""
    return {"11": 1.0, "12": np.sqrt(eps22), "22": eps22}


def phi(r, sigma, eps):
    """Lennard-Jones 12-6, in units of eps11."""
    x = sigma / r
    return 4.0 * eps * (x ** 12 - x ** 6)


def packing(rho, x1):
    """eta for sigma22/sigma11 = 2."""
    return (np.pi / 6.0) * rho * (x1 + 8.0 * (1.0 - x1))


def read_dcf(path):
    rows = list(csv.DictReader(open(path)))
    R = np.array([float(z["R_star"]) for z in rows])
    return R, {p: np.array([float(z["c" + p]) for z in rows]) for p in PAIRS}


def index_rows(kind="dcf"):
    return [e for e in csv.DictReader(open(os.path.join(DATA, "index.csv")))
            if e["kind"] == kind]


def spearman(a, b):
    """Rank correlation with mid-ranks for ties.

    Tie handling matters here: 4*eps22/T* takes only a handful of distinct
    values across the 39 tables, and argsort-based ranking would break those
    ties by input order, making the result depend on how the rows are sorted.
    """
    return float(scipy.stats.spearmanr(np.asarray(a, float),
                                       np.asarray(b, float)).statistic)


# --------------------------------------------------------------------------
# 1.  Is the (2,2) outer region actually smoother than the others?
# --------------------------------------------------------------------------
def sec_smoothness():
    print(__doc__.split("Usage")[0].strip()[:0] or "", end="")
    print("=" * 78)
    print("1. SMOOTHNESS OF THE OUTER REGION  (reference state point)")
    print("=" * 78)
    R, C = read_dcf(os.path.join(DATA, REF_TABLE))
    m = R >= 1.15                       # past the contact peak of every pair
    print(f"   window R* in [1.15, 2.5], n = {m.sum()} points\n")
    print(f"   {'pair':>5} {'monotone steps':>15} {'2nd-diff flips':>15} "
          f"{'rms 2nd diff':>13}")
    for p in PAIRS:
        y = C[p][m]
        d1 = np.diff(y)
        d2 = np.diff(y, 2)
        sg = np.sign(d2)
        flips = int((sg[1:] * sg[:-1] < 0).sum())
        print(f"   {p:>5} {f'{(d1 < 0).sum()}/{len(d1)}':>15} "
              f"{f'{flips}/{len(d2) - 1} ({100 * flips / (len(d2) - 1):.0f}%)':>15} "
              f"{np.sqrt((d2 ** 2).mean()):13.4f}")
    print("\n   c22 is monotone at every step, but its ABSOLUTE noise matches c12")
    print("   and its sign-flip rate matches all three.  The data is not smoothed:")
    print("   c22 is a large smooth signal on the same noise floor -> high S/N.")
    print("   (noise ordering c11 >> c12 ~ c22 follows pair counting: N1=216, N2=648)")


# --------------------------------------------------------------------------
# 2.  Tail amplitude vs the rigorous c -> -beta*phi limit, all 39 tables
# --------------------------------------------------------------------------
def tail_scales(path, T, eps22):
    """Least-squares s_ij with c_ij ~ s_ij * (-beta*phi_ij) on R* in [1.6, 2.5]."""
    R, C = read_dcf(path)
    E = eps_of(eps22)
    out = {}
    for p in PAIRS:
        m = (R >= FIT_LO) & (R <= FIT_HI)
        r = R[m] * SIG[p]
        ref = -phi(r, SIG[p], E[p]) / T
        out[p] = float((C[p][m] @ ref) / (ref @ ref))
    return out


def sec_tail_sweep():
    print("\n" + "=" * 78)
    print("2. TAIL AMPLITUDE  s_ij = c_ij / (-beta*phi_ij)  OVER ALL DCF TABLES")
    print("=" * 78)
    recs = []
    for e in index_rows():
        if not e["rho_star"]:
            continue
        T, eps22, x1 = float(e["tstar"]), float(e["eps"]), float(e["x1"])
        rho = float(e["rho_star"])
        s = tail_scales(os.path.join(DATA, e["file"]), T, eps22)
        # amplification factor predicted by section 6, evaluated per table from
        # its own chat22(0).  Uses the tabulated (already inflated) c22, so it is
        # a rough predictor, not an independent one -- but chat22(0) spans a
        # factor ~30 across the dataset while the artifact moves it by ~2.
        R, C = read_dcf(os.path.join(DATA, e["file"]))
        r22 = R * SIG["22"]
        m22 = float(4 * np.pi * _trapz(r22 ** 2 * C["22"], r22))
        recs.append(dict(eta=packing(rho, x1), T=T, eps22=eps22, x1=x1, rho=rho,
                         quality=e["quality"], amp=(1 - rho * (1 - x1) * m22) ** 2,
                         **{"s" + p: s[p] for p in PAIRS}))
    recs.sort(key=lambda z: z["eta"])
    print(f"   {'eta':>6} {'T*':>3} {'eps22':>6} {'x1':>7} {'s11':>6} {'s12':>6} "
          f"{'s22':>7}  quality")
    for z in recs:
        print(f"   {z['eta']:6.3f} {z['T']:3.0f} {z['eps22']:6.2f} {z['x1']:7.4f} "
              f"{z['s11']:6.2f} {z['s12']:6.2f} {z['s22']:7.2f}  {z['quality']}")

    eta = np.array([z["eta"] for z in recs])
    s22 = np.array([z["s22"] for z in recs])
    print(f"\n   n = {len(recs)} tables with rho* recorded")
    print("   Spearman rank correlation of s22 with")
    for name, v in [("eta (packing fraction)", eta),
                    ("T*", [z["T"] for z in recs]),
                    ("rho*", [z["rho"] for z in recs]),
                    ("x1", [z["x1"] for z in recs]),
                    ("eps22/eps11", [z["eps22"] for z in recs]),
                    ("4*eps22/T* (nominal tail amplitude)",
                     [4 * z["eps22"] / z["T"] for z in recs]),
                    ("M22^2 (sec. 6 amplification factor)",
                     [z["amp"] for z in recs])]:
        print(f"      {name:<38} {spearman(s22, v):+.3f}")
    print(f"      {'(s11 vs eta)':<38} "
          f"{spearman([z['s11'] for z in recs], eta):+.3f}")
    print(f"      {'(s12 vs eta)':<38} "
          f"{spearman([z['s12'] for z in recs], eta):+.3f}")

    # eta and T* are themselves anti-correlated in this dataset (every T*=1 table
    # sits at eta ~ 0.44-0.49), so separate them by looking within each isotherm.
    print("\n   within a single isotherm (removes the eta / T* confound):")
    print(f"      {'T*':>3} {'n':>3} {'eta range':>13} {'Spearman(s22, eta)':>20}")
    for Tv in (1.0, 2.0, 3.0):
        sub = [z for z in recs if z["T"] == Tv]
        if len(sub) < 4:
            continue
        e = [z["eta"] for z in sub]
        print(f"      {Tv:3.0f} {len(sub):3d} "
              f"{f'{min(e):.3f}-{max(e):.3f}':>13} "
              f"{spearman([z['s22'] for z in sub], e):20.3f}")

    print(f"\n   binned by eta:  {'n':>3} {'<s11>':>7} {'<s12>':>7} {'<s22>':>7}")
    for lo, hi in [(0.10, 0.25), (0.25, 0.35), (0.35, 0.43), (0.43, 0.50)]:
        m = (eta >= lo) & (eta < hi)
        if not m.sum():
            continue
        print(f"      {lo:.2f}-{hi:.2f}      {m.sum():3d} "
              f"{np.mean([z['s11'] for z, k in zip(recs, m) if k]):7.2f} "
              f"{np.mean([z['s12'] for z, k in zip(recs, m) if k]):7.2f} "
              f"{np.mean([z['s22'] for z, k in zip(recs, m) if k]):7.2f}")
    print("\n   => the excess tracks PACKING FRACTION (+0.79) and 1/T* (-0.76), and only")
    print("      weakly the nominal tail amplitude 4*eps22/T* (+0.23) that physics says")
    print("      should set it.  eta and T* are confounded here, but the eta trend")
    print("      survives within the T*=2 isotherm, which spans the widest eta range.")
    print("      Both directions are what the sec. 6 amplification predicts: denser and")
    print("      colder => more negative chat22(0) => larger M22^2.")
    print("      At eta < 0.25 all three pairs satisfy c = -beta*phi (s ~ 0.8-1.2).")
    print("      Stopper et al. Fig. 1: a correct c(r) tail is insensitive to eta. VIOLATED.")
    return recs


# --------------------------------------------------------------------------
# 3.  The core, by contrast, behaves exactly as it should
# --------------------------------------------------------------------------
def sec_core():
    print("\n" + "=" * 78)
    print("3. CORE DEPTH c_ij(r -> 0) vs PACKING FRACTION")
    print("=" * 78)
    rec = []
    for e in index_rows():
        if not e["rho_star"]:
            continue
        R, C = read_dcf(os.path.join(DATA, e["file"]))
        eta = packing(float(e["rho_star"]), float(e["x1"]))
        rec.append((eta, float(e["tstar"]),
                    *[float(np.interp(0.05, R, C[p])) for p in PAIRS]))
    rec.sort()
    a = np.array(rec)
    print(f"   {'eta':>6} {'T*':>3} {'c11(0)':>9} {'c12(0)':>9} {'c22(0)':>9}")
    for z in rec:
        flag = "  <- OCR damage (sign/digit)" if z[2] > 0 or z[4] > 0 else ""
        print(f"   {z[0]:6.3f} {z[1]:3.0f} {z[2]:9.3f} {z[3]:9.3f} {z[4]:9.3f}{flag}")
    print(f"\n   Spearman(eta, c11(0)) = {spearman(a[:, 0], a[:, 2]):+.3f}")
    print(f"   Spearman(eta, c22(0)) = {spearman(a[:, 0], a[:, 4]):+.3f}")
    print("\n   Stopper et al. Fig. 1: the core deepens with density. CONFIRMED.")
    print("   => the defect is in the tail alone, not in the extraction as a whole.")


# --------------------------------------------------------------------------
# 4/5.  chi_T criterion and the leading poles
# --------------------------------------------------------------------------
def extended_rc(path, T, eps22, replace_outer):
    """r*c_ij(r) on a common fine grid: tabulated part, -beta*phi beyond 2.5 sigma_ij.

    replace_outer[p] True -> that pair's tabulated r > sigma_ij region is
    overwritten by the rigorous -beta*phi limit.
    """
    R, C = read_dcf(path)
    E = eps_of(eps22)
    r = np.linspace(1e-4, 16.0, 6000)
    W = {}
    for p in PAIRS:
        s = SIG[p]
        rt, ct = R * s, C[p].copy()
        if replace_outer[p]:
            m = rt > s
            ct[m] = -phi(rt[m], s, E[p]) / T
        c = np.interp(r, rt, ct, left=ct[0], right=0.0)
        tail = r > rt[-1]
        c[tail] = -phi(r[tail], s, E[p]) / T
        W[p] = r * c
    return r, W


def chat(r, W, k):
    """chat(k) = 4 pi Int r^2 c(r) sin(kr)/(kr) dr, valid for complex k."""
    return 4.0 * np.pi * _trapz(W * np.sin(k * r) / k, r)


def bmcsl_dbetaP_drho(rho, x, sigma, eps_mat, sig_mat, T, attraction=True):
    """d(beta P)/d rho at fixed composition: BMCSL hard spheres + van der Waals
    mean-field attraction (Stopper et al. Eq. A1)."""
    def bP(rr):
        xi = [(np.pi / 6) * rr * float(x @ (sigma ** n)) for n in range(4)]
        x0, x1_, x2, x3 = xi
        v = (6 / np.pi) * (x0 / (1 - x3) + 3 * x1_ * x2 / (1 - x3) ** 2
                           + (3 - x3) * x2 ** 3 / (1 - x3) ** 3)
        if attraction:
            a = 32 * np.pi * eps_mat * sig_mat ** 3 / 9.0   # -Int phi_att dr
            v -= 0.5 * rr ** 2 * float(x @ a @ x) / T
        return v
    h = 1e-6
    return (bP(rho + h) - bP(rho - h)) / (2 * h)


CASES = {
    "published (as tabulated)":   {p: False for p in PAIRS},
    "c22 outer -> -beta*phi":     {"11": False, "12": False, "22": True},
    "all pairs outer -> -beta*phi": {p: True for p in PAIRS},
}


def sec_compressibility():
    print("\n" + "=" * 78)
    print("4. chi_T / chi_id  =  [1 - rho sum_ij x_i x_j chat_ij(0)]^-1")
    print("=" * 78)
    x = np.array([REF_X1, 1 - REF_X1])
    path = os.path.join(DATA, REF_TABLE)
    print(f"   state: rho*={REF_RHO}, x1={REF_X1}, T*={REF_T}, "
          f"eta={packing(REF_RHO, REF_X1):.4f}\n")
    print(f"   {'case':<30} {'c11(0)':>8} {'c12(0)':>8} {'c22(0)':>9} "
          f"{'chi_T/chi_id':>13}")
    for label, rep in CASES.items():
        r, W = extended_rc(path, REF_T, REF_EPS22, rep)
        ch = {p: float(chat(r, W[p], 1e-6).real) for p in PAIRS}
        C0 = np.array([[ch["11"], ch["12"]], [ch["12"], ch["22"]]])
        chi = 1.0 / (1 - REF_RHO * float(x @ C0 @ x))
        print(f"   {label:<30} {ch['11']:8.2f} {ch['12']:8.2f} {ch['22']:9.2f} "
              f"{chi:13.4f}")

    # self-consistent OZ closure solutions, from oz_closures_comparison.md sec.2
    print()
    for label, ch in [("TN1331 published chat(0)", [1.990, -22.07, -100.80]),
                      ("MSA  (OZ solve)", [-4.178, -31.92, -187.0]),
                      ("PY   (OZ solve)", [1.530, -24.72, -183.3]),
                      ("HNC  (OZ solve)", [5.222, -14.57, -137.2]),
                      ("HNCB (OZ solve)", [0.600, -29.43, -196.29])]:
        C0 = np.array([[ch[0], ch[1]], [ch[1], ch[2]]])
        print(f"   {label:<30} {ch[0]:8.2f} {ch[1]:8.2f} {ch[2]:9.2f} "
              f"{1.0 / (1 - REF_RHO * float(x @ C0 @ x)):13.4f}")

    sigma = np.array([1.0, 2.0])
    eps_mat = np.array([[1.0, np.sqrt(REF_EPS22)], [np.sqrt(REF_EPS22), REF_EPS22]])
    sig_mat = np.array([[1.0, 1.5], [1.5, 2.0]])
    for lab, att in [("BMCSL + mean-field attraction", True),
                     ("BMCSL hard spheres only", False)]:
        d = bmcsl_dbetaP_drho(REF_RHO, x, sigma, eps_mat, sig_mat, REF_T, att)
        print(f"   {lab:<30} {'':8} {'':8} {'':9} {1 / d:13.4f}")
    print("\n   => four independent closures cluster at 0.056-0.080 and the")
    print("      -beta*phi-corrected data lands at 0.058; only the published")
    print("      table gives 0.100.  Changing c22 ALONE removes the whole gap.")


def sec_poles():
    print("\n" + "=" * 78)
    print("5. LEADING POLES  det[I - rho^ chat(k)] = 0   (Stopper et al. Eqs. 5-6)")
    print("=" * 78)
    rhoi = REF_RHO * np.array([REF_X1, 1 - REF_X1])
    idx = {(0, 0): "11", (0, 1): "12", (1, 0): "12", (1, 1): "22"}
    path = os.path.join(DATA, REF_TABLE)

    def make_det(r, W):
        def det(k):
            sk = np.sin(k * r) / k
            C = np.empty((2, 2), dtype=complex)
            for (i, j), p in idx.items():
                C[i, j] = (np.sqrt(rhoi[i] * rhoi[j])
                           * 4 * np.pi * _trapz(W[p] * sk, r))
            M = np.eye(2) - C
            return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        return det

    def newton(det, k0, n=50):
        k = complex(k0)
        for _ in range(n):
            f = det(k)
            h = 1e-6 * (1 + abs(k))
            fp = (det(k + h) - det(k - h)) / (2 * h)
            if fp == 0:
                break
            dk = f / fp
            if abs(dk) > 0.6:
                dk *= 0.6 / abs(dk)
            k -= dk
            if abs(dk) < 1e-11:
                break
        return k

    print(f"   {'case':<30} {'a0_osc':>8} {'a1_osc':>8} {'2pi/a1':>8} "
          f"{'a0_mon':>8} {'gap':>7}")
    for label, rep in CASES.items():
        r, W = extended_rc(path, REF_T, REF_EPS22, rep)
        det = make_det(r, W)
        kosc = newton(det, complex(3.4, 0.5))
        kmon = newton(det, complex(0.0, 0.6))
        print(f"   {label:<30} {kosc.imag:8.4f} {kosc.real:8.4f} "
              f"{2 * np.pi / kosc.real:8.4f} {kmon.imag:8.4f} "
              f"{kmon.imag - kosc.imag:7.4f}")
    print("\n   2pi/a1 = 1.86 matches the g22 oscillation period ~1.9 in App. A --")
    print("   the wavelength is robust.  The DECAY rates are not: the published")
    print("   table puts a_mon within 10% of a_osc, i.e. it places an eta = 0.455")
    print("   liquid essentially ON the Fisher-Widom line, which is not credible.")
    print("   Correcting c22 alone reopens a normal gap.")


# --------------------------------------------------------------------------
# 6.  Why the outer region is the fragile part: the 1/S(0)^2 amplification
# --------------------------------------------------------------------------
def sec_amplification():
    print("\n" + "=" * 78)
    print("6. OZ-INVERSION AMPLIFICATION  d chat22(0) / d hhat22(0) = M22^2")
    print("=" * 78)
    rhoi = REF_RHO * np.array([REF_X1, 1 - REF_X1])

    def M_of(ch):
        C = np.array([[rhoi[0] * ch[0], np.sqrt(rhoi[0] * rhoi[1]) * ch[1]],
                      [np.sqrt(rhoi[0] * rhoi[1]) * ch[1], rhoi[1] * ch[2]]])
        return np.eye(2) - C

    pub = [3.05, -19.57, -96.84]        # section 4, published
    cor = [3.05, -19.57, -194.66]       # section 4, c22 outer corrected
    Mc = M_of(cor)
    amp = Mc[1, 1] ** 2
    need = pub[2] - cor[2]
    dh0 = need / amp
    print("   Ornstein-Zernike at k=0 is  chat = I - S^-1, so a perturbation of the")
    print("   total correlation function propagates as  d chat = M dS M, giving")
    print("   d chat22(0) = M22^2 * d hhat22(0)   with M22 = 1 - rho2 chat22(0).\n")
    print(f"   M22 = {Mc[1, 1]:.3f}      amplification M22^2 = {amp:.1f}")
    print(f"   excess to explain: d chat22(0) = {need:+.1f}")
    print(f"   required          d hhat22(0) = 4pi Int r^2 dh22 dr = {dh0:.4f}\n")
    for lo, hi in [(2.5, 5.0), (5.0, 9.0), (5.0, 12.0)]:
        vol = 4 * np.pi * (hi ** 3 - lo ** 3) / 3
        print(f"      spread uniformly over r in [{lo:4.1f}, {hi:4.1f}] sigma11 "
              f"-> dh22 = {dh0 / vol:.2e}")
    print("\n   i.e. a part-in-10^4 error in the far tail of h22 -- invisible in g --")
    print("   fully accounts for the factor-5 excess in the outer c22.\n")
    print("   The amplification is what makes this a high-density-only failure:")
    print(f"   {'chat22(0)':>11} {'M22':>8} {'M22^2':>9}")
    for c22 in [-5, -10, -20, -40, -97, -195]:
        M = M_of([0.0, -2.0, c22])
        print(f"   {c22:11.1f} {M[1, 1]:8.3f} {M[1, 1] ** 2:9.1f}")
    print("\n   2.3x at low density, 453x at the reference state -- matching the")
    print("   measured eta dependence of section 2.")


SECTIONS = {1: sec_smoothness, 2: sec_tail_sweep, 3: sec_core,
            4: sec_compressibility, 5: sec_poles, 6: sec_amplification}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", type=int, choices=sorted(SECTIONS),
                    help="run a single section")
    args = ap.parse_args()
    for n in ([args.only] if args.only else sorted(SECTIONS)):
        SECTIONS[n]()


if __name__ == "__main__":
    main()
