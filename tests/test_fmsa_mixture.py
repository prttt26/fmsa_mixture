"""Self-checks for the FMSA mixture package.

Each test is an identity that must hold to (near) machine precision, or a
cross-check between two independent computations of one quantity.  Run either

    python tests/test_fmsa_mixture.py          # plain Python, prints a summary
    python -m pytest tests                       # if pytest is installed
"""
import os
import sys

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fmsa_double_prop import FMSA_double_prop      # noqa: E402
from fmsa_free_energy import FMSAFreeEnergy        # noqa: E402
import fmsa_compressibility as comp                # noqa: E402

SIG, EPS, X, RHO, T = np.array([1.0, 2.0]), np.array([1.0, 0.5]), np.array([0.25, 0.75]), 0.139, 1.0
PAIRS = ((0, 0), (0, 1), (1, 1))


def model(input_type='lj', hs_fmt='wb1', T_=T, n_tails=2):
    return FMSA_double_prop(SIG, EPS, X, RHO, T_, n_tails=n_tails,
                            input_type=input_type, hs_fmt=hs_fmt, verbose=False)


def _quad(f, a, b, knot=None):
    pts = [knot] if knot and a < knot < b else None
    return quad(f, a, b, points=pts, limit=400, epsabs=1e-12, epsrel=1e-11)[0]


# ---------------------------------------------------------------- the DCF ----
def test_chat0_two_paths():
    """c_hat_ij(0): closed-form transform == 4 pi Int r^2 c_ij(r) dr in real space."""
    for it in ('lj', 'bh'):
        m = model(it)
        ck = np.real(np.asarray(m.get_chat_k(np.array([1e-7])))[0])
        R, lam = m.radii, np.abs(m.lambda_ij)
        for i, j in PAIRS:
            c = lambda r: 4*np.pi*r*r*float(m.get_c_total(np.array([r]), hs_ref='py')[i, j, 0])
            val = _quad(c, 0.0, R[i, j], lam[i, j]) + _quad(c, R[i, j], 80.0)
            assert abs(val / ck[i, j] - 1.0) < 1e-10, (it, i, j, val, ck[i, j])


def test_dcf_symmetric_and_continuous_at_knot():
    """c_12 = c_21 everywhere, and c is continuous at the unlike-pair knot |lambda_12|."""
    m = model()
    L = float(abs(m.lambda_ij[0, 1]))
    r = np.array([0.1, L*(1-1e-10), L, L*(1+1e-10), 1.2, 2.5])
    for ref in ('py', 'fmt'):
        c = m.get_c_total(r, hs_ref=ref)
        assert np.allclose(c[0, 1], c[1, 0], rtol=0, atol=1e-12)
        assert abs(c[0, 1, 2] - c[0, 1, 1]) < 1e-8 and abs(c[0, 1, 2] - c[0, 1, 3]) < 1e-8


def test_two_identical_species_reduce_to_one():
    """Two identical species at half the density each == one species (all quantities)."""
    one = FMSA_double_prop(np.array([1.0]), np.array([1.0]), np.array([1.0]), 0.5, 1.5,
                           n_tails=2, input_type='bh', verbose=False)
    two = FMSA_double_prop(np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]),
                           0.5, 1.5, n_tails=2, input_type='bh', verbose=False)
    r = np.array([0.1, 0.5, 0.9, 1.3, 2.0])
    c1, c2 = one.get_c_total(r, 'py')[0, 0], two.get_c_total(r, 'py')
    for i, j in ((0, 0), (0, 1), (1, 1)):
        assert np.allclose(c2[i, j], c1, rtol=1e-10, atol=1e-12)
    fe1, fe2 = FMSAFreeEnergy(one), FMSAFreeEnergy(two)
    assert abs(fe1.get_free_energy('none') - fe2.get_free_energy('none')) < 1e-10
    assert abs(comp.dbetaP_drho(one) - comp.dbetaP_drho(two)) < 1e-9


# ------------------------------------------------------------ the RDF -----
def test_rdf_vanishes_in_core_and_contact_positive():
    m = model()
    for i, j in PAIRS:
        R = m.radii[i, j]
        r = np.linspace(0.05, 0.9 * R, 12)
        g = m.get_g_r(r, route='fourier')[i, j]
        assert np.max(np.abs(g)) < 5e-3, (i, j, np.max(np.abs(g)))   # inversion accuracy
        assert m.get_g_contact(i, j) > 0.0


# ------------------------------------------------- compressibility route ---
def test_hs_block_is_hessian_of_betaf_hs():
    """Analytic c_hat^HS(0) == -FD Hessian of the energy route's beta F_HS/V."""
    for v in ('rosenfeld', 'wb1', 'wb2'):
        m = model(hs_fmt=v)
        an = comp._hs_block_fmt(m.rho, m.diag_r, v)
        h, fd = 1e-4, np.zeros((2, 2))
        f = lambda rr: FMSAFreeEnergy(m._make_at_rho(rr)).betaf_hs()
        for i in range(2):
            for j in range(2):
                e = lambda a, b: f(m.rho + np.eye(2)[i]*a*h + np.eye(2)[j]*b*h)
                fd[i, j] = -(e(1, 1) - e(1, -1) - e(-1, 1) + e(-1, -1)) / (4*h*h)
        assert np.max(np.abs(an / fd - 1.0)) < 1e-4, v


def test_fmt_identity_real_space():
    """4 pi Int r^2 c^HS_FMT(r) dr == -d^2(beta F_HS/V)/d rho_i d rho_j exactly."""
    m = model()
    an = comp._hs_block_fmt(m.rho, m.diag_r, 'wb1')
    for i, j in PAIRS:
        val = _quad(lambda r: 4*np.pi*r*r*m.get_HS_FMT(r, i, j), 0.0, m.radii[i, j],
                    abs(m.lambda_ij[i, j]))
        assert abs(val / an[i, j] - 1.0) < 1e-10


def test_ray_integration_reproduces_hs_free_energy():
    """Integrating only the HS block along the ray returns beta F_HS/V itself."""
    m = model()
    rt, x = m.rhostar, m.x
    t, w = np.polynomial.legendre.leggauss(32)
    s, w = 0.5*rt*(t + 1.0), 0.5*rt*w
    g = np.array([-si * float(x @ comp._hs_block_fmt(x*si, m.diag_r, 'wb1') @ x) for si in s])
    assert abs(float(w @ (g*(rt - s)/s)) / FMSAFreeEnergy(m).betaf_hs() - 1.0) < 1e-12


def test_ray_quadrature_converged():
    m = model()
    assert abs(comp.ray_pressure(m, n_gl=16) - comp.ray_pressure(m, n_gl=64)) < 1e-10
    assert abs(comp.ray_free_energy(m, n_gl=16) - comp.ray_free_energy(m, n_gl=64)) < 1e-10


def test_stability_matrix_positive_at_reference_state():
    assert np.all(np.linalg.eigvalsh(comp.stability_matrix(model())) > 0.0)


# -------------------------------------------------------- energy route ----
def test_energy_route_gibbs_duhem():
    """beta P = sum_i rho_i beta mu_i - beta Phi  (totals, ideal parts included)."""
    m = model()
    fe = FMSAFreeEnergy(m)
    for incore in ('none', 'lj'):
        P = fe.get_pressure(incore=incore)
        mu = fe.get_chemical_potentials_total(incore=incore)
        Phi = fe.get_total_free_energy_density(incore)
        assert abs(P - (float(m.rho @ mu) - Phi)) < 1e-6, incore


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
