#!/usr/bin/env python3
"""Check of the explicit closed form of the first-order DCF printed in the main text (Eqs. phim-Wexplicit).

The polynomial-exponential closed form is rebuilt here from the Baxter coefficients Q', Q'' and the Yukawa
residues d_{mn,a} alone, with the elementary functions
    phi_m(t; z) = Theta(t) (-1)^m z^-m [ e^{-zt} - sum_{j<m} (-zt)^j / j! ],
the truncated-quadratic kernels pi_im(x) = alpha x + beta x^2 on [0, sigma_i], and
    W_ij(r) = B_ij(r) - sum_{n,a} d_in,a Phi_jn,za(r + lam_ij) - sum_{m,a} d_mj,a Phi_im,za(r - lam_ij)
              + sum_{m,n,a} d_mn,a Gamma_im,jn,za(r + R_ij),   c1_ij(r) = [W(r) - W(-r)] / (2 pi sqrt(rho_i rho_j) r).
The in-core form of the main text (0<r<R_ij, differences at r and -r, no B_ij term) is compared with FMSA_double_prop.get_c1_inner, whose convolution algebra is independent of this one,
for a binary and a ternary mixture.  The algebra here is not shared with the production code.

    python code/run.py code/fmsa_dcf/validation/verify_explicit_closed_form.py
"""
import numpy as np, warnings, math
from math import factorial
warnings.simplefilter('ignore')
from fmsa_double_prop import FMSA_double_prop

def phi(m,t,z):
    if t<0: return 0.0
    s=sum((-z*t)**j/factorial(j) for j in range(m))
    return (-1)**m * z**(-m) * (math.exp(-z*t)-s)

def trunc_coeffs(alpha,beta,sig):
    a=[0.0,alpha,beta]; at=[alpha*sig+beta*sig**2, alpha+2*beta*sig, beta]
    return a,at

def Phi(s,pi,sig,z):
    a,at=trunc_coeffs(*pi,sig)
    return sum(a[k]*factorial(k)*phi(k+1,s,z) for k in range(3)) - sum(at[k]*factorial(k)*phi(k+1,s-sig,z) for k in range(3))

def Gamma(S,pi1,sig1,pi2,sig2,z):
    A=trunc_coeffs(*pi1,sig1); Bc=trunc_coeffs(*pi2,sig2); tot=0.0
    for p in (0,1):
        for q in (0,1):
            sh=p*sig1+q*sig2
            tot+=(-1)**(p+q)*sum(A[p][k]*Bc[q][l]*factorial(k)*factorial(l)*phi(k+l+2,S-sh,z) for k in range(3) for l in range(3))
    return tot

def W_explicit(dp,i,j,r):
    N=dp.N; sig=dp.sigma if hasattr(dp,'sigma') else None
    sg=np.diag(dp.R_ij)*1.0  # core diameters d_i (R_ii)
    R=dp.R_ij; lam=dp.lambda_ij; rho=dp.rho
    def pi(i_,m):
        rr=math.sqrt(rho[i_]*rho[m]); return (-rr*dp.Q0[i_,m], 0.5*rr*dp.Qpp[m])
    def dterms(m,n): return list(zip(dp._b_poles[m][n],dp._b_coeffs[m][n]))
    W=0.0
    if r>=R[i,j]:
        W+=sum(c*math.exp(-z*(r-R[i,j])) for z,c in dterms(i,j))
    for n in range(N):
        for z,c in dterms(i,n):
            W-=c*Phi(r+lam[i,j],pi(j,n),sg[j],z)
    for m in range(N):
        for z,c in dterms(m,j):
            W-=c*Phi(r-lam[i,j],pi(i,m),sg[i],z)
    for m in range(N):
        for n in range(N):
            for z,c in dterms(m,n):
                W+=c*Gamma(r+R[i,j],pi(i,m),sg[i],pi(j,n),sg[j],z)
    return W


def C_core(dp,i,j,r):
    """The in-core form printed in the main text, 0 < r < R_ij: no B_ij term, differences of Phi and Gamma at r and -r."""
    N=dp.N; sg=np.diag(dp.R_ij)*1.0; R=dp.R_ij; lam=dp.lambda_ij; rho=dp.rho
    def pi(i_,m):
        rr=math.sqrt(rho[i_]*rho[m]); return (-rr*dp.Q0[i_,m], 0.5*rr*dp.Qpp[m])
    def dterms(m,n): return list(zip(dp._b_poles[m][n],dp._b_coeffs[m][n]))
    C=0.0
    for n in range(N):
        for z,c in dterms(i,n):
            C-=c*(Phi(r+lam[i,j],pi(j,n),sg[j],z)-Phi(lam[i,j]-r,pi(j,n),sg[j],z))
    for m in range(N):
        for z,c in dterms(m,j):
            C-=c*(Phi(r-lam[i,j],pi(i,m),sg[i],z)-Phi(-r-lam[i,j],pi(i,m),sg[i],z))
    for m in range(N):
        for n in range(N):
            for z,c in dterms(m,n):
                C+=c*(Gamma(R[i,j]+r,pi(i,m),sg[i],pi(j,n),sg[j],z)-Gamma(R[i,j]-r,pi(i,m),sg[i],pi(j,n),sg[j],z))
    return C

for name,sig,eps,x,rho,T in [('binary',[1.,2.],[1.,.5],[.25,.75],0.139,1.0),('ternary',[1.,1.6,2.3],[1.,.7,.5],[.3,.3,.4],0.12,1.2)]:
    dp=FMSA_double_prop(np.array(sig),np.array(eps),np.array(x),rho,T,n_tails=2,verbose=False)
    worst=0.0
    for i in range(dp.N):
        for j in range(i,dp.N):
            Rij=dp.R_ij[i,j]
            for r in np.linspace(0.03*Rij,0.97*Rij,25):
                c=C_core(dp,i,j,r)/(2*np.pi*math.sqrt(dp.rho[i]*dp.rho[j])*r)
                ref=dp.get_c1_inner(r,i,j)
                worst=max(worst,abs(c-ref)/max(1e-12,abs(ref)))
    print(name,'max rel diff explicit vs get_c1_inner:',worst)
    assert worst < 1e-9, worst
