#!/usr/bin/env python3
"""Physics core for interconverting the structure quantities of a binary fluid.

This is the single implementation of the transforms shared by the other scripts
in `data_benchmark/` (`oz_invert.py`, `convert.py`). It has no I/O and no
knowledge of any particular data source.

Quantities, for a binary mixture (species 1, 2) at partial number densities
rho = (rho_1, rho_2):

    real space       g_ij(r)   radial distribution function   (RDF)
                     h_ij(r) = g_ij(r) - 1   total correlation
                     c_ij(r)   direct correlation function     (DCF)
    Fourier space    hhat_ij(k), chat_ij(k)   3-D radial transforms
                     S_ij(k)   partial structure factor

Conventions
-----------
3-D radial (sine) transform of a spherically symmetric f, and its inverse:

    fhat(k) = (4 pi / k) Int_0^inf r sin(kr) f(r) dr
    f(r)    = 1/(2 pi^2 r) Int_0^inf k sin(kr) fhat(k) dk

Ornstein-Zernike, matrix form with D = diag(rho):

    Hhat = Chat + Chat D Hhat   =>   Chat = Hhat (I + D Hhat)^{-1}
                                     Hhat = (I - Chat D)^{-1} Chat

Ashcroft-Langreth partial structure factor (symmetric matrix convention):

    S = I + D^{1/2} Hhat D^{1/2}   =   (I - D^{1/2} Chat D^{1/2})^{-1}
    S_ij(k) = delta_ij + sqrt(rho_i rho_j) hhat_ij(k)

This is the SAME convention used in c22_outer_diagnosis.py (its M_of / sec_poles
build C_ij = sqrt(rho_i rho_j) chat_ij), so results across data_benchmark/ agree.

Bhatia-Thornton (number-number / concentration-concentration / number-charge),
for a binary with mole fractions x = (x1, x2), from the Ashcroft-Langreth
partials S^AL:

    S_NN = x1 S11 + x2 S22 + 2 sqrt(x1 x2) S12
    S_CC = x1 x2 [ x2 S11 + x1 S22 - 2 sqrt(x1 x2) S12 ]
    S_NC = x1 x2 (S11 - S22) + (x2 - x1) sqrt(x1 x2) S12

These follow from N = rho_1 + rho_2, C = x2 rho_1 - x1 rho_2 and the "per total
particle" normalization S_XY = <dX dY>/N, giving the ideal limits S_NN -> 1,
S_CC -> x1 x2, S_NC -> 0 as rho -> 0.

The canonical internal node is Hhat(k): every input maps to it, every output
derives from it, so all pairwise conversions come for free (class Correlations).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dst

PAIRS = [(0, 0), (0, 1), (1, 1)]
PAIR_TAG = {(0, 0): '11', (0, 1): '12', (1, 1): '22'}


# --- grid and 3-D radial transforms --------------------------------------

def make_grid(r_max: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform DST-I conjugate grids (r, k), both excluding the origin.

    r_i = i * dr with dr = r_max/(n+1), and k_i = i * pi/r_max, i = 1..n.
    """
    dr = r_max / (n + 1)
    r = np.arange(1, n + 1) * dr
    k = np.arange(1, n + 1) * np.pi / r_max
    return r, k


def dst_forward(fr: np.ndarray, r: np.ndarray, k: np.ndarray) -> np.ndarray:
    """fhat(k) = (4 pi / k) Int r sin(kr) f(r) dr, by DST-I on the uniform grid."""
    dr = r[1] - r[0]
    return (4.0 * np.pi / k) * dst(r * fr, type=1) * dr / 2.0


def dst_backward(fk: np.ndarray, r: np.ndarray, k: np.ndarray) -> np.ndarray:
    """f(r) = 1/(2 pi^2 r) Int k sin(kr) fhat(k) dk, the same DST-I."""
    dk = k[1] - k[0]
    return dst(k * fk, type=1) * dk / (4.0 * np.pi ** 2 * r)


def _transform_pairs(field: np.ndarray, r: np.ndarray, k: np.ndarray,
                     backward: bool) -> np.ndarray:
    """Apply the radial transform to each unique pair of a (2,2,N) field."""
    out = np.empty_like(field)
    fn = dst_backward if backward else dst_forward
    for i, j in PAIRS:
        out[i, j] = out[j, i] = fn(field[i, j], r, k)
    return out


# --- matrix relations in k space -----------------------------------------

def _apply_per_k(fk: np.ndarray, op) -> np.ndarray:
    """Apply a 2x2-matrix operator `op(M)` at every k point of a (2,2,N) field."""
    out = np.empty_like(fk)
    for n in range(fk.shape[2]):
        out[:, :, n] = op(fk[:, :, n])
    return out


def ck_from_hk(hk: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Chat = Hhat (I + D Hhat)^{-1}."""
    d = np.diag(rho)
    eye = np.eye(2)
    return _apply_per_k(hk, lambda m: m @ np.linalg.inv(eye + d @ m))


def hk_from_ck(ck: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Hhat = (I - Chat D)^{-1} Chat."""
    d = np.diag(rho)
    eye = np.eye(2)
    return _apply_per_k(ck, lambda m: np.linalg.inv(eye - m @ d) @ m)


def S_AL_from_hk(hk: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Ashcroft-Langreth S = I + D^{1/2} Hhat D^{1/2}."""
    sq = np.sqrt(rho)
    outer = np.outer(sq, sq)                    # sqrt(rho_i rho_j)
    return np.eye(2)[:, :, None] + outer[:, :, None] * hk


def hk_from_S_AL(S: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Hhat = D^{-1/2} (S - I) D^{-1/2}."""
    sq = np.sqrt(rho)
    outer = np.outer(sq, sq)
    return (S - np.eye(2)[:, :, None]) / outer[:, :, None]


def S_BT_from_S_AL(S_AL: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    """Bhatia-Thornton (S_NN, S_CC, S_NC) from the Ashcroft-Langreth partials."""
    x1, x2 = x
    s11, s12, s22 = S_AL[0, 0], S_AL[0, 1], S_AL[1, 1]
    root = np.sqrt(x1 * x2)
    return {
        'S_NN': x1 * s11 + x2 * s22 + 2.0 * root * s12,
        'S_CC': x1 * x2 * (x2 * s11 + x1 * s22 - 2.0 * root * s12),
        'S_NC': x1 * x2 * (s11 - s22) + (x2 - x1) * root * s12,
    }


# --- the hub -------------------------------------------------------------

@dataclass
class Correlations:
    """All structure quantities of a binary fluid, from any one of them.

    Build with a `from_*` constructor, then read any representation off the
    properties. Real-space fields are (2,2,len(r)); k-space fields and structure
    factors are (2,2,len(k)); Bhatia-Thornton is a dict of length-len(k) arrays.

    Every constructor reduces its input to the canonical node `hk` = Hhat(k);
    the properties reconstruct the rest. Pure-Fourier conversions (g<->hhat,
    c<->chat) do not use `rho`; OZ and structure-factor conversions do.
    """
    r: np.ndarray
    k: np.ndarray
    rho: np.ndarray | None      # partial number densities; None if unknown
    hk: np.ndarray              # canonical node: Hhat(k), shape (2,2,len(k))

    # -- constructors --
    @classmethod
    def from_h(cls, h, r, k, rho=None):
        return cls(r, k, _as_rho(rho), _transform_pairs(h, r, k, backward=False))

    @classmethod
    def from_g(cls, g, r, k, rho=None):
        return cls.from_h(g - 1.0, r, k, rho)

    @classmethod
    def from_hk(cls, hk, r, k, rho=None):
        return cls(r, k, _as_rho(rho), np.asarray(hk, float))

    @classmethod
    def from_ck(cls, ck, r, k, rho):
        rho = _as_rho(rho, required='ck -> anything (needs OZ)')
        return cls(r, k, rho, hk_from_ck(np.asarray(ck, float), rho))

    @classmethod
    def from_c(cls, c, r, k, rho):
        return cls.from_ck(_transform_pairs(c, r, k, backward=False), r, k, rho)

    @classmethod
    def from_S_AL(cls, S, r, k, rho):
        rho = _as_rho(rho, required='S_AL -> anything')
        return cls(r, k, rho, hk_from_S_AL(np.asarray(S, float), rho))

    # -- Fourier-space views --
    @property
    def ck(self) -> np.ndarray:
        return ck_from_hk(self.hk, self._rho('chat / DCF'))

    @property
    def S_AL(self) -> np.ndarray:
        return S_AL_from_hk(self.hk, self._rho('structure factor'))

    @property
    def S_BT(self) -> dict[str, np.ndarray]:
        rho = self._rho('Bhatia-Thornton structure factor')
        return S_BT_from_S_AL(self.S_AL, rho / rho.sum())

    # -- real-space views --
    @property
    def h(self) -> np.ndarray:
        return _transform_pairs(self.hk, self.r, self.k, backward=True)

    @property
    def g(self) -> np.ndarray:
        return self.h + 1.0

    @property
    def c(self) -> np.ndarray:
        return _transform_pairs(self.ck, self.r, self.k, backward=True)

    def _rho(self, what: str) -> np.ndarray:
        if self.rho is None:
            raise ValueError(f'{what} requires partial densities rho, which '
                             f'were not supplied')
        return self.rho


def _as_rho(rho, required: str | None = None) -> np.ndarray | None:
    if rho is None:
        if required:
            raise ValueError(f'{required} requires rho')
        return None
    rho = np.asarray(rho, float)
    if rho.shape != (2,):
        raise ValueError(f'rho must have shape (2,), got {rho.shape}')
    return rho
