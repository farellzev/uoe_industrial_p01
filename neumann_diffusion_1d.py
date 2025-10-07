
"""
Analytical solution for 1D diffusion with Neumann BC on [-d, d]
================================================================

PDE:
    ∂C/∂t = D ∂²C/∂x²,     x ∈ (-d, d),  t > 0

BC (Neumann, no flux):
    ∂C/∂x (±d, t) = 0

IC (point source at origin):
    C(x, 0) = δ(x)

Eigenfunction expansion (separation of variables):
    On [-d, d], the Neumann eigenfunctions and eigenvalues are
        φ₀(x) = 1 / √(2d),                  λ₀ = 0
        φₙ(x) = cos(nπ x / d) / √d,         λₙ = (nπ/d)²,  n ≥ 1

    Expanding δ(x) at x=0 in this basis gives coefficients φₙ(0),
    so the solution is

        C(x, t) = φ₀(0)φ₀(x) e^{-D λ₀ t}
                 + ∑_{n=1}^∞ φₙ(0) φₙ(x) e^{-D λₙ t}

                 = 1/(2d) + (1/d) ∑_{n=1}^∞ cos(nπ x / d) e^{-D (nπ/d)² t}.

Usage
-----
    from neumann_diffusion_1d import concentration, concentration_grid

    # pointwise
    c = concentration(x=0.1, t=0.05, D=1.0, d=1.0, N=500)

    # vectorized over x (1D array) and scalar t
    xs = np.linspace(-1.0, 1.0, 1001)
    cs = concentration_grid(xs, t=0.05, D=1.0, d=1.0, N=500)

    # mass check (should be ~1.0 numerically)
    # mass = ∫_{-d}^{d} C(x, t) dx = 1, for all t (Neumann conserves mass)
"""
from __future__ import annotations
import numpy as np

def _series_terms(x: np.ndarray, t: float, D: float, d: float, N: int) -> np.ndarray:
    """
    Compute the cosine-series tail for n=1..N at points x (vectorized).

    Returns an array shaped (len(x),) with the sum over n applied.
    """
    n = np.arange(1, N + 1, dtype=float)
    # shape (N,)
    k = (np.pi * n) / d  # wave numbers
    decay = np.exp(-D * (k ** 2) * t)  # shape (N,)

    # cos(nπ x/d) for each x -> shape (len(x), N)
    X = np.atleast_1d(x).reshape(-1, 1)
    cos_part = np.cos((np.pi * n / d) * X)

    # sum over n (axis=1), result shape (len(x),)
    return (decay * cos_part).sum(axis=1)

def concentration(x: np.ndarray | float, t: float, D: float, d: float, N: int = 500) -> np.ndarray:
    """
    Analytical solution C(x, t) with Neumann BC on [-d, d] and point-source IC at x=0.

    Parameters
    ----------
    x : array-like or float
        Spatial position(s) at which to evaluate the concentration.
    t : float
        Time >= 0.
    D : float
        Diffusivity > 0.
    d : float
        Half-length of the domain; domain is [-d, d], d > 0.
    N : int, optional
        Number of terms in the cosine series (truncation). Larger N -> higher fidelity near t≈0.

    Returns
    -------
    np.ndarray
        Concentration evaluated at x (vectorized).
    """
    x = np.atleast_1d(x).astype(float)
    if t < 0:
        raise ValueError("t must be >= 0")
    if D <= 0:
        raise ValueError("D must be > 0")
    if d <= 0:
        raise ValueError("d must be > 0")
    if N < 0:
        raise ValueError("N must be >= 0")

    # C(x,t) = 1/(2d) + (1/d) * sum_{n=1}^N cos(nπ x / d) * exp(-D (nπ/d)^2 t)
    tail = _series_terms(x, t, D, d, N) if N > 0 else np.zeros_like(x)
    return (1.0 / (2.0 * d)) + (1.0 / d) * tail

def concentration_grid(xs: np.ndarray, t: float, D: float, d: float, N: int = 500) -> np.ndarray:
    """
    Convenience wrapper for evaluating on a grid of xs (1D array).
    """
    xs = np.asarray(xs, dtype=float)
    return concentration(xs, t=t, D=D, d=d, N=N)

def mass(xs: np.ndarray, cs: np.ndarray) -> float:
    """
    Numerical mass via trapezoidal rule over grid xs of length M over [-d, d].
    """
    xs = np.asarray(xs, dtype=float)
    cs = np.asarray(cs, dtype=float)
    if xs.ndim != 1 or cs.ndim != 1 or xs.size != cs.size:
        raise ValueError("xs and cs must be 1D arrays of the same length.")
    return np.trapz(cs, xs)
