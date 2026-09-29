"""
Symplectic time integration specialized for LINEAR systems  z' = A z.

Fast drop-in for ``symplectic_integrators`` when the right-hand side is linear
with a *constant* Jacobian ``A`` (e.g. a linear ROM z' = A z, A = Jhat(A2+A2^T),
Jhat^{-T} A_bar, W_hat Jhat, or J A_fom).

Two simplifications over the general (Newton-based) version:

  1. No Newton iteration. The implicit-midpoint update for z' = A z is the exact
     Cayley transform
            (I - h/2 A) z_{n+1} = (I + h/2 A) z_n
        =>  z_{n+1} = C(h) z_n ,   C(h) = (I - h/2 A)^{-1} (I + h/2 A).

  2. Constant operators are precomputed ONCE per (unique) step size. The 4th-order
     Yoshida step is a fixed composition of Cayley steps, so it is applied via a
     single reusable one-step operator.

Dense vs sparse ``A`` is handled automatically:

  * DENSE  A -> the one-step propagator  M = C(g1 dt) C(g2 dt) C(g1 dt)  is formed
    explicitly and the march is  z_k = M z_{k-1}  (one matmul/step). Best for the
    small dense reduced operators.
  * SPARSE A -> M is NEVER formed (it would densify). Instead the two distinct
    denominators (I - g_i dt/2 A) are LU-factorized once (scipy splu) and each
    step is 3 sparse mat-vecs + 3 back-substitutions. Best for large sparse A
    (e.g. an uncompressed FOM), where a dense n x n propagator is infeasible.

For a linear Hamiltonian system this reproduces the general integrator's
trajectory to round-off, only far cheaper.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.interpolate import CubicSpline

# Yoshida triple-jump coefficients (4th order from the 2nd-order midpoint step).
_CBRT2 = np.cbrt(2.0)
_GAMMA1 = 1.0 / (2.0 - _CBRT2)
_GAMMA2 = -_CBRT2 / (2.0 - _CBRT2)


def cayley_step(A: np.ndarray, h: float) -> np.ndarray:
    """Dense implicit-midpoint one-step propagator for z' = A z:
    ``C(h) = (I - h/2 A)^{-1} (I + h/2 A)`` (symplectic for linear Hamiltonian A)."""
    A = np.asarray(A, dtype=float)
    eye = np.eye(A.shape[0])
    return np.linalg.solve(eye - 0.5 * h * A, eye + 0.5 * h * A)


def yoshida4_propagator(A: np.ndarray, h: float) -> np.ndarray:
    """Dense 4th-order symplectic one-step propagator for z' = A z (Yoshida
    triple-jump of the Cayley step): ``z_{n+1} = M z_n``."""
    c1 = cayley_step(A, _GAMMA1 * h)
    c2 = cayley_step(A, _GAMMA2 * h)
    return c1 @ c2 @ c1                       # apply C1, then C2, then C1


def linear_propagator(A, tau, dt=None):
    """Compounded symplectic propagator over a DURATION ``tau`` for z' = A z:
    the single matrix equal to marching ``floor(tau/dt)`` Yoshida steps of size
    ``dt`` plus one fractional step for the remainder,

        P(tau) = M(rem) @ M(dt)^n ,   n = floor(tau/dt),  rem = tau - n*dt,

    so ``z(t + tau) = P(tau) @ z(t)`` in ONE matvec. All factors are rational
    functions of the same A, so they commute and the composition order is
    immaterial. Being a product of symplectic maps, P is symplectic.

    Built once (log2(n) matmuls by repeated squaring) and reused; this replaces
    step-by-step sub-stepping wherever only the state at t + tau is needed.
    ``dt=None`` (or dt >= tau) means a single step of size tau. Dense A only.
    """
    if dt is None or dt >= tau:
        return yoshida4_propagator(A, tau)
    n_full = int(np.floor(tau / dt + 1e-9))          # tolerate roundoff in tau/dt
    rem = tau - n_full * dt
    P = np.linalg.matrix_power(yoshida4_propagator(A, dt), n_full)
    if rem > 1e-9 * dt:                              # skip a zero-length step
        P = yoshida4_propagator(A, rem) @ P
    return P


def _sparse_cayley_ops(A, h):
    """Factor the h-Cayley step for sparse A: return (LU of (I - h/2 A),
    (I + h/2 A) for mat-vec). Applying the step is  lu.solve(B_plus @ z)."""
    n = A.shape[0]
    ident = sp.identity(n, format="csc")
    minus = (ident - 0.5 * h * A).tocsc()
    plus = (ident + 0.5 * h * A).tocsr()
    return splu(minus), plus


def _build_stepper(A, h):
    """Return a callable ``step(z)`` advancing one 4th-order symplectic step for
    z' = A z, with all h-dependent operators precomputed. Sparse A uses LU + back-
    substitution (no dense propagator); dense A uses the explicit propagator."""
    if sp.issparse(A):
        A = A.tocsc()
        lu1, plus1 = _sparse_cayley_ops(A, _GAMMA1 * h)   # used for the two outer C(g1 h)
        lu2, plus2 = _sparse_cayley_ops(A, _GAMMA2 * h)   # inner C(g2 h)

        def step(z):
            z = lu1.solve(plus1 @ z)                      # C(g1 h)
            z = lu2.solve(plus2 @ z)                      # C(g2 h)
            z = lu1.solve(plus1 @ z)                      # C(g1 h)
            return z
        return step

    M = yoshida4_propagator(A, h)                         # dense: one matmul/step
    return lambda z: M @ z


def linear_symplectic_integrate(A, z0: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Integrate z' = A z on a prescribed time grid with the 4th-order symplectic
    (Yoshida) method. Each consecutive gap is one step. The step operator is built
    once per distinct gap (once total for a uniform grid). ``A`` may be dense or
    a scipy sparse matrix."""
    z0 = np.asarray(z0, dtype=float)
    time = np.asarray(time, dtype=float)

    if time.ndim != 1:
        raise ValueError("time must be a one-dimensional array.")
    if len(time) < 2:
        raise ValueError("At least two time points are required.")
    gaps = np.diff(time)
    if np.any(gaps <= 0.0):
        raise ValueError("time grid must be strictly increasing.")

    Z = np.zeros((z0.size, len(time)), dtype=float)
    Z[:, 0] = z0

    if np.allclose(gaps, gaps[0]):                        # uniform grid -> one stepper
        step = _build_stepper(A, gaps[0])
        for k in range(1, len(time)):
            Z[:, k] = step(Z[:, k - 1])
    else:                                                 # non-uniform -> cache per gap
        cache: dict[float, object] = {}
        for k in range(1, len(time)):
            key = round(float(gaps[k - 1]), 15)
            step = cache.get(key)
            if step is None:
                step = _build_stepper(A, gaps[k - 1])
                cache[key] = step
            Z[:, k] = step(Z[:, k - 1])
    return Z


def linear_symplectic_solve(A, z0: np.ndarray, t_eval: np.ndarray,
                            *, dt: float | None = None,
                            return_internal: bool = False):
    """Integrate z' = A z at internal step ``dt`` and return the solution at the
    requested times ``t_eval`` (cubic-spline-interpolated when ``dt`` subdivides
    the grid). Mirrors ``symplectic_solve`` but takes the constant matrix ``A``
    (dense or sparse) instead of (rhs, jacobian).

    Parameters
    ----------
    dt
        Internal symplectic step. If ``None``, integrates directly on ``t_eval``.
    return_internal
        If ``True``, also return ``(fine_time, Z_fine)`` on the internal grid.
    """
    t_eval = np.asarray(t_eval, dtype=float)

    if dt is None:
        Z = linear_symplectic_integrate(A, z0, t_eval)
        return (Z, t_eval, Z) if return_internal else Z

    t0, tf = float(t_eval[0]), float(t_eval[-1])
    fine_time = np.arange(t0, tf + 0.5 * dt, dt)
    if fine_time[-1] < tf:
        fine_time = np.append(fine_time, tf)

    Z_fine = linear_symplectic_integrate(A, z0, fine_time)

    # Fast path: when every requested time coincides with an internal knot
    # (commensurate dt, the recommended setup), return the integrator states
    # directly. Exact -- a spline evaluated at its knots returns the data --
    # and avoids building CubicSpline coefficients over the fine grid, whose
    # storage is ~4x the (already large) fine trajectory itself.
    idx = np.rint((t_eval - t0) / dt).astype(int)
    on_knots = ((idx >= 0).all() and idx.max() < len(fine_time)
                and np.abs(fine_time[idx] - t_eval).max() <= 1e-3 * dt)
    if on_knots:
        Z_eval = Z_fine[:, idx].copy()
    else:
        Z_eval = CubicSpline(fine_time, Z_fine, axis=1)(t_eval)

    return (Z_eval, fine_time, Z_fine) if return_internal else Z_eval
