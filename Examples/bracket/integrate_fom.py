"""
Symplectically integrate the FULL FOM  x' = J A x,  A = blkdiag(K, M^-1),
with the Cayley (implicit midpoint) step, storing X and Xdot at a sampling dt.

Run:  python integrate_fom.py

Sparse Cayley step (M^-1 is NEVER formed)
-----------------------------------------
With x = [q; p] the dynamics are  q' = M^-1 p,  p' = -K q.  The implicit
midpoint step  (I - h/2 S) x_{n+1} = (I + h/2 S) x_n,  S = J A, reads

    q1 - (h/2) M^-1 p1 = q0 + (h/2) M^-1 p0
    p1 + (h/2) K   q1  = p0 - (h/2) K   q0 .

Multiply the first row by M (kills M^-1) and eliminate p1 -> a single SPD
sparse Schur system per step:

    (M + h^2/4 K) q1 = M q0 + h p0 - h^2/4 (K q0)
              p1     = p0 - (h/2) (K q0 + K q1) .

All operators are constant, so (M + h^2/4 K) is LU-factorized ONCE; each step
then costs 1 sparse back-substitution + 1 K mat-vec + 1 M mat-vec (K q1 is
reused as the next step's K q0). This is algebraically IDENTICAL to the dense
Cayley step of linear_symplectic_integrators (same map, hence symplectic and
exactly H-conserving), but at sparse-FOM cost.

Optionally the 4th-order Yoshida triple-jump of the Cayley step is used
(3 substeps/step; the two distinct Schur matrices are each factorized once).

Derivatives are exact rates on the trajectory:  qdot = M^-1 p (prefactored
sparse solve, only at sampling times) and pdot = -K q (mat-vec, reused).
"""
import os
import sys
import time as tlib

import numpy as np
from scipy.io import mmread
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
import exodusii

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from plate_cases import get_case

# ── Config ────────────────────────────────────────────────────────────────────
TESTCASE   = "bracket"
T0, TF     = 0.0, 0.02       # integration window (match the data: [0, 0.02])
DT_SAMPLE  = 1e-4            # store X / Xdot every DT_SAMPLE (= data snapshot dt)
DT_INT     = 2.5e-8          # target internal step; snapped so DT_SAMPLE = n * dt
ORDER      = 2               # 2 = one Cayley step | 4 = Yoshida triple-jump of Cayley
STORE_XDOT = True
OUT_PREFIX = "fom_cayley"    # writes <data_dir>/{OUT_PREFIX}_{x,xdot,time}.npy

_case = get_case(TESTCASE)

# Yoshida triple-jump coefficients (4th order from the 2nd-order Cayley step).
_CBRT2  = np.cbrt(2.0)
_GAMMA1 = 1.0 / (2.0 - _CBRT2)
_GAMMA2 = -_CBRT2 / (2.0 - _CBRT2)


def load_initial_condition(exo_path, M):
    """q0 (interleaved x,y,z) and p0 = M q_dot0 from the first Exodus snapshot."""
    exo = exodusii.ExodusIIFile(exo_path, mode="r")
    N = exo.num_nodes()
    q0, qdot0 = np.zeros(3 * N), np.zeros(3 * N)
    for c, comp in enumerate("xyz"):
        q0[c::3]    = exo.get_node_variable_values(f"disp_{comp}", 1)
        qdot0[c::3] = exo.get_node_variable_values(f"solution_dot_{comp}", 1)
    return q0, M @ qdot0


def make_cayley_substep(M_csr, K_csr, h):
    """One implicit-midpoint (Cayley) substep of size h in Schur form.

    Returns step(q, p, Kq) -> (q1, p1, Kq1) with Kq = K @ q carried between
    calls, so each substep is exactly 1 LU back-substitution + 1 K mat-vec
    + 1 M mat-vec. The Schur matrix M + (h/2)^2*... is SPD for ANY sign of h
    (Yoshida's negative substep included), factorized once here.
    """
    c  = 0.25 * h * h
    lu = splu(csc_matrix(M_csr + c * K_csr))

    def step(q, p, Kq):
        q1  = lu.solve(M_csr @ q + h * p - c * Kq)
        Kq1 = K_csr @ q1
        p1  = p - 0.5 * h * (Kq + Kq1)
        return q1, p1, Kq1

    return step


def build_stepper(M_csr, K_csr, h, order):
    """Full one-step map of size h: a single Cayley substep (order 2) or the
    Yoshida composition C(g1 h) C(g2 h) C(g1 h) (order 4, 3 substeps/step)."""
    if order == 2:
        return make_cayley_substep(M_csr, K_csr, h)
    if order == 4:
        s1 = make_cayley_substep(M_csr, K_csr, _GAMMA1 * h)   # outer substeps
        s2 = make_cayley_substep(M_csr, K_csr, _GAMMA2 * h)   # inner (negative) substep

        def step(q, p, Kq):
            q, p, Kq = s1(q, p, Kq)
            q, p, Kq = s2(q, p, Kq)
            return s1(q, p, Kq)

        return step
    raise ValueError(f"ORDER must be 2 or 4, got {order}")


def main():
    M = csc_matrix(mmread(_case.mass))
    K = csc_matrix(mmread(_case.stiff))
    n = M.shape[0]                                        # = 3N displacement DOFs
    M_csr, K_csr = M.tocsr(), K.tocsr()                   # csr: fastest mat-vec

    # Internal step commensurate with the sampling: DT_SAMPLE = n_sub * dt exactly,
    # so every stored sample is a genuine integrator state (no interpolation).
    n_sub = max(1, int(round(DT_SAMPLE / DT_INT)))
    dt    = DT_SAMPLE / n_sub
    n_samples = int(round((TF - T0) / DT_SAMPLE)) + 1
    t_out     = T0 + DT_SAMPLE * np.arange(n_samples)

    q, p = load_initial_condition(_case.exo, M)
    Kq   = K_csr @ q

    step = build_stepper(M_csr, K_csr, dt, ORDER)
    Mlu  = splu(M)                                        # for qdot = M^-1 p at samples only

    X    = np.empty((2 * n, n_samples))
    Xdot = np.empty((2 * n, n_samples)) if STORE_XDOT else None
    H    = np.empty(n_samples)                            # energy audit (free at samples)

    def record(k):
        qdot = Mlu.solve(p)                               # exact rate  q' = M^-1 p
        X[:n, k], X[n:, k] = q, p
        if STORE_XDOT:
            Xdot[:n, k], Xdot[n:, k] = qdot, -Kq          # p' = -K q (mat-vec reused)
        H[k] = 0.5 * (q @ Kq + p @ qdot)                  # H = 1/2 (q^T K q + p^T M^-1 p)

    print(f"FOM Cayley integration: n = {2 * n} states, order {ORDER}, "
          f"dt = {dt:.4g} ({n_sub} steps/sample), "
          f"{(n_samples - 1) * n_sub:,} steps, {n_samples} samples")
    t_wall = tlib.perf_counter()

    record(0)
    for k in range(1, n_samples):
        for _ in range(n_sub):
            q, p, Kq = step(q, p, Kq)
        record(k)
        if k % max(1, n_samples // 10) == 0:
            rate = k * n_sub / (tlib.perf_counter() - t_wall)
            print(f"  t = {t_out[k]:.6g}  ({k}/{n_samples - 1} samples, "
                  f"{rate:,.0f} steps/s)", flush=True)

    elapsed = tlib.perf_counter() - t_wall
    drift = np.abs(H - H[0]).max() / abs(H[0])
    print(f"Done in {elapsed:.1f} s ({(n_samples - 1) * n_sub / elapsed:,.0f} steps/s)")
    print(f"Relative energy drift max |H - H0| / |H0| = {drift:.3e}  "
          f"(Cayley conserves quadratic H exactly -> expect round-off)")

    out = _case.data_dir
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, f"{OUT_PREFIX}_x.npy"),    X)
    np.save(os.path.join(out, f"{OUT_PREFIX}_time.npy"), t_out)
    np.save(os.path.join(out, f"{OUT_PREFIX}_energy.npy"), H)
    if STORE_XDOT:
        np.save(os.path.join(out, f"{OUT_PREFIX}_xdot.npy"), Xdot)
    print(f"Wrote {OUT_PREFIX}_x/xdot/time/energy .npy to {out}")


if __name__ == "__main__":
    main()
