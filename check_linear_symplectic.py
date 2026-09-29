"""
Standalone sanity checks for ``linear_symplectic_integrators.py``.

Run from the repo root:   python check_linear_symplectic.py

Test system
-----------
A random *linear Hamiltonian* system in canonical coordinates z = (q, p):

    H(z)  = 1/2 z^T S z          (S symmetric positive definite)
    z'    = J grad H = J S z     =>   A = J S,   J = [[0, I], [-I, 0]]

so we know everything in closed form:
  * the exact flow is  z(t) = expm(t A) z0,
  * a one-step map M is symplectic  iff  M^T J M = J,
  * the implicit midpoint / Cayley step conserves the quadratic H *exactly*
    (up to round-off), so energy drift along the trajectory must be ~1e-14.

Checks
------
  1. Symplecticity of the one-step propagators:  || M^T J M - J ||  ~ round-off.
  2. Exact energy conservation along the integrated trajectory.
  3. Agreement with the general Newton-based ``symplectic_integrators`` module,
     both directly on the time grid (dt=None) and through the internal-dt +
     cubic-spline path of ``*_solve``.
  4. Sparse A gives the same trajectory as dense A.
  5. 4th-order convergence toward the exact expm solution as dt is refined.
"""
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from NiTROM.Optimization_Functions.linear_symplectic_integrators import (
    cayley_step,
    yoshida4_propagator,
    linear_symplectic_solve,
)
from NiTROM.Optimization_Functions.symplectic_integrators import symplectic_solve

# ----------------------------------------------------------------- test setup
rng = np.random.default_rng(0)

n_dof = 4                                   # -> state dimension 2*n_dof = 8
n = 2 * n_dof

# Canonical symplectic form J and a random SPD Hamiltonian matrix S.
J = np.block([[np.zeros((n_dof, n_dof)), np.eye(n_dof)],
              [-np.eye(n_dof), np.zeros((n_dof, n_dof))]])
G = rng.standard_normal((n, n))
S = G @ G.T + n * np.eye(n)                 # symmetric positive definite
A = J @ S                                   # linear Hamiltonian vector field

z0 = rng.standard_normal(n)
energy = lambda Z: 0.5 * np.einsum("it,ij,jt->t", Z, S, Z)   # H at every column

t_eval = np.linspace(0.0, 5.0, 201)         # uniform grid, dt_grid = 0.025
dt_int = 0.01                               # internal step for the *_solve path

failures = []

def report(label, value, tol):
    status = "PASS" if value < tol else "FAIL"
    if value >= tol:
        failures.append(label)
    print(f"  [{status}] {label:<55s} {value:9.2e}  (tol {tol:.0e})")


# ------------------------------------------- 1. one-step maps are symplectic
print("\n1) Symplecticity of the one-step propagators:  || M^T J M - J ||_max")
for label, M in [
    ("Cayley (implicit midpoint) step, h = 0.1", cayley_step(A, 0.1)),
    ("Yoshida 4th-order step,          h = 0.1", yoshida4_propagator(A, 0.1)),
    ("Yoshida 4th-order step,          h = 1.0 (big step)", yoshida4_propagator(A, 1.0)),
]:
    report(label, np.abs(M.T @ J @ M - J).max(), 1e-12)

# --------------------------------- 2. energy is exactly conserved (quadratic H)
print("\n2) Energy conservation along the trajectory:  max |H(t) - H(0)| / |H(0)|")
Z_lin = linear_symplectic_solve(A, z0, t_eval)              # dt=None: no spline
H = energy(Z_lin)
report("linear integrator, dt=None (grid steps)", np.abs(H - H[0]).max() / abs(H[0]), 1e-12)

_, t_fine, Z_fine = linear_symplectic_solve(A, z0, t_eval, dt=dt_int, return_internal=True)
Hf = energy(Z_fine)
report(f"linear integrator, internal dt={dt_int}", np.abs(Hf - Hf[0]).max() / abs(Hf[0]), 1e-12)

# ------------------------- 3. agreement with the general (Newton) integrator
print("\n3) Agreement with symplectic_integrators (Newton):  max |Z_lin - Z_gen|")
rhs = lambda z: A @ z
jac = lambda z: A

Z_gen = symplectic_solve(rhs, jac, z0, t_eval)              # dt=None
report("dt=None  (integrate directly on t_eval)", np.abs(Z_lin - Z_gen).max(), 1e-9)

Z_lin_dt = linear_symplectic_solve(A, z0, t_eval, dt=dt_int)
Z_gen_dt = symplectic_solve(rhs, jac, z0, t_eval, dt=dt_int)
report(f"dt={dt_int} (internal grid + cubic spline)", np.abs(Z_lin_dt - Z_gen_dt).max(), 1e-9)

# ------------------------------------------------- 4. sparse == dense path
print("\n4) Sparse-A path matches dense-A path:  max |Z_sparse - Z_dense|")
Z_sp = linear_symplectic_solve(csc_matrix(A), z0, t_eval, dt=dt_int)
report("scipy.sparse csc vs dense ndarray", np.abs(Z_sp - Z_lin_dt).max(), 1e-10)

# --------------------------------------- 5. 4th-order convergence to exact flow
print("\n5) Convergence to the exact flow z(T) = expm(T A) z0  (expect order ~4)")
T = 1.0
z_exact = expm(T * A) @ z0
# Start fine enough that omega_max * dt << 1 (asymptotic regime for the order).
dts = [0.02, 0.01, 0.005, 0.0025]
errs = [np.abs(linear_symplectic_solve(A, z0, np.array([0.0, T]), dt=h)[:, -1] - z_exact).max()
        for h in dts]
orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
for h, e in zip(dts, errs):
    print(f"       dt = {h:<7g} error = {e:.3e}")
print(f"       observed orders: {['%.2f' % p for p in orders]}")
if min(orders) < 3.5:
    failures.append("4th-order convergence")
    print("  [FAIL] convergence order below 3.5")
else:
    print("  [PASS] convergence order ~ 4")

# --------------------------------------------------------------------- summary
print("\n" + "=" * 60)
if failures:
    print(f"FAILED checks: {failures}")
    raise SystemExit(1)
print("All checks passed: linear_symplectic_solve is symplectic and matches")
print("the general Newton-based symplectic integrator to round-off.")
