import sys
import types

import numpy as np


sys.path.insert(0, ".")
sys.modules.setdefault("mpi4py", types.SimpleNamespace(MPI=types.SimpleNamespace(INT=None)))

from NiTROM.Optimization_Functions.classes import HamiltonianROM


def rel_err(a, b):
    return abs(a - b) / max(abs(a), 1e-14)


def rel_vec_err(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-14)


rng = np.random.default_rng(0)

# Hamiltonian H(z) = z^T A2 z + z^T A3 : z z^T
A2 = np.array([[0.8, -0.3], [0.4, 0.6]])
A3 = np.array([
    [[0.20, -0.10], [0.30, 0.05]],
    [[-0.40, 0.25], [0.15, 0.35]],
])
Phi = np.eye(2)
rom = HamiltonianROM(operators=[A2, A3], poly_comp=[1, 2], Phi=Phi)

z = rng.standard_normal(2)
v = rng.standard_normal(2)
v /= np.linalg.norm(v)
eps = 1e-6


# --- gradient check ---
grad_H = rom.compute_hamiltonian_gradient(z)
fd_grad = (
    rom.compute_hamiltonian(z + eps * v)
    - rom.compute_hamiltonian(z - eps * v)
) / (2 * eps)
analytic_grad = grad_H @ v

print("Gradient check")
print(f"FD      : {fd_grad:.16f}")
print(f"analytic: {analytic_grad:.16f}")
print(f"rel err : {rel_err(fd_grad, analytic_grad):.2e}")


# --- hessian-vector check ---
H = rom.compute_hamiltonian_hessian(z)
fd_hess_vec = (
    rom.compute_hamiltonian_gradient(z + eps * v)
    - rom.compute_hamiltonian_gradient(z - eps * v)
) / (2 * eps)
analytic_hess_vec = H @ v

print("\nHessian-vector check")
print("FD      :", fd_hess_vec)
print("analytic:", analytic_hess_vec)
print(f"rel err : {rel_vec_err(fd_hess_vec, analytic_hess_vec):.2e}")


# --- vector-hessian-vector check ---
fd_second = (
    rom.compute_hamiltonian(z + eps * v)
    - 2 * rom.compute_hamiltonian(z)
    + rom.compute_hamiltonian(z - eps * v)
) / (eps ** 2)
analytic_second = v @ H @ v

print("\nSecond derivative check")
print(f"FD      : {fd_second:.16f}")
print(f"analytic: {analytic_second:.16f}")
print(f"rel err : {rel_err(fd_second, analytic_second):.2e}")


# --- symmetry check ---
print("\nSymmetry check")
print(f"||H - H.T|| = {np.linalg.norm(H - H.T):.2e}")
