import sys
import types

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


sys.path.insert(0, ".")
sys.modules.setdefault("mpi4py", types.SimpleNamespace(MPI=types.SimpleNamespace(INT=None)))

from NiTROM.Optimization_Functions.classes import HamiltonianROM


rng = np.random.default_rng(0)

A2 = np.array([[0.8, -0.3], [0.4, 0.6]])
A3 = np.array([
    [[0.20, -0.10], [0.30, 0.05]],
    [[-0.40, 0.25], [0.15, 0.35]],
])

Phi = np.eye(2)
rom = HamiltonianROM(operators=[A2, A3], poly_comp=[1, 2], Phi=Phi)

time = np.linspace(0.0, 1, 100, endpoint=True)
z_target = np.ones(2)
z0 = np.array([0.8, -0.4])
eps = 1e-5


def evaluate_cost(z0_):
    Z = rom.integrate(time, z0_)
    # Z = solve_ivp(
    #     rom.evaluate_rom_rhs,
    #     [0.0, 1.0],
    #     z0_,
    #     method="RK45",
    #     t_eval=time,
    #     atol=1e-12,
    #     rtol=1e-12,
    # ).y

    zT = Z[:, -1]
    eT = zT - z_target
    return 0.5 * np.dot(eT, eT)


def evaluate_gradient_fd(z0_):
    g = np.zeros_like(z0_)
    for j in range(len(z0_)):
        e = np.zeros_like(z0_)
        e[j] = 1.0
        g[j] = (evaluate_cost(z0_ + eps * e) - evaluate_cost(z0_ - eps * e)) / (2 * eps)
    return g


def evaluate_gradient_adjoint(z0_):
    Z = rom.integrate(time, z0_)
    # Z = solve_ivp(
    #     rom.evaluate_rom_rhs,
    #     [0.0, 1.0],
    #     z0_,
    #     method="RK45",
    #     t_eval=time,
    #     atol=1e-12,
    #     rtol=1e-12,
    # ).y

    zT = Z[:, -1]
    lamT = zT - z_target

    fz = interp1d(time, Z, kind="cubic", fill_value="extrapolate")
    # interpolation is not active (Z and time share same time grid))

    Lam = solve_ivp(
        rom.evaluate_rom_adjoint,
        [1.0, 0.0],
        lamT,
        method="RK45",
        t_eval=np.flipud(time),
        atol=1e-12,
        rtol=1e-12,
        args=(fz,) + tuple(rom.operators),
    ).y

    return Lam[:, -1]


g_fd = evaluate_gradient_fd(z0)
g_adj = evaluate_gradient_adjoint(z0)

print("Adjoint check")
print("FD gradient     :", g_fd)
print("Adjoint lambda0 :", g_adj)
print("abs err         :", np.linalg.norm(g_fd - g_adj))
print("rel err         :", np.linalg.norm(g_fd - g_adj) / max(np.linalg.norm(g_fd), 1e-14))
