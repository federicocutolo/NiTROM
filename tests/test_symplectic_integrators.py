import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
from scipy.linalg import expm

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from NiTROM.Optimization_Functions.symplectic_integrators import (
    symplectic_integrate,
)


def test_fourth_order_symplectic_integrator_zero_rhs_keeps_state_constant():
    z0 = np.array([0.3, -0.4])
    time = np.linspace(0.0, 1.0, 6)

    Z = symplectic_integrate(
        rhs=lambda z: np.zeros_like(z),
        jacobian=lambda z: np.zeros((len(z), len(z))),
        z0=z0,
        time=time,
    )

    expected = np.column_stack([z0 for _ in time])
    npt.assert_allclose(Z, expected)


def test_fourth_order_symplectic_integrator_converges_for_nonseparable_quadratic_hamiltonian():
    K = np.array([[2.0, 1.0], [1.0, 3.0]])
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    A = J @ K
    z0 = np.array([0.8, -0.4])
    tf = 1.0

    rhs = lambda z: A @ z
    jacobian = lambda z: A
    z_exact = expm(tf * A) @ z0

    hs = np.array([0.2, 0.1, 0.05, 0.025])
    errors = []

    for h in hs:
        n_steps = int(round(tf / h))
        time = np.linspace(0.0, tf, n_steps + 1)
        Z = symplectic_integrate(rhs, jacobian, z0, time)
        errors.append(np.linalg.norm(Z[:, -1] - z_exact))

    errors = np.asarray(errors)
    slope, _ = np.polyfit(np.log(hs), np.log(errors), 1)
    
    assert np.all(errors > 0.0)
    assert slope > 3.8
