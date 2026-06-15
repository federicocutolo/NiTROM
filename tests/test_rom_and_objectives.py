import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest


def _load_classes_module():
    base = Path(__file__).resolve().parents[1] / "NiTROM" / "Optimization_Functions"

    mpi_module = types.ModuleType("mpi4py")
    mpi_module.MPI = types.SimpleNamespace(INT=None)
    sys.modules.setdefault("mpi4py", mpi_module)

    classes_spec = importlib.util.spec_from_file_location(
        "nitrom_rom_objective_tests",
        base / "classes.py",
    )
    classes_module = importlib.util.module_from_spec(classes_spec)
    assert classes_spec.loader is not None
    classes_spec.loader.exec_module(classes_module)
    return classes_module


classes = _load_classes_module()


class _DummyComm:
    def Get_size(self):
        return 1

    def Get_rank(self):
        return 0

    def Allgather(self, sendbuf, recvbuf):
        recv_array = recvbuf[0]
        recv_array[...] = np.asarray(sendbuf[0])

    def allgather(self, value):
        return [value]


class _DummyPool:
    def __init__(self, X, time, forcing=None):
        self.comm = _DummyComm()
        self.size = 1
        self.rank = 0
        self.my_n_traj = X.shape[0]
        self.X = X
        self.time = time
        self.F = np.zeros((X.shape[1], X.shape[0])) if forcing is None else forcing
        self.weights = np.ones(X.shape[0])


class _IdentityFOM:
    def compute_output(self, q):
        return q

    def compute_output_derivative(self, q):
        return np.eye(q.shape[0])


def _polynomial_setup():
    time = np.array([0.0, 0.05, 0.10])
    Phi = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, -0.5],
    ])
    Psi = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    A = np.array([[-0.2, 0.1], [0.0, -0.3]])
    rom = classes.PolynomialROM(operators=[A], poly_comp=[1], Phi=Phi, Psi=Psi)
    z0 = np.array([0.1, -0.2])
    Z = rom.integrate(time, z0, forcing=np.zeros_like(z0))
    X = rom.reconstruct(Z)[None, :, :]
    pool = _DummyPool(X, time)
    opt_obj = classes.optimization_objects(
        pool,
        which_trajs=np.array([0]),
        which_times=np.arange(len(time)),
        leggauss_deg=3,
        nsave_rom=4,
        poly_comp=[1],
        hamiltonian=False,
    )
    return pool, opt_obj, rom, Phi, Psi, A


def _hamiltonian_setup():
    time = np.array([0.0, 0.05, 0.10])
    Phi = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])
    A = np.array([[0.8, -0.3], [0.4, 0.6]])
    rom = classes.HamiltonianROM(operators=[A], poly_comp=[1], Phi=Phi)
    z0 = np.array([0.15, 0.05])
    Z = rom.integrate(time, z0)
    X = rom.reconstruct(Z)[None, :, :]
    pool = _DummyPool(X, time)
    opt_obj = classes.optimization_objects(
        pool,
        which_trajs=np.array([0]),
        which_times=np.arange(len(time)),
        leggauss_deg=3,
        nsave_rom=4,
        poly_comp=[1],
        hamiltonian=True,
    )
    return pool, opt_obj, rom, Phi, A


def _polynomial_tall_setup():
    time = np.array([0.0, 0.05, 0.10])
    Phi = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, -0.5],
    ])
    Psi = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    A = np.array([[-0.2, 0.1], [0.0, -0.3]])
    rom = classes.PolynomialROM(operators=[A], poly_comp=[1], Phi=Phi, Psi=Psi)
    z0 = np.array([0.1, -0.2])
    Z = rom.integrate(time, z0, forcing=np.zeros_like(z0))
    X = rom.reconstruct(Z)[None, :, :]
    pool = _DummyPool(X, time)
    opt_obj = classes.optimization_objects(
        pool,
        which_trajs=np.array([0]),
        which_times=np.arange(len(time)),
        leggauss_deg=3,
        nsave_rom=4,
        poly_comp=[1],
        hamiltonian=False,
    )
    return pool, opt_obj, rom, Phi, Psi, A


def _hamiltonian_tall_setup():
    time = np.array([0.0, 0.05, 0.10])
    Phi = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ])
    A = 0.5 * np.eye(2)
    rom = classes.HamiltonianROM(operators=[A], poly_comp=[1], Phi=Phi)
    z0 = np.array([0.15, 0.05])
    Z = rom.integrate(time, z0)
    X = rom.reconstruct(Z)[None, :, :]
    pool = _DummyPool(X, time)
    opt_obj = classes.optimization_objects(
        pool,
        which_trajs=np.array([0]),
        which_times=np.arange(len(time)),
        leggauss_deg=3,
        nsave_rom=4,
        poly_comp=[1],
        hamiltonian=True,
    )
    return pool, opt_obj, rom, Phi, A


def test_polynomial_rom_build_projection_operators_matches_formula():
    _, _, rom, Phi, Psi, _ = _polynomial_setup()
    ops = rom.build_projection_operators()

    F_expected = np.linalg.inv(Psi.T @ Phi)
    decoder_expected = Phi @ F_expected
    encoder_expected = Psi.T
    projector_expected = decoder_expected @ encoder_expected

    print("\n[poly projection] expected F:\n", F_expected)
    print("[poly projection] obtained F:\n", ops["F"])
    print("[poly projection] expected decoder:\n", decoder_expected)
    print("[poly projection] obtained decoder:\n", ops["decoder"])
    print("[poly projection] expected encoder:\n", encoder_expected)
    print("[poly projection] obtained encoder:\n", ops["encoder"])
    print("[poly projection] expected projector:\n", projector_expected)
    print("[poly projection] obtained projector:\n", ops["projector"])

    npt.assert_allclose(ops["F"], F_expected)
    npt.assert_allclose(ops["decoder"], decoder_expected)
    npt.assert_allclose(ops["encoder"], encoder_expected)
    npt.assert_allclose(ops["projector"], projector_expected)


def test_polynomial_rom_evaluate_rom_rhs_matches_manual_contraction():
    _, _, rom, _, _, A = _polynomial_setup()
    z = np.array([0.3, -0.2])
    u = np.array([0.1, 0.0])

    rhs = rom.evaluate_rom_rhs(0.0, z, u, A)
    expected = u + A @ z

    print("\n[poly rhs] expected:", expected)
    print("[poly rhs] obtained   :", rhs)

    npt.assert_allclose(rhs, expected)


def test_polynomial_rom_evaluate_rom_adjoint_linear_case_matches_transpose_action():
    _, _, rom, _, _, A = _polynomial_setup()
    lam = np.array([0.4, -0.1])
    z_ref = np.array([0.2, 0.3])
    fq = lambda t: z_ref

    adj = rom.evaluate_rom_adjoint(0.0, lam, fq, A)

    print("\n[poly adjoint] expected:", A.T @ lam)
    print("[poly adjoint] obtained   :", adj)

    npt.assert_allclose(adj, A.T @ lam)


def test_polynomial_rom_integrate_zero_dynamics_keeps_state_constant():
    time = np.array([0.0, 0.1, 0.2])
    z0 = np.array([0.5, -0.25])
    Phi = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, -0.5],
    ])
    Psi = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    rom = classes.PolynomialROM(
        operators=[np.zeros((2, 2))],
        poly_comp=[1],
        Phi=Phi,
        Psi=Psi,
    )

    Z = rom.integrate(time, z0, forcing=np.zeros_like(z0))

    expected = np.column_stack([z0, z0, z0])
    print("\n[poly integrate] expected:\n", expected)
    print("[poly integrate] obtained:\n", Z)
    npt.assert_allclose(Z, expected)


def test_hamiltonian_rom_build_projection_operators_matches_internal_formula():
    _, _, rom, Phi, _ = _hamiltonian_setup()
    ops = rom.build_projection_operators()

    J = ops["J"]
    JU_expected = J @ Phi
    M_expected = JU_expected.T @ Phi
    decoder_expected = Phi @ np.linalg.inv(M_expected)
    encoder_expected = JU_expected.T
    projector_expected = decoder_expected @ encoder_expected

    print("\n[ham projection] expected JU:\n", JU_expected)
    print("[ham projection] obtained JU:\n", ops["JU"])
    print("[ham projection] expected M:\n", M_expected)
    print("[ham projection] obtained M:\n", ops["M"])
    print("[ham projection] expected decoder:\n", decoder_expected)
    print("[ham projection] obtained decoder:\n", ops["decoder"])
    print("[ham projection] expected encoder:\n", encoder_expected)
    print("[ham projection] obtained encoder:\n", ops["encoder"])
    print("[ham projection] expected projector:\n", projector_expected)
    print("[ham projection] obtained projector:\n", ops["projector"])

    npt.assert_allclose(ops["JU"], JU_expected)
    npt.assert_allclose(ops["M"], M_expected)
    npt.assert_allclose(ops["decoder"], decoder_expected)
    npt.assert_allclose(ops["encoder"], encoder_expected)
    npt.assert_allclose(ops["projector"], projector_expected)


def test_hamiltonian_rom_srnk_b6_zero_hamiltonian_keeps_state_constant():
    x0 = np.array([0.3, -0.4])
    time = np.array([0.0, 0.1, 0.2])
    rom = classes.HamiltonianROM(
        operators=[np.zeros((2, 2))],
        poly_comp=[1],
        Phi=np.eye(2),
    )

    X, H = rom.SRNK_b6(x0=x0, t0=time[0], tf=time[-1], h=0.01, t_eval=time)

    expected_X = np.column_stack([x0, x0, x0])
    expected_H = np.zeros(len(time))
    print("\n[srkn6 zero] expected X:\n", expected_X)
    print("[srkn6 zero] obtained X:\n", X)
    print("[srkn6 zero] expected H:", expected_H)
    print("[srkn6 zero] obtained H   :", H)
    npt.assert_allclose(X, expected_X)
    npt.assert_allclose(H, expected_H)


def test_hamiltonian_rom_srnk_b6_convergence_order():
    """
    Logic of the test:
    - use a quadratic Hamiltonian H(q, p) = 0.5 * (q^2 + p^2)
    - the exact solution is the harmonic oscillator:
      q(t) = cos(t), p(t) = -sin(t) for x0 = [1, 0]
    - integrate with decreasing step sizes and estimate the slope of
      log(error) vs log(h)
    """
    rom = classes.HamiltonianROM(
        operators=[0.5 * np.eye(2)],
        poly_comp=[1],
        Phi=np.eye(2),
    )

    x0 = np.array([1.0, 0.0])
    tf = 2.0
    hs = np.array([0.4, 0.2, 0.1, 0.05])
    errors = []

    for h in hs:
        n_steps = int(round(tf / h))
        time = np.linspace(0.0, tf, n_steps + 1)
        X, _ = rom.SRNK_b6(x0=x0, t0=0.0, tf=tf, h=h, t_eval=time)
        x_exact = np.array([np.cos(tf), -np.sin(tf)])
        errors.append(np.linalg.norm(X[:, -1] - x_exact))

    errors = np.asarray(errors)
    slope, _ = np.polyfit(np.log(hs), np.log(errors), 1)

    print("\n[srkn6 convergence] hs:", hs)
    print("[srkn6 convergence] errors:", errors)
    print("[srkn6 convergence] expected order > 4")
    print("[srkn6 convergence] obtained order   :", slope)

    assert np.all(errors > 0.0)
    assert slope > 4.0


def test_polynomial_reduction_objective_cost_and_gradient_are_finite():
    pool, opt_obj, rom, _, _, _ = _polynomial_setup()
    fom = _IdentityFOM()
    objective = classes.PolynomialReductionObjective(rom, fom, opt_obj, pool)

    cost = objective.trajectory_cost(0)
    grad = objective.trajectory_gradient(0)

    print("\n[poly objective] expected finite cost and finite gradient blocks")
    print("[poly objective] obtained cost:", cost)
    print("[poly objective] obtained grad_Phi shape:", grad["grad_Phi"].shape)
    print("[poly objective] obtained grad_Psi shape:", grad["grad_Psi"].shape)
    print("[poly objective] obtained number of tensor gradients:", len(grad["grad_tensors"]))

    assert np.isfinite(cost)
    assert grad["grad_Phi"].shape == rom.Phi.shape
    assert grad["grad_Psi"].shape == rom.Psi.shape
    assert len(grad["grad_tensors"]) == len(rom.operators)
    assert np.all(np.isfinite(grad["grad_Phi"]))
    assert np.all(np.isfinite(grad["grad_Psi"]))
    assert np.all(np.isfinite(grad["grad_tensors"][0]))


def test_polynomial_rom_build_projection_operators_with_non_square_bases():
    _, _, rom, Phi, Psi, _ = _polynomial_tall_setup()
    ops = rom.build_projection_operators()

    F_expected = np.linalg.inv(Psi.T @ Phi)
    decoder_expected = Phi @ F_expected
    encoder_expected = Psi.T
    projector_expected = decoder_expected @ encoder_expected

    print("\n[poly tall projection] expected decoder shape:", decoder_expected.shape)
    print("[poly tall projection] obtained decoder shape:", ops["decoder"].shape)
    print("[poly tall projection] expected projector shape:", projector_expected.shape)
    print("[poly tall projection] obtained projector shape:", ops["projector"].shape)

    npt.assert_allclose(ops["F"], F_expected)
    npt.assert_allclose(ops["decoder"], decoder_expected)
    npt.assert_allclose(ops["encoder"], encoder_expected)
    npt.assert_allclose(ops["projector"], projector_expected)


def test_polynomial_reduction_objective_is_finite_with_non_square_bases():
    pool, opt_obj, rom, _, _, _ = _polynomial_tall_setup()
    fom = _IdentityFOM()
    objective = classes.PolynomialReductionObjective(rom, fom, opt_obj, pool)

    cost = objective.trajectory_cost(0)
    grad = objective.trajectory_gradient(0)

    print("\n[poly tall objective] expected finite cost and finite gradient blocks")
    print("[poly tall objective] obtained cost:", cost)
    print("[poly tall objective] obtained grad_Phi shape:", grad["grad_Phi"].shape)
    print("[poly tall objective] obtained grad_Psi shape:", grad["grad_Psi"].shape)

    assert np.isfinite(cost)
    assert grad["grad_Phi"].shape == rom.Phi.shape
    assert grad["grad_Psi"].shape == rom.Psi.shape
    assert np.all(np.isfinite(grad["grad_Phi"]))
    assert np.all(np.isfinite(grad["grad_Psi"]))


def test_hamiltonian_rom_build_projection_operators_with_non_square_basis():
    _, _, rom, Phi, _ = _hamiltonian_tall_setup()
    ops = rom.build_projection_operators()

    print("\n[ham tall projection] expected Phi shape:", Phi.shape)
    print("[ham tall projection] obtained decoder shape:", ops["decoder"].shape)
    print("[ham tall projection] obtained encoder shape:", ops["encoder"].shape)
    print("[ham tall projection] obtained projector shape:", ops["projector"].shape)
    print("[ham tall projection] obtained Jhat shape:", ops["Jhat"].shape)

    assert ops["decoder"].shape == Phi.shape
    assert ops["encoder"].shape == (Phi.shape[1], Phi.shape[0])
    assert ops["projector"].shape == (Phi.shape[0], Phi.shape[0])
    assert ops["Jhat"].shape == (Phi.shape[1], Phi.shape[1])


def test_hamiltonian_reduction_objective_is_finite_with_non_square_basis():
    pool, opt_obj, rom, _, _ = _hamiltonian_tall_setup()
    fom = _IdentityFOM()
    objective = classes.HamiltonianReductionObjective(rom, fom, opt_obj, pool)

    cost = objective.trajectory_cost(0)
    grad = objective.trajectory_gradient(0)

    print("\n[ham tall objective] expected finite cost:", True)
    print("[ham tall objective] obtained cost:", cost)
    print("[ham tall objective] expected grad_Phi shape:", rom.Phi.shape)
    print("[ham tall objective] obtained grad_Phi shape:", grad["grad_Phi"].shape)
    print("[ham tall objective] obtained grad_Psi shape:", grad["grad_Psi"].shape)

    assert np.isfinite(cost)
    assert grad["grad_Phi"].shape == rom.Phi.shape
    assert grad["grad_Psi"].shape == rom.Phi.shape
    assert np.all(np.isfinite(grad["grad_Phi"]))
    assert np.all(np.isfinite(grad["grad_Psi"]))


def test_hamiltonian_reduction_objective_cost_and_gradient_are_finite():
    pool, opt_obj, rom, _, _ = _hamiltonian_setup()
    fom = _IdentityFOM()
    objective = classes.HamiltonianReductionObjective(rom, fom, opt_obj, pool)

    cost = objective.trajectory_cost(0)
    grad = objective.trajectory_gradient(0)

    print("\n[ham objective] expected finite cost and finite gradient blocks")
    print("[ham objective] obtained cost:", cost)
    print("[ham objective] obtained grad_Phi shape:", grad["grad_Phi"].shape)
    print("[ham objective] obtained grad_Psi shape:", grad["grad_Psi"].shape)
    print("[ham objective] obtained number of tensor gradients:", len(grad["grad_tensors"]))

    assert np.isfinite(cost)
    assert grad["grad_Phi"].shape == rom.Phi.shape
    assert grad["grad_Psi"].shape == rom.Phi.shape
    assert len(grad["grad_tensors"]) == len(rom.operators)
    assert np.all(np.isfinite(grad["grad_Phi"]))
    assert np.all(np.isfinite(grad["grad_Psi"]))
    assert np.all(np.isfinite(grad["grad_tensors"][0]))
