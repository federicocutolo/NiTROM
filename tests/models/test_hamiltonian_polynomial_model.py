"""Tests for the HamiltonianPolynomialModel class."""

import numpy as np
import pytest
import torch

from nitrom.latent_space_models import HamiltonianPolynomialModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 6  # full-order dimension (even)
R = 2  # reduced dimension
M = 4  # batch size


def _canonical_J(n):
    half = n // 2
    J = np.block([
        [np.zeros((half, half)), np.eye(half)],
        [-np.eye(half), np.zeros((half, half))],
    ])
    return torch.tensor(J, dtype=torch.float64)


def _orthonormal_basis(n, r, seed=1):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, r)))
    return torch.tensor(Q, dtype=torch.float64)


def _sym2(A2):
    return 0.5 * (A2 + A2.T)


def _sym3(A3):
    """Move each slot of A3 to the front and average."""
    return (A3 + A3.permute(1, 0, 2) + A3.permute(2, 0, 1)) / 3.0


def _reference_rhs(model, z):
    """Hand-built Jhat @ grad H for poly_comp subsets of [1, 2]."""
    Jhat = model.Phi.T @ model._applyJ(model.Phi)
    g = torch.zeros_like(z)
    if hasattr(model, "A2"):
        g = g + _sym2(model.A2) @ z
    if hasattr(model, "A3"):
        g = g + torch.einsum("ljk,j,k->l", _sym3(model.A3), z, z)
    return Jhat @ g


def _make_model(poly_comp, seed=42, J=None, Phi=None):
    rng = np.random.default_rng(seed)
    if J is None:
        J = _canonical_J(N)
    if Phi is None:
        Phi = _orthonormal_basis(N, R, seed=seed + 1)
    coeffs = [
        torch.tensor(rng.standard_normal((R,) * (k + 1)), dtype=torch.float64)
        for k in poly_comp
    ]
    return HamiltonianPolynomialModel(R, poly_comp, J, Phi, ham_params=coeffs)


@pytest.fixture(params=[[1], [2], [1, 2]], ids=["deg1", "deg2", "deg1+2"])
def model(request):
    return _make_model(request.param)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

class TestBasic:

    def test_param_names_order(self):
        m = _make_model([1, 2])
        assert m.param_names == ["A2", "A3", "Phi"]

    def test_param_names_deg1(self):
        m = _make_model([1])
        assert m.param_names == ["A2", "Phi"]

    def test_odd_dimension_raises(self):
        J = torch.zeros((5, 5), dtype=torch.float64)
        Phi = torch.zeros((5, R), dtype=torch.float64)
        with pytest.raises(ValueError, match="even"):
            HamiltonianPolynomialModel(R, [1], J, Phi)

    def test_nonsquare_J_raises(self):
        J = torch.zeros((N, N + 2), dtype=torch.float64)
        Phi = torch.zeros((N, R), dtype=torch.float64)
        with pytest.raises(ValueError, match="square"):
            HamiltonianPolynomialModel(R, [1], J, Phi)

    def test_nonskew_J_raises(self):
        J = torch.eye(N, dtype=torch.float64)
        Phi = _orthonormal_basis(N, R)
        with pytest.raises(ValueError, match="skew"):
            HamiltonianPolynomialModel(R, [1], J, Phi)

    def test_skew_but_noncanonical_J_raises(self):
        """J must be canonical-type: J = -J^T alone is not enough, it must
        also satisfy J @ J = -I (i.e. J^{-1} = -J, J^{-T} = J)."""
        J = 2.0 * _canonical_J(N)  # skew, but J @ J = -4 I
        Phi = _orthonormal_basis(N, R)
        with pytest.raises(ValueError, match="J @ J = -I"):
            HamiltonianPolynomialModel(R, [1], J, Phi)

    def test_J_canonical_identities(self):
        """The accepted J satisfies J = -J^T = J^{-T} and J = -J^{-1}."""
        J = _canonical_J(N)
        eye = torch.eye(N, dtype=torch.float64)
        np.testing.assert_allclose((-J.T).numpy(), J.numpy(), atol=1e-14)
        np.testing.assert_allclose((J @ J).numpy(), -eye.numpy(), atol=1e-14)
        np.testing.assert_allclose(
            torch.linalg.inv(J).numpy(), (-J).numpy(), atol=1e-14
        )
        np.testing.assert_allclose(
            torch.linalg.inv(J.T).numpy(), J.numpy(), atol=1e-14
        )

    def test_wrong_Phi_shape_raises(self):
        J = _canonical_J(N)
        Phi = torch.zeros((N + 2, R), dtype=torch.float64)
        with pytest.raises(ValueError, match="Phi"):
            HamiltonianPolynomialModel(R, [1], J, Phi)

    def test_degree_zero_raises(self):
        J = _canonical_J(N)
        Phi = _orthonormal_basis(N, R)
        with pytest.raises(ValueError, match="degrees >= 1"):
            HamiltonianPolynomialModel(R, [0, 1], J, Phi)

    def test_wrong_ham_params_length_raises(self):
        J = _canonical_J(N)
        Phi = _orthonormal_basis(N, R)
        A2 = torch.zeros((R, R), dtype=torch.float64)
        with pytest.raises(ValueError, match="ham_params"):
            HamiltonianPolynomialModel(R, [1, 2], J, Phi, ham_params=[A2])

    def test_no_forcing(self, model):
        """The model is autonomous: no B parameter, no forcing flag."""
        assert model.forcing_exists is False
        assert "B" not in model.param_names

    def test_jhat_matches_definition(self, model):
        Jhat_ref = model.Phi.T @ _canonical_J(N) @ model.Phi
        np.testing.assert_allclose(
            model.Jhat.numpy(), Jhat_ref.numpy(), rtol=1e-12, atol=1e-15
        )

    def test_jhat_skew(self, model):
        np.testing.assert_allclose(
            model.Jhat.numpy(), -model.Jhat.T.numpy(), atol=1e-13
        )

    def test_jhat_noncanonical(self, model):
        """Jhat = Phi^T J Phi is only skew: being a *non-canonical*
        symplectic structure, it does NOT inherit Jhat @ Jhat = -I from J
        for a generic basis Phi."""
        eye = torch.eye(R, dtype=torch.float64)
        assert not torch.allclose(model.Jhat @ model.Jhat, -eye, atol=1e-6)


# ---------------------------------------------------------------------------
# RHS and Hamiltonian evaluation
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_rhs_unbatched(self, model):
        rng = np.random.default_rng(10)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        rhs = model.evaluate_rhs(0.0, z)
        np.testing.assert_allclose(
            rhs.numpy(), _reference_rhs(model, z).numpy(), rtol=1e-12
        )

    def test_rhs_batched(self, model):
        rng = np.random.default_rng(11)
        Z = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
        rhs = model.evaluate_rhs(0.0, Z)
        assert rhs.shape == (M, R)
        for j in range(M):
            np.testing.assert_allclose(
                rhs[j].numpy(), _reference_rhs(model, Z[j]).numpy(), rtol=1e-11
            )

    def test_compute_hamiltonian(self):
        m = _make_model([1, 2])
        rng = np.random.default_rng(12)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        H_ref = (
            0.5 * z @ m.A2 @ z
            + torch.einsum("ijk,i,j,k", m.A3, z, z, z) / 3.0
        )
        np.testing.assert_allclose(
            float(m.compute_hamiltonian(z)), float(H_ref), rtol=1e-12
        )

    def test_compute_hamiltonian_batched(self):
        m = _make_model([1, 2])
        rng = np.random.default_rng(13)
        Z = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
        H = m.compute_hamiltonian(Z)
        assert H.shape == (M,)
        for j in range(M):
            np.testing.assert_allclose(
                float(H[j]), float(m.compute_hamiltonian(Z[j])), rtol=1e-12
            )

    def test_hamiltonian_gradient_fd(self, model):
        rng = np.random.default_rng(14)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        dz = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        g = model.compute_hamiltonian_gradient(z)
        eps = 1e-7
        fd = (
            float(model.compute_hamiltonian(z + eps * dz))
            - float(model.compute_hamiltonian(z - eps * dz))
        ) / (2 * eps)
        np.testing.assert_allclose(float(torch.dot(g, dz)), fd, rtol=1e-6)

    def test_rhs_is_tangent_to_H_levels(self, model):
        """grad H . (Jhat grad H) = 0: the flow conserves H pointwise."""
        rng = np.random.default_rng(15)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        g = model.compute_hamiltonian_gradient(z)
        rhs = model.evaluate_rhs(0.0, z)
        np.testing.assert_allclose(float(torch.dot(g, rhs)), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# update_params
# ---------------------------------------------------------------------------

class TestUpdateParams:

    def test_new_phi_changes_jhat_and_rhs(self):
        m = _make_model([1, 2])
        rng = np.random.default_rng(20)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        rhs_before = m.evaluate_rhs(0.0, z).clone()
        Jhat_before = m.Jhat.clone()

        Phi_new = _orthonormal_basis(N, R, seed=99)
        m.update_params([m.A2, m.A3, Phi_new])

        Jhat_ref = Phi_new.T @ _canonical_J(N) @ Phi_new
        np.testing.assert_allclose(
            m.Jhat.numpy(), Jhat_ref.numpy(), rtol=1e-12, atol=1e-15
        )
        assert not np.allclose(m.Jhat.numpy(), Jhat_before.numpy())

        rhs_after = m.evaluate_rhs(0.0, z)
        np.testing.assert_allclose(
            rhs_after.numpy(), _reference_rhs(m, z).numpy(), rtol=1e-12
        )
        assert not np.allclose(rhs_before.numpy(), rhs_after.numpy())

    def test_new_coeffs_change_rhs(self):
        m = _make_model([1])
        rng = np.random.default_rng(21)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        A2_new = torch.tensor(rng.standard_normal((R, R)), dtype=torch.float64)
        m.update_params([A2_new, m.Phi])
        np.testing.assert_allclose(
            m.evaluate_rhs(0.0, z).numpy(), _reference_rhs(m, z).numpy(),
            rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# VJP via finite differences (including the cubic-only case that the old
# NiTROM implementation got wrong)
# ---------------------------------------------------------------------------

class TestVjpEvaluateRhs:

    @pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
    def test_finite_difference_all_params(self, model, batched):
        rng = np.random.default_rng(30)
        if batched:
            z = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
            v = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
        else:
            z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
            v = torch.tensor(rng.standard_normal(R), dtype=torch.float64)

        grads = model.vjp_evaluate_rhs(z, v)
        assert len(grads) == len(model.param_names)

        eps = 1e-7
        for idx, name in enumerate(model.param_names):
            params = model.get_params()
            assert grads[idx].shape == params[idx].shape
            dP = torch.tensor(
                rng.standard_normal(tuple(params[idx].shape)),
                dtype=torch.float64,
            )
            dd_vjp = torch.sum(grads[idx] * dP).item()

            plus = [t.clone() for t in params]
            minus = [t.clone() for t in params]
            plus[idx] = params[idx] + eps * dP
            minus[idx] = params[idx] - eps * dP

            model.update_params(plus)
            J_plus = torch.sum(v * model.evaluate_rhs(0.0, z)).item()
            model.update_params(minus)
            J_minus = torch.sum(v * model.evaluate_rhs(0.0, z)).item()
            model.update_params(params)

            dd_fd = (J_plus - J_minus) / (2 * eps)
            np.testing.assert_allclose(
                dd_vjp, dd_fd, rtol=1e-4,
                err_msg=f"FD mismatch for parameter {name!r}",
            )

    def test_batched_sums_per_record(self, model):
        """Batched VJP equals the sum of per-record VJPs."""
        rng = np.random.default_rng(31)
        Z = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
        V = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)

        grads_batch = model.vjp_evaluate_rhs(Z, V)
        for idx in range(len(model.param_names)):
            total = torch.zeros_like(grads_batch[idx])
            for j in range(M):
                total = total + model.vjp_evaluate_rhs(Z[j], V[j])[idx]
            np.testing.assert_allclose(
                grads_batch[idx].numpy(), total.numpy(), rtol=1e-10
            )

    def test_inner_batched_matches_per_record(self, model):
        rng = np.random.default_rng(32)
        Z = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
        V = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)

        grads_bulk = model.inner_batched_vjp_evaluate_rhs(Z, V)
        n_inner = len(model.inner_params())
        totals = [None] * n_inner
        for j in range(M):
            g_j = model.inner_vjp_evaluate_rhs(Z[j], V[j])
            for idx in range(n_inner):
                totals[idx] = g_j[idx] if totals[idx] is None else totals[idx] + g_j[idx]
        for idx in range(n_inner):
            np.testing.assert_allclose(
                grads_bulk[idx].numpy(), totals[idx].numpy(), rtol=1e-10
            )


# ---------------------------------------------------------------------------
# VJP against torch autograd (independent reimplementation of the forward)
# ---------------------------------------------------------------------------

class TestVjpAgainstAutograd:

    @pytest.mark.parametrize("poly_comp", [[1], [2], [1, 2]],
                             ids=["deg1", "deg2", "deg1+2"])
    @pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
    def test_all_gradients(self, poly_comp, batched):
        m = _make_model(poly_comp, seed=50)
        J = _canonical_J(N)
        rng = np.random.default_rng(51)
        if batched:
            z = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
            v = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
        else:
            z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
            v = torch.tensor(rng.standard_normal(R), dtype=torch.float64)

        grads = m.vjp_evaluate_rhs(z, v)

        # Reimplement the forward pass in torch with requires_grad leaves.
        leaves = [p.clone().requires_grad_(True) for p in m.get_params()]
        by_name = dict(zip(m.param_names, leaves, strict=True))
        Phi = by_name["Phi"]
        Jhat = Phi.T @ J @ Phi

        g = torch.zeros_like(z)
        if "A2" in by_name:
            A2 = by_name["A2"]
            g = g + z @ (0.5 * (A2 + A2.T)).T
        if "A3" in by_name:
            A3 = by_name["A3"]
            S3 = (A3 + A3.permute(1, 0, 2) + A3.permute(2, 0, 1)) / 3.0
            g = g + torch.einsum("ljk,...j,...k->...l", S3, z, z)
        rhs = g @ Jhat.T

        loss = torch.sum(v * rhs)
        loss.backward()

        for name, leaf, grad in zip(m.param_names, leaves, grads, strict=True):
            np.testing.assert_allclose(
                grad.numpy(), leaf.grad.numpy(), rtol=1e-10, atol=1e-12,
                err_msg=f"Autograd mismatch for parameter {name!r}",
            )


# ---------------------------------------------------------------------------
# project_inner_gradients in isolation (pins the slot-permutation adjoints)
# ---------------------------------------------------------------------------

class TestProjectInnerGradients:

    def test_against_autograd(self):
        """Feed random inner-gradient seeds G_k and compare the projected
        gradients with autograd of sum_k <G_k, T_k(A, Phi)>."""
        m = _make_model([1, 2], seed=60)
        J = _canonical_J(N)
        rng = np.random.default_rng(61)
        G1 = torch.tensor(rng.standard_normal((R, R)), dtype=torch.float64)
        G2 = torch.tensor(rng.standard_normal((R, R, R)), dtype=torch.float64)

        grads = m.project_inner_gradients([G1, G2])

        A2 = m.A2.clone().requires_grad_(True)
        A3 = m.A3.clone().requires_grad_(True)
        Phi = m.Phi.clone().requires_grad_(True)
        Jhat = Phi.T @ J @ Phi
        T1 = Jhat @ (0.5 * (A2 + A2.T))
        S3 = (A3 + A3.permute(1, 0, 2) + A3.permute(2, 0, 1)) / 3.0
        T2 = torch.einsum("az,zjk->ajk", Jhat, S3)
        loss = torch.sum(G1 * T1) + torch.sum(G2 * T2)
        loss.backward()

        for grad, leaf, name in zip(
            grads, [A2, A3, Phi], ["A2", "A3", "Phi"], strict=True
        ):
            np.testing.assert_allclose(
                grad.numpy(), leaf.grad.numpy(), rtol=1e-10, atol=1e-12,
                err_msg=f"project_inner_gradients mismatch for {name!r}",
            )


# ---------------------------------------------------------------------------
# Energy conservation along integrated trajectories
# ---------------------------------------------------------------------------

class TestEnergyConservation:

    def test_H_conserved_along_midpoint_trajectory(self):
        from nitrom.time_steppers.time_stepper import evolve

        m = _make_model([1, 2], seed=70)
        rng = np.random.default_rng(71)
        z = torch.tensor(0.1 * rng.standard_normal(R), dtype=torch.float64)
        H0 = float(m.compute_hamiltonian(z))

        dt = 0.01
        drifts = []
        for _ in range(200):
            z = evolve(
                m.evaluate_rhs, 0.0, z, dt, method="implicit_midpoint",
                newton_tol=1e-13, newton_max_iter=50,
            )
            drifts.append(abs(float(m.compute_hamiltonian(z)) - H0))
        # Bounded near-conservation (exact up to O(dt^2) modified-energy
        # oscillation for the nonlinear H).
        assert max(drifts) < 1e-6

    def test_H_conserved_linear_fast_path(self):
        from nitrom.time_steppers.linear_symplectic import (
            linear_symplectic_integrate,
        )

        m = _make_model([1], seed=72)
        A = m.Jhat @ (0.5 * (m.A2 + m.A2.T))  # inner tensor = full RHS matrix
        rng = np.random.default_rng(73)
        z0 = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        time = torch.arange(0, 501, dtype=torch.float64) * 0.05
        Z = linear_symplectic_integrate(A, z0, time)
        H = m.compute_hamiltonian(Z.T)
        drift = (H - H[0]).abs().max().item()
        assert drift <= 1e-10 * max(abs(float(H[0])), 1.0)
