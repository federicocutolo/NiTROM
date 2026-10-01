"""Tests for the linear symplectic (Cayley) fast path."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from nitrom.backend import set_backend
from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.time_steppers.linear_symplectic import (
    cayley_step,
    interval_maps,
    linear_propagator,
    linear_symplectic_integrate,
    linear_symplectic_solve,
    yoshida4_propagator,
)
from nitrom.time_steppers.time_stepper import evolve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 4  # state dimension (even: canonical J needs pairs)
B = 3  # batch size


def _canonical_J_np(n):
    half = n // 2
    return np.block([
        [np.zeros((half, half)), np.eye(half)],
        [-np.eye(half), np.zeros((half, half))],
    ])


def _hamiltonian_A(seed=0):
    """A = J S with S symmetric: linear Hamiltonian dynamics (torch)."""
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((N, N))
    S = S + S.T
    A_np = _canonical_J_np(N) @ S
    return torch.tensor(A_np, dtype=torch.float64), S


@pytest.fixture
def A_torch():
    return _hamiltonian_A()[0]


@pytest.fixture
def z0():
    rng = np.random.default_rng(10)
    return torch.tensor(rng.standard_normal(N), dtype=torch.float64)


@pytest.fixture
def z0_batched():
    rng = np.random.default_rng(11)
    return torch.tensor(rng.standard_normal((B, N)), dtype=torch.float64)


# ---------------------------------------------------------------------------
# Input validation and shapes
# ---------------------------------------------------------------------------

class TestBasic:

    def test_time_must_be_1d(self, A_torch, z0):
        time = torch.zeros((2, 2), dtype=torch.float64)
        with pytest.raises(ValueError, match="one-dimensional"):
            linear_symplectic_integrate(A_torch, z0, time)

    def test_time_needs_two_points(self, A_torch, z0):
        time = torch.tensor([0.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="two time points"):
            linear_symplectic_integrate(A_torch, z0, time)

    def test_time_strictly_increasing(self, A_torch, z0):
        time = torch.tensor([0.0, 0.2, 0.1], dtype=torch.float64)
        with pytest.raises(ValueError, match="strictly increasing"):
            linear_symplectic_integrate(A_torch, z0, time)

    def test_unbatched_shape(self, A_torch, z0):
        time = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
        Z = linear_symplectic_integrate(A_torch, z0, time)
        assert Z.shape == (N, 11)
        np.testing.assert_allclose(Z[:, 0].numpy(), z0.numpy())

    def test_batched_shape(self, A_torch, z0_batched):
        time = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
        Z = linear_symplectic_integrate(A_torch, z0_batched, time)
        assert Z.shape == (B, N, 11)
        np.testing.assert_allclose(Z[:, :, 0].numpy(), z0_batched.numpy())

    def test_batched_matches_unbatched(self, A_torch, z0_batched):
        time = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
        Z = linear_symplectic_integrate(A_torch, z0_batched, time)
        for b in range(B):
            Zb = linear_symplectic_integrate(A_torch, z0_batched[b], time)
            np.testing.assert_allclose(Z[b].numpy(), Zb.numpy(), rtol=1e-12)

    def test_solve_on_knots_matches_integrate(self, A_torch, z0):
        t_eval = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
        dt = 0.02  # commensurate: every t_eval lands on a knot
        Z_eval, _fine_time, Z_fine = linear_symplectic_solve(
            A_torch, z0, t_eval, dt=dt, return_internal=True
        )
        assert Z_eval.shape == (N, 6)
        # On-knot values must equal the marched states exactly.
        for j, te in enumerate(t_eval):
            k = round(float(te) / dt)
            np.testing.assert_allclose(
                Z_eval[:, j].numpy(), Z_fine[:, k].numpy(), rtol=0, atol=0
            )

    def test_solve_off_knots_interpolates(self, A_torch, z0):
        import scipy.linalg as sla

        t_eval = torch.tensor([0.0, 0.333, 0.777, 1.0], dtype=torch.float64)
        dt = 0.01
        Z_eval = linear_symplectic_solve(A_torch, z0, t_eval, dt=dt)
        # Compare against the exact solution (interp + integration error
        # are both O(dt^3) or better on the fine grid).
        Z_exact = np.stack(
            [sla.expm(A_torch.numpy() * float(t)) @ z0.numpy() for t in t_eval],
            axis=1,
        )
        np.testing.assert_allclose(
            Z_eval.numpy(), Z_exact, rtol=1e-4, atol=1e-6
        )

    def test_solve_dt_none_integrates_directly(self, A_torch, z0):
        t_eval = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
        Z_eval = linear_symplectic_solve(A_torch, z0, t_eval)
        Z_direct = linear_symplectic_integrate(A_torch, z0, t_eval)
        np.testing.assert_allclose(Z_eval.numpy(), Z_direct.numpy(), rtol=1e-14)


# ---------------------------------------------------------------------------
# Symplecticity: M^T J M = J
# ---------------------------------------------------------------------------

class TestSymplecticity:

    @pytest.mark.parametrize("h", [0.1, 0.01])
    def test_cayley_step(self, A_torch, h):
        J = torch.tensor(_canonical_J_np(N), dtype=torch.float64)
        M = cayley_step(A_torch, h)
        np.testing.assert_allclose(
            (M.T @ J @ M).numpy(), J.numpy(), atol=1e-13
        )

    @pytest.mark.parametrize("h", [0.1, 0.01])
    def test_yoshida4_propagator(self, A_torch, h):
        J = torch.tensor(_canonical_J_np(N), dtype=torch.float64)
        M = yoshida4_propagator(A_torch, h)
        np.testing.assert_allclose(
            (M.T @ J @ M).numpy(), J.numpy(), atol=1e-13
        )

    def test_linear_propagator(self, A_torch):
        J = torch.tensor(_canonical_J_np(N), dtype=torch.float64)
        P = linear_propagator(A_torch, 2.35, dt=0.1)
        np.testing.assert_allclose(
            (P.T @ J @ P).numpy(), J.numpy(), atol=1e-12
        )

    def test_propagator_composition(self, A_torch):
        """P(tau1 + tau2) built with the same dt equals P(tau2) @ P(tau1)."""
        dt = 0.05
        tau1, tau2 = 0.4, 0.75
        P1 = linear_propagator(A_torch, tau1, dt)
        P2 = linear_propagator(A_torch, tau2, dt)
        P12 = linear_propagator(A_torch, tau1 + tau2, dt)
        np.testing.assert_allclose(
            P12.numpy(), (P2 @ P1).numpy(), rtol=1e-10, atol=1e-12
        )

    def test_convergence_order_4(self, A_torch, z0):
        """The compounded propagator converges at 4th order to expm."""
        import scipy.linalg as sla

        T = 1.0
        exact = sla.expm(A_torch.numpy() * T) @ z0.numpy()
        errs = []
        for nsteps in (8, 16, 32, 64):
            P = linear_propagator(A_torch, T, dt=T / nsteps)
            errs.append(np.linalg.norm(P.numpy() @ z0.numpy() - exact))
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
        np.testing.assert_allclose(orders[-1], 4.0, rtol=5e-2)


# ---------------------------------------------------------------------------
# Agreement with the general tableau path
# ---------------------------------------------------------------------------

class TestAgreementWithTableau:

    def _march_tableau(self, model, z0, dt, nsteps, method):
        z = z0.clone()
        t = 0.0
        for _ in range(nsteps):
            z = evolve(
                model.evaluate_rhs, t, z, dt, method=method,
                newton_tol=1e-14, newton_max_iter=50,
            )
            t += dt
        return z

    @pytest.mark.parametrize("method,builder", [
        ("implicit_midpoint", cayley_step),
        ("yoshida4", yoshida4_propagator),
    ])
    def test_linear_model(self, A_torch, z0, method, builder):
        """On z' = A z the tableau march equals the Cayley march to Newton
        tolerance."""
        model = PolynomialModel(N, [1], tensors=[A_torch])
        dt = 0.05
        nsteps = 20

        z_tab = self._march_tableau(model, z0, dt, nsteps, method)

        M = builder(A_torch, dt)
        z_cay = z0.clone()
        for _ in range(nsteps):
            z_cay = M @ z_cay

        np.testing.assert_allclose(
            z_tab.detach().numpy(), z_cay.numpy(), rtol=1e-10, atol=1e-12
        )


# ---------------------------------------------------------------------------
# Adjoint maps
# ---------------------------------------------------------------------------

class TestAdjointMaps:

    def test_interval_maps_consistency(self, A_torch):
        """M_gap = Q_i @ N_i for every node."""
        gap = 0.8
        offsets = [0.2, 0.5, 0.7]
        M_gap, N_list, Q_list = interval_maps(A_torch, gap, offsets, dt=0.05)
        for N_i, Q_i in zip(N_list, Q_list, strict=True):
            np.testing.assert_allclose(
                (Q_i @ N_i).numpy(), M_gap.numpy(), rtol=1e-10, atol=1e-12
            )

    def test_transposed_propagator_is_discrete_adjoint(self, A_torch):
        """Marching lambda' = -A^T lambda backward over the gap with the same
        Cayley composition equals applying M_gap^T."""
        gap = 0.6
        dt = 0.05
        M_gap, _, _ = interval_maps(A_torch, gap, [], dt=dt)

        rng = np.random.default_rng(20)
        lam_T = torch.tensor(rng.standard_normal(N), dtype=torch.float64)

        # Backward march of the adjoint ODE: stepping tau -> tau + dt in the
        # reversed time variable applies the Cayley step of +A^T.
        M_adj = linear_propagator(A_torch.T, gap, dt)
        lam_0_march = M_adj @ lam_T
        lam_0_transpose = M_gap.T @ lam_T

        np.testing.assert_allclose(
            lam_0_march.numpy(), lam_0_transpose.numpy(), rtol=1e-10, atol=1e-12
        )

    def test_gradient_pairing(self, A_torch, z0):
        """<lambda(t+gap), M_gap z(t)> == <M_gap^T lambda(t+gap), z(t)>."""
        gap = 0.9
        M_gap, _, _ = interval_maps(A_torch, gap, [], dt=0.05)
        rng = np.random.default_rng(21)
        lam = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
        lhs = torch.dot(lam, M_gap @ z0).item()
        rhs = torch.dot(M_gap.T @ lam, z0).item()
        np.testing.assert_allclose(lhs, rhs, rtol=1e-13)


# ---------------------------------------------------------------------------
# Sparse path (numpy backend only)
# ---------------------------------------------------------------------------

class TestSparse:

    def test_sparse_matches_dense(self):
        set_backend("numpy")
        try:
            rng = np.random.default_rng(30)
            S = rng.standard_normal((N, N))
            S = S + S.T
            A_np = _canonical_J_np(N) @ S
            A_sp = sp.csr_matrix(A_np)
            z0 = rng.standard_normal(N)
            time = np.linspace(0.0, 1.0, 21)

            Z_dense = linear_symplectic_integrate(A_np, z0, time)
            Z_sparse = linear_symplectic_integrate(A_sp, z0, time)
            np.testing.assert_allclose(Z_sparse, Z_dense, rtol=1e-10, atol=1e-12)
        finally:
            set_backend("torch")

    def test_sparse_batched_matches_dense(self):
        set_backend("numpy")
        try:
            rng = np.random.default_rng(31)
            S = rng.standard_normal((N, N))
            S = S + S.T
            A_np = _canonical_J_np(N) @ S
            A_sp = sp.csr_matrix(A_np)
            z0 = rng.standard_normal((B, N))
            time = np.linspace(0.0, 1.0, 11)

            Z_dense = linear_symplectic_integrate(A_np, z0, time)
            Z_sparse = linear_symplectic_integrate(A_sp, z0, time)
            np.testing.assert_allclose(Z_sparse, Z_dense, rtol=1e-10, atol=1e-12)
        finally:
            set_backend("torch")

    def test_sparse_on_torch_raises(self, z0):
        rng = np.random.default_rng(32)
        S = rng.standard_normal((N, N))
        S = S + S.T
        A_sp = sp.csr_matrix(_canonical_J_np(N) @ S)
        time = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
        with pytest.raises(TypeError, match="numpy backend"):
            linear_symplectic_integrate(A_sp, z0, time)


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------

class TestEnergyConservation:

    def test_quadratic_H_conserved_exactly(self):
        """Implicit midpoint (hence its Yoshida composition) conserves
        quadratic invariants: H = 1/2 z^T S z stays constant to round-off
        over a long horizon.  S is positive definite so the flow is a
        bounded oscillation (an indefinite S gives exponential growth and
        round-off swamps the invariant)."""
        rng = np.random.default_rng(40)
        R = rng.standard_normal((N, N))
        S = R.T @ R + np.eye(N)  # SPD
        A = torch.tensor(_canonical_J_np(N) @ S, dtype=torch.float64)
        S_t = torch.tensor(S, dtype=torch.float64)
        rng = np.random.default_rng(41)
        z0 = torch.tensor(rng.standard_normal(N), dtype=torch.float64)

        time = torch.arange(0, 2001, dtype=torch.float64) * 0.1  # T = 200
        Z = linear_symplectic_integrate(A, z0, time)

        H = 0.5 * torch.einsum("it,ij,jt->t", Z, S_t, Z)
        drift = (H - H[0]).abs().max().item()
        assert drift <= 1e-10 * max(abs(H[0].item()), 1.0)

    def test_nonlinear_H_bounded_drift(self):
        """For a cubic Hamiltonian the tableau midpoint/yoshida4 show bounded
        (non-secular) energy oscillation, unlike rk2 which drifts
        monotonically."""
        rng = np.random.default_rng(42)
        J_np = _canonical_J_np(N)

        # H(z) = 1/2 z^T S z + 1/3 C(z,z,z) with SPD S and a small fully
        # symmetric C (weak nonlinearity keeps the orbit in the bounded
        # well of the potential); RHS = J grad H.
        R = rng.standard_normal((N, N))
        S = R.T @ R + np.eye(N)  # SPD
        C = 0.01 * rng.standard_normal((N, N, N))
        C = (
            C
            + C.transpose(0, 2, 1)
            + C.transpose(1, 0, 2)
            + C.transpose(1, 2, 0)
            + C.transpose(2, 0, 1)
            + C.transpose(2, 1, 0)
        ) / 6.0

        def hamiltonian(z):
            return 0.5 * z @ S @ z + np.einsum("ijk,i,j,k", C, z, z, z) / 3.0

        def grad_H(z):
            return S @ z + np.einsum("ijk,j,k->i", C, z, z)

        A1 = torch.tensor(J_np @ S, dtype=torch.float64)
        # RHS tensor for the quadratic term of grad H: J @ C summed over
        # the gradient slots.  grad of (1/3)C(z,z,z) with symmetric C is
        # C(:, z, z), so the degree-2 tensor is J @ C.
        A2 = torch.tensor(np.einsum("il,ljk->ijk", J_np, C), dtype=torch.float64)
        model = PolynomialModel(N, [1, 2], tensors=[A1, A2])

        z0 = torch.tensor(0.2 * rng.standard_normal(N), dtype=torch.float64)
        dt = 0.05
        nsteps = 2000  # T = 100

        def march(method):
            H_vals = np.zeros(nsteps + 1)
            z = z0.clone()
            H_vals[0] = hamiltonian(z.numpy())
            t = 0.0
            for k in range(nsteps):
                z = evolve(
                    model.evaluate_rhs, t, z, dt, method=method,
                    newton_tol=1e-12, newton_max_iter=50,
                )
                t += dt
                H_vals[k + 1] = hamiltonian(z.detach().numpy())
            return H_vals

        H0 = float(hamiltonian(z0.numpy()))
        scale = max(abs(H0), 1.0)

        for method in ("implicit_midpoint", "yoshida4"):
            H_vals = march(method)
            drift = np.abs(H_vals - H0) / scale
            half = nsteps // 2
            # No secular growth: the drift in the second half is comparable
            # to the first half (bounded oscillation), not accumulating.
            assert drift[half:].max() <= 3.0 * max(drift[:half].max(), 1e-14)

        # rk2 at the same step drifts monotonically and ends far worse than
        # the symplectic methods.
        H_rk2 = march("rk2")
        drift_rk2 = abs(H_rk2[-1] - H0) / scale
        H_mid = march("implicit_midpoint")
        drift_mid = np.abs(H_mid - H0).max() / scale
        assert drift_rk2 > 10.0 * drift_mid
