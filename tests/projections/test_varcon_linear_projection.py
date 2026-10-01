"""Tests for the VarconLinearProjection class."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from nitrom.backend import set_backend
from nitrom.projections.linear_projection import LinearProjection
from nitrom.projections.varcon_linear_projection import VarconLinearProjection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 10  # full-space dimension (even)
R = 4   # reduced-space dimension (even: U^T J^T U must be invertible)
M = 5   # batch size


def _make_U(seed=42):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((N, R)))
    return torch.tensor(Q, dtype=torch.float64)


@pytest.fixture
def J():
    return VarconLinearProjection.canonical_J(N)


@pytest.fixture
def proj(J):
    return VarconLinearProjection(_make_U(), J)


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------

class TestBasic:

    def test_param_names(self, proj):
        assert proj.param_names == ["Phi"]

    def test_get_params_is_phi_only(self, proj):
        params = proj.get_params()
        assert len(params) == 1
        np.testing.assert_allclose(params[0].numpy(), proj.Phi.numpy())

    def test_psi_is_J_phi(self, proj, J):
        np.testing.assert_allclose(
            proj.Psi.numpy(), (J @ proj.Phi).numpy(), rtol=1e-13
        )

    def test_update_recomputes_psi_and_S(self, proj, J):
        U_new = _make_U(seed=7)
        proj.update([U_new])
        np.testing.assert_allclose(
            proj.Psi.numpy(), (J @ U_new).numpy(), rtol=1e-13
        )
        S_ref = torch.linalg.inv((J @ U_new).T @ U_new)
        np.testing.assert_allclose(proj.S.numpy(), S_ref.numpy(), rtol=1e-10)

    def test_matches_plain_linear_projection(self, proj, J):
        """encode/decode must equal LinearProjection([U, J @ U])."""
        U = proj.Phi
        ref = LinearProjection([U, J @ U])
        q = torch.randn(N, dtype=torch.float64)
        z = torch.randn(R, dtype=torch.float64)
        np.testing.assert_allclose(
            proj.encode(q).numpy(), ref.encode(q).numpy(), rtol=1e-12
        )
        np.testing.assert_allclose(
            proj.decode(z).numpy(), ref.decode(z).numpy(), rtol=1e-12
        )

    def test_encode_decode_identity(self, proj):
        """The symplectic Petrov-Galerkin pair satisfies encode(decode(z)) = z."""
        z = torch.randn(R, dtype=torch.float64)
        np.testing.assert_allclose(
            proj.encode(proj.decode(z)).numpy(), z.numpy(), rtol=1e-10
        )

    def test_canonical_J_structure(self):
        J = VarconLinearProjection.canonical_J(6)
        half = 3
        np.testing.assert_allclose(J[:half, half:].numpy(), np.eye(half))
        np.testing.assert_allclose(J[half:, :half].numpy(), -np.eye(half))
        np.testing.assert_allclose((J + J.T).numpy(), np.zeros((6, 6)))

    def test_canonical_J_odd_raises(self):
        with pytest.raises(ValueError, match="even"):
            VarconLinearProjection.canonical_J(5)

    def test_odd_dimension_raises(self):
        J = torch.zeros((5, 5), dtype=torch.float64)
        U = torch.zeros((5, R), dtype=torch.float64)
        with pytest.raises(ValueError, match="even"):
            VarconLinearProjection(U, J)

    def test_nonsquare_J_raises(self):
        J = torch.zeros((N, N + 2), dtype=torch.float64)
        U = torch.zeros((N, R), dtype=torch.float64)
        with pytest.raises(ValueError, match="square"):
            VarconLinearProjection(U, J)

    def test_nonskew_J_raises(self):
        J = torch.eye(N, dtype=torch.float64)
        with pytest.raises(ValueError, match="skew"):
            VarconLinearProjection(_make_U(), J)

    def test_skew_but_noncanonical_J_raises(self, J):
        """J = -J^T alone is not enough: J must also satisfy J @ J = -I."""
        with pytest.raises(ValueError, match="J @ J = -I"):
            VarconLinearProjection(_make_U(), 2.0 * J)

    def test_canonical_J_identities(self, J):
        """canonical_J satisfies J = -J^T = J^{-T} and J = -J^{-1}."""
        eye = torch.eye(N, dtype=torch.float64)
        np.testing.assert_allclose((-J.T).numpy(), J.numpy(), atol=1e-14)
        np.testing.assert_allclose((J @ J).numpy(), -eye.numpy(), atol=1e-14)
        np.testing.assert_allclose(
            torch.linalg.inv(J).numpy(), (-J).numpy(), atol=1e-14
        )
        np.testing.assert_allclose(
            torch.linalg.inv(J.T).numpy(), J.numpy(), atol=1e-14
        )

    def test_reduced_structure_skew_only(self, proj):
        """Psi^T Phi = U^T J^T U is skew-symmetric but, unlike J, generally
        NOT orthogonal-skew: the reduced structure is non-canonical."""
        Jred = proj.Psi.T @ proj.Phi
        np.testing.assert_allclose(Jred.numpy(), -Jred.T.numpy(), atol=1e-13)
        eye = torch.eye(R, dtype=torch.float64)
        assert not torch.allclose(Jred @ Jred, -eye, atol=1e-6)

    def test_mismatched_U_raises(self, J):
        U = torch.zeros((N + 2, R), dtype=torch.float64)
        with pytest.raises(ValueError, match="U must"):
            VarconLinearProjection(U, J)

    def test_odd_r_raises(self, J):
        rng = np.random.default_rng(1)
        U, _ = np.linalg.qr(rng.standard_normal((N, 3)))
        U = torch.tensor(U, dtype=torch.float64)
        with pytest.raises(ValueError, match="even number of columns"):
            VarconLinearProjection(U, J)

    def test_vjp_tuple_lengths_match_param_names(self, proj):
        """NitromModule zips vjp outputs against param_names with
        strict=True, so both VJPs must return 1-tuples."""
        q = torch.randn(N, dtype=torch.float64)
        z = torch.randn(R, dtype=torch.float64)
        vq = torch.randn(R, dtype=torch.float64)
        vz = torch.randn(N, dtype=torch.float64)
        assert len(proj.vjp_encode(q, vq)) == len(proj.param_names) == 1
        assert len(proj.vjp_decode(z, vz)) == len(proj.param_names) == 1


# ---------------------------------------------------------------------------
# VJPs against autograd (chained Psi -> Phi gradient)
# ---------------------------------------------------------------------------

class TestVjpAgainstAutograd:

    @pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
    def test_vjp_encode(self, proj, J, batched):
        rng = np.random.default_rng(60)
        if batched:
            q = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)
            v = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
        else:
            q = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
            v = torch.tensor(rng.standard_normal(R), dtype=torch.float64)

        (grad_Phi,) = proj.vjp_encode(q, v)

        Phi = proj.Phi.clone().requires_grad_(True)
        Psi = J @ Phi
        z = q @ Psi if batched else Psi.T @ q
        loss = torch.sum(v * z)
        loss.backward()

        np.testing.assert_allclose(
            grad_Phi.numpy(), Phi.grad.numpy(), rtol=1e-10, atol=1e-12
        )

    @pytest.mark.parametrize("batched", [False, True], ids=["single", "batch"])
    def test_vjp_decode(self, proj, J, batched):
        rng = np.random.default_rng(61)
        if batched:
            z = torch.tensor(rng.standard_normal((M, R)), dtype=torch.float64)
            v = torch.tensor(rng.standard_normal((M, N)), dtype=torch.float64)
        else:
            z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
            v = torch.tensor(rng.standard_normal(N), dtype=torch.float64)

        (grad_Phi,) = proj.vjp_decode(z, v)

        Phi = proj.Phi.clone().requires_grad_(True)
        Psi = J @ Phi
        S = torch.linalg.inv(Psi.T @ Phi)
        qhat = (z @ S.T) @ Phi.T if batched else Phi @ (S @ z)
        loss = torch.sum(v * qhat)
        loss.backward()

        np.testing.assert_allclose(
            grad_Phi.numpy(), Phi.grad.numpy(), rtol=1e-9, atol=1e-12
        )

    def test_vjp_decode_finite_difference(self, proj):
        """FD check of the full chained decode gradient in Phi."""
        rng = np.random.default_rng(62)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
        dU = torch.tensor(rng.standard_normal((N, R)), dtype=torch.float64)

        (grad_Phi,) = proj.vjp_decode(z, v)
        dd_vjp = torch.sum(grad_Phi * dU).item()

        eps = 1e-7
        U0 = proj.Phi.clone()
        proj.update([U0 + eps * dU])
        J_plus = torch.dot(v, proj.decode(z)).item()
        proj.update([U0 - eps * dU])
        J_minus = torch.dot(v, proj.decode(z)).item()
        proj.update([U0])

        dd_fd = (J_plus - J_minus) / (2 * eps)
        np.testing.assert_allclose(dd_vjp, dd_fd, rtol=1e-5)

    def test_vjp_decode_state_inherited(self, proj):
        """The state VJP has no parameter dependence and stays the
        LinearProjection one."""
        z = torch.randn(R, dtype=torch.float64)
        v = torch.randn(N, dtype=torch.float64)
        expected = proj.S.T @ (proj.Phi.T @ v)
        np.testing.assert_allclose(
            proj.vjp_decode_state(z, v).numpy(), expected.numpy(), rtol=1e-12
        )


# ---------------------------------------------------------------------------
# Sparse J (numpy backend only)
# ---------------------------------------------------------------------------

class TestSparse:

    def test_sparse_matches_dense(self):
        set_backend("numpy")
        try:
            rng = np.random.default_rng(70)
            U, _ = np.linalg.qr(rng.standard_normal((N, R)))
            J_sp = VarconLinearProjection.canonical_J(N, sparse=True)

            proj_sp = VarconLinearProjection(U, J_sp)
            J_dense = np.asarray(J_sp.todense(), dtype=float)
            proj_d = VarconLinearProjection(U, J_dense)

            q = rng.standard_normal(N)
            z = rng.standard_normal(R)
            np.testing.assert_allclose(
                proj_sp.encode(q), proj_d.encode(q), rtol=1e-12
            )
            np.testing.assert_allclose(
                proj_sp.decode(z), proj_d.decode(z), rtol=1e-12
            )
        finally:
            set_backend("torch")

    def test_sparse_on_torch_raises(self):
        J_sp = sp.csr_matrix(
            VarconLinearProjection.canonical_J(N).numpy()
        )
        with pytest.raises(TypeError, match="numpy backend"):
            VarconLinearProjection(_make_U(), J_sp)
