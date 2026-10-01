"""End-to-end gradient checks for the Hamiltonian ROM pipeline.

Wires HamiltonianPolynomialModel + VarconLinearProjection through
ParamRegistry (which auto-shares "Phi") and NitromModule, and checks the
discrete-adjoint gradient -- including the summed shared-Phi gradient
(decoder + encoder + the Jhat flow inside the model + the Psi = J Phi
chain in the projection) -- against a finite difference of the cost.
"""

import pytest
import torch

from nitrom.latent_space_models import HamiltonianPolynomialModel
from nitrom.optimization import NitromModule
from nitrom.projections.varcon_linear_projection import VarconLinearProjection
from nitrom.roms.param_registry import ParamRegistry

from ._gradcheck import (
    DTYPE,
    Data,
    LinearOutputFOM,
    assert_grad_close,
    finite_diff_grad,
    random_basis,
)

N, R, NTRAJ, NT, NO = 6, 2, 2, 4, 3


def _data(seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(NTRAJ, N, NT, generator=g, dtype=DTYPE)
    weights = 1.0 + 0.5 * torch.rand(NTRAJ, generator=g, dtype=DTYPE)
    return Data(X, weights=weights)


def _fom(seed: int = 4) -> LinearOutputFOM:
    g = torch.Generator().manual_seed(seed)
    return LinearOutputFOM(torch.randn(NO, N, generator=g, dtype=DTYPE))


def _make_rom(seed: int = 2):
    """Model + projection sharing the same J and the same initial Phi."""
    J = VarconLinearProjection.canonical_J(N)
    U = random_basis(N, R, seed=1)
    g = torch.Generator().manual_seed(seed)
    ham_params = [
        0.3 * torch.randn(R, R, generator=g, dtype=DTYPE),      # A2
        0.2 * torch.randn(R, R, R, generator=g, dtype=DTYPE),   # A3
    ]
    model = HamiltonianPolynomialModel(
        R, [1, 2], J, U, dtype=DTYPE, ham_params=ham_params
    )
    proj = VarconLinearProjection(U, J)
    return model, proj


def _module(reg=0.0, time_stepper="rk4") -> NitromModule:
    model, proj = _make_rom()
    registry = ParamRegistry(model, proj)
    registry.sync("model")
    return NitromModule(
        _data(), registry, fom=_fom(), reg=reg,
        n_substeps=10, time_stepper=time_stepper,
        n_leggauss=5, adjoint_method="discrete",
    )


class TestRegistryWiring:

    def test_phi_is_shared(self):
        model, proj = _make_rom()
        registry = ParamRegistry(model, proj)
        assert registry.shared_names == ["Phi"]
        assert registry.names == ["A2", "A3", "Phi"]
        assert registry.sources("Phi") == ("model", "projection")

    def test_scatter_keeps_copies_identical(self):
        model, proj = _make_rom()
        registry = ParamRegistry(model, proj)
        registry.sync("model")
        registry.assert_shared_consistent()

        g = torch.Generator().manual_seed(11)
        Phi_new = random_basis(N, R, seed=12)
        values = [
            0.1 * torch.randn(R, R, generator=g, dtype=DTYPE),
            0.1 * torch.randn(R, R, R, generator=g, dtype=DTYPE),
            Phi_new,
        ]
        registry.scatter(values)
        registry.assert_shared_consistent()

        # Both consumers picked up the new basis.
        J = VarconLinearProjection.canonical_J(N)
        Jhat_ref = Phi_new.T @ J @ Phi_new
        torch.testing.assert_close(model.Jhat, Jhat_ref)
        torch.testing.assert_close(proj.Psi, J @ Phi_new)


class TestGradient:

    @pytest.mark.parametrize(
        "time_stepper", ["rk4", "implicit_midpoint", "yoshida4"]
    )
    def test_discrete_adjoint_matches_finite_difference(self, time_stepper):
        """The decisive end-to-end check: the analytic gradient of the full
        Hamiltonian pipeline (shared Phi included) against a central finite
        difference of the cost."""
        module = _module(time_stepper=time_stepper)
        assert_grad_close(
            module.gradient(),
            finite_diff_grad(module),
            rtol=1e-4, atol=5e-6,
        )

    def test_forward_smoke_with_regularization(self):
        """reg > 0 penalizes the assembled inner quadratic tensor T2 (not A3
        directly); the loss difference must equal reg * ||T2||^2."""
        module_no_reg = _module(reg=0.0)
        module_reg = _module(reg=0.5)

        loss0 = module_no_reg()
        loss1 = module_reg()

        T2 = module_reg.model.inner_params()[
            module_reg.model.poly_comp.index(2)
        ]
        expected = 0.5 * torch.sum(T2 * T2)
        torch.testing.assert_close(loss1, loss0 + expected, rtol=1e-6, atol=1e-10)

    def test_gradient_with_regularization(self):
        """The reg gradient must flow through project_inner_gradients back to
        A3 and Phi consistently."""
        module = _module(reg=0.3)
        assert_grad_close(
            module.gradient(),
            finite_diff_grad(module),
            rtol=1e-4, atol=5e-6,
        )
