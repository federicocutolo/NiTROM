"""Tests for the abstract StructuredPolynomialModel class."""

import numpy as np
import pytest
import torch

from nitrom.latent_space_models.polynomial_model import PolynomialModel
from nitrom.latent_space_models.structured_polynomial_model import (
    StructuredPolynomialModel,
)

R = 3


def _make_complete_class():
    """Minimal concrete subclass: one free matrix ``W`` assembled into the
    linear tensor ``A = W @ W^T`` (so the chain rule is nontrivial)."""

    class Complete(StructuredPolynomialModel):
        def _param_specs(self, r, poly_comp):
            return [("W", (r, r), "randn")]

        def assemble_inner_tensors(self):
            return [self.W @ self.W.T]

        def project_inner_gradients(self, inner_grads):
            (grad_A,) = inner_grads
            return [grad_A @ self.W + grad_A.T @ self.W]

    return Complete


class TestStructuredAbstract:

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            StructuredPolynomialModel(R, [1])

    def test_must_implement_param_specs(self):
        class Incomplete(StructuredPolynomialModel):
            def assemble_inner_tensors(self):
                return []

            def project_inner_gradients(self, inner_grads):
                return inner_grads

        with pytest.raises(TypeError):
            Incomplete(R, [1])

    def test_must_implement_assemble_inner_tensors(self):
        class Incomplete(StructuredPolynomialModel):
            def _param_specs(self, r, poly_comp):
                return []

            def project_inner_gradients(self, inner_grads):
                return inner_grads

        with pytest.raises(TypeError):
            Incomplete(R, [1])

    def test_must_implement_project_inner_gradients(self):
        class Incomplete(StructuredPolynomialModel):
            def _param_specs(self, r, poly_comp):
                return []

            def assemble_inner_tensors(self):
                return []

        with pytest.raises(TypeError):
            Incomplete(R, [1])


class TestCompleteSubclass:

    def test_delegates_to_inner_polynomial_model(self):
        Complete = _make_complete_class()
        model = Complete(R, [1])

        assert model.param_names == ["W"]
        assert isinstance(model.model, PolynomialModel)

        z = torch.randn(R, dtype=torch.float64)
        rhs = model.evaluate_rhs(0.0, z)
        expected = (model.W @ model.W.T) @ z
        np.testing.assert_allclose(rhs.numpy(), expected.numpy(), rtol=1e-12)

    def test_update_params_reassembles(self):
        Complete = _make_complete_class()
        model = Complete(R, [1])

        W_new = torch.randn((R, R), dtype=torch.float64)
        model.update_params([W_new])

        z = torch.randn(R, dtype=torch.float64)
        rhs = model.evaluate_rhs(0.0, z)
        np.testing.assert_allclose(
            rhs.numpy(), ((W_new @ W_new.T) @ z).numpy(), rtol=1e-12
        )

    def test_vjp_chains_through_assembly(self):
        """vjp_evaluate_rhs (inner VJP + projection) matches a central
        finite difference in W."""
        Complete = _make_complete_class()
        model = Complete(R, [1])

        rng = np.random.default_rng(5)
        z = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        v = torch.tensor(rng.standard_normal(R), dtype=torch.float64)
        dW = torch.tensor(rng.standard_normal((R, R)), dtype=torch.float64)

        (grad_W,) = model.vjp_evaluate_rhs(z, v)
        dd_vjp = torch.sum(grad_W * dW).item()

        eps = 1e-7
        W0 = model.W.clone()
        model.update_params([W0 + eps * dW])
        J_plus = torch.dot(v, model.evaluate_rhs(0.0, z)).item()
        model.update_params([W0 - eps * dW])
        J_minus = torch.dot(v, model.evaluate_rhs(0.0, z)).item()
        model.update_params([W0])

        dd_fd = (J_plus - J_minus) / (2 * eps)
        np.testing.assert_allclose(dd_vjp, dd_fd, rtol=1e-6)
