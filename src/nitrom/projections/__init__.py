"""Projection maps between full-order and latent spaces: linear, variationally constrained, and polynomial."""

from .linear_projection import LinearProjection
from .polynomial_projection import PolynomialProjection
from .projection import Projection
from .varcon_linear_projection import VarconLinearProjection

__all__ = [
    "LinearProjection",
    "PolynomialProjection",
    "Projection",
    "VarconLinearProjection",
]
