"""NiTROM: Non-intrusive Trajectory-based Reduced-Order Modelling."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("nitrom")
except PackageNotFoundError:  # not installed (e.g. docs build from a source tree)
    __version__ = "0+unknown"

from . import projections
from .backend import cleanup_distributed, setup_distributed
from .latent_space_models import (
    AtrPolynomialModel,
    GasPolynomialModel,
    HamiltonianPolynomialModel,
    Model,
    PolynomialModel,
    StructuredPolynomialModel,
)
from .optimization import (
    InferenceModule,
    NitromModule,
    OpInfModule,
    PolyManifoldInfModule,
    perform_POD,
    solve_opinf,
    train,
)
from .roms import ParamRegistry
from .time_steppers import (
    DenseSolution,
    evolve,
    linear_symplectic_integrate,
    linear_symplectic_solve,
    solve_adjoint_ivp_discrete,
    solve_ivp,
    solve_ivp_dense,
)
from .training_data import TrainingData, TrainingPool
from .utils import compute_POD, interp_quadratic, validate_symplectic_structure

__all__ = [
    "AtrPolynomialModel",
    "DenseSolution",
    "GasPolynomialModel",
    "HamiltonianPolynomialModel",
    "InferenceModule",
    "Model",
    "NitromModule",
    "OpInfModule",
    "ParamRegistry",
    "PolyManifoldInfModule",
    "PolynomialModel",
    "StructuredPolynomialModel",
    "TrainingData",
    "TrainingPool",
    "__version__",
    "cleanup_distributed",
    "compute_POD",
    "evolve",
    "interp_quadratic",
    "linear_symplectic_integrate",
    "linear_symplectic_solve",
    "perform_POD",
    "projections",
    "setup_distributed",
    "solve_adjoint_ivp_discrete",
    "solve_ivp",
    "solve_ivp_dense",
    "solve_opinf",
    "train",
    "validate_symplectic_structure",
]
