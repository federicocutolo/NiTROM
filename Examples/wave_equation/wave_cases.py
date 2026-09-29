"""
Shared test-case registry for the 1D wave-equation NiTROM example.

Two self-contained cases, each in its own subfolder with identical naming:
  Dirichlet/{trajectories,output,test_results}  <- main.py        (homogeneous Dirichlet BC)
  Periodic/{trajectories,output,test_results}   <- main_gruber.py (Gruber-Tezaur periodic BC)

Consumers select a case by name and read everything off the returned object:
    from wave_cases import get_case
    case = get_case("dirichlet")
    fom, data_dir, output_dir = case.fom, case.data_dir, case.output_dir
"""
import os
from types import SimpleNamespace

from wave_foms import WaveEquationFOM, PeriodicWaveEquationFOM

HERE = os.path.dirname(os.path.abspath(__file__))

# Train / held-out-test RNG streams for the shared IC sampler (wave_sampling).
# Owned here so data generation (main.py, which also PLOTS the test points) and
# evaluation (test_nitrom.py, which generates the test trajectories) agree on
# the same disjoint draws from the same sampling box.
TRAIN_SEED = 42
TEST_SEED  = 7        # != TRAIN_SEED -> non-overlapping draw from the same box
N_TEST     = 10

# NiTROM initialization cases (mirrors the bracket example): the reduced BASIS
# and the tensor seed A2 are chosen independently, and each (basis, tensor)
# pair trains into its own labeled container <output_dir>/<basis>_<tensor>/ so
# the four cases never overwrite each other.
NITROM_BASES   = ("cl", "pod")      # cotangent-lift (symplectic) | plain POD
NITROM_TENSORS = ("pg", "opinf")    # Petrov-Galerkin             | VC-H-OpInf


def nitrom_output_dir(case, basis, tensor):
    """Folder holding the NiTROM solution initialized with (basis, tensor)."""
    return os.path.join(case.output_dir, f"{basis}_{tensor}")


def ensure_nitrom_dirs(case):
    """Create the four labeled NiTROM containers (existing ones are left alone)."""
    for basis in NITROM_BASES:
        for tensor in NITROM_TENSORS:
            os.makedirs(nitrom_output_dir(case, basis, tensor), exist_ok=True)

# FOM physics + folder per case. N/L/c MUST match the data-generation scripts
# (main.py for Dirichlet, main_gruber.py for Periodic).
TESTCASES = {
    "dirichlet": {
        "label":   "Dirichlet wave equation",
        "folder":  "Dirichlet",
        "fom_cls": WaveEquationFOM,
        "N": 128, "L": 2.0, "c": 1.0,
        # Gaussian-pulse IC sampling box (consumed by wave_sampling). sigma floor
        # scales with grid resolution dx; ceiling keeps ~4*sigma inside the walls.
        "sampling": {"amp": (0.1, 1.0), "sigma_dx_factor": 5.0,
                     "sigma_max": 0.15, "x0": (0.5, 1.5)},
        "legacy":  {"trajectories": "trajectories",
                    "output":       "output",
                    "test_results": "test_results"},
    },
    "periodic": {
        "label":   "Periodic (Gruber-Tezaur) wave equation",
        "folder":  "Periodic",
        "fom_cls": PeriodicWaveEquationFOM,
        "N": 500, "L": 1.0, "c": 0.1,
        # Gaussian-pulse IC sampling box (consumed by wave_sampling), drawn the
        # same way as Dirichlet but periodic + zero-momentum: each bump splits
        # into TWO counter-propagating waves under the periodic BC. x0 spans the
        # whole domain (minimal-image wrap keeps the pulse smooth at the seam).
        "sampling": {"amp": (0.05, 2.0), "sigma_dx_factor": 2.0,
                     "sigma_max": 0.1, "x0": (0.0, 1.0),
                     "momentum": "zero", "periodic": True},
        "legacy":  {"trajectories": "trajectories_gruber",
                    "output":       "output_gruber",
                    "test_results": "test_results_gruber"},
    },
}


def _resolve(testcase, sub):
    """Prefer the <Folder>/<sub> subfolder; fall back to the legacy flat folder."""
    cfg = TESTCASES[testcase]
    new = os.path.join(HERE, cfg["folder"], sub)
    if os.path.isdir(new):
        return new
    return os.path.join(HERE, cfg["legacy"][sub])


def case_dir(testcase, sub):
    """Resolve a single subfolder ('trajectories' | 'output' | 'test_results')."""
    return _resolve(testcase, sub)


def get_case(testcase):
    """Resolve a TESTCASE name to its FOM instance, sampling box, and I/O folders."""
    if testcase not in TESTCASES:
        raise ValueError(f"Invalid TESTCASE={testcase!r}; use one of {list(TESTCASES)}")
    cfg = TESTCASES[testcase]
    fom = cfg["fom_cls"](N=cfg["N"], L=cfg["L"], c=cfg["c"])

    # Resolve the sampling box now that fom.dx is known (sigma floor = factor*dx).
    samp = cfg.get("sampling")
    sample_ranges = None if samp is None else {
        "amp":      samp["amp"],
        "sigma":    (fom.dx * samp["sigma_dx_factor"], samp["sigma_max"]),
        "x0":       samp["x0"],
        "momentum": samp.get("momentum", "traveling"),
        "periodic": samp.get("periodic", False),
    }

    return SimpleNamespace(
        name=testcase,
        label=cfg["label"],
        N=cfg["N"], L=cfg["L"], c=cfg["c"],
        fom=fom,
        sample_ranges=sample_ranges,
        data_dir=_resolve(testcase, "trajectories"),
        output_dir=_resolve(testcase, "output"),
        test_results_dir=_resolve(testcase, "test_results"),
    )
