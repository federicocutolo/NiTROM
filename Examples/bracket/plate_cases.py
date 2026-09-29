"""
Test-case registry for the cantilever-plate NiTROM example (mirrors
wave_cases.py). The plate FOM is generated elsewhere (Albany / the reference
HamiltonianOpInf pipeline); here we only need the I/O layout and the canonical
symplectic structure, so there is no FOM class and no IC sampling box.

    Cantilever/{trajectories,output,test_results}

``prepare_data.py`` writes the NiTROM inputs (traj_/deriv_/time/Phi_cl/weight,
+ optional W_fom) into ``trajectories`` from the provided reference-dict .npy.
"""
import os
from types import SimpleNamespace

import numpy as np
from scipy.sparse import identity, bmat, csc_matrix

HERE = os.path.dirname(os.path.abspath(__file__))

TESTCASES = {
    "bracket": {
        "label":   "3D linear-elastic bracket",
        "folder":  "Bracket",
        "raw_dir": "bracket_for_Fede",       # source FE data (mass.mm, stiff.mm, .e)
        "exo":     "bracket_velIC_100.e",
    },
    "cantilever_plate": {
        "label":   "cantilever plate",
        "folder":  "CantileverPlate",
        "raw_dir": "cantilever_data",    # drop your FE data here
        "exo":     "beam_velIC_100.e",     # Exodus snapshot filename
    },
}


def active_case():
    """Active testcase name; overridable via the ``PLATE_CASE`` env var."""
    return os.environ.get("PLATE_CASE", "bracket")


def canonical_J(dim, sparse=True):
    """Block symplectic J = [[0, I], [-I, 0]] of even size ``dim``.

    Sparse by default: the plate state dimension is ~1e4, so a dense J would be
    ~Gb. The wave_roms / plate_roms operators only matmul with J, so a sparse J
    drops in unchanged.
    """
    half = dim // 2
    if sparse:
        I = identity(half, format="csc")
        Z = csc_matrix((half, half))
        return bmat([[Z, I], [-I, Z]], format="csc")
    return np.block([[np.zeros((half, half)), np.eye(half)],
                     [-np.eye(half), np.zeros((half, half))]])


def _resolve(testcase, sub):
    return os.path.join(HERE, TESTCASES[testcase]["folder"], sub)


# ── NiTROM solution containers ────────────────────────────────────────────────
# A NiTROM run is labeled by its initialization: the reduced basis it starts
# from and the tensor used to seed A2. Each of the four combinations gets its
# own solution folder, e.g. Bracket/output/cl_opinf, so differently-initialized
# runs never overwrite each other and can be compared side by side.
NITROM_BASES   = ("cl", "pod")      # cotangent-lift (symplectic) | plain POD
NITROM_TENSORS = ("pg", "opinf")    # Petrov-Galerkin            | VC-H-OpInf


def nitrom_output_dir(case, basis, tensor):
    """Folder holding the NiTROM solution initialized with (basis, tensor)."""
    return os.path.join(case.output_dir, f"{basis}_{tensor}")


def ensure_nitrom_dirs(case):
    """Create the four labeled NiTROM containers (existing ones are left alone)."""
    for basis in NITROM_BASES:
        for tensor in NITROM_TENSORS:
            os.makedirs(nitrom_output_dir(case, basis, tensor), exist_ok=True)


def get_case(testcase=None):
    """Resolve a TESTCASE name to its I/O folders + metadata.

    Single deterministic trajectory (one IC), so ``sample_ranges`` is None and
    ``fom`` is None: VC-H-OpInf and NiTROM consume the saved data/velocities and
    never call a FOM.
    """
    if testcase is None:
        testcase = active_case()
    if testcase not in TESTCASES:
        raise ValueError(f"Invalid TESTCASE={testcase!r}; use one of {list(TESTCASES)}")
    cfg = TESTCASES[testcase]
    raw_dir = os.path.join(HERE, cfg["raw_dir"])
    return SimpleNamespace(
        name=testcase,
        label=cfg["label"],
        fom=None,
        sample_ranges=None,
        data_dir=_resolve(testcase, "trajectories"),
        output_dir=_resolve(testcase, "output"),
        test_results_dir=_resolve(testcase, "test_results"),
        raw_dir=raw_dir,                                  # source FE data folder
        exo=os.path.join(raw_dir, cfg["exo"]),            # Exodus snapshot file
        mass=os.path.join(raw_dir, "mass.mm"),
        stiff=os.path.join(raw_dir, "stiff.mm"),
    )
