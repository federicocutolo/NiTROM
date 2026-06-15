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

# FOM physics + folder per case. N/L/c MUST match the data-generation scripts
# (main.py for Dirichlet, main_gruber.py for Periodic).
TESTCASES = {
    "dirichlet": {
        "label":   "Dirichlet wave equation",
        "folder":  "Dirichlet",
        "fom_cls": WaveEquationFOM,
        "N": 128, "L": 2.0, "c": 1.0,
        "legacy":  {"trajectories": "trajectories",
                    "output":       "output",
                    "test_results": "test_results"},
    },
    "periodic": {
        "label":   "Periodic (Gruber-Tezaur) wave equation",
        "folder":  "Periodic",
        "fom_cls": PeriodicWaveEquationFOM,
        "N": 500, "L": 1.0, "c": 0.1,
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
    """Resolve a TESTCASE name to its FOM instance and I/O folders."""
    if testcase not in TESTCASES:
        raise ValueError(f"Invalid TESTCASE={testcase!r}; use one of {list(TESTCASES)}")
    cfg = TESTCASES[testcase]
    return SimpleNamespace(
        name=testcase,
        label=cfg["label"],
        N=cfg["N"], L=cfg["L"], c=cfg["c"],
        fom=cfg["fom_cls"](N=cfg["N"], L=cfg["L"], c=cfg["c"]),
        data_dir=_resolve(testcase, "trajectories"),
        output_dir=_resolve(testcase, "output"),
        test_results_dir=_resolve(testcase, "test_results"),
    )
