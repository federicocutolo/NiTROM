import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest


def _load_classes_module():
    module_name = "nitrom_classes_for_tests"
    module_path = (
        Path(__file__).resolve().parents[1]
        / "NiTROM"
        / "Optimization_Functions"
        / "classes.py"
    )

    mpi_module = types.ModuleType("mpi4py")
    mpi_module.MPI = types.SimpleNamespace(INT=None)
    sys.modules.setdefault("mpi4py", mpi_module)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


classes = _load_classes_module()
HamiltonianROM = classes.HamiltonianROM


def _make_rom():
    xdim = 2
    A = np.array([[2.0, -1.0], [3.0, 4.0]])
    H = np.array(
        [
            [[1.0, 2.0], [0.0, -1.0]],
            [[-2.0, 1.0], [3.0, 0.5]],
        ]
    )
    Phi = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    return HamiltonianROM(operators=[A, H], poly_comp=[1, 2], Phi=Phi)


def test_hamiltonian_rom_initialization_builds_einsum_and_jhat():
    rom = _make_rom()

    print("\n[init] einsum_ss:", rom.einsum_ss)
    print("[init] Jhat:\n", rom.Jhat)

    assert rom.einsum_ss == (["ab", "b"], ["abc", "b", "c"])
    npt.assert_allclose(rom.Jhat, np.array([[0.0, 1.0], [-1.0, 0.0]]))


def test_compute_hamiltonian_matches_manual_polynomial():
    rom = _make_rom()
    x = np.array([1.5, -0.5])

    quadratic = x @ rom.operators[0] @ x
    cubic = np.dot(x, np.einsum("ijk,j,k->i", rom.operators[1], x, x))

    value = rom.compute_hamiltonian(x)

    print("\n[hamiltonian] x:", x)
    print("[hamiltonian] quadratic contribution:", quadratic)
    print("[hamiltonian] cubic contribution:", cubic)
    print("[hamiltonian] total expected:", quadratic + cubic)
    print("[hamiltonian] total computed:", value)

    npt.assert_allclose(value, quadratic + cubic)


def test_compute_hamiltonian_gradient_matches_explicit_formula():
    rom = _make_rom()
    x = np.array([0.25, -1.5])
    A, H = rom.operators

    expected = (A + A.T) @ x
    expected += np.einsum("ijk,j,k->i", H, x, x)
    expected += np.einsum("ijk,i,k->j", H, x, x)
    expected += np.einsum("ijk,i,j->k", H, x, x)

    gradient = rom.compute_hamiltonian_gradient(x)

    print("\n[gradient] x:", x)
    print("[gradient] expected:", expected)
    print("[gradient] computed:", gradient)

    npt.assert_allclose(gradient, expected)


def test_evaluate_rom_rhs_applies_jhat_to_gradient():
    rom = _make_rom()
    x = np.array([-0.75, 2.0])

    expected = rom.Jhat @ rom.compute_hamiltonian_gradient(x)

    rhs = rom.evaluate_rom_rhs(x)

    print("\n[rhs] x:", x)
    print("[rhs] Jhat:\n", rom.Jhat)
    print("[rhs] expected:", expected)
    print("[rhs] computed:", rhs)

    npt.assert_allclose(rhs, expected)


def test_evaluate_rom_rhs_requires_even_dimension():
    rom = _make_rom()

    try:
        rom.evaluate_rom_rhs(np.array([1.0, 2.0, 3.0]))
    except ValueError as exc:
        print("\n[guard] caught error:", exc)
        assert "even dimension" in str(exc)
    else:
        raise AssertionError("Expected ValueError for odd-dimensional state.")


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-s", __file__]))
