"""
Export the bracket trajectory (x.npy, the Exodus-derived pipeline data written
by prepare_data.py) to a ParaView-readable VTU time series + PVD index.

Run:  python export_paraview.py
Then open  Bracket/paraview/fom_exodus.pvd  in ParaView.

Each frame is a VTK unstructured grid (the TETRA mesh from the source Exodus
file) with point-data vectors:
    displacement  q  (from the first half of the state columns)
    velocity      qdot (from the xdot file, if present)
Use "Warp By Vector" on `displacement` (add a Scale Factor: the peak
displacement is ~8e-3 on a ~0.2 body) and color by displacement magnitude,
then hit Play -- the PVD carries the physical times.
"""
import os
import sys

import numpy as np
import exodusii

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from plate_cases import get_case

# ── Config ────────────────────────────────────────────────────────────────────
TESTCASE = "bracket"
PREFIX   = "fom_exodus"      # names the output frames/.pvd; input is x/xdot/time.npy
STRIDE   = 1                 # keep every STRIDE-th sample (thin long runs)

_case   = get_case(TESTCASE)
OUT_DIR = os.path.join(HERE, "Bracket", "paraview")

VTK_TET = 10                 # VTK cell type: 4-node tetrahedron


def load_mesh(exo_path):
    """Coordinates (N, 3) and 0-based TETRA connectivity (ne, 4) from Exodus."""
    exo = exodusii.ExodusIIFile(exo_path, mode="r")
    coords = np.asarray(exo.get_coords(), dtype=float)
    conn = np.vstack([np.asarray(exo.get_element_conn(bid), dtype=np.int64) - 1
                      for bid in exo.get_element_block_ids()])
    return coords, conn


def write_vtu(path, coords, conn, point_vectors):
    """Minimal ASCII .vtu (unstructured grid, TETRA cells) with named
    3-component point-data vector fields."""
    n_pts, n_cel = len(coords), len(conn)
    fmt = lambda a: "\n".join(" ".join(f"{v:.9g}" for v in row) for row in a)
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n'
                '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
                '<UnstructuredGrid>\n'
                f'<Piece NumberOfPoints="{n_pts}" NumberOfCells="{n_cel}">\n')
        f.write('<Points>\n<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        f.write(fmt(coords))
        f.write('\n</DataArray>\n</Points>\n')
        f.write('<Cells>\n<DataArray type="Int64" Name="connectivity" format="ascii">\n')
        f.write(fmt(conn))
        f.write('\n</DataArray>\n<DataArray type="Int64" Name="offsets" format="ascii">\n')
        f.write(" ".join(str(4 * (i + 1)) for i in range(n_cel)))
        f.write('\n</DataArray>\n<DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write(" ".join(str(VTK_TET) for _ in range(n_cel)))
        f.write('\n</DataArray>\n</Cells>\n')
        f.write('<PointData Vectors="displacement">\n')
        for name, vec in point_vectors.items():
            f.write(f'<DataArray type="Float64" Name="{name}" '
                    f'NumberOfComponents="3" format="ascii">\n')
            f.write(fmt(vec))
            f.write('\n</DataArray>\n')
        f.write('</PointData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n')


def main():
    data_dir = _case.data_dir
    # Exodus-derived pipeline data written by prepare_data.py (x = [q; p]).
    X    = np.load(os.path.join(data_dir, "x.npy"), mmap_mode="r")
    time = np.load(os.path.join(data_dir, "time.npy"))
    _xd  = os.path.join(data_dir, "xdot.npy")
    Xdot = np.load(_xd, mmap_mode="r") if os.path.exists(_xd) else None

    coords, conn = load_mesh(_case.exo)
    N = len(coords)                                    # nodes; q is (3N,) interleaved
    assert X.shape[0] == 6 * N, f"state dim {X.shape[0]} != 2*3N = {6 * N}"

    frames = range(0, X.shape[1], STRIDE)
    frame_dir = os.path.join(OUT_DIR, PREFIX)
    os.makedirs(frame_dir, exist_ok=True)

    print(f"Exporting {len(frames)}/{X.shape[1]} frames "
          f"({N} nodes, {len(conn)} tets) -> {frame_dir}")
    entries = []
    for j, k in enumerate(frames):
        vecs = {"displacement": np.asarray(X[:3 * N, k]).reshape(N, 3)}
        if Xdot is not None:
            vecs["velocity"] = np.asarray(Xdot[:3 * N, k]).reshape(N, 3)
        name = f"{PREFIX}_{j:04d}.vtu"
        write_vtu(os.path.join(frame_dir, name), coords, conn, vecs)
        entries.append(f'<DataSet timestep="{time[k]:.9g}" '
                       f'file="{PREFIX}/{name}"/>')
        if (j + 1) % max(1, len(frames) // 10) == 0:
            print(f"  frame {j + 1}/{len(frames)}  (t = {time[k]:.6g})", flush=True)

    pvd = os.path.join(OUT_DIR, f"{PREFIX}.pvd")
    with open(pvd, "w") as f:
        f.write('<?xml version="1.0"?>\n'
                '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
                '<Collection>\n' + "\n".join(entries) + '\n</Collection>\n</VTKFile>\n')
    print(f"Wrote {pvd}\nOpen it in ParaView, apply 'Warp By Vector' on "
          f"'displacement', color by its magnitude, and press Play.")


if __name__ == "__main__":
    main()
