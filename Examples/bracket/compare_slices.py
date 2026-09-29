"""
3D comparison of the ROMs against the FOM data in the FULL physical space,
via 2D cross-section videos -- one set of videos per NiTROM case.

A NiTROM case is an initialization pair (basis x tensor: cl/pod x pg/opinf)
living in its container Bracket/output/<basis>_<tensor>. For every TRAINED
case (untrained containers are skipped with a notice), each model -- the
VC-H-OpInf baseline built on the same basis, and the trained NiTROM -- is
integrated in its reduced space, decoded to the compressed full state, and
lifted with Uc to the 3N physical displacement DOFs, the same space as the
raw FOM data (read straight from the Exodus file, uncompressed). N_SLICES
cross sections of the bracket parallel to the (Z, Y) plane (i.e. at constant
X) are then animated: one video per slice, all models overlaid (FOM |
VC-H-OpInf | NiTROM). Writes MP4 (if ffmpeg is available) or GIF otherwise.

Run:  python compare_slices.py
Outputs one folder per case:  Bracket/test_results/<basis>/slices/<tensor>/ .
"""
import os
import shutil
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from scipy.sparse import load_npz
import exodusii

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from plate_cases import (get_case, canonical_J, nitrom_output_dir,
                         NITROM_BASES, NITROM_TENSORS)
from NiTROM.Optimization_Functions.linear_symplectic_integrators import linear_symplectic_solve
import plate_roms

# ── Config ────────────────────────────────────────────────────────────────────
TESTCASE   = "bracket"
# NiTROM cases to animate: (basis, tensor) pairs, one video set per trained
# container output/<basis>_<tensor>; untrained cases are skipped with a notice.
CASES      = [(b, t) for b in NITROM_BASES for t in NITROM_TENSORS]
N_SLICES   = 4          # number of (Z,Y)-parallel cross sections along X
SLICE_FRAC = 0.04       # slice half-thickness as a fraction of the X extent
DT_ROM     = 2.5e-8     # internal symplectic step (commensurate: 1e-4 = 4000*dt)
STRIDE     = 1          # keep every STRIDE-th snapshot in the videos
FPS        = 8
# "deformed": ONE overlaid z-y axes per slice; the section deflects vertically
#             by scale*u_z (cartoon, exaggerated). "field": one panel per model,
#             fixed nodes colored by u_z.
PLOT_MODE  = "deformed"
SCALE      = None       # u_z exaggeration; None -> auto (FOM peak = 30% of y extent)
CONNECT_DOTS = True     # connect the slice nodes along MESH EDGES (deforming
                        # wireframe of the section: every dot linked to its
                        # mesh neighbors, disjoint patches stay disjoint)
DRAW_CONTOUR = False    # additionally outline the body SKIN at the plane
                        # (edge/plane intersection points, NOT mesh nodes)

MODEL_STYLE = {"FOM (data)":   dict(color="k",    zorder=3),
               "VC-H-OpInf":   dict(color="#2ca02c", zorder=2),
               "NiTROM-PG":    dict(color="#9467bd", zorder=2),
               "NiTROM-OpInf": dict(color="#d62728", zorder=2)}

_case    = get_case(TESTCASE)
data_dir = _case.data_dir


def make_definite(A, sign="pos", floor=1e-6):
    """Nearest definite matrix to symmetric A: same eigenvectors, clipped eigenvalues."""
    w, U = np.linalg.eigh(0.5 * (A + A.T))
    thr = floor * np.abs(w).max()
    w = np.clip(w, thr, None) if sign == "pos" else np.clip(w, None, -thr)
    return U @ np.diag(w) @ U.T


def load_fom_data(exo_path):
    """Coordinates (N, 3), TRUE (uncompressed) nodal displacements (N, 3, Nt)
    straight from the Exodus snapshots, the snapshot times, and the surface
    edges of the tet mesh (for section contours)."""
    exo = exodusii.ExodusIIFile(exo_path, mode="r")
    N, Nt = exo.num_nodes(), exo.num_times()
    coords = np.asarray(exo.get_coords(), dtype=float)
    u = np.zeros((N, 3, Nt))
    for c, comp in enumerate("xyz"):
        for k in range(Nt):
            u[:, c, k] = exo.get_node_variable_values(f"disp_{comp}", k + 1)
    conn = np.vstack([np.asarray(exo.get_element_conn(bid), dtype=np.int64) - 1
                      for bid in exo.get_element_block_ids()])
    return coords, u, np.asarray(exo.get_times()), surface_triangles(conn), conn


def surface_triangles(conn):
    """BODY SURFACE triangles from tet connectivity (ne, 4): faces belonging
    to exactly one tet -- the true skin of the body, no hull heuristics."""
    faces = conn[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3)
    faces = np.sort(faces, axis=1)
    uniq, counts = np.unique(faces, axis=0, return_counts=True)
    return uniq[counts == 1]                                 # (nb, 3) node triples


def slab_edges(conn, node_idx):
    """Mesh edges with BOTH endpoints among the slice nodes -> (m, 2) global
    node pairs. Connecting the slice dots along these edges gives a deforming
    wireframe of the section that respects the mesh topology (interior nodes
    included, disjoint patches stay disjoint)."""
    in_slab = np.zeros(conn.max() + 1, dtype=bool)
    in_slab[node_idx] = True
    pairs = conn[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2)
    pairs = pairs[in_slab[pairs].all(axis=1)]
    return np.unique(np.sort(pairs, axis=1), axis=0)


def section_contour(coords, surf_tris, xs):
    """EXACT cross-section contour: intersect every surface triangle with the
    plane x = xs (marching triangles). Returns (nodes, t) with nodes (m, 2, 2)
    and t (m, 2): segment endpoint j of segment i lies on mesh edge
    nodes[i, j] = (a, b) at parameter t[i, j], i.e. P = (1-t) P_a + t P_b.
    The SAME weights interpolate any nodal field (e.g. u_z), so the contour
    deforms consistently with the nodes. Segments form the closed section
    ring(s); order is irrelevant for a LineCollection."""
    d = coords[surf_tris, 0] - xs                            # (nb, 3) signed distances
    d[d == 0.0] = 1e-30                                      # avoid exact-hit degeneracy
    pairs = [(0, 1), (1, 2), (0, 2)]
    cross = np.stack([d[:, i] * d[:, j] < 0 for i, j in pairs], axis=1)  # (nb, 3)
    tri_idx = np.where(cross.sum(axis=1) == 2)[0]            # tris cut by the plane
    if not len(tri_idx):
        return None, None
    nodes = np.empty((len(tri_idx), 2, 2), dtype=np.int64)
    t     = np.empty((len(tri_idx), 2))
    for k, ti in enumerate(tri_idx):                         # 2 crossing edges per tri
        for j, (i0, i1) in enumerate([p for p, c in zip(pairs, cross[ti]) if c]):
            a, b = surf_tris[ti, i0], surf_tris[ti, i1]
            nodes[k, j] = a, b
            t[k, j] = d[ti, i0] / (d[ti, i0] - d[ti, i1])
    return nodes, t


def build_case_roms(basis, tensor):
    """Reduced dynamics + full-space displacement decoders for ONE NiTROM case:
    the VC-H-OpInf baseline built on `basis`, and the trained NiTROM from the
    container output/<basis>_<tensor>. Returns (roms, x0) with roms mapping
    name -> (A_dyn, encoder, lift), or (None, None) if the case is untrained.

    lift maps the reduced trajectory Z (r, Nt) to physical displacements
    (N, 3, Nt): reduced state -> compressed full state (decoder) -> q half ->
    Uc lift -> de-interleave. Mirrors the operator construction of test_nitrom.py.
    """
    nit_dir   = nitrom_output_dir(_case, basis, tensor)
    nit_phi_p = os.path.join(nit_dir, "Phi_nit.npy")
    if not os.path.exists(nit_phi_p):
        print(f"[skip] {basis}_{tensor}: no trained NiTROM in {nit_dir}")
        return None, None

    Phi     = np.load(os.path.join(data_dir, f"Phi_{basis}.npy"))
    Uc      = np.load(os.path.join(data_dir, "Uc.npy"))          # (3N, rc)
    J       = canonical_J(Phi.shape[0])
    n_train = int(np.load(os.path.join(data_dir, "n_train.npy"))[0])
    X_tr    = np.load(os.path.join(data_dir, "x.npy"))[:, :n_train]
    Xt_tr   = np.load(os.path.join(data_dir, "xdot.npy"))[:, :n_train]
    rc      = Uc.shape[1]                                        # q half of the compressed state

    def make_lift(decoder):
        def lift(Z):
            q_comp = (decoder @ Z)[:rc]                          # compressed displacements (rc, Nt)
            q_full = Uc @ q_comp                                 # (3N, Nt), interleaved x,y,z
            return q_full.reshape(-1, 3, q_full.shape[1])        # (N, 3, Nt)
        return lift

    roms = {}

    # VC-H-OpInf baseline on the same basis (Galerkin, learned SPD A_bar)
    A_bar = make_definite(plate_roms.vc_h_operator(Phi, J, [X_tr], [Xt_tr]), sign="pos")
    roms["VC-H-OpInf"] = (plate_roms.opinf_dynamics(Phi, J, A_bar), Phi.T, make_lift(Phi))

    # trained NiTROM (trained basis + operator, symplectic Petrov-Galerkin decoder)
    Phi_nit = np.load(nit_phi_p)
    A2_nit  = np.load(os.path.join(nit_dir, "A2_nit.npy"))
    Jhat    = Phi_nit.T @ np.asarray(J @ Phi_nit)
    label   = "NiTROM-PG" if tensor == "pg" else "NiTROM-OpInf"
    roms[label] = (Jhat @ (A2_nit + A2_nit.T),                   # A_dyn
                   np.asarray(J @ Phi_nit).T,                    # encoder
                   make_lift(-Phi_nit @ np.linalg.inv(Jhat)))    # decoder -> lift

    x0 = np.load(os.path.join(data_dir, "x.npy"))[:, 0]         # compressed FOM IC
    return roms, x0


def animate_slice_deformed(coords, fields, times, node_idx, x_pos, tag, vid_dir,
                           surf_tris=None, edges=None):
    """One video: a SINGLE z-y axes where the cross section (drawn as its slice
    nodes) deflects vertically by scale * u_z, all models overlaid. The
    undeformed section is kept in light gray for reference; the exaggeration is
    set from the FOM so its peak deflection spans ~30% of the section width.
    Axis limits cover the LARGEST deflection over all models, so every dot
    stays in frame (a wilder ROM shrinks the apparent FOM motion instead)."""
    names = list(fields)
    y, z0 = coords[node_idx, 1], coords[node_idx, 2]
    uz = {k: fields[k][node_idx, 2, :] for k in names}
    uz_fom_max = np.abs(uz["FOM (data)"]).max() or 1.0
    uz_all_max = max(np.abs(v).max() for v in uz.values()) or 1.0
    s = SCALE if SCALE is not None else 0.30 * (np.ptp(y) or 1.0) / uz_fom_max
    frames = range(0, len(times), STRIDE)

    # EXACT section contour at the plane x = x_pos: endpoints interpolated on
    # the mesh edges the plane crosses; the same weights interpolate u_z, so
    # the ring bends with the section.
    cn, ct = (None, None)
    if DRAW_CONTOUR and surf_tris is not None:
        cn, ct = section_contour(coords, surf_tris, x_pos)

    def contour_segs(uz_nodes):
        """(m, 2, 2) segments of the contour for full-mesh nodal u_z (scaled)."""
        py = (1 - ct) * coords[cn[..., 0], 1] + ct * coords[cn[..., 1], 1]
        pz = ((1 - ct) * (coords[cn[..., 0], 2] + s * uz_nodes[cn[..., 0]])
              + ct * (coords[cn[..., 1], 2] + s * uz_nodes[cn[..., 1]]))
        return np.stack([py, pz], axis=-1)

    uz_full = {k: fields[k][:, 2, :] for k in names}         # full-mesh u_z (contour interp)

    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    ax.scatter(y, z0, s=10, color="0.85", zorder=1, label="undeformed")
    scats = {k: ax.scatter(y, z0 + s * uz[k][:, 0], s=12, edgecolor="none",
                           label=k, **MODEL_STYLE.get(k, {}))
             for k in names}

    # Wireframe: connect the dots along the mesh edges of the slab (one
    # deforming collection per model + a light-gray undeformed reference).
    wires = {}
    if edges is not None and len(edges):
        loc = np.full(coords.shape[0], -1, dtype=np.int64)   # global -> local ids
        loc[node_idx] = np.arange(len(node_idx))
        e_loc = loc[edges]                                   # (m, 2) local pairs

        def wire_segs(uz_nodes):
            P = np.column_stack([y, z0 + s * uz_nodes])      # deformed slice nodes
            return P[e_loc]                                  # (m, 2, 2) segments

        ax.add_collection(LineCollection(wire_segs(0 * y), colors="0.85",
                                         lw=0.8, zorder=1))
        for k in names:
            lc = LineCollection(wire_segs(uz[k][:, 0]), lw=0.9, alpha=0.85,
                                colors=MODEL_STYLE.get(k, {}).get("color", "k"),
                                zorder=MODEL_STYLE.get(k, {}).get("zorder", 2))
            ax.add_collection(lc)
            wires[k] = lc

    contours = {}
    if cn is not None:
        zeros = np.zeros(coords.shape[0])
        ax.add_collection(LineCollection(contour_segs(zeros), colors="0.8",
                                         lw=1.0, zorder=1))
        for k in names:
            lc = LineCollection(contour_segs(uz_full[k][:, 0]), lw=1.2, alpha=0.9,
                                colors=MODEL_STYLE.get(k, {}).get("color", "k"),
                                zorder=MODEL_STYLE.get(k, {}).get("zorder", 2))
            ax.add_collection(lc)
            contours[k] = lc
    pad_y = 0.08 * (np.ptp(y) or 1.0)
    amp = 1.1 * s * uz_all_max + 0.5 * (np.ptp(z0) or 0.0)
    ax.set_xlim(y.min() - pad_y, y.max() + pad_y)
    ax.set_ylim(np.mean(z0) - amp, np.mean(z0) + amp)
    ax.set_xlabel("y")
    ax.set_ylabel(rf"z + {s:.3g} $\cdot$ $u_z$")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    sup = fig.suptitle("")

    def update(j):
        k_t = frames[j]
        for k in names:
            scats[k].set_offsets(np.column_stack([y, z0 + s * uz[k][:, k_t]]))
            if k in wires:
                wires[k].set_segments(wire_segs(uz[k][:, k_t]))
            if k in contours:
                contours[k].set_segments(contour_segs(uz_full[k][:, k_t]))
        sup.set_text(rf"Slice x = {x_pos:.4g}   ($u_z$ exaggerated {s:.3g}x)   "
                     f"t = {times[k_t]:.4e}")
        return list(scats.values()) + list(wires.values()) + list(contours.values())

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    if shutil.which("ffmpeg"):
        path = os.path.join(vid_dir, f"slice_{tag}_deformed.mp4")
        ani.save(path, writer=animation.FFMpegWriter(fps=FPS, bitrate=1800))
    else:
        path = os.path.join(vid_dir, f"slice_{tag}_deformed.gif")
        ani.save(path, writer=animation.PillowWriter(fps=FPS))
    plt.close(fig)
    return path


def animate_slice(coords, fields, times, node_idx, x_pos, tag, vid_dir):
    """One video: panels (one per model) scattering the slice nodes in (y, z),
    colored by u_z(t) on a shared symmetric scale set by the FOM."""
    names = list(fields)
    y, z = coords[node_idx, 1], coords[node_idx, 2]
    uz = {k: fields[k][node_idx, 2, :] for k in names}           # (n_nodes, Nt)
    vmax = np.abs(uz["FOM (data)"]).max() or 1.0
    frames = range(0, len(times), STRIDE)

    # Equal aspect for chunky sections; auto for thin plate-like ones (an
    # equal-aspect thin strip is unreadable -- the axes still give the scale).
    ry, rz = np.ptp(y), np.ptp(z)
    aspect = "equal" if max(ry, rz) < 5 * max(min(ry, rz), 1e-12) else "auto"

    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.6),
                             sharex=True, sharey=True, constrained_layout=True)
    scats = []
    for ax, k in zip(np.atleast_1d(axes), names):
        s = ax.scatter(y, z, c=uz[k][:, 0], s=18, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, edgecolor="none")
        ax.set_title(k)
        ax.set_xlabel("y")
        ax.set_aspect(aspect)
        scats.append(s)
    np.atleast_1d(axes)[0].set_ylabel("z")
    fig.colorbar(scats[-1], ax=np.atleast_1d(axes).tolist(), shrink=0.85,
                 label=r"$u_z$")
    sup = fig.suptitle("")

    def update(j):
        k_t = frames[j]
        for s, name in zip(scats, names):
            s.set_array(uz[name][:, k_t])
        sup.set_text(f"Slice x = {x_pos:.4g}  ({len(node_idx)} nodes)   "
                     f"t = {times[k_t]:.4e}")
        return scats

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    if shutil.which("ffmpeg"):
        path = os.path.join(vid_dir, f"slice_{tag}.mp4")
        ani.save(path, writer=animation.FFMpegWriter(fps=FPS, bitrate=1800))
    else:
        path = os.path.join(vid_dir, f"slice_{tag}.gif")
        ani.save(path, writer=animation.PillowWriter(fps=FPS))
    plt.close(fig)
    return path


def main():
    # FOM data and slice geometry are shared by every case: load/compute once.
    coords, u_fom, times, surf_tris_, conn = load_fom_data(_case.exo)

    # Slice positions along X, inset from the ends by one half-thickness.
    x = coords[:, 0]
    ext = x.max() - x.min()
    tol = SLICE_FRAC * ext
    x_pos = np.linspace(x.min() + tol, x.max() - tol, N_SLICES)

    # (basis, model name) -> lifted displacement field. The VC-H-OpInf baseline
    # depends only on the basis, so it is integrated once and shared by the two
    # tensor cases of that basis.
    field_cache = {}

    for basis, tensor in CASES:
        roms, x0 = build_case_roms(basis, tensor)
        if roms is None:
            continue
        vid_dir = os.path.join(_case.test_results_dir, basis, "slices", tensor)
        os.makedirs(vid_dir, exist_ok=True)
        print(f"=== case {basis}_{tensor}  ->  {vid_dir}")

        # Integrate each ROM at the data times (dt commensurate -> every sample
        # is an exact symplectic state), then lift to physical displacements
        # (N, 3, Nt).
        fields = {"FOM (data)": u_fom}
        for name, (A_dyn, encoder, lift) in roms.items():
            key = (basis, name)
            if key not in field_cache:
                print(f"Integrating {name} (r = {A_dyn.shape[0]}, dt = {DT_ROM:g})...")
                Z = linear_symplectic_solve(A_dyn, encoder @ x0, times, dt=DT_ROM)
                field_cache[key] = lift(Z)
            fields[name] = field_cache[key]
            e = np.linalg.norm(fields[name] - u_fom) / np.linalg.norm(u_fom)
            print(f"  {name}: relative L2 displacement error vs FOM data: {e:.3e}")

        for i, xs in enumerate(x_pos):
            idx = np.where(np.abs(x - xs) <= tol)[0]
            if len(idx) < 3:
                print(f"slice {i} at x = {xs:.4g}: only {len(idx)} nodes, skipped")
                continue
            if PLOT_MODE == "deformed":
                path = animate_slice_deformed(coords, fields, times, idx, xs,
                                              f"{i}_x{xs:.4g}", vid_dir,
                                              surf_tris=surf_tris_,
                                              edges=slab_edges(conn, idx) if CONNECT_DOTS else None)
            else:
                path = animate_slice(coords, fields, times, idx, xs,
                                     f"{i}_x{xs:.4g}", vid_dir)
            print(f"slice {i}: x = {xs:.4g}, {len(idx)} nodes -> {path}")


if __name__ == "__main__":
    main()
