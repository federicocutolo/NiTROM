"""
Compare the raw FOM data (Exodus, FE time-stepper) against the symplectic
re-integration of the same dynamics (fom_cayley_x.npy, integrate_fom.py):
relative L2 difference over time + tip-displacement overlay.

Both trajectories share the same initial condition and operators; they differ
only in the time integrator (damped FE solver vs energy-exact Cayley), so the
gap IS the integrator discrepancy.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse import csc_matrix
import exodusii

HERE    = os.path.dirname(os.path.abspath(__file__))
fe_dir  = os.path.join(HERE, "bracket_for_Fede")
out_dir = os.path.join(HERE, "Bracket", "trajectories")

# ── Exodus data: x = [q; p], p = M q_dot (same assembly as prepare_data) ──────
exo   = exodusii.ExodusIIFile(os.path.join(fe_dir, "bracket_velIC_100.e"), mode="r")
N, Nt = exo.num_nodes(), exo.num_times()
t_exo = np.asarray(exo.get_times())

q    = np.zeros((3 * N, Nt))
qdot = np.zeros((3 * N, Nt))
for c, comp in enumerate("xyz"):
    for k in range(Nt):
        q[c::3, k]    = exo.get_node_variable_values(f"disp_{comp}", k + 1)
        qdot[c::3, k] = exo.get_node_variable_values(f"solution_dot_{comp}", k + 1)

M     = csc_matrix(mmread(os.path.join(fe_dir, "mass.mm")))
x_exo = np.vstack([q, M @ qdot])
print(f"Exodus data: {x_exo.shape}, t in [{t_exo[0]:g}, {t_exo[-1]:g}]")

# ── Cayley re-integration of the same FOM ─────────────────────────────────────
x_cay = np.load(os.path.join(out_dir, "fom_cayley_x.npy"))
t_cay = np.load(os.path.join(out_dir, "fom_cayley_time.npy"))
assert x_cay.shape == x_exo.shape and np.allclose(t_cay, t_exo), \
    f"grids differ: cayley {x_cay.shape} vs exodus {x_exo.shape}"
print(f"Cayley data: {x_cay.shape}")

# ── Relative L2 difference, full state and per q/p half ───────────────────────
n3 = 3 * N
rel = lambda A, B: np.linalg.norm(A - B, axis=0) / (np.linalg.norm(B, axis=0) + 1e-300)
e_full = rel(x_cay, x_exo)
e_q    = rel(x_cay[:n3], x_exo[:n3])
e_p    = rel(x_cay[n3:], x_exo[n3:])
frob   = np.linalg.norm(x_cay - x_exo) / np.linalg.norm(x_exo)
print(f"relative L2 (Frobenius, all t): {frob:.3e}")
print(f"relative L2 at final time:      full={e_full[-1]:.3e}  q={e_q[-1]:.3e}  p={e_p[-1]:.3e}")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.semilogy(t_exo[1:], e_full[1:], "k-",  lw=1.6, label=r"full state $x$")
ax.semilogy(t_exo[1:], e_q[1:],    "k--", lw=1.2, label=r"$q$ half")
ax.semilogy(t_exo[1:], e_p[1:],    "k:",  lw=1.2, label=r"$p$ half")
ax.set_xlabel("t")
ax.set_ylabel(r"$\|x_{\rm Cayley}(t) - x_{\rm Exodus}(t)\| / \|x_{\rm Exodus}(t)\|$")
ax.set_title("FE data vs symplectic re-integration")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "test102_rel_error.png"), dpi=150)
fig.savefig(os.path.join(HERE, "test102_rel_error.pdf"), bbox_inches="tight")
plt.close(fig)

# ── Tip displacement (node saved by prepare_data; dominant component) ─────────
tip  = int(np.load(os.path.join(out_dir, "tip_node.npy"))[0])
u_exo = q[3 * tip: 3 * tip + 3]                        # (3, Nt) exodus tip displacement
u_cay = x_cay[3 * tip: 3 * tip + 3]
comp  = int(np.argmax(np.abs(u_exo).max(axis=1)))      # dominant x/y/z component

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(t_exo, u_exo[comp], "k-",  lw=1.6, label="Exodus (FE data)")
ax.plot(t_cay, u_cay[comp], "r--", lw=1.2, label="Cayley (symplectic)")
ax.set_xlabel("t")
ax.set_ylabel(rf"tip $q_{'xyz'[comp]}$")
ax.set_title(f"Tip displacement (node {tip})")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "test102_tip_disp.png"), dpi=150)
fig.savefig(os.path.join(HERE, "test102_tip_disp.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved test102_rel_error / test102_tip_disp (png+pdf) in {HERE}")

# ── 1D displacement profiles at 5 time instants ───────────────────────────────
# Profile = dominant displacement component of EVERY node vs its coordinate
# along the long (beam) axis; Exodus and Cayley overlaid per instant. Nodes at
# the same axial station (whole cross sections) appear as vertical spreads.
coords = np.load(os.path.join(out_dir, "coords.npy"))       # (N, 3) node coords
axis   = int(np.argmax(coords.max(0) - coords.min(0)))      # long axis
s_ax   = coords[:, axis]
order  = np.argsort(s_ax)

k_snap = np.linspace(0, Nt - 1, 5).astype(int)
fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharex=True, sharey=True,
                         constrained_layout=True)
for ax_p, k in zip(axes, k_snap):
    ax_p.plot(s_ax[order], q[comp::3, k][order],     "k.", ms=2.5,
              label="Exodus (FE data)")
    ax_p.plot(s_ax[order], x_cay[comp::3, k][order], "r.", ms=2.0, alpha=0.6,
              label="Cayley (symplectic)")
    ax_p.set_title(f"t = {t_exo[k]:.4g}", fontsize=11)
    ax_p.set_xlabel(f"${'xyz'[axis]}$")
axes[0].set_ylabel(rf"$u_{'xyz'[comp]}$")
axes[0].legend(fontsize=9, markerscale=3)
fig.suptitle(f"Nodal $u_{'xyz'[comp]}$ profiles along the {'xyz'[axis]}-axis", fontsize=13)
fig.savefig(os.path.join(HERE, "test102_profiles.png"), dpi=150)
fig.savefig(os.path.join(HERE, "test102_profiles.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Saved test102_profiles (png+pdf) in {HERE}")
