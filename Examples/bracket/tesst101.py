import numpy as np
import scipy as sp
from scipy.io import mmread
from scipy.sparse import load_npz, csc_matrix, eye, bmat
from scipy.sparse.linalg import spsolve

fe_dir = "/Users/federicocutolo/WORKSPACE/CODES/NiTROM/Examples/bracket/bracket_for_Fede"
M = csc_matrix(mmread(fe_dir + "/mass.mm"))
K = csc_matrix(mmread(fe_dir + "/stiff.mm"))

Minv = spsolve(M, eye(M.shape[0], format="csc"))          
A_fom_full = bmat([[K, None], [None, Minv]], format="csc")
print(f"assembled A_fom_full: {A_fom_full.shape}, nnz={A_fom_full.nnz:,} "
      f"(K {K.shape} nnz={K.nnz:,}, M {M.shape} nnz={M.nnz:,})")

n = A_fom_full.shape[0] // 2                              
I = eye(n, format="csc")
J = bmat([[None, I], [-I, None]], format="csc")
print(f"J: {J.shape}, skew check ||J + J^T|| = {abs(J + J.T).max():.1e}")

# POD of the full-space Cayley trajectory (2*3N x Nt)
X = np.load("/Users/federicocutolo/WORKSPACE/CODES/NiTROM/Examples/bracket/"
            "Bracket/trajectories/fom_cayley_x.npy")
print(f"loaded fom_cayley_x {X.shape} from disk")
U_pod, s_pod, _ = np.linalg.svd(X, full_matrices=False)
energy = np.cumsum(s_pod ** 2) / np.sum(s_pod ** 2)
print(f"POD of fom_cayley_x {X.shape}: "
      f"99% energy at rank {int(np.argmax(energy > 0.99)) + 1} / {len(s_pod)}")

# Exodus (FE solver) data: x = [q; p], p = M q_dot -- same assembly as
# prepare_data -- and its POD, for comparison with the Cayley POD above.
import exodusii
exo   = exodusii.ExodusIIFile(fe_dir + "/bracket_velIC_100.e", mode="r")
N_nod, Nt = exo.num_nodes(), exo.num_times()
q_exo    = np.zeros((3 * N_nod, Nt))
qdot_exo = np.zeros((3 * N_nod, Nt))
for c, comp in enumerate("xyz"):
    for k in range(Nt):
        q_exo[c::3, k]    = exo.get_node_variable_values(f"disp_{comp}", k + 1)
        qdot_exo[c::3, k] = exo.get_node_variable_values(f"solution_dot_{comp}", k + 1)
X_exo = np.vstack([q_exo, M @ qdot_exo])
print(f"loaded Exodus data {X_exo.shape}")

U_pod_exo, s_pod_exo, _ = np.linalg.svd(X_exo, full_matrices=False)
energy_exo = np.cumsum(s_pod_exo ** 2) / np.sum(s_pod_exo ** 2)
print(f"POD of Exodus data {X_exo.shape}: "
      f"99% energy at rank {int(np.argmax(energy_exo > 0.99)) + 1} / {len(s_pod_exo)}")

r = np.argmax(energy_exo > 0.99) + 1
# r = 100
if r % 2 != 0:
    r += 1  
print(f"using r={r} (even) for the reduced basis")
U = U_pod_exo[:, :2 * r]          # leading 2r POD modes (even count, orthonormal)
print(U.shape)
# U = sp.linalg.orth(np.random.randn(2 * n , 2 * r))  # random test basis
print(np.linalg.norm(U.T @ U - np.eye(2 * r)))  # check orthonormality
E = (J @ U).T
D = U @ np.linalg.inv(E @ U)
print(f"condition number of E @ U = {np.linalg.cond(E @ U):.3e}")
A_rom = E @ J @ (A_fom_full + A_fom_full.T) @ D / 2

evals, _ = np.linalg.eig(A_rom)
print(f"max Re(evals) = {evals.real.max():.3e}")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(evals.real, evals.imag, "ko")
ax = plt.gca()
ax.set_xlabel("Re")
ax.set_ylabel("Im")
# ax.vlines(0, color="r", lw=0.5)
plt.tight_layout()

# ── Integrate the ROM with the Cayley symplectic integrator ───────────────────
# IC: encoded FOM initial state z0 = E @ x_FOM(0) (Exodus data). dt is
# commensurate with the snapshot spacing (1e-4 = 4000*dt), so every sample is
# an exact integrator state (knot fast path, no spline).
import sys, os
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../..")))
from NiTROM.Optimization_Functions.linear_symplectic_integrators import linear_symplectic_solve

t_exo  = np.asarray(exo.get_times())
z0     = E @ X_exo[:, 0]
DT_ROM = 2.5e-8
Z_rom  = linear_symplectic_solve(A_rom, z0, t_exo, dt=DT_ROM)
X_rom  = D @ Z_rom                                    # reconstruct in full space

err = np.linalg.norm(X_exo - X_rom, axis=0) / (np.linalg.norm(X_exo, axis=0) + 1e-300)
print(f"relative error vs FOM data: max = {err.max():.3e}, final = {err[-1]:.3e}")

plt.figure(figsize=(6, 4.5))
plt.semilogy(t_exo[1:], err[1:], "k-", lw=1.5)
plt.xlabel("t")
plt.ylabel(r"$\|x_{\rm FOM}(t) - x_{\rm ROM}(t)\| \, / \, \|x_{\rm FOM}(t)\|$")
plt.title(f"PG ROM (Exodus POD, r = {2 * r}) vs FOM data")
plt.tight_layout()

# ── Tip displacement (x, y, z) vs time, t > 0: Exodus data vs Exodus ROM ──────
tip = int(np.load("/Users/federicocutolo/WORKSPACE/CODES/NiTR" \
"OM/Examples/bracket/"
                  "Bracket/trajectories/tip_node.npy")[0])
u_tip_exo = X_exo[3 * tip: 3 * tip + 3, :]            # (3, Nt) tip displacement (q half)
u_tip_rom = X_rom[3 * tip: 3 * tip + 3, :]

fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True, constrained_layout=True)
for c, ax_c in enumerate(axes):
    ax_c.plot(t_exo[1:], u_tip_exo[c, 1:], "k-",  lw=1.6, label="Exodus (FE data)")
    ax_c.plot(t_exo[1:], u_tip_rom[c, 1:], "r--", lw=1.2, label="Exodus-POD ROM")
    ax_c.set_ylabel(rf"tip $u_{'xyz'[c]}$")
axes[0].legend(fontsize=9)
axes[0].set_title(f"Tip displacement (node {tip}), t > 0")
axes[-1].set_xlabel("t")

# ── Same pipeline on the CAYLEY-integrated data ───────────────────────────────
# Basis from the Cayley POD (U_pod / `energy`, computed above), reference
# trajectory X = fom_cayley_x, IC z0 = E_cay @ x_cay(0). Same time grid.
r_cay = 27 #int(np.argmax(energy > 0.99)) + 1
if r_cay % 2 != 0:
    r_cay += 1
U_cay = U_pod[:, :2 * r_cay]
E_cay = (J @ U_cay).T
D_cay = U_cay @ np.linalg.inv(E_cay @ U_cay)
print(f"[cayley] r={r_cay} (even), cond(E @ U) = {np.linalg.cond(E_cay @ U_cay):.3e}")
A_rom_cay = E_cay @ J @ (A_fom_full + A_fom_full.T) @ D_cay / 2
ev_cay = np.linalg.eigvals(A_rom_cay)
print(f"[cayley] max Re(evals) = {ev_cay.real.max():.3e}")

Z_rom_cay = linear_symplectic_solve(A_rom_cay, E_cay @ X[:, 0], t_exo, dt=DT_ROM)
X_rom_cay = D_cay @ Z_rom_cay

err_cay = np.linalg.norm(X - X_rom_cay, axis=0) / (np.linalg.norm(X, axis=0) + 1e-300)
print(f"[cayley] relative error vs Cayley data: max = {err_cay.max():.3e}, "
      f"final = {err_cay[-1]:.3e}")

plt.figure(figsize=(6, 4.5))
plt.semilogy(t_exo[1:], err_cay[1:], "k-", lw=1.5)
plt.xlabel("t")
plt.ylabel(r"$\|x_{\rm FOM}(t) - x_{\rm ROM}(t)\| \, / \, \|x_{\rm FOM}(t)\|$")
plt.title(f"PG ROM (Cayley POD, r = {2 * r_cay}) vs Cayley data")
plt.tight_layout()
plt.show()

