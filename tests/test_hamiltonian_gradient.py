import numpy as np
import sys
import types
from pathlib import Path

# --- random test point ---
seed = 42 
rng = np.random.default_rng(seed)
n, r = 8, 2
ns   = 100  # number of snapshots
nt = 10  # number of trajectories

Phi   = rng.standard_normal((n, r))   # random test point
X     = rng.standard_normal((n, ns))  # random snapshots
Z     = rng.standard_normal((r, ns))  # random latent coords

# --- build operators ---
half_dim = n // 2
J = np.bmat([[np.zeros((half_dim, half_dim)), np.eye(half_dim)],
            [-np.eye(half_dim), np.zeros((half_dim, half_dim))]])
J = np.asarray(J)  # convert to numpy array
print("J:\n", J.shape, "\n", J, "\n", J.dtype) 

U = Phi
JU = J @ U
M  = (J @ Phi).T @ Phi
M_inv = np.linalg.inv(M)
D = Phi @ M_inv
M_inv_T = M_inv.T

delta = rng.standard_normal((n, r))
delta /= np.linalg.norm(delta)
eps   = 1e-5

def cost_error(Phi_, X, Z):
    U = np.asarray(Phi_)
    M_ = (J @ U).T @ U
    M_inv_ = np.linalg.inv(M_)
    D_ = U @ M_inv_
    E_ = X - D_ @ Z
    return np.sum(E_ * E_)


def grad_Phi_cost_error(Phi_, X, Z):
    U = np.asarray(Phi_)
    M_ = (J @ U).T @ U
    M_inv_ = np.linalg.inv(M_)
    D_ = U @ M_inv_
    E_ = X - D_ @ Z
    G = np.zeros_like(Phi_)
    for i in range(X.shape[1]):
        zi = Z[:, i]
        xi = X[:, i]
        M_inv_zi = M_inv_ @ zi
        Dzi = D_ @ zi
        ei = E_[:, i]
        DTei = D_.T @ ei

        term1 = np.outer(ei, M_inv_zi)        # n x r            
        term2 = np.outer(J.T @ Dzi, DTei)     # nxn @ nxr @ rx1, (rxn @ nx1)^T -> n x r
        term3 = np.outer(JU @ DTei, M_inv_zi) # n x r

        G += -2 * (term1 - term2 - term3)

    return G

fd       = (cost_error(Phi+eps*delta, X, Z) - cost_error(Phi-eps*delta, X, Z)) / (2*eps)
analytic = np.sum(grad_Phi_cost_error(Phi, X, Z) * delta)

# print("Test 1: static gradient check on random point")
# print(f"FD      : {fd:.16f}")
# print(f"analytic: {analytic:.16f}")
# print(f"rel err : {abs(fd - analytic) / abs(fd):.2e}")

# --- full Hamiltonian gradient check ---
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

sys.modules.setdefault("mpi4py", types.SimpleNamespace(MPI=types.SimpleNamespace(INT=None)))
sys.modules.setdefault(
    "pymanopt",
    types.SimpleNamespace(
        function=types.SimpleNamespace(numpy=lambda manifold: (lambda fn: fn))
    ),
)

from NiTROM.Optimization_Functions import classes


class DummyComm:
    def Get_size(self):
        return 1

    def Get_rank(self):
        return 0

    def Allgather(self, sendbuf, recvbuf):
        recvbuf[0][...] = np.asarray(sendbuf[0])

    def allgather(self, value):
        return [value]


class DummyPool:
    def __init__(self, X, time):
        self.comm = DummyComm()
        self.size = 1
        self.rank = 0
        self.my_n_traj = X.shape[0]
        self.X = X
        self.time = time
        self.F = np.zeros((X.shape[1], X.shape[0]))
        self.weights = np.ones(self.my_n_traj)


class IdentityFOM:
    def compute_output(self, q):
        return q

    def compute_output_derivative(self, q):
        return np.eye(q.shape[0])


def random_phi(rng, n=n, r=r):
    half = n // 2
    J = np.block([
        [np.zeros((half, half)), np.eye(half)],
        [-np.eye(half), np.zeros((half, half))],
    ])
    while True:
        Phi = rng.standard_normal((n, r))
        if abs(np.linalg.det((J @ Phi).T @ Phi)) > 1e-2:
            return Phi


def centered_difference(loss, Phi, delta, eps=1e-5):
    # Centered difference for *Phi*
    return (loss(Phi + eps * delta) - loss(Phi - eps * delta)) / (2 * eps)

# Build Hamiltonian gradient check

def build_hamiltonian_problem():
    time = np.linspace(0.0, 1.0, ns)
    A2 = np.array([[0.8, -0.3], [0.4, 0.6]])
    A3 = np.array([[[0.5, -0.2], [0.1, 0.3]], 
                   [[-0.4, 0.2], [0.2, -0.1]]])

    Phi_data = random_phi(rng, n=n, r=r)
    Phi_test = random_phi(rng, n=n, r=r)
    Psi_test = Phi_test.copy()

    rom_data = classes.HamiltonianROM(operators=[A2, A3], poly_comp=[1], Phi=Phi_data)
    X0 = rng.standard_normal((nt, n))
    print("X0 shape:", X0.shape)  # Should be (nt, n)
    Z0 = np.array([rom_data.encode(x0) for x0 in X0])  # shape (nt, r)
    Z = []

    for z0_traj in Z0:
        z_traj = rom_data.integrate(time, z0_traj)
        # print("z_traj shape:", z_traj.shape)
        Z.append(z_traj)

    Z = np.stack(Z, axis=0)
    print("Z shape:", Z.shape)  # Should be (nt, r, len(time))
    X_data = rom_data.reconstruct(Z)
    print("X_data shape:", X_data.shape)  # Should be (n_traj, n, len(time))
    pool = DummyPool(X_data, time)

    opt_obj = classes.optimization_objects(
        pool,
        which_trajs=np.arange(X_data.shape[0]),
        which_times=np.arange(len(time)),
        leggauss_deg=3,
        nsave_rom=4,
        poly_comp=[1],
        hamiltonian=True,
    )

    fom = IdentityFOM()

    delta = rng.standard_normal(Phi_test.shape)
    delta /= np.linalg.norm(delta)
    # print("delta", delta)

    return opt_obj, pool, fom, Phi_data, Phi_test, Psi_test, (A2, A3), delta, X_data, time

opt_obj, pool, fom, Phi_data, Phi_test, Psi_test, tensors, delta, X_data, time = build_hamiltonian_problem()

print("Active weights:", opt_obj.weights)
rom = opt_obj.build_rom(Phi_test, None, operators=tensors)
objective = opt_obj.build_objective(rom, fom, pool)
cost = objective.trajectory_cost(0)
tg = objective.trajectory_gradient(0)
# print("Trajectory gradient keys:", tg.keys())

Xk = opt_obj.X[0]
grad_static = tg["grad_Phi_static"]
# print("grad_static", grad_static)
analytic_static = 0.0
z_ref = []

# Build Z from X0_data
for traj in range(opt_obj.my_n_traj):
    grad_traj = objective.trajectory_gradient(traj)
    analytic_static += np.sum(grad_traj["grad_Phi_static"] * delta)
    z0_traj = rom.encode(opt_obj.X[traj, :, 0])
    z_ref.append(rom.integrate(opt_obj.time, z0_traj))
Z_ref = np.stack(z_ref, axis=0)

def frozen_static_cost(Phi_, opt_obj, tensors, Z_ref):
    rom = opt_obj.build_rom(Phi_, None, operators=tensors)
    D_ = rom.build_projection_operators()["decoder"]
    total = 0.0
    for traj in range(opt_obj.my_n_traj):
        x_traj = opt_obj.X[traj]
        z_traj = Z_ref[traj]
        e_traj = x_traj - D_ @ z_traj
        total += np.sum(e_traj * e_traj) / opt_obj.weights[traj]
    return total

fd_static = centered_difference(
    lambda Phi_: frozen_static_cost(Phi_, opt_obj, tensors, Z_ref),
    Phi_test,
    delta,
)

rel_err_static = abs(fd_static - analytic_static) / max(abs(fd_static), 1e-14)
print("\n#---- Static check ----#")
print(f"FD static           : {fd_static:.10f}")
print(f"analytic static     : {analytic_static:.10f}")
print(f"rel err static      : {rel_err_static:.2e}")

# Check dynamic gradient via Directional derivative wrt U of integrand function -\lambda^\top \hat{J}\nabla{\hatH}}(z)
lam = np.ones(r)                           # dummy lambdas to get a scalar cost

def frozen_dynamic_cost(Phi_, opt_obj, tensors, Z_ref, lam=lam):
    rom = opt_obj.build_rom(Phi_, None, operators=tensors)
    Jhat = rom.build_projection_operators()["Jhat"]
    total = 0.0
    for traj in range(opt_obj.my_n_traj):
        for snap in range(opt_obj.n_snapshots):
            nabla_H_k = rom.compute_hamiltonian_gradient(Z_ref[traj, :, snap])
            total += lam.T @ Jhat @ nabla_H_k
    return total

fd_dynamic = centered_difference(
    lambda Phi_: frozen_dynamic_cost(Phi_, opt_obj, tensors, Z_ref=Z_ref),
    Phi_test,
    delta,
)

def analytical_dynamic_grad(Phi_, opt_obj, tensors, Z_ref, lam):
    rom = opt_obj.build_rom(Phi_, None, operators=tensors)
    J = rom.build_projection_operators()["J"]
    U = Phi_
    total = np.zeros_like(U)
    for traj in range(opt_obj.my_n_traj):
        for snap in range(opt_obj.n_snapshots):
            nabla_H_k = rom.compute_hamiltonian_gradient(Z_ref[traj, :, snap])
            total += J.T @ U @ np.outer(lam, nabla_H_k) + J @ U @ np.outer(nabla_H_k, lam)
    return total

grad_dynamic = analytical_dynamic_grad(Phi_test, opt_obj, tensors, Z_ref, lam)

analytic_dynamic = np.sum(grad_dynamic * delta)
rel_err_dynamic = abs(fd_dynamic - analytic_dynamic) / max(abs(fd_dynamic), 1e-14)

print("\n#---- Dynamic check ----#")
print(f"FD dynamic          : {fd_dynamic:.10f}")
print(f"analytic dynamic    : {analytic_dynamic:.10f}")
print(f"rel err dynamic     : {rel_err_dynamic:.2e}")


def full_cost(Phi_, opt_obj, tensors, fom, pool):
    rom_ = opt_obj.build_rom(Phi_, None, operators=tensors)
    objective_ = opt_obj.build_objective(rom_, fom, pool)
    return sum(objective_.trajectory_cost(k) for k in range(opt_obj.my_n_traj))


fd_full = centered_difference(
    lambda Phi_: full_cost(Phi_, opt_obj, tensors, fom, pool),
    Phi_test,
    delta,
)


analytic_static = 0.0
analytic_dynamic = 0.0
analytic_terminal = 0.0
for k in range(opt_obj.my_n_traj):
    tg_k = objective.trajectory_gradient(k)
    analytic_static += np.sum(tg_k["grad_Phi_static"] * delta)
    analytic_dynamic += np.sum(tg_k["grad_Phi_dynamic"] * delta)
    analytic_terminal += np.sum(tg_k["grad_Phi_terminal"] * delta)

analytic_full = analytic_static + analytic_dynamic + analytic_terminal
rel_err_full = abs(fd_full - analytic_full) / max(abs(fd_full), 1e-14)

print("\n#---- Full gradient check ----#")
print(f"FD full                : {fd_full:.16f}")
print(f"analytic full          : {analytic_full:.16f}")
print(f"    static term            : {analytic_static:.16f}")
print(f"    dynamic term           : {analytic_dynamic:.16f}")
print(f"    terminal term          : {analytic_terminal:.16f}")
print(f"rel err full           : {rel_err_full:.2e}")


# Check gradient of operators A2 
delta_A2 = rng.standard_normal((r, r))
delta_A2 /= np.linalg.norm(delta_A2)

def centered_difference_A2(loss, A2, delta, eps=1e-5):
    A2, A3 = tensors
    # Centered difference for *A2* 
    return (loss(A2 + eps * delta) - loss(A2 - eps * delta)) / (2 * eps)


def cost_A2(A2_, opt_obj, tensors, Z_ref=Z_ref, lam=lam):
    A2, A3 = tensors
    rom = opt_obj.build_rom(Phi_test, None, operators=(A2_, A3))
    Jhat = rom.build_projection_operators()["Jhat"]
    total = 0.0
    for traj in range(opt_obj.my_n_traj):
        for snap in range(opt_obj.n_snapshots):
            total += lam.T @ Jhat @ (A2_ + A2_.T) @ Z_ref[traj, :, snap]
    return total

fd_A2 = centered_difference_A2(
    lambda A2_: cost_A2(A2_, opt_obj, tensors, Z_ref, lam),
    tensors[0],
    delta_A2,
)


def analytical_grad_A2(Phi_, opt_obj, tensors, Z_ref, lam):
    rom = opt_obj.build_rom(Phi_, None, operators=tensors)
    Jhat = rom.build_projection_operators()["Jhat"]
    total = np.zeros_like(tensors[0])
    for traj in range(opt_obj.my_n_traj):
        for snap in range(opt_obj.n_snapshots):
            z = Z_ref[traj, :, snap]
            total += np.outer(Jhat.T @ lam, z) + np.outer(z, lam) @ Jhat
    return total

grad_dynamic_A2 = analytical_grad_A2(Phi_test, opt_obj, tensors, Z_ref, lam)
analytic_dynamic_A2 = np.sum(grad_dynamic_A2 * delta_A2)
rel_err_dynamic_A2 = abs(fd_A2 - analytic_dynamic_A2) / max(abs(fd_A2), 1e-14)

print("\n#---- Dynamic check for A2 ----#")
print(f"FD dynamic A2       : {fd_A2:.10f}")
print(f"analytic dynamic A2 : {analytic_dynamic_A2:.10f}")
print(f"rel err dynamic A2  : {rel_err_dynamic_A2:.2e}")


# Check gradient of operators A3
delta_A3 = rng.standard_normal((r, r, r))
delta_A3 /= np.linalg.norm(delta_A3)

def centered_difference_A3(loss, A3, delta, eps=1e-5):
    A2, A3 = tensors
    # Centered difference for *A3* 
    return (loss(A3 + eps * delta) - loss(A3 - eps * delta)) / (2 * eps)


def cost_A3(A3_, opt_obj, tensors, Z_ref=Z_ref, lam=lam):
    A2, A3 = tensors
    rom = opt_obj.build_rom(Phi_test, None, operators=(A2, A3_))
    Jhat = rom.build_projection_operators()["Jhat"]
    total = 0.0
    for traj in range(opt_obj.my_n_traj):
        for snap in range(opt_obj.n_snapshots):
            z = Z_ref[traj, :, snap]
            result = (np.einsum('ijk, j, k -> i', A3_, z, z)   # H_ijk  z_j z_k
                    + np.einsum('jik, j, k -> i', A3_, z, z)   # H_jik  z_j z_k
                    + np.einsum('kji, j, k -> i', A3_, z, z)   # H_kji  z_j z_k
                    )
            total += lam.T @ Jhat @ result
    return total

fd_A3 = centered_difference_A3(
    lambda A3_: cost_A3(A3_, opt_obj, tensors, Z_ref, lam),
    tensors[1],
    delta_A3,
)


def analytical_grad_A3(Phi_, opt_obj, tensors, Z_ref, lam):
    rom = opt_obj.build_rom(Phi_, None, operators=tensors)
    Jhat = rom.build_projection_operators()["Jhat"]
    total = np.zeros_like(tensors[1])          
    lam_Jhat = Jhat.T @ lam                    # J^T λ, shape (r,)  [= (Jhat^T λ)]
    for traj in range(opt_obj.my_n_traj):
        for snap in range(opt_obj.n_snapshots):
            z = Z_ref[traj, :, snap]
            total += (np.einsum('i,j,k->ijk', lam_Jhat, z, z)  
                    + np.einsum('i,j,k->ijk', z, lam_Jhat, z)
                    + np.einsum('i,j,k->ijk', z, z, lam_Jhat)
                    )
    return total


grad_dynamic_A3     = analytical_grad_A3(Phi_test, opt_obj, tensors, Z_ref, lam)
analytic_dynamic_A3 = np.sum(grad_dynamic_A3 * delta_A3)
rel_err_dynamic_A3  = abs(fd_A3 - analytic_dynamic_A3) / max(abs(fd_A3), 1e-14)

print("\n#---- Dynamic check for A3 ----#")
print(f"FD dynamic A3       : {fd_A3:.10f}")
print(f"analytic dynamic A3 : {analytic_dynamic_A3:.10f}")
print(f"rel err dynamic A3  : {rel_err_dynamic_A3:.2e}")