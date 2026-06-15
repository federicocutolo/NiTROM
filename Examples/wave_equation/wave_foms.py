"""
Shared full-order models for the wave-equation NiTROM example.

Both cases are canonical Hamiltonian systems  x_dot = J grad H = [[0,I],[-I,0]] W x
with  W = blkdiag(-K, I)  and  H = 0.5 x^T W x; they differ only in the spatial
operator K (Dirichlet vs periodic). The case-specific data generators
(Dirichlet/main.py, Periodic/main_gruber.py) and wave_cases.py import from here.
"""
import os
import sys
import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "../.."))   # repo root with the NiTROM package
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from NiTROM.Optimization_Functions import symplectic_solve


class WaveEquationFOM:
    """Homogeneous Dirichlet wave equation u(0)=u(L)=0 on N interior nodes."""

    def __init__(self, N=128, L=1.0, c=0.1):
        self.N, self.L, self.c, = N, L, c
        self.dx = L / (N + 1)
        self.x  = np.linspace(self.dx, L - self.dx, N)
        self.K  = self._build_K_dirichlet()

        # W matrix: H = 0.5 z^T W z where W = blkdiag(-K, I)
        self.W = np.zeros((2*N, 2*N))
        self.W[:N, :N] = -self.K
        self.W[N:, N:] = np.eye(N)

    def _build_K_dirichlet(self):
        # 2nd-order central stencil (1, -2, 1) / dx^2 on the interior nodes.
        # Dirichlet u(0)=u(L)=0 -> the boundary ghosts are zero, so rows 0 and
        # N-1 simply drop their out-of-domain neighbor. K stays symmetric.
        N, dx, c = self.N, self.dx, self.c
        e = np.ones(N)
        K = (np.diag(-2.0*e) + np.diag(e[:-1], 1) + np.diag(e[:-1], -1)) / dx**2
        return c**2 * K

    def rhs(self, t, z):
        N = self.N
        q, p = z[:N], z[N:]
        return np.concatenate([p, self.K @ q])

    def rhs_matrix(self, X):
        """
        Exact time derivative dZ/dt = J grad H = [P; K Q] for a trajectory matrix
        Z of shape (2N, n_t). Column-wise identical to rhs(); used to save analytic
        velocity snapshots for variationally-consistent OpInf (avoids np.gradient).
        """
        N = self.N
        Q, P = X[:N], X[N:]
        return np.concatenate([P, self.K @ Q], axis=0)

    def jacobian(self, z):
        N = self.N
        return np.block([
            [np.zeros((N, N)), np.eye(N)],
            [self.K, np.zeros((N, N))],
        ])

    def integrate(self, z0, t_eval):
        return symplectic_solve(
            rhs=lambda z: self.rhs(None, z),
            jacobian=self.jacobian,
            z0=z0,
            t_eval=t_eval,
        )

    def hamiltonian(self, Z):
        """Z: (2N, n_t) -> H: (n_t,)"""
        N = self.N; Q, P = Z[:N], Z[N:]
        return 0.5*(np.sum(P**2, 0) - np.einsum('ij,ij->j', Q, self.K @ Q))


class PeriodicWaveEquationFOM:
    """Gruber & Tezaur periodic wave equation: the spatial Laplacian wraps around."""

    def __init__(self, N=500, L=1.0, c=0.1):
        self.N, self.L, self.c = N, L, c
        self.x  = np.linspace(0.0, L, N)        # WaveEx grid: linspace incl. endpoints
        self.dx = self.x[1] - self.x[0]
        self.K  = self._build_K_periodic()
        # H = 0.5 z^T W z with W = blkdiag(-K, I)  (== WaveEx's A)
        self.W = np.zeros((2 * N, 2 * N))
        self.W[:N, :N] = -self.K
        self.W[N:, N:] = np.eye(N)

    def _build_K_periodic(self):
        # (c/dx)^2 * periodic second difference (1, -2, 1) with corner wrap-around.
        N, dx, c = self.N, self.dx, self.c
        e = np.ones(N)
        D2 = np.diag(-2.0 * e) + np.diag(e[:-1], 1) + np.diag(e[:-1], -1)
        D2[0, -1] = 1.0                          # periodic wrap
        D2[-1, 0] = 1.0
        return c**2 * D2 / dx**2

    def rhs(self, t, z):
        N = self.N
        q, p = z[:N], z[N:]
        return np.concatenate([p, self.K @ q])

    def rhs_matrix(self, Z):
        """Exact dZ/dt = [P; K Q] for a trajectory matrix Z (2N, n_t)."""
        N = self.N
        Q, P = Z[:N], Z[N:]
        return np.concatenate([P, self.K @ Q], axis=0)

    def jacobian(self, z=None):
        N = self.N
        return np.block([[np.zeros((N, N)), np.eye(N)],
                         [self.K,           np.zeros((N, N))]])

    def integrate(self, z0, t_eval):
        # The FOM is linear: x_dot = J W x = [[0, I],[K, 0]] x. The exact flow over
        # a uniform step is expm(dt * [[0,I],[K,0]]) -- exact and symplectic (energy
        # conserved to machine precision), and much faster than an implicit solver.
        dt   = t_eval[1] - t_eval[0]
        step = expm(dt * self.jacobian(None))
        Z = np.empty((2 * self.N, len(t_eval)))
        Z[:, 0] = z0
        for k in range(1, len(t_eval)):
            Z[:, k] = step @ Z[:, k - 1]
        return Z

    def hamiltonian(self, Z):
        N = self.N
        Q, P = Z[:N], Z[N:]
        return 0.5 * (np.sum(P**2, 0) - np.einsum('ij,ij->j', Q, self.K @ Q))
