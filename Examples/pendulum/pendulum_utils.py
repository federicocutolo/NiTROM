import numpy as np


m = 1.0
l = 1.0
g = 9.81
J = np.array([[0.0, 1.0], [-1.0, 0.0]])
SRKN_REFERENCE_STEP = 0.01


def hamiltonian(z):
    q, p = z
    return p**2 / (2.0 * m * l**2) + m * g * l * (1.0 - np.cos(q))


def qdot(p):
    return p / (m * l * l)


def pdot(q):
    return -m * g * l * np.sin(q)


def srkn_pendulum_b6(x0, t0, tf, h, t_eval):
    a = [
        0.245298957184271,
        0.604872665711080,
        0.5 - (0.245298957184271 + 0.604872665711080),
        0.5 - (0.245298957184271 + 0.604872665711080),
        0.604872665711080,
        0.245298957184271,
    ]
    b = [
        0.0829844064174052,
        0.396309801498368,
        -0.0390563049223486,
        1 - 2 * (0.0829844064174052 + 0.396309801498368 + -0.0390563049223486),
        -0.0390563049223486,
        0.396309801498368,
        0.0829844064174052,
    ]

    if h <= 0:
        raise ValueError("srkn_pendulum_b6 expects a positive step size.")
    if len(x0) != 2:
        raise ValueError("srkn_pendulum_b6 expects x0 = [q0, p0].")

    n_timesteps = max(1, int(np.ceil((tf - t0) / h)))
    h = (tf - t0) / n_timesteps
    tsim = np.linspace(t0, tf, n_timesteps + 1)

    X_full = np.zeros((2, len(tsim)))
    H_full = np.zeros(len(tsim))
    X_full[:, 0] = x0
    H_full[0] = hamiltonian(x0)

    q, p = x0.astype(float)
    for step in range(1, len(tsim)):
        for stage in range(len(a)):
            q = q + h * b[stage] * qdot(p)
            p = p + h * a[stage] * pdot(q)

        q = q + h * b[-1] * qdot(p)

        z = np.array([q, p])
        X_full[:, step] = z
        H_full[step] = hamiltonian(z)

    if np.array_equal(tsim, t_eval):
        return X_full, H_full

    X_interp = np.vstack([
        np.interp(t_eval, tsim, X_full[0]),
        np.interp(t_eval, tsim, X_full[1]),
    ])
    H_interp = np.interp(t_eval, tsim, H_full)
    return X_interp, H_interp
