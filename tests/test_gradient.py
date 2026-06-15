import numpy as np
import numpy.testing as npt

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT) 

from NiTROM.Optimization_Functions.classes import HamiltonianROM

def fd(loss, variable, direction, eps=1e-7):
    return (loss(variable + eps * direction) - loss(variable - eps * direction)) / (2 * eps)

# ------------------------------------------------------
# Test 1: Gradient of a quadratic form f(x) = x^T A x wrt x.
# ------------------------------------------------------
A = np.array([[0.1, 0.5], [0.2, 0.9]])
x = np.array([1.0, 2.0])
dx = np.array([0.3, -0.4])
dx /= np.linalg.norm(dx)

grad_x_analytic = (A + A.T) @ x             # Analytical gradient     
analytic_derivative = grad_x_analytic @ dx  # Analytical directiona derivative

def loss(x_):   
    return x_.T @ A @ x_

numerical_derivative = fd(loss, x, dx)      # Finite differences directional derivative

print("Test 1: Gradient of f(x) = x^T A x")
print(f"Analytic: {analytic_derivative}, Numerical: {numerical_derivative}")

# ------------------------------------------------------
# Test 2: Gradient of L = ||xi - D @ zi||^2 wrt Phi.
# ------------------------------------------------------

alpha = 1
Phi = np.array([
    [1.0, 0.0],
    [0.1, 0.2],
    [0.0, 1.0],
    [0.3, -0.1],
])

xi = np.array([1.2, -0.5, 0.8, 0.3])
zi = np.array([0.4, -0.7])
dPhi = np.array([
    [0.2, -0.1],
    [0.3, 0.4],
    [-0.5, 0.2],
    [0.1, -0.3],
])

dPhi /= np.linalg.norm(dPhi)
rom = HamiltonianROM(operators=[], poly_comp=[], Phi=Phi)
ops = rom.build_projection_operators()

J = ops["J"]
U = Phi
JU = ops["JU"]
D = ops["decoder"]
M = ops["M"]
M_inv_T = ops["M_inv_T"]
M_inv = np.linalg.inv(M)
M_inv_zi = M_inv @ zi
Dzi = D @ zi

ei = xi - Dzi

term1 = np.outer(ei, M_inv_zi)                    
term2 = np.outer(J.T @ Dzi, D.T @ ei)               
term3 = np.outer(JU @ D.T @ ei, M_inv_zi)            

grad_Phi_analytical = -(2 / alpha) * (term1 - term2 - term3)
analytical = np.sum(grad_Phi_analytical * dPhi)     # Analytical directional derivative

def loss(Phi_):
    U_ = Phi_
    JU_ = J @ U_
    M_ = JU_.T @ U_
    M_inv_ = np.linalg.inv(M_)
    D_ = U_ @ M_inv_
    ei = xi - D_ @ zi
    return (ei @ ei) / alpha

# Frobenius inner product <G, dPhi> = sum(G * dPhi).
numerical = fd(loss, Phi, dPhi)                     # Finite differences directional derivative
print("\nTest 2: Gradient of L = 1/alpha * ||xi - Phi @ zi||^2 with respect to Phi")
print(f"Analytic: {analytical}, \nNumerical: {numerical}")
print(f"Relative error: {abs(analytical - numerical)/abs(analytical)}")
