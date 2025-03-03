# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Parameters
Lx, Ly = 1.0, 1.0  # Domain size
Nx, Ny = 20, 20    # Grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)
alpha = 0.01       # Thermal diffusivity
dt = 0.0005        # Time step
T_final = 0.1      # Final simulation time
Nt = int(T_final / dt)

# Grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial Condition
u = np.zeros((Nx, Ny))

# Source term (example: Gaussian heat source)
def source_term(x, y, t):
    return np.exp(-10 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)) * np.sin(5 * t)

# ADI Coefficients
rx, ry = alpha * dt / dx**2, alpha * dt / dy**2
A = np.zeros((Nx, Nx))
B = np.zeros((Ny, Ny))

# Construct tridiagonal matrices for x and y sweeps
for i in range(1, Nx-1):
    A[i, i-1] = -rx / 2
    A[i, i] = 1 + rx
    A[i, i+1] = -rx / 2
A[0, 0] = A[-1, -1] = 1  # Dirichlet BCs

for j in range(1, Ny-1):
    B[j, j-1] = -ry / 2
    B[j, j] = 1 + ry
    B[j, j+1] = -ry / 2
B[0, 0] = B[-1, -1] = 1  # Dirichlet BCs

# ADI Method
for n in range(Nt):
    f = source_term(X, Y, n * dt)

    # X-direction implicit step
    u_half = np.copy(u)
    for j in range(1, Ny-1):
        rhs = (1 - ry) * u[1:-1, j] + (ry / 2) * (u[1:-1, j-1] + u[1:-1, j+1]) + dt * f[1:-1, j] / 2
        rhs[0] = rhs[-1] = 0  # Apply BCs
        u_half[1:-1, j] = np.linalg.solve(A[1:-1, 1:-1], rhs)

    # Y-direction implicit step
    for i in range(1, Nx-1):
        rhs = (1 - rx) * u_half[i, 1:-1] + (rx / 2) * (u_half[i-1, 1:-1] + u_half[i+1, 1:-1]) + dt * f[i, 1:-1] / 2
        rhs[0] = rhs[-1] = 0  # Apply BCs
        u[i, 1:-1] = np.linalg.solve(B[1:-1, 1:-1], rhs)

# Plot final temperature distribution
plt.imshow(u, extent=[0, Lx, 0, Ly], origin="lower", cmap="hot")
plt.colorbar(label="Temperature")
plt.title("2D Heat Equation with Source (ADI)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
