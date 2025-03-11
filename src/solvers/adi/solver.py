# %%
from dataclasses import dataclass
import scipy.sparse as sp
import numpy as np
import scipy.linalg as la

from solvers.config import Config
from solvers.efd.state import State
from solvers.stopper import Stopper

def build_banded_matrix_A(n: int, mu: float):
  subdiag = np.repeat(mu, n)
  subdiag[0] = 0

  supdiag = np.repeat(mu, n)
  supdiag[-1] = 0

  diag = np.repeat(1 + 2 * mu, n)
  diag[0] = diag[-1] = 1 + mu

  return np.array([subdiag, diag, supdiag])

@dataclass
class Solver:

    # solver configuration
    config: Config

    stopper: Stopper

    def __init__(self, config: Config):
      self.config = config
      self.stopper = config.stopper

    def solve(self, c0: np.array) -> np.array:

      # for shorter notation
      nx, ny = self.config.resolution
      dt = self.config.dt
      dx, dy = self.config.dx, self.config.dy
      k, D = self.config.k, self.config.D

      # ADI solver does not support auto time-step
      assert dt is not None
      
      c = np.copy(c0)
      
      state = State(c_curr = c0.copy(), c_init = c0.copy(), c_prev = c0.copy())

      c_half = np.zeros_like(c)
      c_next = np.zeros_like(c)

      mu_x = [ D_i * dt / (2 * dx ** 2) for D_i in D ]
      mu_y = [ D_i * dt / (2 * dy ** 2) for D_i in D ]

      # since diffusion coefficient is different for each element,
      # the coefficients we construct also are going to be different
      mu_m = [ alpha * dt * k / 2 for alpha in [-3, -5, 2] ]

      # different A matrix for each element since diffusion coefficients differ
      Ax_banded = np.array([build_banded_matrix_A(nx, mu_x[m]) for m in [0, 1, 2]])
      Ay_banded = np.array([build_banded_matrix_A(ny, mu_y[m]) for m in [0, 1, 2]])

      captured_frames = [np.copy(c)]
      captured_times = [0]

      time_step = 0

      while True:

        for m in [0, 1, 2]:

          for y in range(ny):
            reaction_term = mu_m[m] * c[0, :, y] * c[1, :, y]
            diffusion_term = (1 - 2 * mu_y[m]) * c[m, :, y] + mu_y[m] * (c[m, :, max(y - 1, 0)] + c[m, :, min(y + 1, ny - 1)])
            c_half[m, :, y] = la.solve_banded((1, 1), Ax_banded[m], diffusion_term + reaction_term)

          for x in range(nx):
            reaction_term = mu_m[m] * c_half[0, x, :] * c_half[1, x, :]
            diffusion_term = (1 - 2 * mu_x[m]) * c_half[m, x, :] + mu_x[m] * (c_half[m, max(x - 1, 0), :] + c_half[m, min(x + 1, nx - 1), :])
            c_next[m, x, :] = la.solve_banded((1, 1), Ay_banded[m], reaction_term + diffusion_term)

        state.c_prev = c
        state.c_curr = c_next

        captured_frames.append(c_next.copy())
        captured_times.append(time_step * dt)

        if self.stopper.should_stop(state):
          break

        c = c_next
        time_step = time_step + 1

      return np.array(captured_times), np.array(captured_frames)
