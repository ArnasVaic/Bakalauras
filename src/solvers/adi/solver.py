from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
import datetime

from solvers.config import Config
from solvers.efd.state import State
from solvers.stopper import Stopper

def build_banded_matrix_A(n: int, mu: float):
  subdiag = np.repeat(-mu, n)
  subdiag[0] = 0

  supdiag = np.repeat(-mu, n)
  supdiag[-1] = 0

  diag = np.repeat(1 + 2 * mu, n)
  diag[0] = diag[-1] = 1 + mu

  return np.array([subdiag, diag, supdiag])

@dataclass
class Solver:

    config: Config

    stopper: Stopper

    def __init__(self, config: Config):
      self.config = config
      self.stopper = config.stopper

    def solve(self, c0: np.array) -> tuple[np.array, np.array]:

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

      total_calc_time = 0

      if self.config.logger:
        self.config.logger.debug(f'starting simulation, dt={dt}, dx={self.config.dx}, dy={self.config.dy}, D={self.config.D}, k={self.config.k}, size={self.config.size}, resolution={self.config.resolution}')

      while True:

        frame_calc_start = datetime.datetime.now()

        for m in [0, 1, 2]:

          for y in range(ny):
            reaction_term = mu_m[m] * c[0, :, y] * c[1, :, y]
            diffusion_term = (1 - 2 * mu_y[m]) * c[m, :, y] + mu_y[m] * (c[m, :, max(y - 1, 0)] + c[m, :, min(y + 1, ny - 1)])
            c_half[m, :, y] = la.solve_banded((1, 1), Ax_banded[m], diffusion_term + reaction_term)

          for x in range(nx):
            reaction_term = mu_m[m] * c_half[0, x, :] * c_half[1, x, :]
            diffusion_term = (1 - 2 * mu_x[m]) * c_half[m, x, :] + mu_x[m] * (c_half[m, max(x - 1, 0), :] + c_half[m, min(x + 1, nx - 1), :])
            c_next[m, x, :] = la.solve_banded((1, 1), Ay_banded[m], reaction_term + diffusion_term)

        frame_calc_end = datetime.datetime.now()

        total_calc_time = frame_calc_end - frame_calc_start

        time_step = time_step + 1

        if self.config.logger:
          
          # every frame_stride frames log the remaining quantity ratio
          if state.time_step % self.config.frame_stride == 0:
            q, q0 = c_next[:2].sum(axis=(1, 2)), c0[:2].sum(axis=(1, 2))
            ratio = (q[0] + q[1]) / (q0[0] + q0[1])
            self.config.logger.debug(f'step: {state.time_step}, avg frame t: {total_calc_time.total_seconds() * 1000 / time_step:.02f}ms, ratio: {ratio:.02f}, r1: {q[0]/q0[0]:.02f}, r1: {q[1]/q0[1]:.02f}')

        np.copyto(state.c_prev, c)
        np.copyto(state.c_curr, c_next)
        state.time_step = time_step

        if state.time_step % self.config.frame_stride == 0:
          captured_frames.append(c_next.copy())
          captured_times.append(time_step * dt)

        # maybe stopper doesn't need to run every frame?
        if self.stopper.should_stop(state):
          break

        c = c_next
        

      return np.array(captured_times), np.array(captured_frames)
