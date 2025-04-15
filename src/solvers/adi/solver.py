from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
import datetime

from solvers.adi.config import Config
from solvers.adi.state import State
from solvers.adi.utils import build_banded_matrix_A

NUM_OF_ELEMENTS = 3

@dataclass
class Solver:

    config: Config

    # Container for the half-step solution
    half: np.ndarray

    # Matrices for finding half-step solution 
    Ax_banded: np.ndarray
    
    # Matrices for finding next-step solution
    Ay_banded: np.ndarray

    # Aggregated coefficients
    mu_x: tuple[float, float, float]
    mu_y: tuple[float, float, float]
    mu_m: tuple[float, float, float]

    def __init__(self, config: Config):
      self.config = config
      self.stopper = config.stopper

      nx, ny = self.config.resolution
      dt = self.config.dt
      dx, dy = self.config.dx, self.config.dy
      k, D = self.config.k, self.config.D

      self.half = np.zeros_like((NUM_OF_ELEMENTS, nx, ny))

      # since diffusion coefficient is different for each element,
      # the coefficients we construct also are going to be different
      self.mu_m = [ alpha * dt * k / 2 for alpha in [-3, -5, 2] ]

      self.mu_x = [ D_i * dt / (2 * dx ** 2) for D_i in D ]
      self.Ax_banded = np.array(
        [build_banded_matrix_A(nx, self.mu_x[m]) 
        for m in range(NUM_OF_ELEMENTS)]
      )

      self.mu_y = [ D_i * dt / (2 * dy ** 2) for D_i in D ]
      self.Ay_banded = np.array(
        [build_banded_matrix_A(ny, self.mu_y[m]) 
        for m in range(NUM_OF_ELEMENTS)]
      )

    # Directly modify the state by solving for the next step
    def solve_step(self, state: State) -> None:
      nx, ny = self.config.resolution
      h = self.half
      c = state.current
      for m in [0, 1, 2]:
        mu_x = self.mu_x[m]
        mu_y = self.mu_y[m]
        mu_m = self.mu_m[m]
        for y in range(ny):
          h[m, :, y] = la.solve_banded(
            (1, 1), 
            self.Ax_banded[m], 
            (1 - 2 * mu_y) * c[m, :, y] + mu_y * (c[m, :, max(y - 1, 0)] + c[m, :, min(y + 1, ny - 1)]) + mu_m * c[0, :, y] * c[1, :, y]
          )

        for x in range(nx):
          c[m, x, :] = la.solve_banded(
            (1, 1), 
            self.Ay_banded[m], 
            mu_m * h[0, x, :] * h[1, x, :] + (1 - 2 * mu_x) * h[m, x, :] + mu_x * (h[m, max(x - 1, 0), :] + h[m, min(x + 1, nx - 1), :])
          )

    def solve(self, c0: np.array) -> tuple[np.array, np.array]:
      dt = self.config.dt
      
      state = State(c0)

      captured_frames = [np.copy(c)]
      captured_times = [0]

      time_step = 0

      total_calc_time = 0

      if self.config.logger:
        self.config.logger.debug(f'starting simulation, dt={dt}, dx={self.config.dx}, dy={self.config.dy}, D={self.config.D}, k={self.config.k}, size={self.config.size}, resolution={self.config.resolution}')

      while True:

        frame_calc_start = datetime.datetime.now()
        self.solve_step()
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
