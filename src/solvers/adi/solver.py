from dataclasses import dataclass
from typing import Callable
import numpy as np
import scipy.linalg as la
import datetime

from solver_utils import validate_frame_stable
from solvers.adi.config import Config
from solvers.adi.state import State
from solvers.adi.utils import build_banded_matrix_A

NUM_OF_ELEMENTS = 3

# - It makes to have the frame capture module have the responsibility
#   to capture the frames and to decide when it needs to do so.
# - Maybe it doesn't because we may want to mix and match the result shapes and the frame skipping rules
# What should be the life time of a frame capture module?
# What would be a simple solution?
# - Have the frame capture module belong to the configuration
#   so it uses the same module for each solution


@dataclass
class Solver:
  """
  ADI Solver for the YAG reaction-diffusion system
  """
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

    self.half = np.zeros((NUM_OF_ELEMENTS, nx, ny))

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

  def solve_step(self, state: State) -> None:
    """Solve for the next step, directly modifies the state
    """
    nx, ny = self.config.resolution
    h = self.half
    c = state.current
    for m in [0, 1, 2]:
      mu_x, mu_y, mu_m = self.mu_x[m], self.mu_y[m], self.mu_m[m]

      for y in range(ny):
        yt, yb = min(y + 1, ny - 1), max(y - 1, 0)
        h[m, :, y] = la.solve_banded(
          (1, 1),
          self.Ax_banded[m],
          (1 - 2 * mu_y) * c[m, :, y] + mu_y * (c[m, :, yb] + c[m, :, yt]) + mu_m * c[0, :, y] * c[1, :, y]
        )

      for x in range(nx):
        xl, xr = max(x - 1, 0), min(x + 1, nx - 1)
        c[m, x, :] = la.solve_banded(
          (1, 1),
          self.Ay_banded[m],
          mu_m * h[0, x, :] * h[1, x, :] + (1 - 2 * mu_x) * h[m, x, :] + mu_x * (h[m, xl, :] + h[m, xr, :])
        )

    state.time_step = state.time_step + 1

  def solve(self, c0: np.array, capture: Callable) -> tuple[np.array, np.array]:

    state = State(c0)

    captured_result = [capture(c0)]
    captured_times = [0]

    if self.config.logger:
      self.config.logger.info(f'starting simulation, dt={self.config.dt}, dx={self.config.dx}, dy={self.config.dy}, D={self.config.D},k={self.config.k}, size={self.config.size}, resolution={self.config.resolution}')

    # O(TN^2)

    while True:

      if state.time_step == 1:
        validate_frame_stable(self.config, state.current)

      solve_start = datetime.datetime.now()
      self.solve_step(state)
      solve_end = datetime.datetime.now()

      if self.config.logger:

        # every frame_stride frames log the remaining quantity ratio
        if state.time_step % self.config.frame_stride == 0:
          q, q0 = state.current[:2].sum(axis=(1, 2)), c0[:2].sum(axis=(1, 2))
          ratio = (q[0] + q[1]) / (q0[0] + q0[1])
          self.config.logger.info(f'step: {state.time_step}, avg frame t: {(solve_end - solve_start).total_seconds() * 1000 / state.time_step:.02f}ms, ratio: {ratio:.02f}, r1: {q[0]/q0[0]:.02f}, r1: {q[1]/q0[1]:.02f}')

      if state.time_step % self.config.frame_stride == 0:
        captured_result.append(capture(state.current))
        captured_times.append(state.time_step * self.config.dt)

      # maybe stopper doesn't need to run every frame?
      if self.stopper.should_stop(state):
        # capture the moment when reaction stops
        captured_result.append(capture(state.current))
        captured_times.append(state.time_step * self.config.dt)
        break

    return np.array(captured_times), np.array(captured_result)
