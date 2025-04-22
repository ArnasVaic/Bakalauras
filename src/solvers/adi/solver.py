from dataclasses import dataclass
import sys
from typing import Callable
import numpy as np
import scipy.linalg as la
import datetime

from solver_utils import validate_frame_stable
from solvers.adi.config import Config
from solvers.adi.state import State
from solvers.adi.utils import build_banded_matrix_A, frame_quantity, initialize_banded

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

  # time step for which the banded matrices (and mu) are initialized
  # if this value changes, the banded matrices and mu will be
  # initialized once more and this value updated.
  initialized_dt: float

  # Matrices for finding half-step solution
  ax_banded: np.ndarray

  # Matrices for finding next-step solution
  ay_banded: np.ndarray

  # Aggregated coefficients (each of these has 3 coefficients)
  mu_x: list[float]
  mu_y: list[float]
  mu_m: list[float]

  def __init__(self, config: Config):
    self.config = config
    self.stopper = config.stopper

    nx, ny = self.config.resolution

    self.half = np.zeros((NUM_OF_ELEMENTS, nx, ny))
    self.ax_banded = np.zeros((3, 3, nx))
    self.ay_banded = np.zeros((3, 3, nx))

    self.mu_x = [0, 0, 0]
    self.mu_y = [0, 0, 0]
    self.mu_m = [0, 0, 0]

    self.initialize_banded(config.time_step_strategy.dt)

  def initialize_banded(self, dt: float) -> None:
    dx, dy = self.config.dx, self.config.dy

    half_dtk = dt * self.config.k / 2
    dt_over_dx2 = dt / (2 * dx ** 2)
    dt_over_dy2 = dt / (2 * dy ** 2)

    self.mu_m[:] = self.config.alpha * half_dtk
    self.mu_x[:] = self.config.D * dt_over_dx2
    self.mu_y[:] = self.config.D * dt_over_dy2

    initialize_banded(self.ax_banded, self.mu_x[0], 0)
    initialize_banded(self.ax_banded, self.mu_x[1], 1)
    initialize_banded(self.ax_banded, self.mu_x[2], 2)

    initialize_banded(self.ay_banded, self.mu_y[0], 0)
    initialize_banded(self.ay_banded, self.mu_y[1], 1)
    initialize_banded(self.ay_banded, self.mu_y[2], 2)

    self.initialized_dt = dt

  def update_half(self, state: State, element_index: int) -> None:
    m = element_index
    _, ny = self.config.resolution
    mu_y = self.mu_y[m]
    mu_m = self.mu_m[m]
    h = self.half
    c = state.current
    for y in range(ny):
      yt, yb = min(y + 1, ny - 1), max(y - 1, 0)
      h[m, :, y] = la.solve_banded(
        (1, 1),
        self.ax_banded[m],
        (1 - 2 * mu_y) * c[m, :, y] + mu_y * (c[m, :, yb] + c[m, :, yt]) + mu_m * c[0, :, y] * c[1, :, y]
      )

  def update_next(self, state: State, element_index: int) -> None:
    m = element_index
    nx, _ = self.config.resolution
    mu_x = self.mu_x[m]
    mu_m = self.mu_m[m]
    h = self.half
    c = state.current
    for x in range(nx):
      xl, xr = max(x - 1, 0), min(x + 1, nx - 1)
      c[m, x, :] = la.solve_banded(
        (1, 1),
        self.ay_banded[m],
        mu_m * h[0, x, :] * h[1, x, :] + (1 - 2 * mu_x) * h[m, x, :] + mu_x * (h[m, xl, :] + h[m, xr, :])
      )

  def solve_step(self, state: State) -> None:
    """Solve for the next step, directly modifies the state
    """
    dt = self.config.time_step_strategy.dt
    self.update_half(state, 0)
    self.update_half(state, 1)
    self.update_half(state, 2)

    self.update_next(state, 0)
    self.update_next(state, 1)
    self.update_next(state, 2)

    state.time_step = state.time_step + 1
    state.time = state.time + dt

    # TODO: this is expensive, maybe think of a way to calling sum each frame
    state.current_quantity = frame_quantity(state.current)
    self.config.time_step_strategy.update_dt(state)

  def solve(self, c0: np.array, capture: Callable) -> tuple[np.array, np.array]:

    logger = self.config.logger

    state = State(c0)

    captured_result = [capture(c0)]
    captured_times = [0]

    if self.config.logger:
      self.config.logger.info(f'starting simulation, dt={self.config.time_step_strategy.dt}, dx={self.config.dx}, dy={self.config.dy}, D={self.config.D},k={self.config.k}, size={self.config.size}, resolution={self.config.resolution}')

    # O(TN^2)

    while True:

      if state.time_step == 1:
        validate_frame_stable(self.config, state.current)

      # check if banded matrices should be reinitialized
      current_dt: float = self.config.time_step_strategy.dt
      if self.initialized_dt - current_dt > sys.float_info.epsilon:
        logger.info(f"Recalculating banded, dt difference = {self.initialized_dt - current_dt:.2e}")
        self.initialize_banded(current_dt)

      # solve_start = datetime.datetime.now()
      self.solve_step(state)
      # solve_end = datetime.datetime.now()

      if self.config.logger:

        # every frame_stride frames log the remaining quantity ratio
        if state.time_step % self.config.frame_stride == 0:
          dt = self.config.time_step_strategy.dt
          q, q0 = state.current[:2].sum(axis=(1, 2)), c0[:2].sum(axis=(1, 2))
          ratio = (q[0] + q[1]) / (q0[0] + q0[1])
          self.config.logger.info(f'step: {state.time_step}, dt: {dt} ratio: {100 * ratio:.02f}, r1: {100 * q[0]/q0[0]:.02f}, r1: {100 * q[1]/q0[1]:.02f}')

      if state.time_step % self.config.frame_stride == 0:
        captured_result.append(capture(state.current))
        captured_times.append(state.time)

      # maybe stopper doesn't need to run every frame?
      if self.stopper.should_stop(state):
        # capture the moment when reaction stops
        captured_result.append(capture(state.current))
        captured_times.append(state.time)
        break

    return np.array(captured_times), np.array(captured_result)
