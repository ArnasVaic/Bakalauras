from dataclasses import dataclass
from logging import Logger
import numpy as np
from scipy.signal import convolve2d
from solvers.adi.utils import frame_quantity
from solvers.ftcs.debug import log_debug_info, log_initial_info
from solvers.ftcs.state import State
from solvers.mixer import Mixer
from solvers.stopper import Stopper
from solvers.ftcs.config import Config

@dataclass
class Solver:
  """Solver based on the explicit foward time centered space method"""
  config: Config
  stopper: Stopper
  mixer: Mixer

  def __init__(self, config: Config):
    self.config = config
    self.stopper = config.stopper
    self.mixer = config.mixer
    self.filter = self.laplacian_filter(config.dx, config.dy)

  def dt_bound(self) -> float:
    """Upper time step bound."""
    dx, dy = self.config.dx, self.config.dy
    D, k, c0 = self.config.D, self.config.k, self.config.c0

    dx2dy2_inv = dx**-2 + dy**-2
    k15c0 = 15 * k * c0

    dt1 = (k15c0 + 2 * D[0] * dx2dy2_inv)**-1
    dt2 = (k15c0 + 2 * D[1] * dx2dy2_inv)**-1
    dt3 = (2 * D[2] * dx2dy2_inv)**-1

    return min(dt1, dt2, dt3)

  def solve(
    self,
    c_init: np.ndarray,
    logger: Logger | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Solve the PDE"""
    self.config.validate()

    dt: float = self.config.dt or self.dt_bound()
    assert dt <= self.dt_bound()

    log_initial_info(logger, dt, self.config)
    state = State(c_init)
    c, p = state.current, state.c_prev
    D, k = self.config.D, self.config.k
    while True:

      log_debug_info(logger, state)

      if self.mixer.should_mix(state, dt):
        if logger is not None:
          logger.info(f'mixing, step = {state.time_step}, time = {state.time_step * self.dt}')
        state.c_prev = self.mixer.mix(state.c_prev)

      if state.time_step % self.config.frame_stride == 0:
        state.capture(dt)
        # pass

      kc1c2 = k * state.c_prev[0] * state.c_prev[1]

      c[0] = p[0] + dt * (-3 * kc1c2 + D[0] * self.laplacian(p[0]))
      c[1] = p[1] + dt * (-5 * kc1c2 + D[1] * self.laplacian(p[1]))
      c[2] = p[2] + dt * ( 2 * kc1c2 + D[2] * self.laplacian(p[2]))
      np.copyto(p, c)

      state.current_quantity = frame_quantity(c)

      if self.stopper.should_stop(state):
        break

      # update time step
      state.time_step = state.time_step + 1
      state.time = state.time + dt

    return np.array(state.captured_steps), np.array(state.captured_c)

  def laplacian_filter(self, dx: float, dy: float) -> np.ndarray[np.float64]:
    """Laplacian filter for an evenly spaced discrete grid."""
    return np.array([
      [      0,            dy**-2 ,      0 ],
      [ dx**-2, -2*(dx**-2+dy**-2), dx**-2 ],
      [      0,            dy**-2 ,      0 ]
    ])

  def laplacian(self, c: np.ndarray[np.float64]):
    """Laplacian operator on an evenly spaced discrete grid."""
    # Extend array to compensate for hrinking after convolution.
    padded = np.pad(c, (1, 1), 'edge')
    return convolve2d(padded, self.filter, mode='valid')
