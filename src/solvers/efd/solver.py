from dataclasses import dataclass
import numpy as np
from scipy.signal import convolve2d
from solvers.efd.debug import log_debug_info, log_initial_info
from solvers.efd.state import State
from solvers.mixer import Mixer
from solvers.stopper import Stopper
from solvers.config import Config

# Explicit finite difference solver.

@dataclass
class Solver:
  config: Config
  stopper: Stopper
  mixer: Mixer
  dt: float

  def __init__(self, config: Config):
    self.config = config
    self.stopper = config.stopper
    self.mixer = config.mixer
    self.filter = self.laplacian_filter(config.dx, config.dy)
    self.dt = self.config.dt or self.dt_bound()

  # Upper time step bound.
  def dt_bound(self) -> float:
    dx, dy = self.config.dx, self.config.dy
    D, k, c0 = self.config.D, self.config.k, self.config.c0
    return 1.0 / (15 * k * c0 + 2 * np.max(D) * (dx**-2 + dy**-2))

  def solve(self, c_init: np.ndarray[np.float64]) -> tuple[np.ndarray, np.ndarray]:
    self.config.validate()
    log_initial_info(self.config.logger, self.dt, self.config)
    state = State(c_curr = c_init.copy(), c_init = c_init.copy(), c_prev = c_init.copy())
    D, k = self.config.D, self.config.k
    while True:

      log_debug_info(self.config.logger, state)

      if self.mixer.should_mix(state.time_step, self.dt):
        if self.config.logger is not None:
          self.config.logger.debug(f'mixing, step = {state.time_step}, time = {state.time_step * self.dt}')
        state.c_prev = self.mixer.mix(state.c_prev)

      if state.time_step % self.config.frame_stride == 0:
        state.capture(self.dt)

      kc1c2 = k * state.c_prev[0] * state.c_prev[1]
 
      state.c_curr[0] = state.c_prev[0] + self.dt * (-3 * kc1c2 + D[0] * self.laplacian(state.c_prev[0]))
      state.c_curr[1] = state.c_prev[1] + self.dt * (-5 * kc1c2 + D[1] * self.laplacian(state.c_prev[1]))
      state.c_curr[2] = state.c_prev[2] + self.dt * ( 2 * kc1c2 + D[2] * self.laplacian(state.c_prev[2]))

      state.c_prev = state.c_curr

      if self.stopper.should_stop(state):
        break

      # update time step
      state.time_step = state.time_step + 1

    return np.array(state.captured_steps), np.array(state.captured_c)

  def laplacian_filter(self, dx: float, dy: float) -> np.ndarray[np.float64]:
    return np.array([
      [      0,            dy**-2 ,      0 ],
      [ dx**-2, -2*(dx**-2+dy**-2), dx**-2 ],
      [      0,            dy**-2 ,      0 ]
    ])
  
  def laplacian(self, c: np.ndarray[np.float64]):
    # extend array to compensate for
    # shrinking after convolution.
    padded = np.pad(c, (1, 1), 'edge')
    return convolve2d(padded, self.filter, mode='valid')