from dataclasses import dataclass
import numpy as np

@dataclass
class State:

  time_step: int = 0
  
  # Current state of the simulation with shape [ 3, t, width, height ]
  c_next: np.ndarray[np.float64]

  # Previous state of the simulation with shape [ 3, t, width, height ]
  c_prev: np.ndarray[np.float64]

  # Initial state of the simulation with shape [ 3, t, width, height ]
  c_init: np.ndarray[np.float64]

  captured_steps: list[int] = []
  captured_c: list[np.ndarray[np.float64]] = []
 
  def capture(self) -> None:
    self.captured_c.append(self.c_next)
    self.captured_steps.append(self.time_step)

  def update(self, lap, D, k, dt) -> None:

    kc1c2 = k * self.c_prev[0] * self.c_prev[1]
 
    self.c_next[0] = self.c_prev[0] + dt * (-3 * kc1c2 + D * lap(self.c_prev[0]))
    self.c_next[1] = self.c_prev[1] + dt * (-5 * kc1c2 + D * lap(self.c_prev[1]))
    self.c_next[2] = self.c_prev[2] + dt *   2 * kc1c2

    self.c_prev = self.c_next
