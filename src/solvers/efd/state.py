from dataclasses import dataclass, field
import numpy as np

@dataclass
class State:

  # Current state of the simulation with shape [ 3, t, width, height ]
  c_curr: np.ndarray[np.float64]

  # Previous state of the simulation with shape [ 3, t, width, height ]
  c_prev: np.ndarray[np.float64]

  # Initial state of the simulation with shape [ 3, t, width, height ]
  c_init: np.ndarray[np.float64]

  time_step: int = 0

  captured_steps: list[int] = field(default_factory=list)
  
  captured_c: list[np.ndarray[np.float64]] = field(default_factory=list)
 
  def capture(self, dt: float) -> None:
    self.captured_c.append(self.c_curr.copy())
    self.captured_steps.append(self.time_step * dt)
