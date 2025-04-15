from dataclasses import dataclass
import numpy as np

@dataclass
class State:

  # Initial state of the simulation with shape [ 3, width, height ]
  initial: np.ndarray[np.float64]

  # Current state of the simulation with shape [ 3, width, height ]
  current: np.ndarray[np.float64]

  # Previous state of the simulation with shape [ 3, width, height ]
  previous: np.ndarray[np.float64]

  # simulation time step
  time_step: int = 0

  def __init__(self, initial: np.ndarray[np.float64]):
    np.copyto(self.initial, initial)
    np.copyto(self.current, initial)
    np.copyto(self.previous, initial)