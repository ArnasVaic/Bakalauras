from dataclasses import dataclass
import numpy as np

from solvers.adi.utils import frame_quantity

@dataclass
class State:
  """State of the solver"""

  # Initial state of the simulation with shape [ 3, width, height ]
  initial: np.ndarray[np.float64]

  # Current state of the simulation with shape [ 3, width, height ]
  current: np.ndarray[np.float64]

  # Previous state of the simulation with shape [ 3, width, height ]
  previous: np.ndarray[np.float64]

  # initial quantity of each element (shape [3])
  initial_quantity: np.ndarray

  # current quantity of each element (shape [3])
  current_quantity: np.ndarray

  # Time steps immediately after which mixing took place
  mixing_steps: list[int]

  # current simulation time
  time: float = 0.0

  # simulation time step
  time_step: int = 0

  def __init__(self, initial: np.ndarray[np.float64]):
    self.initial = np.copy(initial)
    self.current = np.copy(initial)
    self.previous = np.copy(initial)
    self.initial_quantity = frame_quantity(initial)
    self.current_quantity = frame_quantity(initial)
    self.mixing_steps = []
