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

  # optimize calculation by cache'ing initial quantity
  initial_qnt: float

  # current quantity of each element (shape [3])
  quantity: np.ndarray

  # current simulation time
  time: float = 0.0

  # simulation time step
  time_step: int = 0

  def __init__(self, initial: np.ndarray[np.float64]):
    self.initial = np.copy(initial)
    self.current = np.copy(initial)
    self.previous = np.copy(initial)
    self.initial_qnt = initial[:2].sum()