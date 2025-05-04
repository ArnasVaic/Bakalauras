from dataclasses import dataclass, field
import numpy as np

from solvers.adi.utils import frame_quantity

@dataclass
class State:

  # Current state of the simulation
  # Shape: [ 3, t, width, height ]
  current: np.ndarray[np.float64]

  # Previous state of the simulation
  # Shape [ 3, t, width, height ]
  c_prev: np.ndarray[np.float64]

  # Initial state of the simulation
  # Shape [ 3, t, width, height ]
  c_init: np.ndarray[np.float64]

  # optimize calculation by cache'ing initial quantity
  initial_quantity: np.ndarray
  current_quantity: np.ndarray

  time_step: int = 0

  captured_steps: list[int] = field(default_factory=list)

  captured_c: list[np.ndarray[np.float64]] = field(default_factory=list)

  def __init__(self, initial: np.ndarray[np.float64]):
    self.c_init = np.copy(initial)
    self.current = np.copy(initial)
    self.c_prev = np.copy(initial)
    self.initial_quantity = frame_quantity(initial)
    self.current_quantity = frame_quantity(initial)
    self.captured_c = []
    self.captured_steps = []

  def capture(self, dt: float) -> None:
    self.captured_c.append(self.current.copy())
    self.captured_steps.append(self.time_step * dt)
