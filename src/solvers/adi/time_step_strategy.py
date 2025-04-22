from abc import abstractmethod
from dataclasses import dataclass
from solvers.adi.state import State

@dataclass
class TimeStepStrategy:
  """Time stepping strategy"""

  @property
  def dt(self) -> float:
    pass

  @abstractmethod
  def update_dt(self, state: State) -> None:
    pass

class ConstantTimeStep(TimeStepStrategy):

  def __init__(self, dt: float):
    self._dt = dt

  @property
  def dt(self) -> float:
    return self._dt

  def update_dt(self, state: State) -> None:
    pass

class ACAStep(TimeStepStrategy):
  def __init__(self, a1: float, d: float, upper: float, threshold: float, low: float = 1.0):
    self._dt = a1
    self._d = d
    self._upper = upper
    self._threshold = threshold
    self._low = low

  @property
  def dt(self) -> float:
    return self._dt

  def update_dt(self, state: State) -> None:
    self._dt = min(self._dt + self._d, self._upper)
    q = state.current_quantity[0] + state.current_quantity[1]
    q0 = state.initial_quantity[0] + state.initial_quantity[1]
    if q / q0 <= self._threshold:
      self._dt = self._low


class ClampedArithmeticTimeStep(TimeStepStrategy):
  def __init__(self, a1: float, d: float, upper: float):
    self._dt = a1
    self._d = d
    self._upper = upper

  @property
  def dt(self) -> float:
    return self._dt

  def update_dt(self, state: State) -> None:
    self._dt = min(self._dt + self._d, self._upper)