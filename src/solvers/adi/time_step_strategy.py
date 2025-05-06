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

# Stepwise Clamped Arithmetic Time Step with Quantity Detection
# SCAQ 
class SCAQStep(TimeStepStrategy):
  """Clamped arithmetic progression time step with stepwise increase and quantity level detection."""
  def __init__(
    self,
    steps_until_change: int,
    a1: float,
    d: float,
    upper: float,
    threshold: float,
    low: float = 1.0):
    self._steps_until_change = steps_until_change
    self._dt = a1
    self._d = d
    self._upper = upper
    self._threshold = threshold
    self._low = low

  @property
  def dt(self) -> float:
    return self._dt

  def update_dt(self, state: State) -> None:
    # only update dt every steps_until_change steps
    if state.time_step % self._steps_until_change == 0:
      self._dt = min(self._dt + self._d, self._upper)

    # always check quantity level despite the step skip
    q = state.current_quantity[0] + state.current_quantity[1]
    q0 = state.initial_quantity[0] + state.initial_quantity[1]
    if q / q0 <= self._threshold:
      self._dt = self._low

# Stepwise Clamped Geometric Time Step with Quantity Detection
# SCGQ 
class SCGQStep(TimeStepStrategy):
  """Clamped arithmetic progression time step with stepwise increase and quantity level detection."""
  def __init__(
    self,
    steps_until_change: int,
    a1: float,
    r: float,
    upper: float,
    threshold: float,
    low: float = 1.0):
    self._steps_until_change = steps_until_change
    self._dt = a1
    self._r = r
    self._upper = upper
    self._threshold = threshold
    self._low = low

  @property
  def dt(self) -> float:
    return self._dt

  def update_dt(self, state: State) -> None:
    # only update dt every steps_until_change steps
    if state.time_step % self._steps_until_change == 0:
      self._dt = min(self._dt * self._r, self._upper)

    # always check quantity level despite the step skip
    q = state.current_quantity[0] + state.current_quantity[1]
    q0 = state.initial_quantity[0] + state.initial_quantity[1]
    if q / q0 <= self._threshold:
      self._dt = self._low

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