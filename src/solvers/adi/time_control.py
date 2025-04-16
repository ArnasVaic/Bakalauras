
from abc import abstractmethod


class TimeControl:
  """Controls the time steps"""

  @abstractmethod
  def step(self, t: float) -> float:
    """Perform a time step

    Args:
        t (float): Current time

    Returns:
        float: New time
    """
    pass

class FixedTimeStep(TimeControl):

  def __init__(self, dt: float):
    self.dt = dt

  def step(self, t: float) -> float:
    return t + self.dt