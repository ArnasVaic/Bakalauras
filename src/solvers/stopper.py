from abc import abstractmethod

class Stopper:
  @abstractmethod
  def should_stop(self, state) -> bool:
    pass

class ThresholdStopper(Stopper):

  def __init__(self, threshold: float):
    self.threshold = threshold

  def should_stop(self, state) -> bool:
    q = state.current_quantity[0] + state.current_quantity[1]
    q0 = state.initial_quantity[0] + state.initial_quantity[1]
    return q / q0 <= self.threshold

class TotalStepsStopper(Stopper):

  def __init__(self, total_steps):
    self.total_steps = total_steps

  def should_stop(self, state) -> bool:
    return state.time_step + 1 == self.total_steps
