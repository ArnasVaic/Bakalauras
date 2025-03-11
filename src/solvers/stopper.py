from abc import abstractmethod
import numpy as np

class Stopper:
  @abstractmethod
  def should_stop(state) -> bool:
    pass

class ThresholdStopper(Stopper):

  def __init__(self, threshold):
    self.threshold = threshold

  def should_stop(self, state) -> bool:

    q0 = np.sum(state.c_init[0] + state.c_init[1])
    q = np.sum(state.c_curr[0] + state.c_curr[1])

    print(q / q0)

    return q / q0 <= self.threshold
    
class TotalStepsStopper(Stopper):

  def __init__(self, total_steps):
    self.total_steps = total_steps

  def should_stop(self, state) -> bool:
    return state.step == self.total_steps