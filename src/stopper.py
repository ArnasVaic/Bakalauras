from abc import abstractmethod
import numpy as np

def build_stopper(config):
  if config['threshold']:
    return ThresholdStopper(config)
  elif config['total_steps']:
    return TotalStepsStopper(config)
  else:
    raise Exception("Stopper could not be built because proper configuration is missing.")

class Stopper:
  @abstractmethod
  def should_stop(state):
    pass

class ThresholdStopper(Stopper):

  def __init__(self, config):
    self.threshold = config['threshold']

  def should_stop(self, state):

    q0 = np.sum(state.c1[0] + state.c2[0])
    q = np.sum(state.c1[-1] + state.c1[-1])

    return q / q0 <= self.threshold
    
class TotalStepsStopper(Stopper):

  def __init__(self, config):
    self.total_steps = config['total_steps']

  def should_stop(self, state):
    return state.step == self.total_steps