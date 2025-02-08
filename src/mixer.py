import numpy as np
from efd.solver import dt_bound

class Mixer:

  def __init__(self, config):
    self.B = config['B']
    self.t_mix = config['t_mix']
    self.optimal_mix = config['optimal_mix']
    self.dt = config['dt'] or dt_bound(config)

  def should_mix(self, time_step):

    if self.t_mix is None:
      return False

    # true if any discrete time points are less 
    # than a half time step away from point of mixing.
    return np.any(abs(time_step * self.dt - self.t_mix) <= self.dt / 2)

  def mix(self, state):
     
    state.c1

    c1_last, c2_last, c3_last = mix(
      [c1_last, c2_last, c3_last], 
      B=B, 
      optimal_mix=optimal_mix, 
      debug=False)