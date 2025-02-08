from dataclasses import dataclass
import numpy as np
from scipy.signal import convolve2d
from mixer import Mixer, mix, should_mix
from stopper import build_stopper

from efd_config import dxdy, reaction_constants
from efd_config import DEFAULT_CONFIG

# Explicit finite difference solver.

@dataclass
class State:
  time_step: int = 0
  c1 = 0.0
  c2 = 0.0
  c3 = 0.0

def laplacian_filter(dx, dy):
  return np.array([
    [      0,            dy**-2 ,      0 ],
    [ dx**-2, -2*(dx**-2+dy**-2), dx**-2 ],
    [      0,            dy**-2 ,      0 ]
  ])

def laplacian(c, filter):
  # extend array to compensate for
  # shrinking after convolution.
  padded = np.pad(c, (1, 1), 'edge')
  return convolve2d(padded, filter, mode='valid')

# Upper time step bound.
def dt_bound(config):
  dx, dy = dxdy(config)
  D, k, c0 = reaction_constants(config)
  return 1.0 / (15 * k * c0 + 2 * D * (dx**-2 + dy**-2))

def solve(c_init, config=DEFAULT_CONFIG):

  filter = laplacian_filter(dx, dy)
  lap = lambda c : laplacian(c, filter)

  mixer = Mixer(config)
  stopper = build_stopper(config)

  D, k, _ = reaction_constants(config)

  dx, dy = dxdy(config)
  dt = config['dt'] or dt_bound(config)

  captured_steps = [0] # always capture initial condition
  state = State(*c_init)

  while True:

    if mixer.should_mix(state.time_step):
     
      c1_last, c2_last, c3_last = mix(
        [c1_last, c2_last, c3_last], 
        B=B, 
        optimal_mix=optimal_mix, 
        debug=False)

    c1_next = c1_last + dt * (-3 * k * c1_last * c2_last + D * lap(c1_last))
    c2_next = c2_last + dt * (-5 * k * c1_last * c2_last + D * lap(c2_last))
    c3_next = c3_last + dt *   2 * k * c1_last * c2_last

    if time_step % frame_stride == 0:
      _ = c1.append(c1_next), c2.append(c2_next), c3.append(c3_next)

      # register that we calculated data for next time step
      ts.append(time_step + 1)

    if stopper.should_stop(state):
      break

    # update time step
    state.time_step = state.time_step + 1
    # update last values
    c1_last, c2_last, c3_last = c1_next, c2_next, c3_next

  # convert python lists to numpy lists
  c1 = np.stack(c1, axis=0)
  c2 = np.stack(c2, axis=0)
  c3 = np.stack(c3, axis=0)

  # shape [ 3, t, width, height ]
  c = np.stack((c1, c2, c3), axis=0)

  return np.array(captured_steps), c
