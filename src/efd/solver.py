import numpy as np
from scipy.signal import convolve2d
from efd.state import State
from mixer import Mixer, mix, should_mix
from stopper import Stopper

from efd.config import Config, dxdy, reaction_constants
from efd.config import DEFAULT_CONFIG

# Explicit finite difference solver.

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
def dt_bound(config) -> float:
  dx, dy = dxdy(config)
  D, k, c0 = reaction_constants(config)
  return 1.0 / (15 * k * c0 + 2 * D * (dx**-2 + dy**-2))

def solve(c_init, config: Config):
 
  D, k = config.D, config.k
  dx, dy = config.dx, config.dy
  dt = config.dt or dt_bound(config)

  state = State(c_init = c_init, c_prev = c_init)

  filter = laplacian_filter(dx, dy)
  lap = lambda c : laplacian(c, filter)

  mixer = Mixer(config)
  stopper: Stopper = config['stopper']

  while True:

    if mixer.should_mix(state.time_step * dt):
      mixer.mix(state)

    if state.time_step % config.frame_stride == 0:
      state.capture()

    if stopper.should_stop(state):
      break

    state.update(lap, D, k, dt)

    # update time step
    state.time_step = state.time_step + 1

  return np.array(state.captured_steps), np.array(state.captured_c)
