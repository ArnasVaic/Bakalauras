# %%

import logging
from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition 
from solvers.config import Config
from solvers.efd.solver import Solver
from solvers.mixer import SubdivisionMixer

# This script will run the simulations on increasingly
# larger spaces (numbers discrete points also increasing)
# and compare how reaction completion time changes with
# respect to the time of mixing.

logging.basicConfig(
  filename='debug.log',
  filemode='w',
  format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.DEBUG)

config = Config()
config.logger = logging.getLogger(__name__)

mix_hours = np.arange(0, 11, 0.5)

# reaction completion time without mixing

# c0 = initial_condition(config, (1, 1))
# config.mixer.mix_times = []
# solver = Solver(config)
# steps, _ = solver.solve(c0)

# ts = np.repeat(steps[-1] * solver.dt, len(mix_hours))
# np.save(f'assets/{1}x{1}.npy', ts)

# reaction completion times for 
# multiple grid sizes with mixing

sizes = []

base_size = 2.154434690031884

for s in sizes:
  ts = [] # reaction completion times
  for index, mix_hour in enumerate(mix_hours):

    # discrete points per particle
    pts_per_particle = 40
    config.size = ( s * base_size, s * base_size )
    config.resolution = ( s * pts_per_particle, s * pts_per_particle) 
    config.mixer = SubdivisionMixer(
      (2 * s, 2 * s), 'perfect', [ mix_hour * 3600 ])
    
    c0 = initial_condition(config, (s, s))
    solver = Solver(config)
    steps, c = solver.solve(c0)

    # we only care about the end time
    ts.append(steps[-1] * solver.dt)
  ts = np.array(ts)
  np.save(f'larger-efd/assets/{s}x{s}.npy', ts)

# %%

# visualize

base_size = 2.154434690031884
pts_per_particle = 40

for s in [1, 2, 4, 8, 16]:
  ts = np.load(f'larger-efd/assets/{s}x{s}.npy')
  hours = ts / 3600
  plt.plot(mix_hours, hours, label=f'{s}x{s}')
plt.xlabel('$t_{mix}$ [h]')
plt.ylabel('$t_{end}$ [h]')
plt.legend()
plt.show()