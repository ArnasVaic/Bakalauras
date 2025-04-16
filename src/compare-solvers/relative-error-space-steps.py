# %%
# Observe the relative error between a high resolution solution and lower resolution solutions

from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition
from solver_utils import validate_solution_stable, get_quantity_over_time, timed
from solvers.adi.config import Config
from solvers.adi.solver import Solver


# %% Cache results for given parameters

RESOLUTION = 120

config = Config()
config.dt = 1
config.frame_stride = 50
config.resolution = (RESOLUTION, RESOLUTION)
solver = Solver(config)

c0 = initial_condition(config)

with timed(f"ADI Solve time {config.resolution}") as elapsed:
  t, c = solver.solve(c0, capture_frame)
q = get_quantity_over_time(config, c)
validate_solution_stable(config, c)

np.save(f'compare-solvers/assets/adi-{RESOLUTION}x{RESOLUTION}-t', t)
np.save(f'compare-solvers/assets/adi-{RESOLUTION}x{RESOLUTION}-q', q)

del config, solver, c0, t, q


# %% Plot errors, ahah no

RESOLUTIONS = [ 40, 200 ]

for resolution in RESOLUTIONS:
  t = np.load(f'compare-solvers/assets/adi-{resolution}x{resolution}-t.npy')
  q = np.load(f'compare-solvers/assets/adi-{resolution}x{resolution}-q.npy')

  plt.plot(t / 3600, q[:,2], linestyle='dashed', label = f'{resolution}x{resolution}')

plt.xlabel('t [h]')
plt.ylabel('quantity [g]')
plt.legend()

# %% Plot relative errors

true_t = np.load(f'compare-solvers/assets/adi-200x200-t.npy')
true_q = np.load(f'compare-solvers/assets/adi-200x200-q.npy') 

RESOLUTIONS = [ 40, 60, 80, 120 ]

ELEMENT = 1

for index, resolution in enumerate(RESOLUTIONS):

  t = np.load(filepath(resolution, 't'))
  q = np.load(filepath(resolution, 'q'))

  T = min(t.shape[0], true_t.shape[0])

  error = np.abs(true_q[:T, :] - q[:T, :]) / np.abs(true_q[:T, :])

  plt.plot(
    t[:T] / 3600,
    error[:, ELEMENT],
    label=f'{resolution}x{resolution}'
  )

plt.title(f'Pointwise relative error between ADI solutions of different resolutions ($c_{ELEMENT + 1}$)')
plt.xlabel('t [h]')
plt.ylabel(f'error, %')
plt.legend()
plt.show()

# %% Plot absolute errors

true_t = np.load(f'compare-solvers/assets/adi-200x200-t.npy')
true_q = np.load(f'compare-solvers/assets/adi-200x200-q.npy') 

RESOLUTIONS = [ 40, 60, 80, 120 ]

ELEMENT = 2

for index, resolution in enumerate(RESOLUTIONS):

  t = np.load(filepath(resolution, 't'))
  q = np.load(filepath(resolution, 'q'))

  T = min(t.shape[0], true_t.shape[0])

  error = np.abs(true_q[:T, :] - q[:T, :])

  plt.plot(
    t[:T] / 3600,
    error[:, ELEMENT],
    label=f'{resolution}x{resolution}'
  )

plt.title(f'Absolute error between ADI solutions of different resolutions ($c_{ELEMENT + 1}$)')
plt.xlabel('t [h]')
plt.ylabel(f'difference in quantity [g]')
plt.legend()
plt.show()

# %% L norms

import numpy.linalg as la

true_q = np.load(f'compare-solvers/assets/adi-200x200-q.npy') 

RESOLUTIONS = [ 40, 60, 80, 120 ]
ELEMENT = 2

for index, resolution in enumerate(RESOLUTIONS):
  q = np.load(filepath(resolution, 'q'))
  T = min(q.shape[0], true_q.shape[0])

  diff = true_q[:T, ELEMENT] - q[:T, ELEMENT]

  l2 = la.norm(diff, 2)
  linf = la.norm(diff, np.inf)

  print(f'{resolution:3}x{resolution:3}, L2: {l2:8.2e}, Linf: {linf:8.2e}')