# %% This file is supposed to be an upgrade from the last by cache'ing solve times

# We'll use one numpy array file with multiple
# entries to save solve times for different grid sizes

import os
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from solver_utils import timed
from solvers.initial_condition import initial_condition
from solvers.adi.config import default_config as  default_adi_config
from solvers.ftcs.config import default_config as default_ftcs_config
from solvers.adi.solver import Solver as ADISolver
from solvers.ftcs.solver import Solver as FTCSSolver

SAMPLE_POINTS = 20

FILE_PATH = 'compare-solvers/assets/solve-times.npy'

if not os.path.isfile(FILE_PATH):
  # if the number of sample points
  # changes, the file will have to
  # be deleted manually
  array = np.zeros(SAMPLE_POINTS)
  np.save(FILE_PATH, array)
  del array

INITIAL_RESOLUTION = 40
RESOLUTION_STEP = 10
RESOLUTIONS = np.linspace(
  INITIAL_RESOLUTION,
  INITIAL_RESOLUTION + RESOLUTION_STEP * (SAMPLE_POINTS - 1),
  SAMPLE_POINTS
)

RESOLUTIONS = [ int(r) for r in RESOLUTIONS ]

def measure_solve_time(i: int, r: int, dt: float, solver: Literal['adi', 'ftcs']) -> float:

  if solver == 'adi':
    config = default_adi_config()
    config.resolution = (r, r)
    config.dt = dt
    solver = ADISolver(config)
    c0 = initial_condition(config)
    with timed(f"ADI {(r, r)} solve time (dt={dt})") as elapsed:
      _ = solver.solve(c0, lambda _: 0)
    return elapsed()

  if solver == 'ftcs':
    config = default_ftcs_config()
    config.resolution = (r, r)
    config.dt = dt
    solver = FTCSSolver(config)
    c0 = initial_condition(config)
    with timed(f"ADI {(r, r)} solve time (dt={dt})") as elapsed:
      _ = solver.solve(c0)
    return elapsed()
  return False

def write_solve_time(i: int, t: float) -> None:
  array = np.load(FILE_PATH)
  array[i] = t
  np.save(FILE_PATH, array)

# %% Calculate solve times
dt = 30 # initial time step

time_steps = np.zeros(SAMPLE_POINTS)

for i, r in enumerate(RESOLUTIONS):
  while True:
    try:
      t = measure_solve_time(i, r, dt, 'adi')
      time_steps[i] = dt
      break
    except:
      dt = dt - 0.1 # try reducing the time step

  write_solve_time(i, t)

np.save('compare-solvers/assets/solve-time-steps.npy', time_steps)

# %% EFD Solve times



# %% Plot times

ts = np.load('compare-solvers/assets/solve-times-backup.npy')

efd_ts = np.load('compare-solvers/assets/efd-solver-duration.npy')

plt.title(f'ADI solver calculation time')
plt.plot([ r*r for r in RESOLUTIONS ], ts / 60)

EFD_R = [ r*r for r in np.linspace(40, 230, 21)]
plt.plot(EFD_R, efd_ts / 60)
plt.xlabel(f'Number of discrete grid points [units]')
plt.ylabel(f'time [min]')
