# %% This file is supposed to be an upgrade from the last by cache'ing solve times

# We'll use one numpy array file with multiple
# entries to save solve times for different grid sizes

import os
import numpy as np
import matplotlib.pyplot as plt

from solver_utils import timed
from solvers.initial_condition import initial_condition
from solvers.adi.config import Config
from solvers.adi.solver import Solver

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
 
def capture_none(_):
  return 0

def measure_solve_time(i: int, r: int, dt: float) -> float:
  config = Config()
  config.resolution = (r, r)
  config.dt = dt
  solver = Solver(config)
  c0 = initial_condition(config)
  with timed(f"ADI {(r, r)} solve time (dt={dt})") as elapsed:
    _ = solver.solve(c0, capture_none)
  return elapsed()

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
      t = measure_solve_time(i, r, dt)
      time_steps[i] = dt
      break
    except:
      dt = dt - 0.1 # try reducing the time step

  write_solve_time(i, t)

np.save('compare-solvers/assets/solve-time-steps.npy', time_steps)

# %% Plot times

ts = np.load('compare-solvers/assets/solve-times-backup.npy')

plt.title(f'ADI solver calculation time')
plt.plot([ r*r for r in RESOLUTIONS ], ts / 60)
plt.xlabel(f'Number of discrete grid points [units]')
plt.ylabel(f'time [min]')
