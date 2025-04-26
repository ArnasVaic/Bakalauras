# Compare duration of different solvers

# %% Imports & constants
import os
import logging

import numpy as np

from solver_utils import timed
from solvers.efd.config import default_config as default_efd_config
from solvers.efd.solver import Solver as FTCSSolver
from solvers.initial_condition import initial_condition

SAMPLE_POINTS = 20

INITIAL_RESOLUTION = 40
RESOLUTION_STEP = 10
RESOLUTIONS = np.linspace(
  INITIAL_RESOLUTION,
  INITIAL_RESOLUTION + RESOLUTION_STEP * (SAMPLE_POINTS - 1),
  SAMPLE_POINTS
)
RESOLUTIONS = [ int(r) for r in RESOLUTIONS ]

# %% Create file for cache'ing results

FILE_PATH = 'compare-solvers/assets/efd-solver-duration.npy'

if not os.path.isfile(FILE_PATH):
  # if the number of sample points
  # changes, the file will have to
  # be deleted manually
  array = np.zeros(SAMPLE_POINTS)
  np.save(FILE_PATH, array)
  del array

def write_solve_time(i: int, t: float) -> None:
  array = np.load(FILE_PATH)
  array[i] = t
  np.save(FILE_PATH, array)
  del array

# %% Cache EFD solver durations for different resolutions

def measure_ftcs_solve_time(i: int, r: int) -> float:
  config = default_efd_config()
  config.resolution = (r, r)
  solver = FTCSSolver(config)
  c0 = initial_condition(config)
  with timed(f'FTCS {r}x{r}') as elapsed:
    _ = solver.solve(c0, logging.getLogger(__name__))
  return elapsed()


for i, r in enumerate(RESOLUTIONS):
  if i < 11:
    continue
  t = measure_ftcs_solve_time(i, r)
  write_solve_time(i, t)
