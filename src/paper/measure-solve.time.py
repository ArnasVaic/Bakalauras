# %% This file is supposed to be an upgrade from the last by cache'ing solve times

# We'll use one numpy array file with multiple
# entries to save solve times for different grid sizes

import logging
import os
import sys
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt

from solver_utils import timed
from solvers.adi.time_step_strategy import SCGQStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import default_config as  default_adi_config
from solvers.ftcs.config import default_config as default_ftcs_config
from solvers.adi.solver import Solver as ADISolver
from solvers.ftcs.solver import Solver as FTCSSolver

def initialize_file(filename: str, points: int) -> None:
  if not os.path.isfile(filename):
    array = np.zeros(points)
    np.save(filename, array)
    del array

  array = np.load(filename)
  if array.shape[0] != points: #  resize the array
    assert array.shape[0] > points, "Shrinking the array is not supported"
    array = np.resize(array, points)
    np.save(filename, array)
    del array

def write_result(filepath: str, index: int, solve_time: float) -> None:
  array = np.load(filepath)
  array[index] = solve_time
  np.save(filepath, array)

def generate_resolutions(
  initial_resolution: int,
  resolution_step: int,
  sample_points: int
) -> list[int]:
  return np.linspace(
    initial_resolution,
    initial_resolution + resolution_step * (sample_points - 1),
    sample_points
  ).astype(int).tolist()

def measure_adi(resolution: int) -> float:
  config = default_adi_config()
  config.resolution = (resolution, resolution)
  config.time_step_strategy = SCGQStep(200, 1, 2, 250, 0.0301, 10)
  solver = ADISolver(config, logging.getLogger("__name__"))
  c0 = initial_condition(config)
  with timed(f"ADI {(resolution, resolution)} solve time (dt={config.dt})") as elapsed:
    _ = solver.solve(c0, lambda _: 0)
  return elapsed()

def measure_ftcs(resolution: int) -> float:
  config = default_ftcs_config()
  config.resolution = (resolution, resolution)
  config.frame_stride = sys.maxsize # disable frame capture
  solver = FTCSSolver(config)
  c0 = initial_condition(config)
  with timed(f"ADI {(resolution, resolution)} solve time (dt={solver.dt_bound()})") as elapsed:
    _ = solver.solve(c0)
  return elapsed()

SAMPLE_POINTS = 20
INITIAL_RESOLUTION = 40
RESOLUTION_STEP = 10

ADI_FILE_PATH = 'compare-solvers/assets/adi-solve-times.npy'
FTCS_FILE_PATH = 'compare-solvers/assets/ftcs-solve-times.npy'

initialize_file(ADI_FILE_PATH, SAMPLE_POINTS)
initialize_file(FTCS_FILE_PATH, SAMPLE_POINTS)

RESOLUTIONS = generate_resolutions(INITIAL_RESOLUTION, RESOLUTION_STEP, SAMPLE_POINTS)

# %% ADI method

for index, resolution in enumerate(RESOLUTIONS):
  t = measure_adi(resolution)
  write_result(ADI_FILE_PATH, index, t)

# %% FTCS Solve times

for index, resolution in enumerate(RESOLUTIONS):
  t = measure_ftcs(resolution)
  write_result(FTCS_FILE_PATH, index, t)

# %% Plot times

points = [ r * r for r in RESOLUTIONS ]
adi_solve_ts = np.load(ADI_FILE_PATH) / 60
ftcs_solve_ts = np.load(FTCS_FILE_PATH) / 60

plt.plot(points, adi_solve_ts)
plt.plot(points, ftcs_solve_ts)

plt.xlabel(f'Diskrečių taškų skaičius [vnt]', fontsize=14)
plt.ylabel(f'Skaičiavimo laikas [min]', fontsize=14)

plt.savefig('../paper/images/ftcs-adi-perf.png', dpi=300, bbox_inches='tight')
plt.show()
