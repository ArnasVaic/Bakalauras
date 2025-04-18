# %% Measure EFD solver time in comparison to ADI

from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition 
from solver_utils import timed
from solvers.adi.config import Config
from solvers.adi.solver import Solver as ADISolver
from solvers.efd.solver import Solver as EFDSolver

COMMON_TIME_STEP = 25

def capture_none(frame: np.ndarray) -> np.ndarray:
  return np.array([0.0])

sizes = [ 40, 60, 80, 100, 110, 120, 200 ]
steps = [ 25, 20, 20, 20,  20,  20,  20]


def measure_solver(config: Config, solver_type: str) -> float:

  if solver_type == 'adi':
    solver = ADISolver(config)
  elif solver_type == 'efd':
    solver = EFDSolver(config)
  else:
    raise Exception("Unknown solver type. Supported types: adi, efd.")

  c0 = initial_condition(config)
  with timed(f"{solver_type} {config.resolution} Solve time") as elapsed:
    _ = solver.solve(c0, capture_none) # discard results
    t = elapsed()
  return t

efd_ts, adi_ts = [], []

for size in sizes:

  config = Config()
  config.resolution = (size, size)

  # let EFD method choose it's own time step because
  # upper bound gets smaller with resolution
  # efd_t = measure_solver(config, 'efd')

  config.dt = COMMON_TIME_STEP
  adi_t = measure_solver(config, 'adi')

  # efd_ts.append(efd_t)
  adi_ts.append(adi_t)

# plt.plot(sizes, efd_ts, label=f'EFD')
plt.plot(sizes, adi_ts, label=f'ADI')

plt.xlabel(f'grid side-length [units]')
plt.ylabel(f'solution time [s]')
plt.legend()