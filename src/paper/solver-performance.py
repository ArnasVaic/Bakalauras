# %% 

from typing import Literal

from solvers.adi.solver import Solver as ADISolver
from solvers.adi.config import default_config as  default_adi_config

from solvers.ftcs.solver import Solver as FTCSSolver
from solvers.ftcs.config import default_config as default_ftcs_config

from solvers.initial_condition import initial_condition
from solver_utils import timed

def measure_adi(resolution: int) -> float:

  config = default_adi_config()
  config.resolution = (resolution, resolution)
  config.dt = FTCSSolver(config).dt_bound()
  solver = ADISolver(config)
  c0 = initial_condition(config)
  with timed(f"ADI {(resolution, resolution)} solve time (dt={config.dt})") as elapsed:
    _ = solver.solve(c0, lambda _: 0)
  return elapsed()

def measure_ftcs(resolution: int) -> float:

  config = default_ftcs_config()
  config.resolution = (resolution, resolution)
  solver = FTCSSolver(config)
  c0 = initial_condition(config)
  with timed(f"ADI {(resolution, resolution)} solve time (dt={solver.dt_bound()})") as elapsed:
    _ = solver.solve(c0)
  return elapsed()

measure_ftcs(40)