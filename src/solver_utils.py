
import numpy as np
from solvers.config import Config
from solvers.efd.solver import Solver
from solvers.initial_condition import initial_condition

def reaction_end_time(config: Config, t: float) -> float:
  config.mixer.mix_times = [ t ]
  solver = Solver(config)
  c0 = initial_condition(config)
  ts, _ = solver.solve(c0)
  t_end = ts[-1] * solver.dt
  return t_end

def get_quantity_over_time(config: Config, solution: np.ndarray) -> np.ndarray:
  # shape of the solution is assumed to be [T, 3, W, H]
  total_points = config.resolution[0] * config.resolution[1]
  return solution.sum(axis=(2, 3)) / total_points