
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