# %%

import numpy as np
from solver_utils import show_solution_frame, validate_solution_stable
from solvers.initial_condition import initial_condition 
from solvers.adi.config import Config
from solvers.adi.solver import Solver
from solvers.stopper import TotalStepsStopper

config = Config()
config.dt = 10.0
config.resolution = (120, 120)

config.stopper = TotalStepsStopper(10)

c0 = initial_condition(config)

solver = Solver(config)

def capture_frame(frame: np.ndarray) -> np.ndarray:
  return np.copy(frame)

t, c = solver.solve(c0, capture_frame)


# %%

show_solution_frame(config, t, c, frame=1, element=0)

validate_solution_stable(config, c)
