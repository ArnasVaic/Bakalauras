# %%

import numpy as np
from solver_utils import get_quantity_over_time, show_solution_frame, timed, validate_solution_stable
from solvers.adi.time_step_strategy import ClampedArithmeticTimeStep, ConstantTimeStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import Config
from solvers.adi.solver import Solver
from solvers.stopper import TotalStepsStopper

config = Config()
config.time_step_strategy = ConstantTimeStep(25)
# ClampedArithmeticTimeStep(25, 0.1, 200)
config.frame_stride = 20

c0 = initial_condition(config)

with timed("solved in"):
  t, c = Solver(config).solve(c0, lambda f: f.copy())


# %%

show_solution_frame(config, t, c, frame=-1, element=2)
print(c.shape)
q = get_quantity_over_time(config, c)
print(f'final quantity of c3: {q[-1, 2]:.2e} [g]')