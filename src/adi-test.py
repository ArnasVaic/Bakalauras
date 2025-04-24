# %%

import numpy as np
from solver_utils import get_quantity_over_time, show_solution_frame, timed, validate_solution_stable
from solvers.adi.time_step_strategy import ClampedArithmeticTimeStep, ConstantTimeStep, ACAStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import default_config
from solvers.adi.solver import Solver
from solvers.stopper import TotalStepsStopper

config = default_config()
config.time_step_strategy = ACAStep(25, 0.0000, 50, 0.0301)
c0 = initial_condition(config)

with timed("solved in"):
  t, c = Solver(config).solve(c0, lambda f: f.copy())

# %%

show_solution_frame(config, t, c, frame=-1, element=2)
print(c.shape)
q = get_quantity_over_time(config, c)
print(f'final quantity of c3: {q[-1, 2]:.2e} [g]')
# %%
