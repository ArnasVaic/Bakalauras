# %% 

import numpy as np
from solver_utils import show_solution_frames, timed
from solvers.adi.time_step_strategy import SCGQMStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import large_config
from solvers.adi.solver import Solver

T = 1000

frame_strides = {
  1000: 100,
  1200: 100,
  1600: 100
}

config = large_config(1, T, 'random', [1.5 * 3600])
config.time_step_strategy = SCGQMStep(100, 0.1, 2, 60, 0.0301, 5)
config.frame_stride = frame_strides[T]
c0 = initial_condition(config)
 
# %% Solve
with timed("solved in"):
  t, c = Solver(config).solve(c0, lambda f: f.copy())

print(c.shape)

# %%
frames = [0,4,10,15,18]
show_solution_frames(config, t, c, frames, 1)