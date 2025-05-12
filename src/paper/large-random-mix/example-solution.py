# %%
import numpy as np
from solver_utils import show_solution_frames, pretty_time
from solvers.adi.time_step_strategy import SCGQMStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import MixConfig, large_config
from solvers.adi.solver import Solver
import matplotlib.pyplot as plt

# %% Solve

mix_config = MixConfig('random', [ 1.0 * 3600 ])
config = large_config(order=1, temperature=1000, mix_config=mix_config)
config.frame_stride = 100
c0 = initial_condition(config)

t, c = Solver(config).solve(c0, lambda f: f.copy())

# %% Show

frames = [0, 12, 13, 20, 30]
show_solution_frames(t, c, frames, 0, config=config, filename='../paper/images/mixing/random-mix-ord-1-c1.png')
show_solution_frames(t, c, frames, 1, config=config, filename='../paper/images/mixing/random-mix-ord-1-c2.png')
show_solution_frames(t, c, frames, 2, config=config, filename='../paper/images/mixing/random-mix-ord-1-c3.png')