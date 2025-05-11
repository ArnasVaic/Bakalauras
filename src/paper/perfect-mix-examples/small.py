# %% 

from statistics import stdev
import numpy as np
from tqdm import tqdm
from solver_utils import get_quantity_over_time, show_solution_frames, pretty_time
from solvers.adi.time_step_strategy import ConstantTimeStep, SCGQMStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import MixConfig, large_config
from solvers.adi.solver import Solver
import matplotlib.pyplot as plt

PATH_BASE = 'paper/large-perfect-mix/'

frame_strides = {
  1000: 50,
  1200: 100,
  1600: 100
}

step_strategy = SCGQMStep(100, 0.1, 1.5, 30, 0.0301, 5)

# %% Basic example on smallest space

T = 1000
mix_cfg = MixConfig('perfect', [1 * 3600])
cfg = large_config(order=0, temperature=T, mix_config=mix_cfg)
cfg.time_step_strategy = step_strategy
cfg.frame_stride = frame_strides[T]
c0 = initial_condition(cfg)

# %%
t, c = Solver(cfg).solve(c0, lambda f: f.copy())
np.save(PATH_BASE + 'example_1_t.npy', t)
np.save(PATH_BASE + 'example_1_c.npy', c)

# %% Visualize small example

t = np.load(PATH_BASE + 'example_1_t.npy')
c = np.load(PATH_BASE + 'example_1_c.npy')

frames = [0, 25, 26, 40, 50]
show_solution_frames(t, c, frames, 0, PATH_BASE + 'perfect-mix-ord-0-c1.png', config=cfg)
show_solution_frames(t, c, frames, 2, PATH_BASE + 'perfect-mix-ord-0-c3.png', config=cfg)

# %% Quantity in small example vs normal

