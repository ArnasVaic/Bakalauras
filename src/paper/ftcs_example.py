# %% Imports

import numpy as np
from solver_utils import get_quantity_over_time, show_solution_frames, timed
from solvers.initial_condition import initial_condition
from solvers.efd.config import default_config
from solvers.efd.solver import Solver

T = 1600

config = default_config(temperature=T)
config.size = (80, 80)
c0 = initial_condition(config)

# %% Solve
with timed("solved in"):
  t, c = Solver(config).solve(c0)

np.save(f'paper/ftcs_example_{T}C_t.npy', t)
np.save(f'paper/ftcs_example_{T}C_c.npy', c)

# %% Generate visualization

t = np.load(f'paper/ftcs_example_{T}C_t.npy')
c = np.load(f'paper/ftcs_example_{T}C_c.npy')

frames = [0, 10, 50, 100, -1] if T == 1000 else [0, 50, 200, 400, -1]

show_solution_frames(config, t, c, frames=frames, element=2, filename=f'../paper/images/ftcs/c3-{T}C.png', cmap='inferno')
show_solution_frames(config, t, c, frames=frames, element=0, filename=f'../paper/images/ftcs/c1-{T}C.png', cmap='inferno')
print(c.shape)
q = get_quantity_over_time(config, c)
print(f'final quantity of c3: {q[-1, 2]:.2e} [g]')
