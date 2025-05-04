# %% Imports

import numpy as np
from solver_utils import get_quantity_over_time, show_solution_frames, timed, validate_solution_stable
from solvers.adi.time_step_strategy import ClampedArithmeticTimeStep, ConstantTimeStep, ACAStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import default_config
from solvers.adi.solver import Solver
from solvers.stopper import TotalStepsStopper

T = 1600

config = default_config(temperature=T)
config.size = (80, 80)
config.time_step_strategy = ConstantTimeStep(10)
c0 = initial_condition(config)

# %% Solve
with timed("solved in"):
  t, c = Solver(config).solve(c0, lambda f: f.copy())

np.save(f'paper/adi_example_{T}C_t.npy', t)
np.save(f'paper/adi_example_{T}C_c.npy', c)

# %% Generate visualization

t = np.load(f'paper/adi_example_{T}C_t.npy')
c = np.load(f'paper/adi_example_{T}C_c.npy')

frames = [0, 10, 50, 100, -1] if T == 1000 else [0, 50, 200, 400, -1]

show_solution_frames(config, t, c, frames=frames, element=2, filename=f'../paper/images/adi/c3-{T}C.png')
show_solution_frames(config, t, c, frames=frames, element=0, filename=f'../paper/images/adi/c1-{T}C.png')
print(c.shape)
q = get_quantity_over_time(config, c)
print(f'final quantity of c3: {q[-1, 2]:.2e} [g]')
