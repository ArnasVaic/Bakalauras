# %%

import numpy as np
from solver_utils import get_quantity_over_time, show_solution_frame, show_solution_frames
from solvers.initial_condition import initial_condition
from solvers.adi.config import MixConfig, large_config
from solvers.adi.solver import Solver

mix = MixConfig('perfect', [ 1.2345 * 3600 ])
config = large_config(order=0, temperature=1000, mix_config=mix)
c0 = initial_condition(config)

# %%
t, c = Solver(config).solve(c0, lambda f: f.copy())

print(c.shape)
# %%

import matplotlib.pyplot as plt
from solver_utils import pretty_time

dt = np.diff(t)
plt.plot(dt)
plt.axline((1330, 0), (1330, 30), linestyle='dashed')

# %%

show_solution_frames(t, c, [0,1320,1330,2000,-1], 2)
# %%
