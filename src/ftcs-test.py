# %%
from matplotlib import pyplot as plt
import numpy as np  
from solver_utils import get_quantity_over_time
from solvers.initial_condition import initial_condition 
from solvers.ftcs.config import Config
from solvers.ftcs.solver import Solver

# %%
config = Config()
c0 = initial_condition(config)
solver = Solver(config)
t, c = solver.solve(c0)

# %% 

q = get_quantity_over_time(config, c)
ts = t / 3600
N = t.shape[0]

label = lambda x: f'$c_{x}$'
plt.plot(ts, q[:, 0], label=label(1))
# plt.plot(ts, q[:, 1], label=label(2))
plt.plot(ts, 0.03 * np.repeat(q[0, 0], N), linestyle='dashed', label='Reakcijos stabdymo slenkstis')
# plt.plot(ts, 0.03 * np.repeat(q[0, 1], N), linestyle='dashed')
plt.xlabel('t [h]')
plt.ylabel('quantity [g]')
plt.legend()
plt.show()