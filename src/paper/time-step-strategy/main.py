# %% 

from statistics import stdev
import numpy as np
from tqdm import tqdm
from solver_utils import show_solution_frames, pretty_time
from solvers.adi.time_step_strategy import SCGQMStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import MixConfig, large_config
from solvers.adi.solver import Solver
import matplotlib.pyplot as plt

config = large_config(order=0, temperature=1000)
config.time_step_strategy = SCGQMStep(100, 0.1, 2, 60, 0.0301, 5)
c0 = initial_condition(config)

t, c = Solver(config).solve(c0, lambda _: 0)

dt = np.diff(t)

# %%

plt.ylabel('Laiko žingsnio dydis [s]')
plt.xlabel('Laiko žingsnio indeksas [vnt]')
plt.plot(dt[:-1])
plt.savefig('../paper/images/timestep/strategy-no-mix.png', dpi=300, bbox_inches='tight')