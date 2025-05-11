# %% 
import os
from statistics import stdev
import numpy as np
from tqdm import tqdm
from solver_utils import get_quantity_over_time, show_solution_frames, pretty_time
from solvers.adi.time_step_strategy import ConstantTimeStep, SCGQMStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import MixConfig, large_config
from solvers.adi.solver import Solver
import matplotlib.pyplot as plt

T = 1000
ORDERS = [0, 1, 2, 3, 4, 5, 6]
step_strategy = SCGQMStep(100, 0.1, 1.5, 30, 0.0301, 5)

durations = np.zeros_like(ORDERS)

for index, order in enumerate(ORDERS):
  config = large_config(order, T)
  config.time_step_strategy = step_strategy
  # dont need to save any frames, except last
  config.frame_stride = 1000000
  c0 = initial_condition(config)

  t, c = Solver(config).solve(c0, lambda _: 0)
  durations[index] = t[-1]

np.save('paper/duration-order/durations.npy', durations)

# %% Show difference based on order

durations = np.load('paper/duration-order/durations.npy')
sizes = 2 ** np.array(ORDERS)
plt.plot(sizes, durations / 3600)
plt.ylabel("Reakcijos pabaigos laikas [val]")
plt.xlabel("Kiek kartų padidinta erdvė")
plt.savefig('../paper/images/mixing/duration-order-T1000.png', dpi=300, bbox_inches='tight')
