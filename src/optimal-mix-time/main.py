# %%

import numpy as np
from golden_search import gss
from solver_utils import show_solution_frames, pretty_time, timed
from solvers.adi.time_step_strategy import SCGQMStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import MixConfig, large_config
from solvers.adi.solver import Solver
from solvers.adi.config import Config
import matplotlib.pyplot as plt

def duration(order: int, mix_time: float):
  mix_config = MixConfig('perfect', [ mix_time ])
  config = large_config(order=order, temperature=1000, mix_config=mix_config)
  config.frame_stride = 100
  c0 = initial_condition(config)
  solver = Solver(config)
  with timed("solved in") as elapsed:
    t, _ = solver.solve(c0, lambda _: 0)
  print(pretty_time(t[-1]))
  return t[-1]

ts = []
orders = [2, 3]
for order in orders:
  t_start, t_end = 35/60 * 3600, 45/60 * 3600
  optimal_mix_time = gss(lambda t: duration(order, t), t_start, t_end, 1)
  ts.append(optimal_mix_time)

  print(f't = {pretty_time(optimal_mix_time)} ({optimal_mix_time})')


# %%

sizes = [1, 4, 16, 64]
mix_times = np.repeat(2432.141720425366, 4)
plt.plot(sizes, mix_times)
