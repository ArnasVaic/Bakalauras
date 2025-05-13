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
orders = [0]
for order in orders:
  t_start, t_end = 35/60 * 3600, 45/60 * 3600
  optimal_mix_time = gss(lambda t: duration(order, t), t_start, t_end, 5)
  ts.append(optimal_mix_time)

  print(f't = {pretty_time(optimal_mix_time)} ({optimal_mix_time})')


# %%

sizes = [1, 4, 16, 64]
mix_times = [
  2332.7335937499997,
  2432.141720425366,
  2319.7787366300154,
  2272.185302734374
]
plt.plot(sizes, mix_times/3600)

# %%
import matplotlib.pyplot as plt

sizes = [1, 4, 16, 64]
mix_times = [
  2332.7335937499997,
  2432.141720425366,
  2319.7787366300154,
  2272.185302734374
]

# Convert to minutes
mix_times_min = [t / 60 for t in mix_times]

# Plotting
plt.figure(figsize=(7,5))
plt.plot(sizes, mix_times_min, marker='o')
plt.xscale('log', base=2)
plt.xlabel('Kiek kartų padidinta sritis')
plt.ylabel('Optimalus maišymo laikas [min]')

# Set custom x-ticks
xticks = sizes
xtick_labels = [f"$2^{int(np.log2(x))}$" for x in xticks]
plt.xticks(xticks, labels=xtick_labels)

# Annotate each point with formatted time, add vertical padding
for x, t in zip(sizes, mix_times_min):
  time_str = pretty_time(t * 60)  # convert back to seconds for formatting
  plt.text(x, t + 0.05, time_str, ha='left', va='bottom')  # +20 min offset for vertical padding

# Add margins so annotations aren't clipped
plt.margins(y=0.1)  # 10% headroom on y-axis
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('../paper/images/mixing/opt-mix-time-.png', dpi=300, bbox_inches='tight')
plt.show()