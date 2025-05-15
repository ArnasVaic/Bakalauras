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
import os


# 2241.5308158844955

# %%
def duration(order: int, mix_time: float):
  mix_config = MixConfig('perfect', [ mix_time ])
  config = large_config(order=order, temperature=1000, mix_config=mix_config)
  config.frame_stride = 100
  c0 = initial_condition(config)
  solver = Solver(config)
  t, _ = solver.solve(c0, lambda _: 0)
  print(f'Mix moment: {pretty_time(mix_time)}, Duration: {pretty_time(t[-1])}')
  return t[-1]

orders = [0]

for index, order in enumerate(orders):
  t_start, t_end = 35/60 * 3600, 40/60 * 3600
  optimal_mix_time = gss(lambda t: duration(order, t), t_start, t_end, 1)
  
  ts = np.load('optimal-mix-time/T1000.npy')
  ts[index] = optimal_mix_time
  np.save('optimal-mix-time/T1000.npy', ts)
  
  print(f'Optimal mix time for order {order}: {pretty_time(optimal_mix_time)} ({optimal_mix_time})')

# %%

ts = np.load('optimal-mix-time/T1000.npy')
tr = np.zeros_like(ts)
for ord, t in enumerate(ts): 
  tr[ord] = duration(ord, t)
np.save('optimal-mix-time/dur-T1000.npy', tr)

# %%

sizes = [1, 4, 16, 64]
ts = np.load('optimal-mix-time/T1000.npy')
plt.plot(sizes, ts/3600)

# %%
sizes = [1, 4, 16, 64]
ts = np.load('optimal-mix-time/dur-T1000.npy')
plt.plot(sizes, ts/3600)

# %%
plt.rcParams.update({
  'font.size': 14,         # General font size
  'axes.titlesize': 16,    # Title font size
  'axes.labelsize': 14,    # X and Y label size
  'xtick.labelsize': 12,   # X tick label size
  'ytick.labelsize': 12,   # Y tick label size
  'legend.fontsize': 12,   # Legend font size, if you use legends
})

sizes = [1, 4, 16, 64]

mix_times = np.load('optimal-mix-time/T1000.npy')

# Convert to minutes
mix_times_min = [t / 60 for t in mix_times]

# Plotting
plt.figure(figsize=(7,5))
plt.plot(sizes, mix_times_min, marker='o')
plt.xscale('log', base=4)
plt.xlabel('Kiek kartų padidinta sritis')
plt.ylabel('Optimalus maišymo laikas [min]')

# Set custom x-ticks
xticks = sizes
xtick_labels = [f"$4^{int(np.log(x) / np.log(4))}$" for x in xticks]
plt.xticks(xticks, labels=xtick_labels)

# Annotate each point with formatted time, add vertical padding
for x, t in zip(sizes, mix_times_min):
  time_str = pretty_time(t * 60)  # convert back to seconds for formatting
  plt.text(x, t + 0.05, time_str, ha='left', va='bottom')  # +20 min offset for vertical padding

# Add margins so annotations aren't clipped
plt.margins(y=0.1)  # 10% headroom on y-axis
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('../paper/images/mixing/opt-mix-time.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
plt.rcParams.update({
  'font.size': 14,         # General font size
  'axes.titlesize': 16,    # Title font size
  'axes.labelsize': 14,    # X and Y label size
  'xtick.labelsize': 12,   # X tick label size
  'ytick.labelsize': 12,   # Y tick label size
  'legend.fontsize': 12,   # Legend font size, if you use legends
})

sizes = [1, 4, 16, 64]

mix_times = np.load('optimal-mix-time/dur-T1000.npy')

# Convert to minutes
mix_times_min = [t / 3600 for t in mix_times]

# Plotting
plt.figure(figsize=(7,5))
plt.plot(sizes, mix_times_min, marker='o')
plt.xscale('log', base=4)
plt.xlabel('Kiek kartų padidinta sritis')
plt.ylabel('Optimali reakcijos trukmė [val]')

# Set custom x-ticks
xticks = sizes
xtick_labels = [f"$4^{int(np.log(x) / np.log(4))}$" for x in xticks]
plt.xticks(xticks, labels=xtick_labels)

# Annotate each point with formatted time, add vertical padding
for x, t in zip(sizes, mix_times_min):
  time_str = pretty_time(t * 3600)  # convert back to seconds for formatting
  plt.text(x, t + 0.002, time_str, ha='left', va='bottom')  # +20 min offset for vertical padding

# Add margins so annotations aren't clipped
plt.margins(y=0.1)  # 10% headroom on y-axis
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('../paper/images/mixing/opt-duration.png', dpi=300, bbox_inches='tight')
plt.show()