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

T = 1600
ORDER = 0

frame_strides = {
  1000: 100,
  1200: 100,
  1600: 100
}

step_strategy = SCGQMStep(100, 0.1, 1.5, 30, 0.0301, 5)

# %% Baseline: duration when mixing does not occur

config = large_config(ORDER, T)
config.time_step_strategy = step_strategy
c0 = initial_condition(config)

t, c = Solver(config).solve(c0, lambda _: 0)

print(t[-1])
np.save(f'paper/large-perfect-mix/baseline-ord{ORDER}-T{T}.npy', [t[-1]])

# %% Mixing: reaction duration when perfect mixing occurs

baseline = np.load(f'paper/large-perfect-mix/baseline-ord{ORDER}-T{T}.npy')

# First part: from 0 to 1.5 with 20 points
moments_1 = np.linspace(0, 2 * 3600, 25)

# Second part: from 1.5 to baseline with 20 points
moments_2 = np.linspace(2 * 3600, *baseline, 15)

# Concatenate them (excluding duplicate 1.5 at the start of moments_2)
MOMENTS = np.concatenate((moments_1, moments_2[1:]))

# %%
duration = np.zeros_like(MOMENTS)

for index, moment in tqdm(enumerate(MOMENTS)):
  mix_config = MixConfig('perfect', [moment])
  config = large_config(ORDER, T, mix_config)
  config.frame_stride = 100
  config.time_step_strategy = step_strategy
  c0 = initial_condition(config)

  t, c = Solver(config).solve(c0, lambda _: 0)
  duration[index] = t[-1]

  print(f"mix moment: {pretty_time(moment)}, duration: {pretty_time(t[-1])}")

np.save(f'paper/large-perfect-mix/duration-ord{ORDER}-T{T}.npy', duration)

# %%

ORDER = 0
baseline = np.load(f'paper/large-perfect-mix/baseline-ord{ORDER}-T{T}.npy')
duration = np.load(f'paper/large-perfect-mix/duration-ord{ORDER}-T{T}.npy')

# Assuming baseline is a scalar or single-value array
baseline_value = baseline.item()  # in case it's a 0-d array

plt.figure(figsize=(8, 5))

plt.plot(MOMENTS / 3600, duration / 3600, label='Reakcijos trukmė maišant')
plt.plot(
  MOMENTS / 3600,
  np.full(len(MOMENTS), baseline_value) / 3600,
  label='Reakcijos trukmė nemaišant')

plt.xlabel('Maišymo momentas [val]')
plt.ylabel('Reakcijos trukmė [val]')
plt.legend()
plt.grid(True)
# plt.savefig('../paper/images/mixing/duration-mix-moment-dependance-ord0.png', dpi=300, bbox_inches='tight')
plt.show()


# %% Manual test

def runc(order, temperature, mix_config):
  cfg = large_config(order, temperature, mix_config)
  cfg.time_step_strategy = step_strategy
  cfg.frame_stride = 100
  c0 = initial_condition(cfg)
  return Solver(cfg).solve(c0, lambda f: f.copy())

def runq(order, temperature, mix_config):
  cfg = large_config(order, temperature, mix_config)
  cfg.time_step_strategy = step_strategy
  cfg.frame_stride = 100
  c0 = initial_condition(cfg)
  t, c = Solver(cfg).solve(c0, lambda f: f.copy())
  return t, get_quantity_over_time(cfg, c)

# %%
ORDER = 0
T = 1000
cfg = MixConfig('perfect', [1.0 * 3600])
# t1, c1 = runc(ORDER, T, cfg)
t1, q1 = runq(ORDER, T, cfg)
t2, q2 = runq(ORDER, T, None)

# %% View solution
frames = [0, 10, 11, 30, 50]
show_solution_frames(t1, c1, frames, 1)
show_solution_frames(t1, c1, frames, 0)
show_solution_frames(t1, c1, frames, 2)

# %%

# Set global font size
plt.rcParams.update({'font.size': 14})

mix_frame = 13
plt.figure(figsize=(8, 5))

# Plot lines
plt.plot(t1 / 3600, q1[:, 2], label='Medžiagos maišomos', color='blue')
plt.plot(t2 / 3600, q2[:, 2], label='Medžiagos nemaišomos', color='orange')

# Labels with increased font size
plt.xlabel('Laikas [val]', fontsize=16)
plt.ylabel('Produkto kiekis sistemoje [g]', fontsize=16)

# Highlight points
plt.scatter(t1[mix_frame] / 3600, q1[mix_frame, 2], color='red', label='Maišymo momentas')
plt.scatter(t1[-1] / 3600, q1[-1, 2], color='blue', label='Reakcijos pabaiga maišant')
plt.scatter(t2[-1] / 3600, q2[-1, 2], color='orange', label='Reakcijos pabaiga nemaišant')

# Legend with larger font
plt.legend(fontsize=14)

# Optional: increase tick label size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Save figure
plt.savefig('../paper/images/mixing/perfect-mix-vs-no-mix-ord-0-q3.png', dpi=300, bbox_inches='tight')

# Show it
plt.show()