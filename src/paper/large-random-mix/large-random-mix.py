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

T = 1000
ORDER = 2

frame_strides = {
  1000: 100,
  1200: 100,
  1600: 100
}

step_strategy = SCGQMStep(100, 0.1, 2, 60, 0.0301, 5)


# %% Baseline: duration when mixing does not occur

config = large_config(ORDER, T)
config.time_step_strategy = step_strategy
c0 = initial_condition(config)

t, c = Solver(config).solve(c0, lambda _: 0)

print(t[-1])
np.save(f'paper/large-random-mix/baseline-{ORDER}.npy', [t[-1]])

# %% Mixing: average reaction duration when random mixing occurs

baseline = np.load(f'paper/large-random-mix/baseline-{ORDER}.npy')

SAMPLES = 10
MOMENTS = np.linspace(0, baseline, 20)

# %%
avg_dur = np.zeros_like(MOMENTS)
std_dur = np.zeros_like(MOMENTS)

for index, moment in tqdm(enumerate(MOMENTS)):
  mix_config = MixConfig('random', [moment])
  config = large_config(ORDER, T, mix_config)
  config.frame_stride = 100
  config.time_step_strategy = step_strategy
  c0 = initial_condition(config)

  durations = []

  for _ in range(SAMPLES):
    t, c = Solver(config).solve(c0, lambda _: 0)
    durations.append(t[-1])

  avg = sum(durations) / SAMPLES
  stddev = stdev(durations)
  print(f"moment: {pretty_time(moment)}, avg dur: {avg}, stdev: {stddev}")

  avg_dur[index] = avg
  std_dur[index] = stddev

np.save(f'paper/large-random-mix/random-{ORDER}.npy', avg_dur)
np.save(f'paper/large-random-mix/random-stddev-{ORDER}.npy', std_dur)

print(c.shape)

# %%

baseline = np.load(f'paper/large-random-mix/baseline-{ORDER}.npy')
avg_dur = np.load(f'paper/large-random-mix/random-{ORDER}.npy')
plt.plot(MOMENTS[10:] / 3600, avg_dur[10:] / 3600)
plt.plot(MOMENTS[10:] / 3600, np.repeat(*baseline, len(MOMENTS[10:]))/3600)

# %%
frames = [0,4,10,15,18]
show_solution_frames(config, t, c, frames, 1)