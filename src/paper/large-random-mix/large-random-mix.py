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

T = 1000
ORDER = 2

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
ORDER = 2
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
plt.plot(t1 / 3600, q1[:,2], label='mix')
plt.plot(t2 / 3600, q2[:,2],label='no mix')
plt.plot(np.repeat(1.0, 2), [0, 2.5 * 10**-6], linestyle='dashed' )
plt.legend()