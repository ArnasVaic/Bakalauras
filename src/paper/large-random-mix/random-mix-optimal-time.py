# %%
import numpy as np
from solver_utils import show_solution_frames, pretty_time
from solvers.adi.time_step_strategy import SCGQMStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import MixConfig, large_config
from solvers.adi.solver import Solver
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% Baseline
config = large_config(order=1, temperature=1000)
config.frame_stride = 100
c0 = initial_condition(config)
t, c = Solver(config).solve(c0, lambda f: f.copy())
baseline = np.array([t[-1]])
np.save('paper/large-random-mix/baseline-T1000-ord01.npy', baseline)

# %% Solve

OPTIMAL_MIX_TIME = 2/3 * 3600
SAMPLES = 20

durations = np.zeros(SAMPLES)

for sample in tqdm(range(SAMPLES)):
  mix_config = MixConfig('random', [ OPTIMAL_MIX_TIME ])
  config = large_config(order=1, temperature=1000, mix_config=mix_config)
  config.frame_stride = 100
  c0 = initial_condition(config)

  t, c = Solver(config).solve(c0, lambda f: f.copy())
  durations[sample] = t[-1]

np.save('paper/large-random-mix/sample-durations.npy', durations)
# %% Show

xs = np.arange(SAMPLES)
plt.plot(xs, np.repeat(baseline, SAMPLES))
plt.plot(xs, durations)

# %%
frames = [0, 12, 13, 20, 30]
show_solution_frames(t, c, frames, 0, config=config, filename='../paper/images/mixing/random-mix-ord-1-c1.png')
show_solution_frames(t, c, frames, 1, config=config, filename='../paper/images/mixing/random-mix-ord-1-c2.png')
show_solution_frames(t, c, frames, 2, config=config, filename='../paper/images/mixing/random-mix-ord-1-c3.png')