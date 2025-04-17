# %%
# We want to see how reaction end time depends 
# on the resolution ran with the same parameters.

import datetime
import numpy as np
import matplotlib.pyplot as plt

from solver_utils import timed
from solvers.initial_condition import initial_condition
from solvers.adi.config import Config
from solvers.adi.solver import Solver

SAMPLE_POINTS = 15

INITIAL_RESOLUTION = 40
RESOLUTION_STEP = 10
RESOLUTIONS = np.linspace(
  INITIAL_RESOLUTION,
  INITIAL_RESOLUTION + RESOLUTION_STEP * (SAMPLE_POINTS - 1),
  SAMPLE_POINTS
)

RESOLUTIONS = [ int(r) for r in RESOLUTIONS ]

# %%
# Generate solutions with varying resolutions

dts = np.load('compare-solvers/assets/solve-time-steps.npy')

end_times = np.zeros(SAMPLE_POINTS)

for i, r in enumerate(RESOLUTIONS):
  config = Config()
  config.resolution = (r, r)
  config.dt = dts[i]

  solver = Solver(config)
  c0 = initial_condition(config)
  with timed(f"ADI {(r, r)} solve time (dt={dts[i]})") as elapsed:
    t, _ = solver.solve(c0, lambda _: 0)
  end_times[i] = t[-1]

np.save('compare-solvers/assets/end-times', end_times)

# %% Plot data

end_times = np.load('compare-solvers/assets/end-times.npy')
plt.plot(RESOLUTIONS, end_times / 3600)

plt.plot(
  RESOLUTIONS,
  np.repeat(end_times.min() / 3600, len(RESOLUTIONS)),
  label=str(datetime.timedelta(seconds=int(end_times.min()))),
  linestyle='dashed')

plt.plot(
  RESOLUTIONS,
  np.repeat(end_times.max() / 3600, len(RESOLUTIONS)),
  label=str(datetime.timedelta(seconds=int(end_times.max()))),
  linestyle='dashed')

plt.ylabel(f'reaction end time [h]')
plt.xlabel(f'square grid sidelength [units]')
plt.legend()