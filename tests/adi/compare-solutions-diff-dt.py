# %%

from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition 
import datetime
from solvers.ftcs.config import Config
from solvers.adi.solver import Solver as ADISolver
from solvers.ftcs.solver import Solver as FTCSSolver

efd_config, adi_config = Config()

efd_config.dt = 25
adi_config.dt = 250 # explicit dt for ADI

t1, c1 = ADISolver(adi_config).solve(initial_condition(adi_config), lambda f: f.copy())
t2, c2 = FTCSSolver(efd_config).solve(initial_condition(efd_config))

q1, q2 = c1.sum(axis=(2, 3)), c2.sum(axis=(2, 3))

print(t1.shape, t2.shape)

# %% Compare quantities over time
for m in range(3):
  plt.plot(t1, q1[:, m], label=f"Explicit FTCS, $q_{m + 1}$")
  plt.plot(t2, q2[:, m], linestyle='dashed', label=f"ADI, $q_{m + 1}$")
plt.legend()

# %% Compare individual frames
frames = [1, 4, 10, 20]
element = 0
T = min(c1.shape[0], c2.shape[0])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
dc = c1[:T] - c2[:T]

for ax, frame in zip(axes, frames): 
  im = ax.imshow(dc[frame, element], cmap='RdBu', vmin=dc.min(), vmax=dc.max())
  ax.set_title(f't = {frame * config.dt / 3600:.02f}h')
  fig.colorbar(im, ax=ax)

plt.tight_layout()