# %%

import logging
from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition 
import datetime
from solvers.ftcs.config import Config
from solvers.adi.solver import Solver as ADISolver
from solvers.ftcs.solver import Solver as EFDSolver
from solvers.stopper import TotalStepsStopper

logging.basicConfig(
  filename='debug.log',
  filemode='w',
  format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.DEBUG)

efd_config, adi_config = Config(),  Config()

# no reaction
efd_config.logger = logging.getLogger(__name__)
efd_config.k = 0
adi_config.k = 0

efd_config.resolution = (100, 100)
adi_config.resolution = (100, 100)

# need to use total step stopper
efd_config.stopper = TotalStepsStopper(1000)
adi_config.stopper = TotalStepsStopper(100)

efd_config.dt = 4  # explicit dt for EFD
adi_config.dt = 400 # explicit dt for ADI

t2, c2 = EFDSolver(efd_config).solve(initial_condition(efd_config))
t1, c1 = ADISolver(adi_config).solve(initial_condition(adi_config))

q1, q2 = c1.sum(axis=(2, 3)), c2.sum(axis=(2, 3))

print(t1.shape, t2.shape)

# %% Compare quantities over time
for m in range(3):
  plt.plot(t1, q1[:, m], label=f"Explicit FTCS, $q_{m + 1}$")
  plt.plot(t2, q2[:, m], linestyle='dashed', label=f"ADI, $q_{m + 1}$")
plt.legend()

# %% Compare individual frames
frames = [1, 10, 20, 999]
element = 0
T = min(c1.shape[0], c2.shape[0])

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

c_max = max(c1.max(), c2.max())
c_min = min(c1.min(), c2.min())

for ax, frame in zip(axes, frames): 

  adi_frame = int(efd_config.dt / adi_config.dt * frame)

  s1 = c1[adi_frame, element, :, :]
  s2 = c2[frame, element, :, :]
  eps = 1e-3 * c1.max()

  err = s2 - s1

  im = ax.imshow(err, cmap='RdBu', vmin=c_min, vmax=c_max)
  ax.set_title(f't = {frame * efd_config.dt / 3600:.02f}h')
  fig.colorbar(im, ax=ax)

plt.tight_layout()

# plt.imshow(dc[frame, element], cmap='RdBu', vmin=dc.min(), vmax=dc.max())
# plt.colorbar()
# plt.legend()