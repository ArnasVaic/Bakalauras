# %%

import logging
import numpy as np
import matplotlib.pyplot as plt
from solver_utils import get_quantity_over_time
from solvers.adi.time_step_strategy import ConstantTimeStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import default_config as default_adi_config
from solvers.ftcs.config import default_config as default_ftcs_config
from solvers.adi.solver import Solver as ADISolver
from solvers.ftcs.solver import Solver as FTCSSolver

# %% Generate data with ADI solver
for N in [10, 100]:
  config = default_adi_config(temperature=1000)
  config.resolution = (N, N)
  config.time_step_strategy = ConstantTimeStep(2.41)
  config.frame_stride = 100 if N == 100 else 1
  c0 = initial_condition(config)
  t, c = ADISolver(config).solve(c0, lambda f: f.copy())
  q = get_quantity_over_time(config, c)
  np.save(f'LMD/data/adi_q_{N}.npy', q)
  np.save(f'LMD/data/adi_t_{N}.npy', t)
  
# %% Generate data with explicit solver

for N in [10, 100]:
  config = default_ftcs_config(temperature=1000)
  config.resolution = (N, N)
  config.dt = 2.41
  config.frame_stride = 100 if N == 100 else 1
  c0 = initial_condition(config)
  t, c = FTCSSolver(config).solve(c0, logging.getLogger(__name__))
  q = get_quantity_over_time(config, c)
  np.save(f'LMD/data/ftcs_q_{N}.npy', q)
  np.save(f'LMD/data/ftcs_t_{N}.npy', t)
  
# %% Visualize quantity in plt

  
plt.figure(figsize=(6, 4))
plt.xlabel('laikas [val]')
plt.ylabel('kiekis [$\\mu g$]')
  
for N in [10, 100]:  
  q = np.load(f'LMD/data/ftcs_q_{N}.npy')
  t = np.load(f'LMD/data/ftcs_t_{N}.npy')
  plt.plot(t / 3600, q[:, 2], label=f'FTCS N={N}')
  
  q = np.load(f'LMD/data/adi_q_{N}.npy')
  t = np.load(f'LMD/data/adi_t_{N}.npy')
  plt.plot(t / 3600, q[:, 2], label=f'ADI N={N}', linestyle='--')

plt.legend()
plt.savefig('LMD/images/quantity_comparison.png', dpi=300)
plt.show()

# %% Absolute error between FTCS and ADI

plt.figure(figsize=(6, 4))
plt.xlabel('laikas [val]')
plt.ylabel('Absoliutus kiekio skirtumas\n(ADI vs išreikštinis) [$\\mu g$]')
  
for N in [10, 100]:
  
  q1 = np.load(f'LMD/data/ftcs_q_{N}.npy')
  q2 = np.load(f'LMD/data/adi_q_{N}.npy')
  
  t1 = np.load(f'LMD/data/ftcs_t_{N}.npy')
  t2 = np.load(f'LMD/data/adi_t_{N}.npy')
  
  T = min(len(t1), len(t2))
  
  dq = np.abs(q2[:T, 2] - q1[:T, 2])
  
  plt.plot(t1[:T] / 3600, dq, label=f'N={N}')

plt.legend()
plt.savefig('LMD/images/abs_error.png', dpi=300)
plt.show()