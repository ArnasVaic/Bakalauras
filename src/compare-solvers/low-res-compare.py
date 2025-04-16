# %% Compare solution of low resolution between different methods

import numpy as np
from solvers.initial_condition import initial_condition
from solvers.adi.config import Config as ADIConfig
from solvers.adi.solver import Solver as ADISolver
from solvers.efd.config import Config as EFDConfig
from solvers.efd.solver import Solver as EFDSolver

RESOLUTIONS = [ 10, 20, 30 ]

# %% Generate low resolution solutions with EFD solver

for r in RESOLUTIONS:
  config = EFDConfig()
  config.resolution = (r, r)
  config.dt = None
  solver = EFDSolver(config)
  c0 = initial_condition(config)
  t, c = solver.solve(c0)
  path = f'compare-solvers/assets/efd-{r}x{r}'
  np.save(f'{path}-t', t)
  np.save(f'{path}-c', c)

# %% Generate low resolution solutions with ADI solver

for r in RESOLUTIONS:
  config = ADIConfig()
  config.resolution = (r, r)
  config.dt = EFDSolver(config).dt_bound()
  solver = ADISolver(config)
  c0 = initial_condition(config)
  t, c = solver.solve(c0)
  path = f'compare-solvers/assets/adi-{r}x{r}'
  np.save(f'{path}-t', t)
  np.save(f'{path}-c', c)