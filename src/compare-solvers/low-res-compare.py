# %% Compare solution of low resolution between different methods

import numpy as np
from solver_utils import get_quantity_over_time
from solvers.initial_condition import initial_condition
from solvers.adi.config import Config as ADIConfig
from solvers.adi.solver import Solver as ADISolver
from solvers.efd.config import Config as EFDConfig
from solvers.efd.solver import Solver as EFDSolver
import matplotlib.pyplot as plt

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
  t, c = solver.solve(c0, lambda f: f.copy())
  path = f'compare-solvers/assets/adi-{r}x{r}'
  np.save(f'{path}-t', t)
  np.save(f'{path}-c', c)

# %% Load data

ELEMENT = 0
RES = 30

def filepath(method: str, r: int, attribute: str):
  return f'compare-solvers/assets/{method}-{r}x{r}-{attribute}.npy'

t1 = np.load(filepath('efd', RES, 't'))
c1 = np.load(filepath('efd', RES, 'c'))

t2 = np.load(filepath('adi', RES, 't'))
c2 = np.load(filepath('adi', RES, 'c'))

# %% Plot comparison in frames

FRAME = 10
plt.imshow(c1[FRAME, ELEMENT, :, :] - c2[FRAME, ELEMENT, :, 
:], cmap='rainbow')
plt.colorbar()

# %% Plot comparison in quantity

q1 = get_quantity_over_time(config, c1)
q2 = get_quantity_over_time(config, c2)

plt.plot(t1 / 3600, q1, label=f'EFD {RES}x{RES}')
plt.plot(t2 / 3600, q2, label=f'ADI {RES}x{RES}', linestyle='dashed')
plt.legend()
plt.xlabel("t [h]")
plt.ylabel("quantity [g]")

# %% Plot absolute quantity error

q1 = get_quantity_over_time(config, c1)
q2 = get_quantity_over_time(config, c2)

N = min(q1.shape[0], q2.shape[0])

for i in range(3):
  plt.plot(t1[:N] / 3600, np.abs(q1[:N, i] - q2[:N, i]), label=f'$c_{i}$')
plt.legend()
plt.xlabel("t [h]")
plt.ylabel("error [g]")
plt.title(f"Absolute error between low resolution $({RES}\\times{RES})$\n numeric solutions (ADI vs EFD)")