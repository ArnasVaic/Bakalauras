# %%

from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition 
import datetime
from solvers.config import Config
from solvers.adi.solver import Solver as ADISolver
from solvers.efd.solver import Solver as EFDSolver

config = Config()

config.dt = 25

c0 = initial_condition(config)

adi_solver = ADISolver(config)
efd_solver = EFDSolver(config)

t1, c1 = adi_solver.solve(c0)
t2, c2 = efd_solver.solve(c0)

q1, q2 = c1.sum(axis=(2, 3)), c2.sum(axis=(2, 3))

print(t1.shape, t2.shape)

# %%

for m in range(3):
  plt.plot(t1, q1[:, m], label=f"Explicit FTCS, $q_{m + 1}$")
  plt.plot(t2, q2[:, m], linestyle='dashed', label=f"ADI, $q_{m + 1}$")
plt.legend()