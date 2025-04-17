# %%

from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition 
from solvers.efd.config import Config
from solvers.adi.solver import Solver

config = Config()
config.dt = 25

c0 = initial_condition(config)

solver = Solver(config)
t, c = solver.solve(c0, lambda f: f.copy())
q = c.sum(axis=(2, 3))

print(t.shape)

# %% Compare individual frames
frame = 3
element = 0
plt.imshow(c[frame, element])
plt.colorbar()
plt.legend()