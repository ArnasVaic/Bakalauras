# %%

from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition 
from solvers.config import Config
from solvers.adi.solver import Solver

config = Config()
config.dt = 251.1

c0 = initial_condition(config)

solver = Solver(config)
t, c = solver.solve(c0)
q = c.sum(axis=(2, 3))

print(t.shape)

# %% Compare individual frames
frame = 3
element = 0
plt.imshow(c[frame, element])
plt.colorbar()
plt.legend()