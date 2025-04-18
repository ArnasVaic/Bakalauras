# %% Compare solutions with the same time step, but different spatial steps and solvers. Find a way to measure the solutions because different resolutions make it hard to do frame by frame difference. Perhaps to compare frame by frame would be possible by scaling up the smaller space and then performing the subtraction. Also produce q(t) graphs because they are very informative. Compare solution times of different solvers.

from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition 
from solver_utils import *
from solvers.efd.config import Config
from solvers.adi.solver import Solver as ADISolver
from solvers.efd.solver import Solver as EFDSolver

COMMON_TIME_STEP = 25

# %% Generate a single with EFD solver with default resolution (40x40)

config = Config()
config.dt = COMMON_TIME_STEP
solver = EFDSolver(config)
c0 = initial_condition(config)
with timed("EFD Solve time") as elapsed:
    t, c = solver.solve(c0)
print(t.shape, c.shape)
q = get_quantity_over_time(config, c)

np.save('compare-solvers/efd-default-t', t)
np.save('compare-solvers/efd-default-q', q)

del config, solver, c0, t, c, q

# %% Generate multiple solutions with ADI solver and different resolutions

markers = [ 'x', 'o', '+', '^' ]
RESOLUTIONS = [ (10, 10), (20, 20), (40, 40), (100, 100) ]

config = Config()
config.dt = COMMON_TIME_STEP

for resolution in RESOLUTIONS:   
    config.resolution = resolution
    solver = ADISolver(config)
    c0 = initial_condition(config)
    with timed("ADI Solve time") as elapsed:
        t, c = solver.solve(c0)
    q = get_quantity_over_time(config, c)

    np.save(f'compare-solvers/adi-{resolution[0]}x{resolution[1]}-t', t)
    np.save(f'compare-solvers/adi-{resolution[0]}x{resolution[1]}-q', q)

# %% Plot default EFD and different ADI solutions

for index, resolution in enumerate(RESOLUTIONS): 
    t = np.load(f'compare-solvers/adi-{resolution[0]}x{resolution[1]}-t.npy')
    q = np.load(f'compare-solvers/adi-{resolution[0]}x{resolution[1]}-q.npy')

    visual_stride = int(q.shape[0] / 20)

    # show the quantity only for the product of the reaction
    plt.plot(
        t[::visual_stride] / 3600, 
        q[::visual_stride, 2], 
        markers[index], 
        label=f'ADI {resolution[0]}x{resolution[1]}')

t = np.load(f'compare-solvers/efd-default-t.npy')
q = np.load(f'compare-solvers/efd-default-q.npy')

visual_stride = int(q.shape[0] / 20)

plt.plot(
    t[::visual_stride] / 3600, 
    q[::visual_stride, 2],
    label=f'EFD {40}x{40}')

plt.xlabel('t [h]')
plt.ylabel('q [g]')
plt.legend()


# %% visualize individual frames of solutions where spatial step sizes differ

config = Config()
config.dt = COMMON_TIME_STEP
config.resolution = (80, 100)

solver = ADISolver(config)
c0 = initial_condition(config)
with timed("ADI Solve time") as elapsed:
    t, c = solver.solve(c0)

# %%
show_solution_frame(config, t, c, 500, 2)