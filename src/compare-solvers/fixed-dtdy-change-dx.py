# %% Imports

from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition 
from solver_utils import *
from solvers.config import Config
from solvers.adi.solver import Solver as ADISolver
from solvers.efd.solver import Solver as EFDSolver

COMMON_TIME_STEP = 25
markers = [ 'x', 'o', '+', '^' ]
Y_RESOLUTION = 40
X_RESOLUTIONS = [ 40, 80, 120 ]

# %% Generate a single with EFD solver with default resolution (40x40)

config = Config()
config.dt = COMMON_TIME_STEP
solver = EFDSolver(config)
c0 = initial_condition(config)
with timed("EFD Solve time") as elapsed:
    t, c = solver.solve(c0)
print(t.shape, c.shape)
q = get_quantity_over_time(config, c)

np.save('compare-solvers/assets/efd-default-t', t)
np.save('compare-solvers/assets/efd-default-q', q)

del config, solver, c0, t, c, q


# %% Generate solutions by only changing spatial step in one axis
config = Config()
config.dt = COMMON_TIME_STEP

for resolution in X_RESOLUTIONS:   
    config.resolution = (resolution, Y_RESOLUTION)
    solver = ADISolver(config)
    c0 = initial_condition(config)
    with timed(f"ADI Solve time {config.resolution}") as elapsed:
        t, c = solver.solve(c0)
    q = get_quantity_over_time(config, c)

    np.save(f'compare-solvers/assets/adi-{resolution}x{Y_RESOLUTION}-t', t)
    np.save(f'compare-solvers/assets/adi-{resolution}x{Y_RESOLUTION}-q', q)

del config, solver, c0, t, c, q

# %% Plot default EFD and different ADI solutions for single changing spatial step

for index, resolution in enumerate(X_RESOLUTIONS): 
    t = np.load(f'compare-solvers/assets/adi-{resolution}x{Y_RESOLUTION}-t.npy')
    q = np.load(f'compare-solvers/assets/adi-{resolution}x{Y_RESOLUTION}-q.npy')

    visual_stride = int(q.shape[0] / 20)

    # show the quantity only for the product of the reaction
    plt.plot(
        t[::visual_stride] / 3600, 
        q[::visual_stride, 2], 
        markers[index], 
        label=f'ADI {resolution}x{Y_RESOLUTION}')

t = np.load(f'compare-solvers/assets/efd-default-t.npy')
q = np.load(f'compare-solvers/assets/efd-default-q.npy')

visual_stride = int(q.shape[0] / 20)

plt.plot(
    t[::visual_stride] / 3600, 
    q[::visual_stride, 2],
    label=f'EFD {40}x{40}')

plt.xlabel('t [h]')
plt.ylabel('q [g]')
plt.legend()

# %% Visualize single frame
config = Config()
config.dt = COMMON_TIME_STEP
config.resolution = (80, 100)

solver = ADISolver(config)
c0 = initial_condition(config)
with timed("ADI Solve time") as elapsed:
    t, c = solver.solve(c0)

# %%
show_solution_frame(config, t, c, 500, 2)