# %%

# Compare solutions with the same time step, but different spatial steps and solvers. Find a way to measure the solutions because different resolutions make it hard to do frame by frame difference. Perhaps to compare frame by frame would be possible by scaling up the smaller space and then performing the subtraction. Also produce q(t) graphs because they are very informative. Compare solution times of different solvers.

from contextlib import contextmanager
import time
from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition 
from solver_utils import get_quantity_over_time
from solvers.config import Config
from solvers.adi.solver import Solver as ADISolver
from solvers.efd.solver import Solver as EFDSolver

COMMON_TIME_STEP = 25
markers = [ 'x', 'o', '+', '^' ]
RESOLUTIONS = [ (10, 10), (20, 20), (40, 40), (100, 100) ]

@contextmanager
def timed(msg="Elapsed"):
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    end = time.perf_counter()
    print(f"{msg}: {end - start:.6f} seconds")

# %% Generate a single with EFD solver with default resolution (40x40)

config = Config()
config.dt = COMMON_TIME_STEP
solver = EFDSolver(config)

c0 = initial_condition(config)

with timed("EFD Solve time") as elapsed:
    efd_solution = solver.solve(c0)

del config, solver, c0

# %% Generate multiple solutions with ADI solver and different resolutions

plt.xlabel('t [h]')
plt.ylabel('q [g]')

for index, resolution in enumerate(RESOLUTIONS):

    config = Config()
    config.dt = COMMON_TIME_STEP
    config.resolution = resolution

    solver = ADISolver(config)
    c0 = initial_condition(config)

    with timed("ADI Solve time") as elapsed:
        t, c = solver.solve(c0)

    q = get_quantity_over_time(config, c)

    # show the quantity only for the product of the reaction
    label=f'ADI {resolution[0]}x{resolution[1]}'

    visual_stride = int(q.shape[0] / 20)

    plt.plot(t[::visual_stride] / 3600, q[::visual_stride, 2], markers[index], label=label)

q = get_quantity_over_time(Config(), efd_solution[1])
visual_stride = int(q.shape[0] / 20)
plt.plot(
    efd_solution[0][::visual_stride] / 3600, 
    q[::visual_stride, 2],
    label=f'EFD {40}x{40}')
plt.legend()

# %% Measure EFD solver time in comparison

plt.xlabel('t [h]')
plt.ylabel('q [g]')
for index, resolution in enumerate(RESOLUTIONS):

    config = Config()
    # config.dt = COMMON_TIME_STEP
    config.resolution = resolution
    solver = EFDSolver(config)

    c0 = initial_condition(config)

    with timed(f"EFD Solve time") as elapsed:
        t, c = solver.solve(c0)

    q = get_quantity_over_time(config, c)

    # show the quantity only for the product of the reaction
    label=f'EFD {resolution[0]}x{resolution[1]}'

    visual_stride = max(int(q.shape[0] / 20), 1)
    
    plt.plot(t[::visual_stride] / 3600, q[::visual_stride, 2], markers[index], label=label)

plt.legend()

# %% Compare running times

sizes = [ 40, 60, 80, 100, 110, 120 ]

def measure_solver(config: Config, solverType: str) -> float:

    if solverType == 'adi':
        solver = ADISolver(config)
    elif solverType == 'efd':
        solver = EFDSolver(config)
    else:
        raise Exception("Unknown solver type. Supported types: adi, efd.")
    
    c0 = initial_condition(config)
    with timed(f"{solverType} {config.resolution} Solve time") as elapsed:
        _ = solver.solve(c0) # discard results
        t = elapsed()
    return t

efd_ts, adi_ts = [], []

for size in sizes:

    config = Config()
    config.resolution = (size, size)

    # let EFD method choose it's own time step because
    # upper bound gets smaller with resolution
    efd_t = measure_solver(config, 'efd')

    config.dt = COMMON_TIME_STEP
    adi_t = measure_solver(config, 'adi')

    efd_ts.append(efd_t)
    adi_ts.append(adi_t)

plt.plot(sizes, efd_ts, label=f'EFD')
plt.plot(sizes, adi_ts, label=f'ADI')

plt.xlabel(f'grid side-length [units]')
plt.ylabel(f'solution time [s]')
plt.legend()
