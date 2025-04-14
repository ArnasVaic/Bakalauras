# %% 
# Observe the relative error between a high resolution solution and lower resolution solutions

import logging
from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition
from solver_utils import *
from solvers.config import Config
from solvers.adi.solver import Solver as ADISolver

COMMON_TIME_STEP = 25
RESOLUTIONS = [ 240 ]

logging.basicConfig(
  filename='debug.log',
  filemode='w',
  format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.DEBUG)

# %% Generate solutions for all resolutions
config = Config()
config.logger = logging.getLogger(__name__)
config.frame_stride = 200
config.dt = COMMON_TIME_STEP

for resolution in RESOLUTIONS:   

    config.resolution = (resolution, resolution)
    solver = ADISolver(config)

    c0 = initial_condition(config)

    show_solution_frame(config, [0], np.array([c0]), 0, 0)

    # print("solver started")
    # with timed(f"ADI Solve time {config.resolution}") as elapsed:
    #     t, c = solver.solve(c0)

    # print(c.shape)
    
    # q = get_quantity_over_time(config, c)

    # np.save(f'compare-solvers/assets/adi-{resolution}x{resolution}-t', t)
    # np.save(f'compare-solvers/assets/adi-{resolution}x{resolution}-q', q)

# del config, solver, c0, t, c, q