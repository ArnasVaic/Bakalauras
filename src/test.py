# %%

import logging
from matplotlib import pyplot as plt
from solvers.initial_condition import initial_condition 
import datetime
from solvers.config import Config
from solvers.efd.solver import Solver

# def quantity(solver: Solver, config: Config):
#   return solver.solve(initial_condition(config))

logging.basicConfig(
  filename='logs.txt',
  filemode='w',
  format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.DEBUG)

config = Config()
config.logger = logging.getLogger(__name__)
config.mixer.mix_times = [ 2 * 3600 ]

c0 = initial_condition(config)

solver = Solver(config)

t, c = solver.solve(c0)

show_step = -1
actual_step = t[show_step]
time = str(datetime.timedelta(seconds=int(actual_step * solver.dt)))
plt.title(f't = {time}')
plt.imshow(c[show_step, 2, :, :])
