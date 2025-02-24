# %%

import datetime
import logging
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from solver_utils import reaction_end_time
from solvers.config import large_config
from concurrent.futures import ProcessPoolExecutor

from solvers.efd.solver import Solver
from solvers.initial_condition import initial_condition

# This script will run the simulations on increasingly
# larger spaces (numbers discrete points also increasing)
# and compare how reaction completion time changes with
# respect to the time of mixing.

logging.basicConfig(
  filename='debug.log',
  filemode='w',
  format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.DEBUG)

ts_mix = 3600 * np.arange(1.5, 2.5, 0.01)
orders = [1]

for order in orders:

  config = large_config(order)
  config.dt = 5
  config.frame_stride = 50
  config.logger = logging.getLogger(__name__)
  # ts_end = [ reaction_end_time(config, t_mix) for t_mix in ts_mix ]
  with ProcessPoolExecutor() as executor:
    futures = [ executor.submit(reaction_end_time, config, t) for t in ts_mix ]
    ts_end = [ future.result() for future in tqdm(futures, total=len(ts_mix)) ]

  # show_step = 100
  # solver = Solver(config)
  # c0 = initial_condition(config)
  # t, c = solver.solve(c0)

  # actual_step = t[show_step]

  # pretty_time = str(datetime.timedelta(seconds=int(actual_step * solver.dt)))
  # print(pretty_time)
  # img = np.transpose(c[show_step], (1, 2, 0))
  # plt.imshow(img / np.max(c))

  np.save(f'larger-efd/assets/{order}x{order}-detailed.npy', ts_end)

# %%

for s in [1, 2, 4, 8, 16]:
  ts = np.load(f'larger-efd/assets/{s}x{s}.npy')
  hours = ts / 3600
  plt.plot(np.arange(0, 11, 0.5), hours, label=f'{s}x{s}')

for order in [0, 1]:
  ts_end = np.load(f'larger-efd/assets/{order}x{order}-detailed.npy')
  hours = ts_end / 3600
  plt.plot(np.arange(1, 3, 0.01), hours, label=f'{order}x{order}-d')

# , 8401.53
# for order, opt_t in enumerate([7086.69]):
#   config = large_config(0)
#   t_end = reaction_end_time(config, opt_t)
#   plt.scatter([opt_t / 3600], [t_end / 3600])

plt.xlabel('$t_{mix}$ [h]')
plt.ylabel('$t_{end}$ [h]')
plt.legend()
plt.show()