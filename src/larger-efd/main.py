# %%

import logging
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from solver_utils import reaction_end_time
from solvers.config import large_config
from concurrent.futures import ProcessPoolExecutor

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
order = 0

config = large_config(order)
config.dt = 1
config.logger = logging.getLogger(__name__)

with ProcessPoolExecutor() as executor:
  futures = [ executor.submit(reaction_end_time, config, t) for t in ts_mix ]
  ts_end = [ future.result() for future in tqdm(futures, total=len(ts_mix)) ]

np.save(f'larger-efd/assets/{order}x{order}-detailed.npy', ts_end)

# %%

ts_end = np.load(f'larger-efd/assets/1x1.npy')
hours = ts_end / 3600
plt.plot(np.arange(0, 11, 0.5), hours, label=f'1x1')

ts_end = np.load(f'larger-efd/assets/0x0-detailed.npy')
hours = ts_end / 3600
plt.plot(np.arange(1.5, 2.5, 0.01), hours, label=f'{order}x{order}-d')

t_mix_opt = 7086.693097530343
t_end = reaction_end_time(config, t_mix_opt)

plt.xlabel('$t_{mix}$ [h]')
plt.ylabel('$t_{end}$ [h]')
plt.legend()
plt.show()