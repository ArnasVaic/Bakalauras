# %%

import logging
from matplotlib import pyplot as plt, ticker
import numpy as np
from solvers.initial_condition import initial_condition 
import datetime
from solvers.config import Config
from solvers.adi.solver import Solver
from solvers.stopper import TotalStepsStopper

logging.basicConfig(
  filename='debug.log',
  filemode='w',
  format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.DEBUG)

# make parameters super simple
config = Config()

config.size = (1, 1)
config.resolution = (40, 40)
config.dt = 0.1 # default is auto time step
config.k = 0    # turn off reaction rate
config.c0 = 1
config.D = (0.01, 0.01, 0.01)

config.stopper = TotalStepsStopper(10)

config.logger = logging.getLogger(__name__)
c0 = initial_condition(config)

solver = Solver(config)

t, c = solver.solve(c0)

q = c.sum(axis=(2, 3))

# %%

show_step = -1
element = 0
show_time = t[show_step]

print(f'total steps: {t.shape[0]}')
vmin, vmax = np.min(c[:,element,:,:]), np.max(c[:,element,:,:])

time = str(datetime.timedelta(seconds=int(show_time)))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

plt.title(f't = {time}')

img = c[show_step, element, :, :]
im = axes[0].imshow(img, vmin=max(vmin, 0), vmax=vmax)

axes[1].plot(t, q[:, element])
#axes[1].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#axes[1].ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))

fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
plt.show()
