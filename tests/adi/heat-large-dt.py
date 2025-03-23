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
config.resolution = (100, 100)
config.dt = 1 # default is auto time step
config.k = 0    # turn off reaction rate
config.c0 = 1
config.D = (0.01, 0.01, 0.01)

config.stopper = TotalStepsStopper(100)

config.logger = logging.getLogger(__name__)

x = np.linspace(0, config.size[0], config.resolution[0])
y = np.linspace(0, config.size[1], config.resolution[1])
X, Y = np.meshgrid(x, y)

mean, disp = 0.5, 0.1

c1_0 = 1 / ( disp**2 * 2 * 3.1414) * 2.7182 ** ( -0.5 * ((X - mean)/disp) ** 2 - 0.5 * ((Y - mean)/disp) ** 2)
c2_0 = 1 / ( disp**2 * 2 * 3.1414) * 2.7182 ** ( -0.5 * ((X - mean)/disp) ** 2 - 0.5 * ((Y - mean)/disp) ** 2)

c0 = np.stack((c1_0, c2_0, np.zeros_like(c1_0)))

solver = Solver(config)

t, c = solver.solve(c0)

q = c.sum(axis=(2, 3))

# %%

show_step = 10
element = 0
show_time = t[show_step]

print(f'total steps: {t.shape[0]}')
vmin, vmax = np.min(c[:,element,:,:]), np.max(c[:,element,:,:])

time = str(datetime.timedelta(seconds=int(show_time)))

plt.title(f't = {time}')
img = c[show_step, element, :, :]
plt.imshow(img, vmin=max(vmin, 0), vmax=vmax)
plt.colorbar()
plt.show()
