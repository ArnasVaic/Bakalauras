# %%

import logging
from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition 
import datetime
from solvers.config import Config
from solvers.efd.solver import Solver
from solvers.mixer import SubdivisionMixer

logging.basicConfig(
  filename='debug.log',
  filemode='w',
  format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.DEBUG)

config = Config()
config.logger = logging.getLogger(__name__)
config.mixer = SubdivisionMixer((2, 2), 'random', [ ])
c0 = initial_condition(config)

solver = Solver(config)

t, c = solver.solve(c0)

# %%

show_step = 20
element = 1
actual_step = t[show_step]

print(t.shape[0])
vmin, vmax = np.min(c[:,element,:,:]), np.max(c[:,element,:,:])

time = str(datetime.timedelta(seconds=int(actual_step * solver.dt)))
plt.title(f't = {time}')
plt.imshow(c[show_step, element, :, :], vmin=max(vmin, 0), vmax=vmax)
plt.colorbar()
plt.show()
