# %%

import logging
import numpy as np
import matplotlib.pyplot as plt
from solver_utils import get_quantity_over_time
from solvers.adi.time_step_strategy import ConstantTimeStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import MixConfig, large_config as adi_config
from solvers.ftcs.config import default_config as default_ftcs_config
from solvers.adi.solver import Solver as ADISolver
from solvers.ftcs.solver import Solver as FTCSSolver

# %%

moments = [[], [3600], [4 * 3600]]

D = 3

for moment in moments:
  mix_cfg = MixConfig('perfect', moment)
  config = adi_config(order=0, temperature=1000, mix_config=mix_cfg)
  config.resolution = (40, 40)
  config.time_step_strategy = ConstantTimeStep(2.41)
  config.frame_stride = 10
  c0 = initial_condition(config)
  t, c = ADISolver(config).solve(c0, lambda f: f.copy())
  q = get_quantity_over_time(config, c)
  name_part = 'none' if not moment else f'{moment[0] / 3600}'
  np.save(f'LMD/data/adi_q-mix-at-{name_part}-T1000-{D}D.npy', q)
  np.save(f'LMD/data/adi_t-mix-at-{name_part}-T1000-{D}D.npy', t)
  
# %%


plt.rcParams.update({'font.size': 14})

moments = [[3600], [4 * 3600], []]

for i in range(3):
  name_part = 'none' if not moments[i] else f'{moments[i][0] / 3600}'
  t = np.load(f'LMD/data/adi_t-mix-at-{name_part}-T1000-3D.npy')
  q = np.load(f'LMD/data/adi_q-mix-at-{name_part}-T1000-3D.npy')
  plt.plot(t[10:] / 3600, q[10:, 2], label= f'$t_\\text{{mix}}={name_part}$ val.' if i != 2 else f'nėra maišymo')
plt.xlabel("laikas [val]")
plt.ylabel("medžiagos kiekis sistemoje [val]")
plt.title(f"$T={1000}^\\circ C$")
plt.legend()

# plt.savefig(
#   f'LMD/images/perfect-q.png',
#   dpi=300,
#   bbox_inches='tight'
# )
