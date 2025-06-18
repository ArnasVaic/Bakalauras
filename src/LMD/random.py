# %%

import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from solver_utils import get_quantity_over_time
from solvers.adi.time_step_strategy import ConstantTimeStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import MixConfig, large_config as adi_config
from solvers.ftcs.config import default_config as default_ftcs_config
from solvers.adi.solver import Solver as ADISolver
from solvers.ftcs.solver import Solver as FTCSSolver
import matplotlib.pyplot as plt

# %%
moments = np.linspace(0, np.sqrt(15) * 60, 30) ** 2

# %%
D = 3

for moment in tqdm(moments):
  mix_cfg = MixConfig('random', moment)
  config = adi_config(order=0, temperature=1000, mix_config=mix_cfg)
  config.resolution = (40, 40)
  # config.time_step_strategy = ConstantTimeStep(2.41)
  config.frame_stride = 10
  c0 = initial_condition(config)
  t, c = ADISolver(config).solve(c0, lambda f: f.copy())
  q = get_quantity_over_time(config, c)
  name_part = f'{int(moment / 60)}'
  np.save(f'LMD/data/adi_q-random-mix-at-{name_part}-T1000-{D}D.npy', q)
  np.save(f'LMD/data/adi_t-random-mix-at-{name_part}-T1000-{D}D.npy', t)
  
# %%


plt.rcParams.update({'font.size': 14})

t_end = []
for moment in moments:
  name_part = f'{int(moment / 60)}'
  t = np.load(f'LMD/data/adi_t-random-mix-at-{name_part}-T1000-3D.npy')
  t_end.append(t[-1] / 3600)

plt.plot(moments[1:] / 3600, t_end[1:])
plt.xlabel("maišymo momentas [val]")
plt.ylabel("reakcijos trukmė [val]")

# plt.savefig(
#   f'LMD/images/perfect-q.png',
#   dpi=300,
#   bbox_inches='tight'
# )
