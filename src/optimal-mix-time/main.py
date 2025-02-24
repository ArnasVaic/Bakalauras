# %%

import datetime
from golden_search import gss
from solver_utils import reaction_end_time
from solvers.config import large_config

ts = []
orders = [1, 2, 3]
for order in orders:

  cfg = large_config(order)
  cfg.dt = 0.5 # large steps decrease detail
  cfg.frame_stride = 500 # acount for increased size of result

  t_start, t_end = 1 * 3600, 3 * 3600 
  optimal_mix_time = gss(lambda t: reaction_end_time(cfg, t), t_start, t_end, 0.01)
  ts.append(optimal_mix_time)

  pretty_time = str(datetime.timedelta(seconds=int(optimal_mix_time)))
  print(f't = {pretty_time} ({optimal_mix_time})')
