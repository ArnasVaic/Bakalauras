# %%

import datetime
from golden_search import gss
from solvers.initial_condition import initial_condition 
from solvers.config import Config
from solvers.efd.solver import Solver
from solvers.mixer import SubdivisionMixer
import matplotlib.pyplot as plt
import numpy as np

# physical size of particle
PART_SZ = 2.154434690031884 
# particle resolution
PART_RS = 40 

# %%

s = 4

config = Config()
config.size = ( s * PART_SZ, s * PART_SZ )
config.resolution = ( s * PART_RS, s * PART_RS) 
config.mixer = SubdivisionMixer((2 * s, 2 * s), 'perfect', [ 3 * 3600 ])

c0 = initial_condition(config, (s, s))

solver = Solver(config)

print(solver.dt)

t, c = solver.solve(c0)

# %%

show_step = 45
actual_step = t[show_step]

pretty_time = str(datetime.timedelta(seconds=int(actual_step * solver.dt)))
print(pretty_time)

img = np.transpose(c[show_step], (1, 2, 0))

plt.imshow(img / np.max(img), vmin=0, vmax=1)
plt.show()

# %%

def reaction_end_time(s: int, config: Config, t: float) -> float:
    config.mixer.mix_times = [ t ]
    solver = Solver(config)
    c0 = initial_condition(config, (s, s))
    ts, _ = solver.solve(c0)
    return ts[-1] * solver.dt

ts = []
sizes = [1, 2, 4, 8, 16]
for s in sizes:

    config = Config()
    config.size = ( s * PART_SZ, s * PART_SZ )
    config.resolution = ( s * PART_RS, s * PART_RS) 
    config.mixer = SubdivisionMixer((2 * s, 2 * s), 'perfect', [ ])

    f = lambda t: reaction_end_time(s, config, t)

    optimal_mix_time = gss(f, 1 * 3600, 3 * 3600)

    ts.append(optimal_mix_time)

    pretty_time = str(datetime.timedelta(seconds=int(optimal_mix_time)))

    print(f't = {pretty_time} ({optimal_mix_time})')

# %%
import matplotlib.pyplot as plt
import numpy as np

plt.plot(np.array(ts) / 3600)