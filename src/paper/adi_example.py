# %% Imports

from matplotlib import pyplot as plt
import numpy as np
from solver_utils import get_quantity_over_time, show_solution_frames, timed
from solvers.adi.time_step_strategy import ACAStep, ConstantTimeStep
from solvers.initial_condition import initial_condition
from solvers.adi.config import default_config, large_config
from solvers.adi.solver import Solver

T = 1000

frame_strides = {
  1000: 100,
  1200: 100,
  1600: 100
}

config = large_config(order=0, temperature=T)
config.frame_stride = frame_strides[T]
c0 = initial_condition(config)

# %% Solve
with timed("solved in"):
  t, c = Solver(config).solve(c0, lambda f: f.copy())

np.save(f'paper/adi_example_{T}C_t.npy', t)
np.save(f'paper/adi_example_{T}C_c.npy', c)

# %% Generate visualization

t = np.load(f'paper/adi_example_{T}C_t.npy')
c = np.load(f'paper/adi_example_{T}C_c.npy')

frames_to_show = {
  1000: [0, 2, 10, 20, -1],
  1200: [0, 2, 10, 20, -1],
  1600: [0, 2, 10, 20, -1],
}

show_solution_frames(
  t, c,
  frames=frames_to_show[T],
  element=0,
  #filename=f'../paper/images/adi/c1-{T}C.png',
  config=config
)
show_solution_frames(
  t, c,
  frames=frames_to_show[T],
  element=2,
  #filename=f'../paper/images/adi/c3-{T}C.png',
  config=config
)

print(c.shape)
q = get_quantity_over_time(config, c)

print(f'initial quantity of c1 and c2: {q[0, 0] + q[0, 1]:.2e} [g]')
print(f'final quantity of c3: {q[-1, 2]:.2e} [g]')


# %% Quantity graphs

plt.rcParams.update({'font.size': 14})

for i in range(3):
  plt.plot(t / 3600, q[:, i], label=f'Medžiaga $c_{i + 1}$')
plt.xlabel("laikas [val]")
plt.ylabel("medžiagos kiekis sistemoje [val]")
plt.title(f"$T={T}^\circ C$")
plt.legend()

plt.savefig(
  f'../paper/images/adi/q-{T}C.png',
  dpi=300,
  bbox_inches='tight'
)
