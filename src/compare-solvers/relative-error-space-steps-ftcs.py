# %%
# Observe the relative error between a high resolution solution and lower resolution solutions

from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition
from solver_utils import validate_solution_stable, get_quantity_over_time, timed
from solvers.ftcs.config import default_config
from solvers.ftcs.solver import Solver

def filepath(r: int, attribute: str):
  return f'compare-solvers/assets/ftcs-{r}x{r}-{attribute}.npy'


# %% Cache results for given parameters

RESOLUTION = 120

config = default_config()
config.frame_stride = 50
config.dt = 0.6005908612893365
config.resolution = (RESOLUTION, RESOLUTION)
solver = Solver(config)

c0 = initial_condition(config)

# %%

with timed(f"FTCS Solve time {config.resolution}") as elapsed:
  t, c = solver.solve(c0)
q = get_quantity_over_time(config, c)

np.save(f'compare-solvers/assets/ftcs-{RESOLUTION}x{RESOLUTION}-t', t)
np.save(f'compare-solvers/assets/ftcs-{RESOLUTION}x{RESOLUTION}-q', q)

del config, solver, c0, t, q


# %% Plot errors, ahah no

RESOLUTIONS = [ 40, 200 ]

for resolution in RESOLUTIONS:
  t = np.load(f'compare-solvers/assets/adi-{resolution}x{resolution}-t.npy')
  q = np.load(f'compare-solvers/assets/adi-{resolution}x{resolution}-q.npy')

  plt.plot(t / 3600, q[:,2], linestyle='dashed', label = f'{resolution}x{resolution}')

plt.xlabel('t [h]')
plt.ylabel('quantity [g]')
plt.legend()

# %% Plot relative errors

true_t = np.load(f'compare-solvers/assets/adi-200x200-t.npy')
true_q = np.load(f'compare-solvers/assets/adi-200x200-q.npy') 

RESOLUTIONS = [ 40, 60, 80, 120 ]

ELEMENT = 1

for index, resolution in enumerate(RESOLUTIONS):

  t = np.load(filepath(resolution, 't'))
  q = np.load(filepath(resolution, 'q'))

  T = min(t.shape[0], true_t.shape[0])

  error = np.abs(true_q[:T, :] - q[:T, :]) / np.abs(true_q[:T, :])

  plt.plot(
    t[:T] / 3600,
    error[:, ELEMENT],
    label=f'{resolution}x{resolution}'
  )

plt.title(f'Pointwise relative error between ADI solutions of different resolutions ($c_{ELEMENT + 1}$)')
plt.xlabel('t [h]')
plt.ylabel(f'error, %')
plt.legend()
plt.show()

# %% Plot absolute errors

true_t = np.load(f'compare-solvers/assets/ftcs-200x200-t.npy')
true_q = np.load(f'compare-solvers/assets/ftcs-200x200-q.npy') 

RESOLUTIONS = [ 40, 60, 80, 120 ]

ELEMENT = 2

for index, resolution in enumerate(RESOLUTIONS):

  t = np.load(filepath(resolution, 't'))
  q = np.load(filepath(resolution, 'q'))

  T = min(t.shape[0], true_t.shape[0])

  error = np.abs(true_q[:T, :] - q[:T, :])

  plt.plot(
    t[:T] / 3600,
    error[:, ELEMENT],
    label=f'{resolution}x{resolution}'
  )

# plt.title(f'Absolute error between ADI solutions of different resolutions ($c_{ELEMENT + 1}$)')
plt.xlabel('laikas [val]')
plt.ylabel(f'medžiagos kiekio skirtumas [g]')
plt.legend()
plt.savefig('../paper/images/adi/absolute-error-ftcs.png', dpi=300, bbox_inches='tight')
plt.show()

# %% L norms

import numpy.linalg as la

true_q = np.load(f'compare-solvers/assets/adi-200x200-q.npy') 

RESOLUTIONS = [ 40, 60, 80, 120 ]
ELEMENT = 2

for index, resolution in enumerate(RESOLUTIONS):
  q = np.load(filepath(resolution, 'q'))
  T = min(q.shape[0], true_q.shape[0])

  diff = true_q[:T, ELEMENT] - q[:T, ELEMENT]

  l2 = la.norm(diff, 2)
  linf = la.norm(diff, np.inf)

  print(f'{resolution:3}x{resolution:3}, L2: {l2:8.2e}, Linf: {linf:8.2e}')
  
# %% 

import numpy as np
import matplotlib.pyplot as plt

# Set global font sizes for readability
plt.rcParams.update({
  'font.size': 18,
  'axes.titlesize': 20,
  'axes.labelsize': 18,
  'legend.fontsize': 14,
  'xtick.labelsize': 14,
  'ytick.labelsize': 14
})

true_t = np.load(f'compare-solvers/assets/ftcs-200x200-t.npy')
true_q = np.load(f'compare-solvers/assets/ftcs-200x200-q.npy') 

RESOLUTIONS = [40, 60, 80, 120]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 plots

for ELEMENT in range(3):  # ELEMENT = 0, 1, 2
  ax = axes[ELEMENT]

  for resolution in RESOLUTIONS:
    t = np.load(filepath(resolution, 't'))
    q = np.load(filepath(resolution, 'q'))

    T = min(t.shape[0], true_t.shape[0])

    error = np.abs(true_q[:T, :] - q[:T, :])

    ax.plot(
        t[:T] / 3600,
        error[:, ELEMENT],
        label=f'{resolution}×{resolution}'
    )

  # Set subplot title
  ax.set_title(f'$c_{{{ELEMENT+1}}}$', fontsize=22)

  # Remove individual axis labels
  ax.set_xlabel('')
  ax.set_ylabel('' if ELEMENT != 0 else '')

  # Reduce number of axis ticks
  ax.locator_params(axis='x', nbins=4)  # ~4 ticks on x-axis
  ax.locator_params(axis='y', nbins=5)  # ~5 ticks on y-axis

# Add common X and Y labels for the whole figure
fig.text(0.5, 0.02, 'Laikas [val]', ha='center', fontsize=20)
fig.text(0.01, 0.5, 'Medžiagos kiekio skirtumas [g]', va='center', rotation='vertical', fontsize=20)

# One shared legend for all plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=16, bbox_to_anchor=(0.5, 1.05))

# Adjust layout to fit labels and legend nicely
fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])

plt.savefig('../paper/images/ftcs/absolute-error-multi.png', dpi=300, bbox_inches='tight')
plt.show()
