# %% 

from matplotlib import pyplot as plt
import numpy as np
from solvers.initial_condition import initial_condition
from solver_utils import validate_solution_stable, get_quantity_over_time, timed
from solvers.adi.config import Config
from solvers.adi.solver import Solver

def capture_frame(frame : np.ndarray) -> np.ndarray:
  return frame.copy()

def filepath(resolution: int, kind: str) -> str:
  return f'compare-solvers/assets/adi-{resolution}x{resolution}-{kind}.npy'

# %%

RESOLUTIONS = [ 40, 60, 80, 120, 200 ]
ts_end = []

for index, resolution in enumerate(RESOLUTIONS):
  t = np.load(filepath(resolution, 't'))
  ts_end.append(t[-1])

ts_end = np.array(ts_end)

plt.plot(
  RESOLUTIONS,
  ts_end / 3600
)

plt.ylabel('t [h]')
plt.xlabel(f'Square grid sidelength [units]')
plt.show()