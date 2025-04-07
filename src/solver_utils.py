
from contextlib import contextmanager
import datetime
import time
from matplotlib import pyplot as plt
import numpy as np
from solvers.config import Config
from solvers.efd.solver import Solver
from solvers.initial_condition import initial_condition

def reaction_end_time(config: Config, t: float) -> float:
  config.mixer.mix_times = [ t ]
  solver = Solver(config)
  c0 = initial_condition(config)
  ts, _ = solver.solve(c0)
  t_end = ts[-1] * solver.dt
  return t_end

def get_quantity_over_time(config: Config, solution: np.ndarray) -> np.ndarray:
  # shape of the solution is assumed to be [T, 3, W, H]
  total_points = config.resolution[0] * config.resolution[1]
  return solution.sum(axis=(2, 3)) / total_points

def show_solution_frame(
    config: Config, 
    t: np.array, 
    solution: np.ndarray, 
    frame: int, 
    element: int) -> None:
  extent = [
    0, config.dx * (config.resolution[0] - 1), 
    0, config.dy * (config.resolution[1] - 1)
  ]

  time = str(datetime.timedelta(seconds=int(t[frame])))

  plt.xlabel('x [μm]')
  plt.ylabel('y [μm]')
  plt.title(f'$c_{element}$ at $t=$ {time}')

  plt.imshow(
    solution[frame, element, :, :], 
    aspect=1,
    extent=extent
  )

@contextmanager
def timed(msg="Elapsed"):
  start = time.perf_counter()
  yield lambda: time.perf_counter() - start
  end = time.perf_counter()
  print(f"{msg}: {end - start:.6f} seconds")