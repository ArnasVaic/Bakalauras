
from contextlib import contextmanager
import datetime
import time
from matplotlib import pyplot as plt
import numpy as np
from solvers.ftcs.config import Config
from solvers.ftcs.solver import Solver
from solvers.initial_condition import initial_condition

def pretty_time(seconds: float) -> str:
  return str(datetime.timedelta(seconds=int(seconds)))

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
  # shape of the result is [T, 3]
  return solution.sum(axis=(2, 3)) / total_points

def show_solution_frame(
    config: Config,
    t: np.ndarray,
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
  plt.title(f'$c_{element + 1}$ at $t=$ {time}')

  plt.imshow(
    solution[frame, element, :, :], 
    aspect=1,
    extent=extent,
    vmin = solution[:, element].min(),
    vmax = solution[:, element].max()
  )

  plt.colorbar()

def show_solution_frames(
    t: np.ndarray,
    solution: np.ndarray,
    frames: list[int],
    element: int,
    filename: str | None = None,
    cmap: str = 'viridis',
    config: Config | None = None) -> None:
  
  extent = [
    0, config.dx * (config.resolution[0] - 1),
    0, config.dy * (config.resolution[1] - 1)
  ] if config is not None else [0,1,0,1]

  fig, axes = plt.subplots(1, 5, figsize=(15, 3), constrained_layout=True)  # 1 row, 5 columns

  fig.text(-0.03, 0.5, f'$c_{element + 1}$', va='center', rotation='horizontal', fontsize=24)

  for ax, frame in zip(axes, frames):

    time = str(datetime.timedelta(seconds=int(t[frame])))

    ax.set_title(f'$t=$ {time}', fontsize=18)
    im = ax.imshow(
      solution[frame, element, :, :],
      aspect=1,
      extent=extent,
      vmin = solution[:, element].min(),
      vmax = solution[:, element].max(),
      cmap=cmap
    )
    ax.axis('off')  # Remove axes for a cleaner look

  cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, orientation='vertical')
  cbar.set_label('Koncentracija', fontsize=12)

  if filename is not None:
    fig.savefig(filename, dpi=300, bbox_inches='tight')  # Save with high DPI
  plt.show()

def validate_solution_stable(config: Config, solution: np.ndarray):
  # this check is very specific to the initial conditions
  # take a line at about a quarter of the total height from
  # the bottom (specific height doesn't really matter since
  # the initial conditions are very symmetric)
  y = int(config.resolution[1] / 4)
  df = np.diff(solution[1, 0, :, y])
  assert np.sum((df[:-1] > 0) & (df[1:] < 0)) < 1, "Simulation not stable"

def validate_frame_stable(config: Config, frame: np.ndarray):
  # this check is very specific to the initial conditions
  # take a line at about a quarter of the total height from
  # the bottom (specific height doesn't really matter since
  # the initial conditions are very symmetric)
  y = int(config.resolution[1] / 4)
  df = np.diff(frame[0, :, y])
  assert np.sum((df[:-1] > 0) & (df[1:] < 0)) < 1, "Simulation not stable"

@contextmanager
def timed(msg="Elapsed"):
  start = time.perf_counter()
  yield lambda: time.perf_counter() - start
  end = time.perf_counter()
  print(f"{msg}: {end - start:.6f} seconds")
  