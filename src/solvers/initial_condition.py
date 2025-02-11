
import numpy as np

from solvers.config import Config

def initial_condition(config: Config) -> np.ndarray[np.float64]:
  assert config.resolution[0] % 2 == 0, f'Even resolution width is required, but is {config.resolution[0]}'
  assert config.resolution[1] % 2 == 0, f'Even resolution height is required, but is {config.resolution[1]}'

  c0 = config.c0

  half_w, half_h = config.resolution[0] // 2, config.resolution[1] // 2

  c1 = np.zeros(config.resolution)
  c1[:half_w, :half_h] = 3 * c0
  c1[half_w:, half_h:] = 3 * c0

  c2 = np.zeros(config.resolution)
  c2[half_w:, :half_h] = 5 * c0
  c2[:half_w, half_h:] = 5 * c0

  c3 = np.zeros(config.resolution)

  return np.array([c1, c2, c3])