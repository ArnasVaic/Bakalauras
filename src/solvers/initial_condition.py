
import numpy as np

from solvers.config import Config

def initial_condition(config: Config, subdivisions: tuple[int, int]) -> np.ndarray[np.float64]:
  assert config.resolution[0] % subdivisions[0] == 0
  assert config.resolution[1] % subdivisions[1] == 0

  # fill with checker board pattern

  c0 = config.c0

  assert subdivisions[0] > 0
  assert subdivisions[1] > 0

  subs_x, subs_y = subdivisions

  shift_x = True
  if subs_x == 1:
    shift_x = False
    subs_x = 2

  shift_y = True
  if subs_y == 1:
    shift_y = False
    subs_y = 2

  square_w = config.resolution[0] / subs_x
  square_h = config.resolution[1] / subs_y

  assert square_w == square_h

  c1 = 3 * c0 * checkerboard(config.resolution, square_w)
  c2 = 5 * c0 * checkerboard(config.resolution, square_w, False)
  c3 = np.zeros(config.resolution)

  if shift_x:
    c1 = np.roll(c1, square_w / 2, 0)
    c2 = np.roll(c2, square_w / 2, 0)

  if shift_y:
    c2 = np.roll(c2, square_h / 2, 1)
    c1 = np.roll(c1, square_h / 2, 1)

  return np.array([c1, c2, c3])

def checkerboard(shape, a, fill_odd=True):
  """
  Create a checkerboard pattern with an option to fill either odd or even squares.
  
  Parameters:
  shape : tuple -> (rows, cols) of the output array
  a : int -> size of a single square
  fill_odd : bool -> If True, fills odd squares (1); If False, fills even squares (1).
  
  Returns:
  numpy array with checkerboard pattern
  """
  rows, cols = shape
  x = np.arange(rows) // a  
  y = np.arange(cols) // a  
  pattern = (x[:, None] + y) % 2  # Create checkerboard pattern
  
  return pattern if fill_odd else 1 - pattern  # Invert for even squares