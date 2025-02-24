
import numpy as np

from solvers.config import Config

def initial_condition(config: Config) -> np.ndarray[np.float64]:

  rx = 2 ** config._order[0] # Resolution in x axis
  ry = 2 ** config._order[1] # Resolution in y axis

  assert config.resolution[0] % rx == 0
  assert config.resolution[1] % ry == 0

  # fill with checker board pattern

  c0 = config.c0

  assert rx > 0
  assert ry > 0
  
  shift_x = True
  if rx == 1:
    shift_x = False
    rx = 2

  shift_y = True
  if ry == 1:
    shift_y = False
    ry = 2

  square_w = config.resolution[0] / rx
  square_h = config.resolution[1] / ry

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