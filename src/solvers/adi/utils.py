import logging
import numpy as np

logging.basicConfig(
  filename='debug.log',
  filemode='w',
  format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO)

def initialize_banded(array: np.ndarray, mu: float, element_index: int) -> None:
  i = element_index # alias for brevity
  # initialize first and last element of sup & sub-diagonals as zeros
  # this is required format by scipy
  array[i, 0, 0] = array[i, 2, -1] = 0

  # initialize sup & sub-diagonals
  array[i, 0, 1:] = array[i, 2, :-1] = - mu

  # initialize the main diagonal
  array[i, 1, 0] = array[i, 1, -1] = 1 + mu
  array[i, 1, 1:-1] = 1 + 2 * mu

def build_banded_matrix_A(n: int, mu: float) -> np.ndarray:
  sub_diag = np.repeat(-mu, n)
  sub_diag[0] = 0

  sup_diag = np.repeat(-mu, n)
  sup_diag[-1] = 0

  diag = np.repeat(1 + 2 * mu, n)
  diag[0] = diag[-1] = 1 + mu

  return np.array([sub_diag, diag, sup_diag])