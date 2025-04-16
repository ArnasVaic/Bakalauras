import logging
import numpy as np

logging.basicConfig(
  filename='debug.log',
  filemode='w',
  format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
  level=logging.INFO)

def build_banded_matrix_A(n: int, mu: float):
  sub_diag = np.repeat(-mu, n)
  sub_diag[0] = 0

  sup_diag = np.repeat(-mu, n)
  sup_diag[-1] = 0

  diag = np.repeat(1 + 2 * mu, n)
  diag[0] = diag[-1] = 1 + mu

  return np.array([sub_diag, diag, sup_diag])