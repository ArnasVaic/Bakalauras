import numpy as np

def validate_dt(dt, dx, dy, D, c0, k):
  print(f'dx={dx},dy={dy},D={D},c0={c0},k={k}')
  dt_upper_bound = get_upper_dt_bound(dx, dy, D, c0, k)

  if dt_upper_bound < dt:
    msg = f'error: given dt={dt:.04f} may produce unstable results! Use dt <= {dt_upper_bound:04f}.'
    print(msg)
  assert dt_upper_bound >= dt

def validate_solution(c):

  c1, c2, c3 = c

  if c1.min() < 0:
    id = np.unravel_index(c1.argmin(), c1.shape)
    print(f'c1 min: {c1.min()} at {id}')

  if c2.min() < 0:
    id = np.unravel_index(c2.argmin(), c2.shape)
    print(f'c2 min: {c2.min()} at {id}')

  if c3.min() < 0:
    id = np.unravel_index(c3.argmin(), c3.shape)
    print(f'c3 min: {c3.min()} at {id}')

  # material quantities over time
  q1 = np.sum(c1, axis=(1, 2))
  q2 = np.sum(c2, axis=(1, 2))
  q3 = np.sum(c3, axis=(1, 2))

  # rates of change of material quantities through time
  q1dt = np.diff(q1, 1, axis=0)
  q2dt = np.diff(q2, 1, axis=0)
  q3dt = np.diff(q3, 1, axis=0)

  if np.any(q1dt >= sys.float_info.epsilon):
    id = np.unravel_index(q1dt.argmax(), q1dt.shape)
    print(f'q1dt[{id[0]}]={q1dt[id]}')

  if np.any(q2dt >= sys.float_info.epsilon):
    id = np.unravel_index(q2dt.argmax(), q2dt.shape)
    print(f'q2dt[{id[0]}]={q2dt[id]}')

  if np.any(q3dt < sys.float_info.epsilon):
    id = np.unravel_index(q3dt.argmin(), q3dt.shape)
    print(f'q3dt[{id[0]}]={q3dt[id]}')

  # assert 1st & 2nd materials quantities are always non-negative and non-increasing
  assert np.all(c1 >= 0)
  assert np.all(q1dt <= sys.float_info.epsilon)

  assert np.all(c2 >= 0)
  assert np.all(q2dt <= sys.float_info.epsilon)

  # assert 3rd material quantity is non-negative and non-decreasing
  assert np.all(c3 >= 0)
  assert np.all(q3dt >= -sys.float_info.epsilon)

def validate_inputs(
  W,
  H,
  N,
  M,
  D,
  c0,
  k,
  c1_init,
  c2_init,
  c3_init,
  threshold,
  t_mix,
  T,
  dt):
  assert W > 0
  assert H > 0

  assert N > 0
  assert isinstance(N, int)

  assert M > 0
  assert isinstance(M, int)

  assert D >= 0

  assert np.all(c1_init >= 0)
  assert np.all(c2_init >= 0)
  assert np.all(c3_init >= 0)

  assert threshold is None or 0 <= threshold <= 1

  #assert t_mix is None or np.all(t_mix >= 0)

  assert T is None or (isinstance(T, int) and T > 0)

  assert threshold or T

  dx = W / (N - 1)
  dy = H / (M - 1)

  dt_upper_bound = get_upper_dt_bound(dx, dy, D, c0, k)
  if dt is not None and dt_upper_bound < dt:
    print(f'warning: it must hold that dt <= {dt_upper_bound}')
  assert dt is None or dt_upper_bound >= dt