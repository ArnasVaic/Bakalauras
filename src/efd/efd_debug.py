def print_initial_debug_info(T, dt, dt_upper_bound, t_mix, threshold):

  if T is None:
    print(f'Will stop when quantity of initial elements reaches threshold ({threshold}).')
  else:
    print(f'Will stop at time step T={T}.')

  print(f'Upper dt bound: {dt_upper_bound}, given dt={dt}')

  if t_mix is not None:
    print(f"Mixing times: {t_mix}")

def print_sim_debug_info(t, dt, c1_init, c2_init, c1_last, c2_last):
  q = (c1_last + c2_last).sum() / (c1_init + c2_init).sum()
  print(f'[t={t * dt:.02f},step={t}] q={q:.02f}')