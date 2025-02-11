from logging import Logger
from solvers.efd.state import State

def log_debug_info(logger: Logger | None, state: State) -> None:
    if logger is None:
        return
    
    q = state.c_curr.sum(axis=(1, 2))
    q0 = state.c_init.sum(axis=(1, 2))
    q1p, q2p = q[0] / q0[0], q[1] / q0[1]

    logger.debug(f'step = {state.time_step}, q1 % = {q1p}, q2 % = {q2p}, q3 = {q[2]}')