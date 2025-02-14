from logging import Logger
from solvers.config import Config
from solvers.efd.state import State

def log_debug_info(logger: Logger | None, state: State) -> None:
    if logger is None:
        return
    
    q = state.c_curr.sum(axis=(1, 2))
    q0 = state.c_init.sum(axis=(1, 2))
    q1p, q2p = q[0] / q0[0], q[1] / q0[1]

    logger.debug(f'step = {state.time_step}, q1 % = {q1p:.02f}, q2 % = {q2p:.02f}, q3 = {q[2]:.02f}')

def log_initial_info(logger: Logger | None, dt: float, config: Config):
    if logger is None:
        return
    
    logger.debug(f'starting simulation, dt={dt}, dx={config.dx}, dy={config.dy}, D={config.D}, k={config.k}, size={config.size}, resolution={config.resolution}')