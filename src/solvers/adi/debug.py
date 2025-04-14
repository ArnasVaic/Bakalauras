from logging import Logger

import numpy as np

from solvers.config import Config


# def log_debug_info(logger: Logger | None, c0: np.ndarray, c: np.ndarray, step: int) -> None:
#     if logger is None:
#         return
    
#     # doing this each frame is a lil much
#     q, q0 = c.sum(axis=(1, 2)), c0.sum(axis=(1, 2))
#     q1p, q2p = q[0] / q0[0], q[1] / q0[1]

#     logger.debug(f'step = {step}, q1 % = {q1p:.02f}, q2 % = {q2p:.02f}, q3 = {q[2]:.02f}')

# def log_initial_info(logger: Logger | None, dt: float, config: Config):
#     if logger is None:
#         return
    
#     logger.debug(f'starting simulation, dt={dt}, dx={config.dx}, dy={config.dy}, D={config.D}, k={config.k}, size={config.size}, resolution={config.resolution}')