# %%
from dataclasses import dataclass
import scipy.sparse as sp
import numpy as np

from solvers.config import Config

@dataclass
class Solver:

    # solver configuration
    config: Config


    def solve(self, c: np.array) -> np.array:

        assert self.config.dt is not None
        
        # since diffusion coefficient is different for each
        # element, the coefficients we construct also are going
        # to be different
        mu_x = [ D * self.config.dt / (2 * self.config.dx ** 2) for D in self.config.D ]
        mu_y = [ D * self.config.dt / (2 * self.config.dy ** 2) for D in self.config.D ]

        A_x = construct_tridiagonal_matrix(self.config.resolution[0])
        B_x = construct_tridiagonal_matrix(self.config.resolution[1])

        time_step = 0

        while True:

            c_half = np.zeros_like(c)

            # iterate each element
            for m in range(3):
                for i in range(self.config.resolution[1]):
                    
                

            rhs = np.eye() 

            time_step = time_step + 1

        pass

def construct_tridiagonal_matrix(n: int):
    mat = sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n)).tocsc()
    mat[0, 0] = mat[-1, -1] = -1
    return sp.csc_matrix(mat)
