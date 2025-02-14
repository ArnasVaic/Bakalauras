# %%

import numpy as np
from solvers.efd.state import State
from solvers.mixer import SubdivisionMixer

mixer = SubdivisionMixer((2, 2), 'perfect', [])

A = np.arange(48).reshape((3, 4, 4))

B = mixer.mix(A)

target = np.array([
    [
        [10, 11, 8, 9],
        [14, 15, 12, 13],
        [2, 3, 0, 1],
        [6, 7, 4, 5]
    ],
    [
        [26, 27, 24, 25],
        [30, 31, 28, 29],
        [18, 19, 16, 17],
        [22, 23, 20, 21]
    ],
    [
        [42, 43, 40, 41],
        [46, 47, 44, 45],
        [34, 35, 32, 33],
        [38, 39, 36, 37]
    ]
])

assert np.array_equal(target, B)