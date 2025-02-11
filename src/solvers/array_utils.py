
import numpy as np

def rect_values(values: np.ndarray, x: int, y: int, l: int) -> np.ndarray:
    left, right = x * l, l * (x + 1) - 1
    top, bottom = y * l, l * (y + 1) - 1
    return values[:, left : right + 1, top : bottom + 1]