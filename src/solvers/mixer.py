from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from solvers.array_utils import rect_values
from solvers.efd.state import State

class Mixer:
  @abstractmethod
  def should_mix(self, time_step: int, dt: float) -> bool:
    pass

  @abstractmethod
  def mix(self, state: State) -> None:
    pass

@dataclass
class SubdivisionMixer:

  # Subdivisions of space along each axis.
  subdivisions: tuple[int, int] = (2, 2)

  # Mixing mode.
  # Supported values: 'random', 'perfect'
  mode: str = 'random'

  # Moments in time when (not time steps)
  # when the reaction space is going to be mixed.
  mix_times: np.ndarray[float] = np.array([])

  @property
  def subdivision_count(self) -> int:
    return self.subdivisions[0] * self.subdivisions[1]

  def should_mix(self, time_step: int, dt: float) -> bool:
    # true if any discrete time points are less 
    # than a half time step away from point of mixing.
    return np.any(abs(time_step * dt - self.mix_times) <= dt / 2)

  def mix(self, c: np.ndarray[np.float64]) -> None:
    if self.mode == 'random':
      self.random_mix(c)
    elif self.mode == 'perfect':
      self.perfect_mix(c)
    else:
      raise Exception("mix mode not supported")
  
  def random_mix(self, c: np.ndarray[np.float64]) -> np.ndarray[float]:
    rotations = np.random.randint(4, size=self.subdivision_count)
    positions = np.random.permutation(self.subdivision_count)
    return self.mix_with_params(c, rotations, positions)

  def perfect_mix(self, c: np.ndarray[np.float64]) -> np.ndarray[float]:
    indices = np.arange(self.subdivision_count).reshape(self.subdivisions[0], self.subdivisions[1])    
    for i in range(0, self.subdivisions[0], 2):
      for j in range(0, self.subdivisions[1], 2):
        indices[i:i+2, j:j+2] = indices[i:i+2, j:j+2][::-1, ::-1]
    rotations = np.zeros(self.subdivision_count)
    return self.mix_with_params(c, rotations, indices)

  def mix_with_params(
    self, 
    c: np.ndarray[np.float64], 
    rotations: np.ndarray[int],
    positions: np.ndarray[int]):

    # blocks make up the reaction space and are
    # of identical dimensions so that they could
    # be swapped with one another and/or rotated
    sub_size = np.astype(c.shape[1:] / np.array(self.subdivisions), int)
    assert sub_size[0] == sub_size[1]
    sidelength = sub_size[0]

    flat_blocks = []

    for index in range(self.subdivision_count):
      x, y = index % self.subdivisions[0], index // self.subdivisions[0]
      block = rect_values(c, x, y, sidelength)
      flat_blocks.append(block)

    # rotate
    flat_blocks = np.array([ 
      np.array([ np.rot90(c_i, angle) for c_i in c ]) 
      for c, angle 
      in zip(flat_blocks, rotations) 
    ])

    # reposition
    flat_blocks = flat_blocks[positions]  

    # reshape back into a grid
    blocks_grid = np.reshape(flat_blocks, (self.subdivisions[0], self.subdivisions[1], *sub_size))

    # concatenate columns and rows back into N x M matrix 
    columns = np.concatenate(blocks_grid, axis=1)
    return np.concatenate(columns, axis=1)