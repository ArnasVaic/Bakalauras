from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from efd.state import State

class Mixer:
  @abstractmethod
  def should_mix(state) -> bool:
    pass

  def mix(state) -> None:
    pass

@dataclass
class SubdivisionMixer:

  # Subdivisions of space along each axis.
  subdivisions: int = 2

  # Mixing mode.
  # Supported values: 'random', 'perfect'
  mode: str = 'random'

  # Moments in time when (not time steps)
  # when the reaction space is going to be mixed.
  mix_times: np.ndarray[float] = np.array([])

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
    
  def random_mix(self, c: np.ndarray[float]) -> np.ndarray[float]:
    total_blocks = self.subdivisions ** 2
    rotations = np.random.randint(4, size=total_blocks)
    positions = np.random.permutation(total_blocks)
    
    # returns array with shape [3, t, w, h]
    return np.array([ self.mix_single(c_i, rotations, positions) for c_i in c ]) 

  def mix_single(
    self, 
    c_i: np.ndarray[np.float64], 
    rotations: np.ndarray[int],
    positions: np.ndarray[int]):

    # blocks make up the reaction space and are
    # of identical dimensions so that they could
    # be swapped with one another and/or rotated
    block_size = np.astype(c_i.shape / self.subdivisions, int)
    assert block_size[0] == block_size[1]
    sidelength = block_size[0]

    total_blocks = self.subdivisions ** 2

    block_grid = []
    for i in range(self.subdivisions):
      left, right = i * sidelength, sidelength * (i + 1) - 1
      block_columns = []
      for j in range(self.subdivisions):
        top, bottom = j * sidelength, sidelength * (j + 1) - 1
        block_columns.append(c_i[left : right + 1, top : bottom + 1])
      block_grid.append(block_columns)

    block_grid = np.array(block_grid)

    # reshape grid of blocks to rotate and reindex
    blocks_1d = np.reshape(block_grid, (total_blocks, *block_size))

    # rotate
    blocks_1d = np.array([ np.rot90(b, k) for b, k in zip(blocks_1d, rotations) ])

    # reposition
    blocks_1d = blocks_1d[positions]  

    # reshape back into a grid
    blocks_grid = np.reshape(
      blocks_1d, 
      (B, B, *block_size)
    )

    # concatenate columns and rows back into N x M matrix 
    columns = np.concatenate(blocks_grid, axis=1)
    return np.concatenate(columns, axis=1)