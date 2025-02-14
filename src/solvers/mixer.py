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
  def mix(self, c: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
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

  def mix(self, c: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    if self.mode == 'random':
      c_mixed = self.random_mix(c)
    elif self.mode == 'perfect':
      c_mixed = self.perfect_mix(c)
    else:
      raise Exception("mix mode not supported")
    return c_mixed
  
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
    flat_indices = indices.flatten()
    return self.mix_with_params(c, rotations, flat_indices)

  def mix_with_params(
    self, 
    c: np.ndarray[np.float64], 
    rotations: np.ndarray[int],
    positions: np.ndarray[int]):

    # Split space [3, W, H] into chunks of equal sidelength.
    # Chunk shape will look like [3, a, a] 
    chunk_size = \
      int(c.shape[1] / self.subdivisions[0]), \
      int(c.shape[2] / self.subdivisions[1])
    assert chunk_size[0] == chunk_size[1]

    a = chunk_size[0]
    num_chunks_w, num_chunks_h = self.subdivisions

    chunks = c.reshape(3, num_chunks_w, a, num_chunks_h, a)
    # shape [num_chunks_w, num_chunks_h, 3, a, a]
    chunks = chunks.transpose(1, 3, 0, 2, 4)
    # shape [num_chunks_w * num_chunks_h, 3, a, a]
    flat_chunks = chunks.reshape(-1, 3, a, a)

    for i in range(flat_chunks.shape[0]):
      for c in range(3):
        flat_chunks[i, c] = np.rot90(flat_chunks[i, c], k=rotations[i])

    # reindex
    flat_chunks = flat_chunks[positions]

    flat_chunks = flat_chunks.reshape(num_chunks_w, num_chunks_h, 3, a, a)
    mixed = flat_chunks.transpose(2, 0, 3, 1, 4).reshape(3, num_chunks_w * a, num_chunks_h * a)
    return mixed
