from abc import abstractmethod
from dataclasses import dataclass
import numpy as np

class Mixer:
  @abstractmethod
  def should_mix(self, time_step: int, dt: float) -> bool:
    pass

  @abstractmethod
  def mix(self, c: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    pass

@dataclass
class SubdivisionMixer:

  # Mixing will be performed on a discrete grid of 
  # blocks by swapping them. Grid shape is the resolution.
  resolution: tuple[int, int] = (2, 2)

  # Mixing mode.
  # Supported values: 'random', 'perfect'
  mode: str = 'random'

  # Moments in time when (not time steps)
  # when the reaction space is going to be mixed.
  mix_times: np.ndarray[float] = np.array([])

  def should_mix(self, time_step: int, dt: float) -> bool:
    # true if any discrete time points are less 
    # than a half time step away from point of mixing.

    self.mix_times = np.array(self.mix_times)
    return np.any(abs(time_step * dt - self.mix_times) <= dt / 2)

  def mix(self, c: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    """Creates a new state by mixing given state c."""
    if self.mode == 'random':
      c_mixed = self.random_mix(c)
    elif self.mode == 'perfect':
      c_mixed = self.perfect_mix(c)
    else:
      raise Exception("mix mode not supported")
    return c_mixed
  
  def random_mix(self, c: np.ndarray[np.float64]) -> np.ndarray[float]:
    particle_cnt = self.resolution[0] * self.resolution[1]
    rotations = np.random.randint(4, size=particle_cnt)
    positions = np.random.permutation(particle_cnt)
    return self.mix_with_params(c, rotations, positions)

  def perfect_mix(self, c: np.ndarray[np.float64]) -> np.ndarray[float]:
    particle_cnt = self.resolution[0] * self.resolution[1]
    indices = np.arange(particle_cnt).reshape(self.resolution[0], self.resolution[1])    
    for i in range(0, self.resolution[0], 2):
      for j in range(0, self.resolution[1], 2):
        indices[i:i+2, j:j+2] = indices[i:i+2, j:j+2][::-1, ::-1]
    rotations = np.zeros(particle_cnt)
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
      int(c.shape[1] / self.resolution[0]), \
      int(c.shape[2] / self.resolution[1])
    assert chunk_size[0] == chunk_size[1]

    a = chunk_size[0]
   
    chunks = c.reshape(3, self.resolution[0], a, self.resolution[1], a)
    # shape [num_chunks_w, num_chunks_h, 3, a, a]
    chunks = chunks.transpose(1, 3, 0, 2, 4)
    # shape [num_chunks_w * num_chunks_h, 3, a, a]
    flat_chunks = chunks.reshape(-1, 3, a, a)

    for i in range(flat_chunks.shape[0]):
      for c in range(3):
        flat_chunks[i, c] = np.rot90(flat_chunks[i, c], k=rotations[i])

    # reindex
    flat_chunks = flat_chunks[positions]

    flat_chunks = flat_chunks.reshape(self.resolution[0], self.resolution[1], 3, a, a)
    mixed = flat_chunks.transpose(2, 0, 3, 1, 4).reshape(3, self.resolution[0] * a, self.resolution[1] * a)
    return mixed
