from dataclasses import dataclass
from logging import Logger
from solvers.mixer import Mixer, SubdivisionMixer
from solvers.stopper import Stopper, ThresholdStopper

@dataclass
class Config:

  # logging
  logger: Logger | None = None

  # Physical size of the simulation space.
  size: tuple[float, float] = (2.154434690031884, 2.154434690031884)

  # Number of discrete points in each axis
  resolution: tuple[int, int] = (40, 40)

  # Diffusion coefficients for each element
  D: tuple[float, float, float] = (28e-6, 28e-6, 28e-8)

  # Reaction speed
  k: float = 192

  # Initial concentration
  c0: float = 1e-6

  # Simulation time step. Setting this 
  # value to None will signal to the 
  # solver to choose the largest step
  # that would ensure stability. 
  dt: None | float = None

  # Controls when to stop the simulation
  stopper: Stopper = ThresholdStopper(0.03)

  # Controls how and when to mix reagents
  mixer: Mixer = SubdivisionMixer((2, 2), 'perfect', [])

  # Reduce the size of the resulting array by saving a 
  # small number of frames spaced evenly throughout the time.
  frame_stride: int = 1

  # You should NEVER set this parameter explicitly.
  # It is used to signal to the initial configuration
  # creator how many particles should be in the 
  _order: tuple[int, int] = (0, 0)

  @property
  def dx(self) -> float:
    return self.size[0] / (self.resolution[0] - 1)
  
  @property
  def dy(self) -> float:
    return self.size[1] / (self.resolution[1] - 1) 

  def validate(self) -> None:
    assert self.size[0] > 0, f"Width must be positive, but is {self.size[0]}."
    assert self.size[1] > 0, f"Height must be positive, but is {self.size[1]}."
    assert self.resolution[0] > 0, f"Resolution width must be positive, but is {self.resolution[0]}."
    assert self.resolution[1] > 0, f"Resolution width must be positive, but is {self.resolution[1]}."
    assert self.dt is None or self.dt > 0, f"Time step must be None ir positive, but is {self.dt}."
    # TODO: add more validations

def large_config(order: int) -> Config:
  """Create a configuration for a larger space. 
  Do not change configuration that has been created
  with this method, it could lead to unexpected results. 
  Order of magnitude in each axis. Min value is 0 which 
  results in smallest possible configuration. Each subsequent value 
  mirrors the space in specified axis making it 4 times larger."""

  # Resolution multiplier
  res_mul = 2 ** order

  config = Config() # construct default config object
  config._order = (order, order)
  config.size = (config.size[0] * res_mul, config.size[1] * res_mul)
  config.resolution = (config.resolution[0] * res_mul, config.resolution[1] * res_mul)
  config.mixer = SubdivisionMixer((2 * res_mul, 2 * res_mul), 'perfect', [])

  return config