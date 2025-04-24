from dataclasses import dataclass
from solvers.mixer import Mixer, SubdivisionMixer
from solvers.stopper import Stopper, ThresholdStopper

@dataclass
class Config:
  """Configuration for the EFD solver."""

  # Physical size of the simulation space.
  size: tuple[float, float]

  # Number of discrete points in each axis
  resolution: tuple[int, int]

  # Diffusion coefficients for each element
  D: tuple[float, float, float]

  # Reaction speed
  k: float

  # Initial concentration
  c0: float

  # Simulation time step. Setting this
  # value to None will signal to the
  # solver to choose the largest step
  # that would ensure stability.
  dt: None | float

  # Controls when to stop the simulation
  stopper: Stopper

  # Controls how and when to mix reagents
  mixer: Mixer

  # Reduce the size of the resulting array by saving a
  # small number of frames spaced evenly throughout the time.
  frame_stride: int

  # You should NEVER set this parameter explicitly.
  # It is used to signal to the initial configuration
  # creator how many particles should be in the
  _order: tuple[int, int]

  @property
  def dx(self) -> float:
    """Distance between two points in the x axis."""
    return self.size[0] / (self.resolution[0] - 1)

  @property
  def dy(self) -> float:
    """Distance between two points in the y axis."""
    return self.size[1] / (self.resolution[1] - 1) 

  def validate(self) -> None:
    """Validate the configuration parameters."""
    sx, sy = self.size
    assert sx > 0, f"Width must be positive, but is {sx}."
    assert sy > 0, f"Height must be positive, but is {sy}."
    rx, ry = self.resolution
    assert rx > 0, f"Resolution width must be positive, but is {rx}."
    assert ry > 0, f"Resolution width must be positive, but is {ry}."
    assert self.dt is None or self.dt > 0, f"Time step must be None ir positive, but is {self.dt}."
    # TODO: add more validations

def default_config() -> Config:
  """Create a default configuration."""

  config = Config(
    _order = (0, 0),
    size = (1, 1),
    resolution = (40, 40),
    D = (28e-6, 28e-6, 28e-8),
    k = 192,
    c0 = 1e-6,
    dt = None,
    stopper = ThresholdStopper(0.03),
    frame_stride = 1,
    mixer = SubdivisionMixer((2, 2), 'perfect', [])
  )

  return config

def large_config(order: int) -> Config:
  """Create a configuration for a larger space. 
  Do not change configuration that has been created
  with this method, it could lead to unexpected results. 
  Order of magnitude in each axis. Min value is 0 which 
  results in smallest possible configuration. Each subsequent value 
  mirrors the space in specified axis making it 4 times larger."""

  # Resolution multiplier
  res_mul = 2 ** order

  config = default_config()
  config._order = (order, order)
  config.size = (config.size[0] * res_mul, config.size[1] * res_mul)
  config.resolution = (config.resolution[0] * res_mul, config.resolution[1] * res_mul)
  config.mixer = SubdivisionMixer((2 * res_mul, 2 * res_mul), 'perfect', [])

  return config
