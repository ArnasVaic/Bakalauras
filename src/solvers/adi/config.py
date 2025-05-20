from dataclasses import dataclass
from typing import Literal
import numpy as np
from solvers.adi.time_step_strategy import ConstantTimeStep, SCGQMStep, TimeStepStrategy
from solvers.mixer import Mixer, SubdivisionMixer
from solvers.stopper import Stopper, ThresholdStopper

@dataclass
class Config:
  """Configuration for ADI solver"""

  # Physical size of the simulation space.
  size: tuple[float, float]

  # Number of discrete points in each axis
  resolution: tuple[int, int]

  # Diffusion coefficients for each element
  D: np.ndarray

  # Reaction coefficients
  alpha: np.ndarray

  # Reaction speed
  k: float

  # Initial concentration
  c0: float

  # Time step strategy
  time_step_strategy: TimeStepStrategy

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
    """Step size in x axis is a function of the size and resolution"""
    return self.size[0] / (self.resolution[0] - 1)

  @property
  def dy(self) -> float:
    """Step size in y axis is a function of the size and resolution"""
    return self.size[1] / (self.resolution[1] - 1)

  @property
  def total_points(self) -> int:
    """Total number of discrete points in spatial axes"""
    return self.resolution[0] * self.resolution[1]

  def validate(self) -> None:
    """Validate the configuration parameters."""
    sx, sy = self.size
    assert sx > 0, f"Width must be positive, but is {sx}."
    assert sy > 0, f"Height must be positive, but is {sy}."
    rx, ry = self.resolution
    assert rx > 0, f"Resolution width must be positive, but is {rx}."
    assert ry > 0, f"Resolution width must be positive, but is {ry}."
    assert self.dt > 0, f"Time step must be positive, but is {self.dt}."
    # TODO: add more validations

def default_config(temperature: int = 1000) -> Config:
  """Create a default configuration."""

  assert temperature in [1000, 1200, 1600], f"Temperature {temperature} is not supported."

  size_map = {
    1000: (1, 1),
    1200: (1, 1),
    1600: (10**(1/3), 10**(1/3))
  }

  diffusion_map = {
    1000: [10.5e-6, 10.5e-6, 10.5e-8],
    1200: [15e-6, 15e-6, 15e-8],
    1600: [28e-6, 28e-6, 28e-8]
  }

  k_map = {
    1000: 119,
    1200: 146,
    1600: 192
  }

  config = Config(
    _order = (0, 0),
    size = size_map[temperature],
    resolution = (40, 40),
    D = np.array(diffusion_map[temperature]),
    k = k_map[temperature],
    c0 = 1e-6,
    stopper = ThresholdStopper(0.03),
    frame_stride = 1,
    mixer = SubdivisionMixer(np.array([]), (2, 2), 'perfect'),
    time_step_strategy = ConstantTimeStep(1.0),
    alpha = np.array([-3, -5, 2])
  )

  return config

@dataclass
class MixConfig:
  """Mixing configuration"""
  mode: Literal['perfect', 'random']

  moments: list[float]

def large_config(
  order: int,
  temperature: Literal[1000, 1200, 1600],
  mix_config: MixConfig | None = None) -> Config:
  """Create a configuration for a larger space. 
  Do not change configuration that has been created
  with this method, it could lead to unexpected results. 
  Order of magnitude in each axis. Min value is 0 which 
  results in smallest possible configuration. Each subsequent value 
  mirrors the space in specified axis making it 4 times larger."""

  # Resolution multiplier
  res_mul = 2 ** order

  config = default_config(temperature) # construct default config object
  config._order = (order, order)
  config.size = (config.size[0] * res_mul, config.size[1] * res_mul)
  config.resolution = (config.resolution[0] * res_mul, config.resolution[1] * res_mul)
  # experimental parameters that usually work out
  config.time_step_strategy = SCGQMStep(100, 0.1, 1.5, 30, 0.0301, 5)

  if mix_config is not None:
    config.mixer = SubdivisionMixer(
      np.array(mix_config.moments),
      (2 * res_mul, 2 * res_mul),
      mix_config.mode
    )

  return config
