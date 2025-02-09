from dataclasses import dataclass
from mixer import Mixer, SubdivisionMixer
from stopper import Stopper, ThresholdStopper

@dataclass
class Config:

  # Physical size of the simulation space.
  size: tuple[float, float] = (2.154434690031884, 2.154434690031884)

  # Number of discrete points in each axis
  resolution: tuple[int, int] = (80, 80)

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
  dt: None | float

  # Controls when to stop the simulation
  stopper: Stopper = ThresholdStopper(0.03)

  # Controls how and when to mix reagents
  mixer: Mixer = SubdivisionMixer(2, 'perfect', None)

  # Reduce the size of the resulting array by saving a 
  # small number of frames spaced evenly throughout the time.
  frame_stride: int

  @property
  def dx(self) -> float:
    return self.size[0] / (self.resolution[0] - 1)
  
  @property
  def dy(self) -> float:
    return self.size[1] / (self.resolution[1] - 1) 

# 'B': 2, # Number of space subdivisions in each axis.
# 't_mix': None, #[ 1.5 * 3600 ], # Timestamps when to perform mixing
# 'optimal_mix': False, # Indicator whether to apply perfect mixing
