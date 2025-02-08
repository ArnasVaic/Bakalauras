DEFAULT_CONFIG = {

  # Physical size of the simulation space.
  'size': (2.154434690031884, 2.154434690031884),
  'resolution': (80, 80), # Number of discrete points in each axis
 
  'D': 28e-6, # Diffusion coefficient
  'k': 192, # Reaction speed
  'c0': 1e-6, # Initial concentration value

  # Simulation time step. Setting this 
  # value to None will signal to the 
  # solver to choose the largest step
  # that would ensure stability. 
  'dt': None,

  # Stopping condition (one of these has to be non-None)
  # Stops the reactions when combined quantity
  # of initial elements reaches threshold.
  'threshold': 0.03,
  'total_steps': None, # Time steps to simulate.
  
  # Mixing parameters
  
  'B': 2, # Number of space subdivisions in each axis.
  't_mix': None, #[ 1.5 * 3600 ], # Timestamps when to perform mixing
  'optimal_mix': False, # Indicator whether to apply perfect mixing

  # Reduce the size of the array by
  # saving a small number of frames
  # spaced evenly throughout the time.
  'frame_stride': 10,
}

def dxdy(config):
  W, H, N, M = config['size'], config['resolution']

  dx, dy = W / (N - 1) = H / (M - 1)

  return dx, dy

def reaction_constants(config):
  return config['D'], config['k'], config['c0']