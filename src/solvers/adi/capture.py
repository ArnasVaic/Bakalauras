from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
from solvers.adi.state import State

class SnapshotStrategy:

  @abstractmethod
  def capture(self, state: State) -> None:
    pass

  @abstractmethod
  def result(self) -> np.ndarray:
    pass

@dataclass
class FrameSnapshot(SnapshotStrategy):

  result = []

  def capture(self, state: State):
    return 


class FrameSkip:
  
  """Abstract class for a component which 
  indicates to the solver which solution steps
  should be skipped and which should be captured.
  """

  @abstractmethod
  def should_skip(self, state: State) -> bool:
    pass

class StrideSkip(FrameSkip):
  """Skip a set amount of frames for each captured frame.
  """

  def __init__(self, stride: int):
    self.stride = stride

  def should_skip(self, state: State) -> bool:
    return state.time_step % self.stride != 0

class FixedThresholdQSkip(FrameSkip):
  """Try to capture a fixed number of evenly spaced frames
  until the initial quantity of elements within the system
  reaches a given threshold.
  """

  def __init__(self, threshold: float, target_frame_cnt: int):
    self.target_frame_cnt = target_frame_cnt
    self.threshold = threshold

  def should_capture(self, state: State) -> bool:
    # we know the current time step
    # and we know the current element ratio
    # we can approximately calculate from the given
    # parameters the specific percentages at which
    # to capture.
    # 
    # Problem 1. Multiple frames might have the same percentage, how to avoid duplicate captures?

    # try to capture frames at the exact percentages between 100% and the given threshold
    capture_points = np.linspace(self.threshold, 1.0, self.target_frame_cnt)

    # we don't need to sum the third channel
    q_current = state.current[:2].sum(axis=(1, 2))
    q_initial = state.initial[:2].sum(axis=(1, 2))

    ratio = (q_current[0] + q_current[1]) / (q_initial[0] + q_initial[1])

    return capture_points.any()