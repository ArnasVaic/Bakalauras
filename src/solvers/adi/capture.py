from abc import abstractmethod
from dataclasses import dataclass
import numpy as np

class FrameCapture:
  
  @abstractmethod
  def should_capture(self, frame: np.ndarray) -> bool:
    pass
  
  @abstractmethod
  def result() -> tuple[np.ndarray, np.ndarray]:
    pass
  
# auto frame is designed to work with threshold stopper.
# It will automatically calculate the stride needed to
# save approximately the number of target frames, from the 
# current ratio of elements remaining
class AutoFrameCapture(FrameCapture):
  
  def __init__(self, targetFrameCount: int = 100):
    self.targetFrameCount = targetFrameCount

  def should_capture(self, frame: np.ndarray):
    return 