import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union
from enum import Enum

class Matching(Enum):
    LINEAR = 1
    KD_TREE = 2

class IterativeClosestPoint():
    def __init__(self, match_mode: Matching) -> None:
        self.match_mode: Matching = match_mode

    def __call__(self):
        pass

    def match(self):
        pass

    def _linear_match(self):
        pass

    def _kd_match(self):
        pass