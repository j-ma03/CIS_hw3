import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union
from enum import Enum
from meshgrid import Meshgrid

class Matching(Enum):
    LINEAR = 1
    KD_TREE = 2

class IterativeClosestPoint():
    def __init__(self, match_mode: Matching) -> None:
        # Define the algorithm used to find closest points
        self.match_mode: Matching = match_mode

    def __call__(self):
        pass

    def match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ):
        """
        Finds the closest point pairs given a point cloud and meshgrid
        """
        if self.match_mode == Matching.LINEAR:
            return self._linear_match(pt_cloud_A, pt_cloud_B)

    def _linear_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ):
        for 
        pass

    def _kd_match(self):
        pass