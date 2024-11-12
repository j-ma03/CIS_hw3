import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union
from enum import Enum
from utils.meshgrid import Meshgrid

class Matching(Enum):
    LINEAR = 1
    KD_TREE = 2

class IterativeClosestPoint():
    def __init__(self, match_mode: Matching = Matching.LINEAR) -> None:
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
            return self._linear_match(pt_cloud, meshgrid)

    def _linear_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ):
        # Populate a matrix of closest distances to the meshgrid
        min_dist = np.empty(pt_cloud.shape[0])
        min_dist.fill(np.inf)

        # Populate a matrix of closest points on the meshgrid
        closest_pt = np.zeros_like(pt_cloud)

        # Iterate through all the triangles in the meshgrid
        for triangle in meshgrid:
            # Extract the bounding box of the triangle
            box = triangle.box()

            # Extend the bounding box by a margin determined by the
            # current minimum distance from each point
            box.top_left = box.top_left.reshape(1, 3)
            box.top_left -= np.repeat(min_dist.reshape(-1, 1), 3, axis=1)

            box.bottom_right = box.bottom_right.reshape(1, 3)
            box.bottom_right += np.repeat(min_dist.reshape(-1, 1), 3, axis=1)

            # Check if there are any candidates to consider
            candidates = box.contains(pt_cloud)
            
            if candidates.any():
                # Compute closest distance on the triangle for all candidates
                dist, pt = triangle.closest_distance_to(pt_cloud[candidates])

                # Find candidates where distance to triangle is less than
                # the previously recorded minimum distance
                new_mins = dist < min_dist[candidates]

                # Update the closest point and minimum distance
                closest_pt[candidates][new_mins] = pt[new_mins]
                min_dist[candidates][new_mins] = dist[new_mins]

        return closest_pt, min_dist
                

    def _kd_match(self):
        raise NotImplementedError