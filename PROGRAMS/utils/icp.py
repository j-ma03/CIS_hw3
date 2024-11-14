import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union
from enum import Enum
from utils.meshgrid import Meshgrid, BoundingBox
from utils.octree import Octree

class Matching(Enum):
    SIMPLE_LINEAR = 1
    VECTORIZED_LINEAR = 2
    SIMPLE_OCTREE = 3

class IterativeClosestPoint():
    def __init__(self, match_mode: Matching = Matching.SIMPLE_LINEAR) -> None:
        # Define the algorithm used to find closest points
        self.match_mode: Matching = match_mode

    def __call__(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ):
        """
        Runs the full ICP algorithm given a point cloud and meshgrid.
        """
        raise NotImplementedError

    def match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ):
        """
        Finds the closest point and distance from points on a point cloud 
        to a triangle meshgrid.
        """

        if not isinstance(meshgrid, Meshgrid):
            raise Exception(f'Expected input meshgrid should be of type Meshgrid but got \'{meshgrid.__class__.__name__}\'!')

        if self.match_mode == Matching.SIMPLE_LINEAR:
            return self._simple_linear_match(pt_cloud, meshgrid)
        
        elif self.match_mode == Matching.VECTORIZED_LINEAR:
            return self._vectorized_linear_match(pt_cloud, meshgrid)
        
        elif self.match_mode == Matching.SIMPLE_OCTREE:
            return self._octree_match(pt_cloud, meshgrid)

    def _simple_linear_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Implementation of a simple linear search ICP algorithm containing 
        loops over all data points and Triangles to find the closest point 
        and distance to the meshgrid.
        """

        # Populate a matrix of closest distances to the meshgrid
        min_dist = np.empty(pt_cloud.shape[0])
        min_dist.fill(np.inf)

        # Populate a matrix of closest points on the meshgrid
        closest_pt = np.zeros_like(pt_cloud)

        # Iterate through all the points and triangles in the meshgrid
        for i, point in enumerate(pt_cloud):
            for triangle in meshgrid:
                # Extract the bounding box of the triangle
                box = triangle.box()

                # Extend the bounding box by a margin determined by the
                # current minimum distance from each point
                box.enlarge(min_dist[i])

                # Check if there are any candidates to consider
                if box.contains(point[None,]):
                    # Compute closest distance on the triangle for all candidates
                    dist, pt = triangle.closest_distance_to(point[None,])

                    # Find candidates where distance to triangle is less than
                    # the previously recorded minimum distance
                    if dist[0] < min_dist[i]:
                        # Update the closest point and minimum distance
                        closest_pt[i] = pt[0]
                        min_dist[i] = dist[0]

        return closest_pt, min_dist
    
    def _vectorized_linear_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Implementation of a fast vectorized linear search ICP algorithm
        containing a only single loop over all Triangles in the meshgrid.
        Closest point and distance for all data points are updated at once 
        for each Triangle.
        """

        # Populate a matrix of closest distances to the meshgrid
        min_dist = np.empty(pt_cloud.shape[0])
        min_dist.fill(np.inf)

        # Populate a matrix of closest points on the meshgrid
        closest_pt = np.zeros_like(pt_cloud)

        # Stack all points to manage them together
        pt_cloud_expanded = pt_cloud.reshape(-1, 3)

        # Iterate through all the triangles in the meshgrid
        for triangle in meshgrid:
            # Extract the bounding box of the triangle
            box = triangle.box()

            # Extend the bounding box by a margin determined by the
            # current minimum distance from each point
            expanded_min = box.min_xyz.reshape(1, 3) - min_dist.reshape(-1, 1)
            expanded_max = box.max_xyz.reshape(1, 3) + min_dist.reshape(-1, 1)

            # Identify candidate points within the bounding box
            candidates = np.all((expanded_min <= pt_cloud_expanded) & \
                                (pt_cloud_expanded <= expanded_max), axis=1)

            # Check if there are any candidates to consider
            if candidates.any():
                # Compute closest distance on the triangle for all candidates
                candidate_points = pt_cloud_expanded[candidates]
                dist, pt = triangle.closest_distance_to(candidate_points)

                # Find candidates where distance to triangle is less than
                # the previously recorded minimum distance
                closer_mask = dist < min_dist[candidates]

                # Select indices where new distances are closer from candidate indices
                indices = np.where(candidates)[0]
                closer_indices = indices[closer_mask]

                # Update the closest point and minimum distance
                min_dist[closer_indices] = dist[closer_mask]
                closest_pt[closer_indices] = pt[closer_mask]

        return closest_pt, min_dist
    
    def _octree_match(
        self,
        pt_cloud: NDArray[np.float32],
        meshgrid: Meshgrid,
        tree: Octree = None,
        closest_pt: NDArray[np.float32] = None,
        min_dist: NDArray[np.float32] = None
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Performs an ICP match using an octree with iterative point processing.
        Each point iteratively checks nodes within the octree to locate the closest
        point on the meshgrid.
        """

        if closest_pt is None:
            # Populate a matrix of closest points on the meshgrid
            closest_pt = np.zeros_like(pt_cloud)
            
        if min_dist is None:
            # Populate a matrix of closest distances to the meshgrid
            min_dist = np.empty(pt_cloud.shape[0])
            min_dist.fill(np.inf)

        if tree is None:
            # Create a new tree containing triangles from the meshgrid
            tree = Octree(meshgrid.triangles)

        if tree.num_elements == 0:
            return closest_pt, min_dist

        # Iterate through all the points and subtrees
        if tree.have_subtrees:
            for i, point in enumerate(pt_cloud):
                for subtree in tree:
                    # Extract the bounding box of the subtree
                    box = subtree.box()

                    # Extend the bounding box by a margin determined by the
                    # current minimum distance from each point
                    box.enlarge(min_dist[i])

                    # Check if there are any candidates to consider
                    if box.contains(point[None,]):
                        # Find candidates where distance to tree's centroid is less than
                        # the previously recorded minimum distance to a triangle
                        if np.linalg.norm(point - subtree.center, ord=2) < min_dist[i]:
                            # Recursively search the subtree
                            closest_pt, min_dist = self._octree_match(
                                pt_cloud, meshgrid, subtree, closest_pt, min_dist
                            )

        # Iterate through all elements of the subtree if node is a child
        if not tree.have_subtrees:
            for i, point in enumerate(pt_cloud):
                for triangle in tree.elements:
                    # Extract the bounding box of the triangle
                    box = triangle.box()

                    # Extend the bounding box by a margin determined by the
                    # current minimum distance from each point
                    box.enlarge(min_dist[i])

                    # Check if there are any candidates to consider
                    if box.contains(point[None,]):
                        # Compute closest distance on the triangle for all candidates
                        dist, pt = triangle.closest_distance_to(point[None,])

                        # Find candidates where distance to triangle is less than
                        # the previously recorded minimum distance
                        if dist[0] < min_dist[i]:
                            # Update the closest point and minimum distance
                            closest_pt[i] = pt[0]
                            min_dist[i] = dist[0]

        return closest_pt, min_dist
