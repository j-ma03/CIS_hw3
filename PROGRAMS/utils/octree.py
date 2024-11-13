import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Iterator, Union
from utils.meshgrid import Triangle, BoundingBox

class OctreeNode:
    def __init__(
        self,
        elements: List[Triangle],
        min_count: int = 1,
        min_diag=0.1
    ):
        
        # Save Triangle elements
        self.elements: List[Triangle] = elements
        self.num_elements: int = len(elements)

        # Save bounding box of the triangle centers and compute 
        self.bb = self.get_bounding_box()
        self.center = (self.bb.min_xyz + self.bb.max_xyz) / 2.0
        self.have_subtrees = False
        self.subtrees = None

        # Build subtrees if necessary
        self.construct_subtrees(min_count, min_diag)

    def get_bounding_box(self) -> BoundingBox:
        """
        Computes the bounding box of the Triangle center points.
        """

        # Compute centers and find minimum and maximum corners
        centers = np.array([ele.center() for ele in self.elements])
        box = BoundingBox(np.min(centers, axis=0), np.max(centers, axis=0))

        return box
    
    def construct_subtrees(self, min_count, min_diag):
        """
        Construct subtrees recursively by splitting elements into eight regions.
        """

        # Stop constructing subtrees if too free elements or the 
        # split diagonal is too small
        if self.num_elements <= min_count or \
            np.linalg.norm(self.bb.min_xyz - self.bb.max_xyz) <= min_diag:
            self.have_subtrees = False
            return
        
        # Add subtrees
        self.have_subtrees = True
        counts = [0] * 8  # Counts for each octant
        
        # Split and sort the things into 8 subtrees based on the center
        self.split_sort(self.center, counts)

        # Add padding to list of counts
        counts = [0] + counts

        # Construct subtrees for each partition
        self.subtrees = [OctreeNode(self.elements[counts[i]:counts[i]+counts[i+1]]) 
                         for i in range(len(counts) - 1)]
    
    def split_sort(self, splitting_point, counts):
        """
        Split the elements into 8 regions based on comparison with the splitting point.
        This method updates the nnn, npn, etc., counts for each octant.
        """

        for ele in self.elements:
            sort_point = ele.center()
            index = (sort_point[0] >= splitting_point[0]) * 4 + (sort_point[1] >= splitting_point[1]) * 2 + (sort_point[2] >= splitting_point[2])


