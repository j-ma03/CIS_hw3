import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Iterator
from utils.meshgrid import Triangle, BoundingBox

class Octree:
    def __init__(
        self,
        elements: List[Triangle],
        min_count: int = 1,
        min_diag: float = 0.1
    ) -> None:
        
        # Save Triangle elements
        self.elements: List[Triangle] = elements
        self.num_elements: int = len(elements)

        # Save bounding box of the triangle centers and compute 
        self.bb: BoundingBox = self.box()
        self.center: NDArray[np.float32] = (self.bb.min_xyz + self.bb.max_xyz) / 2.0
        self.have_subtrees: bool = False
        self.subtrees: List[Octree] = None

        # Build subtrees if necessary
        self.construct_subtrees(min_count, min_diag)

    def box(self) -> BoundingBox:
        """
        Computes the bounding box of the Triangle center points.
        """

        # Compute centers and find minimum and maximum corners
        centers = np.array([ele.center() for ele in self.elements])

        # No bounding box if there are no elements in tree
        if centers.size == 0:
            return None
        
        box = BoundingBox(np.min(centers, axis=0), np.max(centers, axis=0))

        return box
    
    def construct_subtrees(
        self,
        min_count: int,
        min_diag: float
    ) -> None:
        """
        Construct subtrees recursively by splitting elements into eight regions.
        """

        # Stop constructing subtrees if too few elements or the 
        # split diagonal is too small
        if self.num_elements <= min_count or \
            np.linalg.norm(self.bb.min_xyz - self.bb.max_xyz) <= min_diag:
            self.have_subtrees = False
            return
        
        # Add subtrees
        self.have_subtrees = True
        
        # Split and sort the things into 8 subtrees based on the center
        partitions = self.split_sort(self.center)

        # Construct subtrees for each partition
        self.subtrees = [ Octree(partitions[k]) for k in partitions if len(partitions[k]) > 0 ]
    
    def split_sort(self, splitting_point: NDArray[np.float32]) -> Dict[List[bool], int]:
        """
        Split the elements into octants based on comparison with the splitting point.
        This method updates the nnn, npn, etc., counts for each octant.
        """

        # Create empty partion as dictionary
        partitions = self.create_partitions()

        # Iterate through all elements 
        for ele in self.elements:
            # Extract center point of the element
            center = ele.center()

            # Add element to the respective parititons
            octant = (center >= splitting_point).tolist()
            partitions[tuple(octant)].append(ele)

        return partitions
    
    def create_partitions(self) -> Dict[List[bool], List[Triangle]]:
        """
        Creates eight partitions for the Octree as a Dictionary, which 
        contains the octants represented by a 3D list of booleans.
        """

        partitions = {}

        # Create the 8 quadrants represented by a 3D list of booleans
        # where True represents + and False represents -
        for x in [True, False]:
            for y in [True, False]:
                for z in [True, False]:
                    partitions[(x, y, z)] = []

        return partitions
    
    def __iter__(self) -> Iterator['Octree']:
        """
        Creates an iterator for accessing all the subtrees in
        the Octree.
        """

        # Returns iterator from list of subtrees
        return self.subtrees.__iter__()


