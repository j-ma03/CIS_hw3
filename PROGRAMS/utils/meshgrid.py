import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Iterator

class BoundingBox():
    def __init__(
        self,
        min_xyz: NDArray[np.float32],
        max_xyz: NDArray[np.float32]
    ):
        """
        Stores a bounding box by its minimum and maximum (x, y, z) corners.
        """

        # Top-left coordinate should be a 3D vector of (x, y, z) coordinates
        if len(min_xyz.shape) != 1 or min_xyz.shape[0] != 3:
            raise ValueError('Minimum corner coordinate should be a 3D vector of (x, y, z) coordinates!')
        
        # Bottom-left coordinate should be a 3D vector of (x, y, z) coordinates
        if len(max_xyz.shape) != 1 or max_xyz.shape[0] != 3:
            raise ValueError('Maximum corner coordinate should be a 3D vector of (x, y, z) coordinates!')

        # Stores the top left and bottom right corners of the bounding box
        self.min_xyz: NDArray[np.float32] = min_xyz
        self.max_xyz: NDArray[np.float32] = max_xyz

    def contains(self, points: NDArray[np.float32]) -> NDArray[np.bool8]:
        """
        Determines whether points, given as an Nx3 matrix, are contained within 
        or on the edge of the bounding box.
        """

        # Input should be an Nx3 matrix of points
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError('Points should be an Nx3 matrix!')
        
        # Finds which points are between the minimum and maximum values
        # for each coordinate axis
        bounds = (self.min_xyz[None,] <= points) & (points <= self.max_xyz[None,])
        bounds = np.all(bounds, axis=1)
        
        return bounds

    def overlaps(self, bounding_box: 'BoundingBox') -> bool:
        """
        Determines if the bounding box overlaps with another bounding box.
        """

        # Checks if the x-axis, y-axis, and z-axis overlaps
        if (self.bottom_right < bounding_box.top_left).any() or \
            (bounding_box.bottom_right < self.top_left).any():
            # One of the axes do not overlap
            return False
        
        # All axes overlap
        return True

class Triangle():
    def __init__(
        self,
        v1: NDArray[np.float32],
        v2: NDArray[np.float32],
        v3: NDArray[np.float32],
    ):
        """
        Stores a triangle as a set of vertices.
        """

        # Triangle vertices should be a 3D vector of (x, y, z) coordinates
        if len(v1.shape) != 1 or v1.shape[0] != 3:
            raise ValueError(f'Expected v1 to be a 3D vector of (x, y, z) coordinates but got {v1.shape}.')
        
        if len(v2.shape) != 1 or v2.shape[0] != 3:
            raise ValueError(f'Expected v2 to be a 3D vector of (x, y, z) coordinates but got {v2.shape}.')
        
        if len(v3.shape) != 1 or v3.shape[0] != 3:
            raise ValueError(f'Expected v3 to be a 3D vector of (x, y, z) coordinates but got {v3.shape}.')
        
        # Save triangle vertices
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def box(self) -> BoundingBox:
        """
        Computes a bounding box for the triangle based on its vertices.
        """

        # Stack vertices into a 3x3 matrix
        vertices = np.stack([self.v1, self.v2, self.v3], axis=0)

        # Compute coordinates of the top-left and bottom-right corners 
        # of the bounding box
        top_left = np.min(vertices, axis=0)
        bottom_right = np.max(vertices, axis=0)

        return BoundingBox(top_left, bottom_right)

    def closest_distance_to(
        self,
        points: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Computes the closest distance from each point in an Nx3 matrix
        to the triangle by solving a constrained least-squares problem.
        Returns an N-dimensional vector of the closest distance to 
        the triangle and an Nx3 matrix of the closest point on the triangle.
        """

        # Input should be an Nx3 matrix of points
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise ValueError('Points should be an Nx3 matrix!')

        # Construct 3x3 matrix of vertices in Barycentric form
        # Approach based on Dr. Taylor's slides on finding closest points:
        # https://ciis.lcsr.jhu.edu/lib/exe/fetch.php?media=courses:455-655:lectures:finding_point-pairs.pdf
        A = np.array([
            self.v1,
            self.v2,
            self.v3
        ]).T

        # Add an additional row of ones to system to enforce λ + μ + v = 0
        A = np.concatenate([A, np.ones((1, A.shape[1]))], axis=0)
        c_points = np.concatenate([points.T, np.ones((1, points.T.shape[1]))], axis=0)

        # Solve the least squares problem to get λ, μ, v 
        b = np.linalg.lstsq(A, c_points, rcond=None)[0]

        # Find the point projected onto the plane of the triangle
        closest_pt = (A @ b).T[:,:3]

        # Find point that is outside the λ, μ, v constraints
        λ_constraint = b[0] < 0
        μ_constraint = b[1] < 0
        v_constraint = b[2] < 0

        # Handle the boundary cases where λ < 0
        if λ_constraint.any():
            closest_pt[λ_constraint] = self._project_on_seg(
                closest_pt[λ_constraint].reshape(-1, 3), self.v2, self.v3
            )

        # Handle the boundary cases where μ < 0
        if μ_constraint.any():
            closest_pt[μ_constraint] = self._project_on_seg(
                closest_pt[μ_constraint].reshape(-1, 3),self.v3, self.v1
            )
        
        # Handle the boundary cases where v < 0
        if v_constraint.any():
            closest_pt[v_constraint] = self._project_on_seg(
                closest_pt[v_constraint].reshape(-1, 3), self.v1, self.v2
            )

        # Compute closest distance as magnitude of the difference vector
        closest_dist = np.linalg.norm(closest_pt - points, ord=2, axis=1)

        return closest_dist, closest_pt

    def _project_on_seg(self, c: NDArray[np.float32], p: NDArray[np.float32], q: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Project points `c` onto the line segment defined by `p` and `q`.
        """
        # Vector from p to c
        pc = c - p
        pq = q - p

        # Compute the scalar projection of pc onto pq
        λ = np.sum(pc * pq, axis=1) / np.sum(pq * pq)

        # Clamp λ to the range [0, 1] to ensure the projection lies on the segment
        λ = np.clip(λ, 0.0, 1.0)

        # Compute the projection point
        c_star = p + λ[:, None] * pq

        return c_star
    
    def __repr__(self) -> str:
        """
        Define Triangle object as a string representation
        """

        # Print out triangle vertices as string
        return str((self.v1, self.v2, self.v3))
        
class Meshgrid():
    def __init__(
        self,
        vertices: NDArray[np.float32],
        triangle_indices: NDArray[np.float32]
    ) -> None:
        """
        Stores a meshgrid as a set of vertices and Triangles.
        """
        
        # Vertices should be an Nx3 matrix of (x, y, z) coordinates
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError('Vertices should be provided as an Nx3 matrix!')
        
        # Triangle indices should be a Tx3 matrix of indices cooresponding
        # to the three vertices of the triangle
        if len(triangle_indices.shape) != 2 or triangle_indices.shape[1] != 3:
            raise ValueError('Triangle indices should be provided as a Tx3 matrix!')

        # Save all vertices in meshgrid as an Nx3 matrix
        self.vertices: NDArray[np.float32] = vertices

        # Save triangle vertex indices as a Tx3 matrix
        self.triangle_indices: NDArray[np.float32] = triangle_indices

        # Save triangles as a list of Triangle objects
        self.trangles: List[Triangle] = []

        # Construct Triangles and add them to the list
        for i in range(self.triangle_indices.shape[0]):
            # Extract the vertices as (x, y, z) coordinates
            v1, v2, v3 = self.vertices[self.triangle_indices[i]]

            self.trangles.append(
                Triangle(v1, v2, v3)
            )

    def __iter__(self) -> Iterator[Triangle]:
        """
        Creates an iterator for accessing all the triangles in
        the meshgrid.
        """

        # Returns iterator from list of Triangles
        return self.trangles.__iter__()
