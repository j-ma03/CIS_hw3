import numpy as np
from numpy.typing import NDArray

class Triangle():
    def __init__(
        self,
        v1: NDArray[np.float32],
        v2: NDArray[np.float32],
        v3: NDArray[np.float32],
    ):
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

    def closest_distance_to(
        self,
        points: NDArray[np.float32]
    ) -> NDArray[np.float32]:
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

        # Solve the least squares problem to get λ, μ, v 
        b = np.linalg.lstsq(A, points.T, rcond=None)[0]

        # Find the point projected onto the plane of the triangle
        closest_pt = (A @ b).T

        # Find point that is outside the λ, μ, v constraints
        λ_constraint = np.nonzero(b[0] < 0)
        μ_constraint = np.nonzero(b[1] < 0)
        v_constraint = np.nonzero(b[2] < 0)

        # Handle the boundary cases where λ < 0
        closest_pt[λ_constraint] = self._project_on_seg(
            closest_pt[λ_constraint].reshape(-1, 3), self.v2, self.v3
        )

        # Handle the boundary cases where μ < 0
        closest_pt[μ_constraint] = self._project_on_seg(
            closest_pt[μ_constraint].reshape(-1, 3),self.v3, self.v1
        )
        
        # Handle the boundary cases where v < 0
        closest_pt[v_constraint] = self._project_on_seg(
            closest_pt[v_constraint].reshape(-1, 3), self.v1, self.v2
        )

        # Compute closest distance as magnitude of the difference vector
        closest_dist = np.linalg.norm(closest_pt - points, ord=2, axis=1)

        return closest_dist, closest_pt

    def _project_on_seg(
        self,
        c: NDArray[np.float32],
        p: NDArray[np.float32],
        q: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Given an set of points c as an Nx3 matrix, find the projection
        of the closest point onto the line segment pq, where p and q are
        3D vectors of (x, y, z) coordinates.
        """

        # c should be an Nx3 matrix of points
        if len(c.shape) != 2 or c.shape[1] != 3:
            raise ValueError('c should be an Nx3 matrix!')
        
        # p should be a 3D vector
        if len(p.shape) != 1 or p.shape[0] != 3:
            raise ValueError(f'p should be a 3D vector of (x, y, z) coordinates!')
        
        # q should be a 3D vector
        if len(q.shape) != 1 or q.shape[0] != 3:
            raise ValueError(f'q should be a 3D vector of (x, y, z) coordinates!')

        # Compute the degree to which segment pc aligns with pc*,
        # where c* are the points c projected onto pq
        λ = c.reshape(-1, 3) - p.reshape(1, 3)
        λ = λ @ (q - p).reshape(3, 1)

        # Normalize by the degree which the segment pq aligns with pq
        λ /= np.dot(q - p, q - p)
        λ = np.maximum(0.0, np.minimum(λ, 1.0))

        # Compute c*, the projection of c onto pq
        c_star = p + λ * (q - p)

        return c_star
        

class Meshgrid():
    def __init__(
        self,
        vertices: NDArray[np.float32],
        triangle_indices: NDArray[np.float32]
    ) -> None:
        
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError('Vertices should be provided as an Nx3 matrix!')
        
        if len(triangle_indices.shape) != 2 or triangle_indices.shape[1] != 3:
            raise ValueError('Triangle indices should be provided as a Tx3 matrix!')

        # Save all vertices in meshgrid as an Nx3 matrix
        self.vertices: NDArray[np.float32] = vertices

        # Save triangle vertex indices as a Tx3 matrix
        triangle_indices: NDArray[np.float32] = triangle_indices




