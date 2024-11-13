import unittest

from utils.dataloader import *
# from utils.icp import *
from utils.meshgrid import Triangle
import numpy as np

class TestClosestPoint(unittest.TestCase):    
    def test_closest_point_above_plane(self):
        # generate 3 random points that make up a triangle
        # generate random point p that lies on the plane of the triangle -- ground truth point
        # project a vector p' from p of random distance perpendicular to the plane of the triangle
        # find experimental closest point on triangle by calling closest_distance_to(p')
        # compare the two
        # assert that experimental closest point and p are the same

        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v3 = np.random.rand(3)
        triangle = Triangle(v1, v2, v3) # create a triangle 

        # generate random point that lies on the plane of the triangle
        r1, r2 = np.random.rand(2)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        p = (1 - r1 - r2) * v1 + r1 * v2 + r2 * v3

        # get unit vector perpendicular to the plane of the triangle
        n = np.cross(v2 - v1, v3 - v1)
        n = n / np.linalg.norm(n)
        # project a vector p' from p of random distance perpendicular to the plane of the triangle
        distance = np.random.rand()
        p_prime = p + distance * n

        # find experimental closest point on triangle by calling closest_distance_to(p')
        closest_dist, closest_point = triangle.closest_distance_to(p_prime)

        # compare the two
        np.testing.assert_almost_equal(closest_point, p, decimal=5)


        pass

    def test_closest_point_outside_plane(self):
        # generate 3 random points that make up a triangle
        # generate random point p that lies outside the plane of the triangle -- ground truth point
        # project a vector p' from p of random distance perpendicular to the plane of the triangle
        # find experimental closest point on triangle by calling closest_distance_to(p')
        # compare the two
        # assert that experimental closest point and p are the same

        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v3 = np.random.rand(3)
        triangle = Triangle(v1, v2, v3)

        # generate random point that lies on edge of the triangle
        # Randomly choose an edge: 0 for AB, 1 for BC, 2 for CA
        edge_choice = np.random.choice([0, 1, 2])
        
        # Generate a random value for t between 0 and 1
        t = np.random.rand()
        
        # Calculate the point on the chosen edge
        if edge_choice == 0:
            # Point on edge AB
            P = (1 - t) * v1 + t * v2
        elif edge_choice == 1:
            # Point on edge BC
            P = (1 - t) * v2 + t * v3
        else:
            # Point on edge CA
            P = (1 - t) * v3 + t * v1

        # get unit vector perpendicular to P
        n = np.cross(v2 - v1, v3 - v1)
        n = n / np.linalg.norm(n)
        




# class TestDataloader(unittest.TestCase):
#     def test_read_file(self):
#         """
#         Test the read_file method of the RigidBodyDataloader class
#         """
#         # create a dataloader object
#         dl = RigidBodyDataloader.read_file('data/1.txt')
#         self.assertEqual(dl.N_markers, 4)
#         self.assertEqual(dl.markers.shape, (4, 3))
#         self.assertEqual(dl.tip.shape, (1, 3))
    
# class TestOutputAccuracy(unittest.TestCase):
#     def test_output_accuracy(self):
#         """
#         Test the accuracy of the output of the dataloader
#         """
#         # create a dataloader object
#         dl = RigidBodyDataloader.read_file('data/1.txt')
#         self.assertEqual(dl.N_markers, 4)
#         self.assertEqual(dl.markers.shape, (4, 3))
#         self.assertEqual(dl.tip.shape, (1, 3))
