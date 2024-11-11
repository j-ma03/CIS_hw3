import unittest

from utils.dataloader import *
from utils.icp import *
import numpy as np

class TestClosestPoint(unittest.TestCase):
    def generate_triangle():
        return np.random.rand(3, 3)
    
    def generate_point_cloud():
        return np.random.rand(10, 3)
    
    def test_closest_point(self):
        # generate 3 random points that make up a triangle
        # generate random point cloud of 10 points
        # find ground truth closest point
        # find experimental closest point
        # compare the two

        triangle = self.generate_triangle()
        pt_cloud = self.generate_point_cloud()

        pass

class TestDataloader(unittest.TestCase):
    def test_read_file(self):
        """
        Test the read_file method of the RigidBodyDataloader class
        """
        # create a dataloader object
        dl = RigidBodyDataloader.read_file('data/1.txt')
        self.assertEqual(dl.N_markers, 4)
        self.assertEqual(dl.markers.shape, (4, 3))
        self.assertEqual(dl.tip.shape, (1, 3))
    
class TestOutputAccuracy(unittest.TestCase):
    def test_output_accuracy(self):
        """
        Test the accuracy of the output of the dataloader
        """
        # create a dataloader object
        dl = RigidBodyDataloader.read_file('data/1.txt')
        self.assertEqual(dl.N_markers, 4)
        self.assertEqual(dl.markers.shape, (4, 3))
        self.assertEqual(dl.tip.shape, (1, 3))
