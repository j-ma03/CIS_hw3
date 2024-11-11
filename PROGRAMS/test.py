import unittest

from utils.dataloader import *
# from utils.icp import *
from utils.meshgrid import Triangle
import numpy as np

class TestClosestPoint(unittest.TestCase):    
    def test_closest_point(self):
        # generate 3 random points that make up a triangle
        # generate random point
        # find ground truth closest point on triangle
        # find experimental closest point on triangle
        # compare the two

        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v3 = np.random.rand(3)

        triangle = Triangle(v1, v2, v3)

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
