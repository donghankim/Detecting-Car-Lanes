import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, pdb

class Lanes():
    def __init__(self, config):
        self.config = config
        self.binary_img = None
        self.left_base = None
        self.right_base = None

    def find_starting_point(self, binary_img):
        self.binary_img = binary_img
        bottom_half = binary_img[binary_img.shape[0]//2:, :]
        histogram = np.sum(bottom_half, axis=0)
        midpoint = np.int(histogram.shape[0]//2)
        self.left_base = np.argmax(histogram[:midpoint])
        self.right_base = np.argmax(histogram[midpoint:]) + midpoint

    def sliding_window(self):
        pass
    






