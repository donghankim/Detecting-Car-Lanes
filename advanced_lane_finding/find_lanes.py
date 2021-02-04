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
        self.left_pts = []
        self.right_pts = []

    def find_starting_point(self, binary_img):
        self.binary_img = binary_img
        bottom_half = binary_img[binary_img.shape[0]//2:, :]
        histogram = np.sum(bottom_half, axis=0)
        midpoint = np.int(histogram.shape[0]//2)
        self.left_base = np.argmax(histogram[:midpoint])
        self.right_base = np.argmax(histogram[midpoint:]) + midpoint

    def sliding_window(self):
        window_height = np.int(self.binary_img.shape[0]//self.config.windows_num)
        nonzero_xy = self.binary_img.nonzero()
        nonzeroy = np.array(nonzero_xy[0])
        nonzerox = np.array(nonzero_xy[1])

        leftx_current = self.left_base
        rightx_current = self.right_base
        left_lane_xy = []
        right_lane_xy = []

        for window in range(self.config.windows_num):
            win_y_low = self.binary_img.shape[0] - (window+1)*window_height
            win_y_high = self.binary_img.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.config.window_margin
            win_xleft_high = leftx_current + self.config.window_margin
            win_xright_low = rightx_current - self.config.window_margin
            win_xright_high = rightx_current + self.config.window_margin

            # Identify the nonzero pixels in x and y within the window
            good_left_xy = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_xy = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_xy.append(good_left_xy)
            right_lane_xy.append(good_right_xy)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_xy) > self.config.min_pix:
                leftx_current = np.int(np.mean(nonzerox[good_left_xy]))
            if len(good_right_xy) > self.config.min_pix:
                rightx_current = np.int(np.mean(nonzerox[good_right_xy]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_xy = np.concatenate(left_lane_xy)
            right_lane_xy = np.concatenate(right_lane_xy)
        except ValueError:
            pass

        leftx = nonzerox[left_lane_xy]
        lefty = nonzeroy[left_lane_xy]
        rightx = nonzerox[right_lane_xy]
        righty = nonzeroy[right_lane_xy]

        for i in range(len(leftx)):
            x = leftx[i]
            y = lefty[i]
            new_point = [x, y]
            self.left_pts.append(new_point)

        for i in range(len(rightx)):
            x = rightx[i]
            y = righty[i]
            new_point = [x, y]
            self.right_pts.append(new_point)

        self.left_pts = np.array(self.left_pts, np.int32)
        self.right_pts = np.array(self.right_pts, np.int32)

    def fit_poly_lines(self):
        pass








