import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, pdb

class Lanes():
    def __init__(self, config):
        self.config = config
        self.binary_img = None
        self.y_coords = None
        self.left_base = None
        self.right_base = None
        self.left_pts = []
        self.right_pts = []
        self.left_coef = np.array([])
        self.right_coef = np.array([])

        self.left_curvature = None
        self.right_curvature = None
        self.camera_position = None
        self.offset = None
        self.ym_per_pix = 30/720
        self.xm_per_pix = 3.7/700

    def find_starting_point(self, binary_img):
        self.binary_img = binary_img
        self.y_coords = np.linspace(0, self.binary_img.shape[0]-1, self.binary_img.shape[0])
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

        if self.left_coef.size == 0 and self.right_coef.size == 0:
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

        # In the case we already have a fitted lanes
        else:
            left_lane_xy = ((nonzerox > (self.left_coef[0]*(nonzeroy**2) + self.left_coef[1]*nonzeroy + self.left_coef[2] - self.config.window_margin)) & (nonzerox < (self.left_coef[0]*(nonzeroy**2) + self.left_coef[1]*nonzeroy + self.left_coef[2] + self.config.window_margin)))
            right_lane_xy = ((nonzerox > (self.right_coef[0]*(nonzeroy**2) + self.right_coef[1]*nonzeroy +self.right_coef[2] - self.config.window_margin)) & (nonzerox < (self.right_coef[0]*(nonzeroy**2) + self.right_coef[1]*nonzeroy + self.right_coef[2] + self.config.window_margin)))

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


    def fit_poly_lines(self):
        np_left_pts = np.array(self.left_pts)
        np_right_pts = np.array(self.right_pts)
        self.left_coef = np.polyfit(np_left_pts[:,1], np_left_pts[:,0], 2)
        self.right_coef = np.polyfit(np_right_pts[:,1], np_right_pts[:,0], 2)

        try:
            left_fitx = self.left_coef[0]*self.y_coords**2 + self.left_coef[1]*self.y_coords + self.left_coef[2]
            right_fitx = self.right_coef[0]*self.y_coords**2 + self.right_coef[1]*self.y_coords + self.right_coef[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*self.y_coords**2 + 1*self.y_coords
            right_fitx = 1*self.y_coords**2 + 1*self.y_coords

        # update points
        self.left_pts = []
        self.right_pts = []
        for i in range(len(self.y_coords)):
            left_x = int(left_fitx[i])
            right_x = int(right_fitx[i])
            y = self.y_coords[i]
            left = [left_x, y]
            right = [right_x, y]
            self.left_pts.append(left)
            self.right_pts.append(right)

    def get_curvature(self):
        self.left_curvature = ((1+(2*self.left_coef[0]*self.binary_img.shape[0]*self.ym_per_pix + self.left_coef[1])**2)**(3/2))/np.absolute(2*self.left_coef[0])
        self.right_curvature = ((1+(2*self.right_coef[0]*self.binary_img.shape[0]*self.ym_per_pix + self.right_coef[1])**2)**(3/2))/np.absolute(2*self.right_coef[0])
        self.camera_position = (self.binary_img.shape[1]/2)
        y_pos = self.binary_img.shape[0]
        lane_center = (self.left_pts[y_pos-1][0] + self.right_pts[y_pos-1][0])/2
        self.offset = abs(self.camera_position - lane_center)*self.xm_per_pix
        
    def get_color_warp(self):
        warp_zero = np.zeros_like(self.binary_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        y_coords = np.linspace(10, self.binary_img.shape[0], self.binary_img.shape[0]-10+1)
        left = self.left_coef[0]*y_coords**2 + self.left_coef[1]*y_coords + self.left_coef[2]
        right = self.right_coef[0]*y_coords**2 + self.right_coef[1]*y_coords + self.right_coef[2]
        plot_left = np.array([np.transpose(np.vstack([left, y_coords]))])
        plot_right = np.array([np.flipud(np.transpose(np.vstack([right, y_coords])))])
        pts = np.hstack((plot_left, plot_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        return color_warp












