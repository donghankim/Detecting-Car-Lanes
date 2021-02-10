import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, pdb

class Threshold():
    def __init__(self, config):
        self.config = config
        self.img = None
        self.gray_smoothed = None
        self.hls = None
        self.sobelx = None
        self.sobely = None
        self.sobel_mag = None
        self.gradient_dir = None
        self.final_output = None

    def sobel_x(self, thresh_min = None, thresh_max = None):
        sobelx = cv2.Sobel(self.gray_smoothed, cv2.CV_64F, 1, 0, ksize = self.config.ksize)
        sobel_abs = np.absolute(sobelx)
        uintSobel = np.uint8(255*sobel_abs/np.max(sobel_abs))
        binary_output = np.zeros_like(uintSobel)

        if thresh_min == None or thresh_max == None:
            binary_output[(uintSobel >= self.config.thresh_min) & (uintSobel <= self.config.thresh_max)] = 1
        else:
            binary_output[(uintSobel >= thresh_min) & (uintSobel <= thresh_max)] = 1

        self.sobelx = binary_output


    def sobel_y(self, thresh_min = None, thresh_max = None):
        sobely = cv2.Sobel(self.gray_smoothed, cv2.CV_64F, 0, 1, ksize = self.config.ksize)
        sobel_abs = np.absolute(sobely)
        uintSobel = np.uint8(255*sobel_abs/np.max(sobel_abs))
        binary_output = np.zeros_like(uintSobel)

        if thresh_min == None or thresh_max == None:
            binary_output[(uintSobel >= self.config.thresh_min) & (uintSobel <= self.config.thresh_max)] = 1
        else:
            binary_output[(uintSobel >= thresh_min) & (uintSobel <= thresh_max)] = 1

        self.sobely = binary_output


    def sobel_xy(self, thresh_min = None, thresh_max = None):
        sobelx = cv2.Sobel(self.gray_smoothed, cv2.CV_64F, 1, 0, ksize = self.config.ksize)
        sobely = cv2.Sobel(self.gray_smoothed, cv2.CV_64F, 0, 1, ksize = self.config.ksize)
        sobel_mag = (sobelx**2 + sobely**2)**(1/2)
        scale_factor = np.max(sobel_mag)/255
        sobel_mag = (sobel_mag/scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(sobel_mag)

        if thresh_min == None or thresh_max == None:
            binary_output[(sobel_mag >= self.config.thresh_min) & (sobel_mag <= self.config.thresh_max)] = 1
        else:
            binary_output[(sobel_mag >= thresh_min) & (sobel_mag <= thresh_max)] = 1

        self.sobel_mag = binary_output
        # self.show_img(binary_output)

    def sobel_direction(self, angle_min = None, angle_max = None):
        sobelx = cv2.Sobel(self.gray_smoothed, cv2.CV_64F, 1, 0, ksize = self.config.ksize)
        sobely = cv2.Sobel(self.gray_smoothed, cv2.CV_64F, 0, 1, ksize = self.config.ksize)
        x_abs = np.absolute(sobelx)
        y_abs = np.absolute(sobely)
        angle = np.arctan2(y_abs, x_abs)
        binary_output = np.zeros_like(angle)

        if angle_min == None or angle_max == None:
            binary_output[(angle >= self.config.angle_min) & (angle <= self.config.angle_max)] = 1
        else:
            binary_output[(angle >= angle_min) & (angle <= angle_max)] = 1
        self.gradient_dir = binary_output


    def grad_method(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.gray_smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
        self.sobel_x()
        self.sobel_y()
        self.sobel_xy()
        self.sobel_direction()
        grad_output = np.zeros_like(self.gray_smoothed)
        grad_output[(self.sobel_mag == 1) & (self.sobelx == 1)] = 1
        return grad_output

    def hls_method(self):
        self.hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
        h = self.hls[:,:,0]
        l = self.hls[:,:,1]
        s = self.hls[:,:,2]
        r = self.img[:,:,0]
        color_output = np.zeros_like(self.gray_smoothed)
        # color_output[((s > 1) & (l == 0)) | ((s == 0) & (h > 1) & (l > 1))] = 1
        color_output[(s > 90) & (s < 255)] = 1
        return color_output


    def get_bin(self, img):
        """
        Combining the results from gradient threhsolding and color thresholding (to make it more robus to shadows).
        """
        self.img = img
        grad_output = self.grad_method()
        color_output = self.hls_method()

        self.final_output = np.zeros_like(self.gray_smoothed)
        self.final_output[(grad_output == 1) | (color_output == 1)] = 1
        #pdb.set_trace()
        return self.final_output

    # for debugging
    def show_img(self, img):
        plt.imshow(img, cmap = 'gray')
        plt.show()



