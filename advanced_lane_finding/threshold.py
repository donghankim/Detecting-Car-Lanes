import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, pdb

class Threshold():
    def __init__(self, config):
        self.config = config
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
        # self.show_img(binary_output)

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
        # self.show_img(binary_output)

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
        #self.show_img(self.gradient_dir)

    def get_bin(self, img):
        """
        Combining the results from Sobel_x and the gradient magitude will
        probably produce the best results.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.gray_smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
        self.hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        self.S = self.hls[:, :, 2]
        self.sobel_x()
        self.sobel_y()
        self.sobel_xy()
        self.sobel_direction()
        self.final_output = np.zeros_like(self.gradient_dir)
        self.final_output[(self.sobel_mag == 1) & (self.gradient_dir == 1)] = 1
        # self.show_img(self.final_output)
    
        return self.sobelx


    def show_img(self, img):
        plt.imshow(img, cmap = 'gray')
        plt.show()

    # for documentation, must send 3 images
    def save_fig(self, filename, *img):
        fig, axes = plt.subplots(1, 3, figsize = (16,8))
        fig.subplots_adjust(hspace = 0.1, wspace = 0.2)
        axes.ravel()

        """
        for i in range(len(img)):
            axes[i].imshow(img[i], cmap = 'gray')
        """

        axes[0].imshow(img[0])
        axes[1].imshow(img[1])
        axes[2].imshow(img[2], cmap = 'gray')
        axes[0].set_title("Original")
        axes[1].set_title("Warped")
        axes[2].set_title("Binary Output")
        #plt.savefig(os.path.join(self.config.github_save, filename))
        plt.show()



