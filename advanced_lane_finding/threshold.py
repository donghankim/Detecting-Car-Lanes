import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, pdb

class Threshold():
    def __init__(self, config, img):
        self.config = config
        self.img = img
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        self.gray_smoothed = cv2.GaussianBlur(gray, (3, 3), 0)

        self.sobelx = None
        self.sobely = None
        self.sobel_mag = None
        self.gradient_dir = None
        self.final_output = None

        self.sobel_x(20, 100)
        #self.sobel_y()

    def sobel_x(self, thresh_min, thresh_max):
        self.sobelx = cv2.Sobel(self.gray_smoothed, cv2.CV_64F, 1, 0, ksize = 3)
        sobel_abs = np.absolute(self.sobelx)
        uintSobel = np.uint8(255*sobel_abs/np.max(sobel_abs))
        binary_output = np.zeros_like(uintSobel)
        binary_output[(uintSobel >= thresh_min) & (uintSobel <= thresh_max)] = 1
        self.show_img(binary_output)

    def sobel_y(self, threh_min, thresh_max):
        self.sobely = cv2.Sobel(gray_smoothed, cv2.CV_64F, 0, 1, ksize = 3)
        sobel_abs = np.absolute(sobel_y)
        uintSobel = np.uint8(255*sobel_abs/np.max(sobel_abs))
        binary_output = np.zeros_like(uintSobel)
        binary_output[(uintSobel >= thresh_min) & (uintSobel <= thresh_max)] = 1
        show_img(binary_output)

    def sobel_xy(self, thresh_min, thresh_max):
        self.sobel_mag = (self.sobelx**2 + self.sobely**2)**(1/2)
        scale_factor = np.max(self.sobel_mag)/255
        self.sobel_mag = (self.sobel_mag/self.scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(self.sobel_mag)
        binary_output[(self.sobel_mag >= threh_min) & (self.sobel_mag <= thresh_max)] = 1
        show_img(binary_output)




    def show_img(self, img):
        plt.imshow(img, cmap = 'gray')
        plt.show()

    # for documentation
    def save_fig(self, img):
        fig, axes = plt.subplots(1,3, figsize = (16,8))
        fig.subplots_adjust(hspace = 0.1, wspace = 0.2)
        axes.ravel()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #axes[0].

        cv2.imwrite(os.path.join(self.config.github_save, filename), img)

