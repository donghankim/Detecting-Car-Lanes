import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, pdb


class Funcs():
    def __init__(self, config):
        self.config = config

    def get_test_img(self):
        path = '../media/undistorted_images/test_git.jpg'
        img = cv2.imread(path)
        return img

    def show_img(self, img):
        plt.imshow(img, cmap='gray')
        plt.show()

    def draw_poly_lines(self, img, *pts):
        for i in range(len(pts)):
            image = cv2.polylines(img, [pts[i]], True, (255, 0, 0), 3)

        return image

    def color_poly(self, img, pts):
        cv2.fillPoly(img, pts, (0, 255, 0))
        return img

    # for documentation, must send 3 images
    def save_fig(self, filename, *img):
        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        fig.subplots_adjust(hspace=0.1, wspace=0.2)
        axes.ravel()

        axes[0].imshow(img[0])
        axes[1].imshow(img[1], cmap = 'gray')
        axes[2].imshow(img[2])
        axes[0].set_title("Warped")
        axes[1].set_title("Binary Output")
        axes[2].set_title("Lanes Output")
        plt.savefig(os.path.join(self.config.github_save, filename))
        #plt.show()

