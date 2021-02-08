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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

    def draw_poly_lines(self, img, *pts):
        for i in range(len(pts)):
            image = cv2.polylines(img, [pts[i]], True, (255, 0, 0), 3)

        return image

    # for documentation, must send 2 images
    def save_fig(self, filename, *img):
        fig, axes = plt.subplots(1, len(img), figsize=(16, 8))
        fig.subplots_adjust(hspace=0.1, wspace=0.2)
        axes.ravel()
        img1 = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img[1], cv2.COLOR_BGR2RGB)
        axes[0].imshow(img1)
        axes[1].imshow(img2)

        axes[0].set_title("Warped Lanes")
        axes[1].set_title("Output")

        plt.savefig(os.path.join(self.config.github_save, filename))
        #plt.show()

