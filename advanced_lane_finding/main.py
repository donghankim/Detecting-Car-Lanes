from config import get_args
from calibrations import Camera
from threshold import Threshold
import numpy as np
import cv2
import os, pdb

def get_test_img():
    path = '../media/undistorted_images/test6.jpg'
    img = cv2.imread(path)
    return img

def main():
    config = get_args()
    test_img = get_test_img()
    Thresh = Threshold(config, test_img)
    Cam = Camera(config)
    Cam.calibrate_camera()

    # Only need to run once
    if config.undistort_test:
        Cam.undistort_all()










if __name__ == '__main__':
    main()
