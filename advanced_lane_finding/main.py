from config import get_args
from calibrations import Camera
from threshold import Threshold
from find_lanes import Lanes
import numpy as np
import cv2
import os, pdb

def get_test_img():
    path = '../media/undistorted_images/test_git.jpg'
    img = cv2.imread(path)
    return img

def main():
    config = get_args()
    Thresh = Threshold(config)
    Cam = Camera(config)
    Cam.calibrate_camera()
    Lane = Lanes(config)

    # Only need to run once
    if config.undistort_test:
        Cam.undistort_all()

    test_img = get_test_img()
    wraped = Cam.perspective_transform(test_img)
    bin_out = Thresh.get_bin(wraped)
    Lane.find_starting_point(bin_out)
    #Thresh.save_fig("test",test_img, wraped, bin_out)





if __name__ == '__main__':
    main()
