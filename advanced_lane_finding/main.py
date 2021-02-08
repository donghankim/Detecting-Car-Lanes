from config import get_args
from utils import Funcs
from calibrations import Camera
from threshold import Threshold
from find_lanes import Lanes
import numpy as np
import cv2
import os, pdb


def main():
    config = get_args()
    utils = Funcs(config)
    thresh = Threshold(config)
    cam = Camera(config)
    cam.calibrate_camera()
    lane = Lanes(config)

    # Only need to run once
    if config.undistort_test:
        Cam.undistort_all()

    # Pipeline
    test_img = utils.get_test_img()
    warped = cam.perspective_transform(test_img)
    bin_out = thresh.get_bin(warped)
    lane.find_starting_point(bin_out)
    lane.sliding_window()
    lane.fit_poly_lines()
    lane.get_curvature()

    color_warp = lane.get_color_warp()
    lanes_colored = cv2.warpPerspective(color_warp, cam.perspective_inv, (color_warp.shape[1], color_warp.shape[0]))
    result = cv2.addWeighted(test_img, 1, lanes_colored, 0.3, 0)




if __name__ == '__main__':
    main()

