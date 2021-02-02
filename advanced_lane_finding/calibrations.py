import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os, pdb


class Camera():
    def __init__(self, config):
        self.config = config
        self.mtx = None
        self.dist = None
        self.ret = False

    def calibrate_camera(self):
        image_paths = glob.glob(self.config.calibrate_dir + 'calibration*.jpg')
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        objpoints = []
        imgpoints = []

        for fname in image_paths:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if ret:
            self.ret = True
            self.mtx = mtx
            self.dist = dist

    # for debugging
    def undistort_single(self, img):
        if not self.ret:
            print("Camera has not been calibrated.")
            return
        else:
            # do this for all images in test_images folder
            undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
            plt.imshow(img)
            plt.show()

    def undistort_all(self):
        if not self.ret:
            print("Camera has not been calibrated.")
            return
        else:
            image_paths = os.listdir(self.config.test_img)
            for fname in image_paths:
                img = cv2.imread(os.path.join(self.config.test_img, fname))
                undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
                cv2.imwrite(self.config.undistorted_save + fname, undistorted)




