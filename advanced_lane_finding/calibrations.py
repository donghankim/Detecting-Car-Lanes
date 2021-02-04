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

    def perspective_transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width = img.shape[1]
        height = img.shape[0]
        src_pts = np.array([[0, height], [width // 2 - 76, height * 0.625], [width // 2 + 76, height * 0.625], [width, height]], np.float32)
        dst_pts = np.array([[0, height], [100, 0], [width-100, 0], [width, height]], np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        wraped = cv2.warpPerspective(img, M, (width, height), flags = cv2.INTER_LINEAR)
        return wraped


"""
FOR GITHUB DOCUMENTATION

        test_img = cv2.imread("../media/undistorted_images/test_git.jpg")
        line_img = cv2.imread("../media/undistorted_images/test_git.jpg")
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
        width = test_img.shape[1]
        height = test_img.shape[0]

        src_pts = np.array([[0, height], [width // 2 - 76, height * 0.625], [width // 2 + 76, height * 0.625], [width, height]], np.float32)
        dst_pts = np.array([[0, height], [100, 0], [width-100, 0], [width, height]], np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        wraper = cv2.warpPerspective(test_img, M, (width, height), flags = cv2.INTER_LINEAR)
        cv2.polylines(line_img, [src_pts.astype(np.int32)], True, (0, 255, 255), 3)

        fig, axes = plt.subplots(1,3, figsize = (16,8))
        fig.subplots_adjust(hspace=0.1, wspace=0.2)
        axes.ravel()

        axes[0].imshow(test_img)
        axes[0].set_title("Original Image")
        axes[1].imshow(line_img)
        axes[1].set_title("ROI")
        axes[2].imshow(wraper)
        axes[2].set_title("Perspective Transform")

        plt.savefig(os.path.join(self.config.github_save, "testtt.png"))

"""
