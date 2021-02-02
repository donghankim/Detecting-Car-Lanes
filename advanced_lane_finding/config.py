
import argparse


def get_args():
    argp = argparse.ArgumentParser(description='adversarial examples', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # GENERAL
    argp.add_argument('--undistort_test', action = 'store_true')

    # PATH
    argp.add_argument('--root_dir', type=str, default='../media/')
    argp.add_argument('--calibrate_dir', type=str, default='../media/camera_cal/')
    argp.add_argument('--test_img', type=str, default='../media/test_images/')
    argp.add_argument('--test_vid', type=str, default='../media/test_video/')
    argp.add_argument('--undistorted_save', type=str, default='../media/undistorted_images/')
    argp.add_argument('--github_save', type=str, default='../media/github/')

    # PARAMETERS
    argp.add_argument('--ksize', type = int, default = 3)
    argp.add_argument('--thresh_min', type = int, default = 20)
    argp.add_argument('--thresh_max', type = int, default = 100)
    argp.add_argument('--angle_min', type = float, default = 0.9)
    argp.add_argument('--angle_max', type = float, default = 1.0)


    return argp.parse_args()
