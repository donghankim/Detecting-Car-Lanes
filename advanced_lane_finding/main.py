from config import get_args
from utils import Funcs
from calibrations import Camera
from threshold import Threshold
from find_lanes import Lanes
import numpy as np
import cv2
from tqdm import tqdm
import os, pdb

# algorihtm pipeline (driver)
def process_vid(vid_path, config, thresh, cam, lane):
    cap = cv2.VideoCapture(vid_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    save_path = config.vid_save + config.vid_name
    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MP4V'), 30, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            warped = cam.perspective_transform(frame)
            bin_out = thresh.get_bin(warped)
            lane.find_starting_point(bin_out)
            lane.sliding_window()
            lane.fit_poly_lines()
            lane.get_curvature()

            color_warp = lane.get_color_warp()
            lanes_colored = cv2.warpPerspective(color_warp, cam.perspective_inv, (color_warp.shape[1], color_warp.shape[0]))
            result = cv2.addWeighted(frame, 1, lanes_colored, 0.3, 0)

            cv2.imshow("Window", result)
            writer.write(result)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

# just plays the input video
def play_vid(vid_path):
    cap = cv2.VideoCapture(vid_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Window", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    config = get_args()
    thresh = Threshold(config)
    cam = Camera(config)
    cam.calibrate_camera()
    lane = Lanes(config)

    # Only need to run once
    if config.undistort_test:
        Cam.undistort_all()

    vid_files = [f for f in os.listdir(config.test_vid) if os.path.isfile(os.path.join(config.test_vid, f))]
    for files in tqdm(vid_files):
        config.vid_name = files
        vid_path = config.test_vid + config.vid_name
        process_vid(vid_path, config, thresh, cam, lane)
        print(f"Completed: {config.vid_name}")


if __name__ == '__main__':
    main()

