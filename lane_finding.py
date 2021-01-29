import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import pdb
import os
from collections.abc import Mapping


def show_img(img_dict, filename):
    if not isinstance(img_dict, Mapping):
        plt.imshow(img_dict)
        return
    elif len(img_dict) == 1:
        plt.imshow(img_dict.values[0])
        return
    else:
        col = 3
        row = 1
        values_list = list(img_dict.values())

    fig, axes = plt.subplots(row, col, figsize=(16, 8))
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    axes.ravel()

    for idx, name in enumerate(img_dict.keys()):
        img = img_dict[name]
        if name == 'gray':
            axes[idx].imshow(img, cmap = 'gray')
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(img)

    plt.savefig(filename)


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, default):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if default:
        draw_default_lines(line_img, lines)
    else:
        draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def draw_default_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


# called by hough_lines
def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    """
    # Track gradient and intercept of left and right lane
    left_slope = []
    left_intercept = []
    left_y = []

    right_slope = []
    right_intercept = []
    right_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2-y1)/(x2-x1)
            intercept = y2 - (slope*x2)

            # right lane
            if slope > 0.0 and slope < math.inf and abs(slope) > 0.3:
                right_slope.append(slope)
                right_intercept.append(intercept)
                right_y.append(y1)
                right_y.append(y2)

            # left lane
            elif slope < 0.0 and slope > -math.inf and abs(slope) > 0.3:
                left_slope.append(slope)
                left_intercept.append(intercept)
                left_y.append(y1)
                left_y.append(y2)

    y_min = min(min(left_y), min(right_y)) + 40
    y_max = img.shape[0]
    l_m = np.mean(left_slope)
    l_c = np.mean(left_intercept)
    r_m = np.mean(right_slope)
    r_c = np.mean(right_intercept)

    l_x_max = int((y_max - l_c)/l_m)
    l_x_min = int((y_min - l_c)/l_m)
    r_x_max = int((y_max - r_c)/r_m)
    r_x_min = int((y_min - r_c)/r_m)

    cv2.line(img, (l_x_max, y_max), (l_x_min, y_min), color, thickness)
    cv2.line(img, (r_x_max, y_max), (r_x_min, y_min), color, thickness)


def interpolate(lanes, img):
    # Interpolating lines
    result = weighted_img(lanes, img)
    return result


def main():
    img_dict = {}
    img = cv2.imread("test_img.jpg")
    gray = grayscale(img)
    gray_blur = gaussian_blur(gray, 3)
    edges = canny(gray_blur, low_threshold=75, high_threshold=150)
    points = np.array([[130, 600], [380, 300], [650, 300], [900, 550]], dtype=np.int32)
    ROI = region_of_interest(edges, [points])
    lines = hough_lines(ROI, 2, np.pi/180, 15, 5, 25, True)
    lanes = hough_lines(ROI, 2, np.pi/180, 15, 5, 25, False)
    res = interpolate(lanes, img)

    img_dict['img'] = img
    img_dict['gray'] = gray_blur
    img_dict['edges'] = edges
    show_img(img_dict, "edges.png")

    img_dict = {}
    img_dict['img'] = img
    img_dict['edges'] = edges
    img_dict['roi'] = ROI
    show_img(img_dict, "roi.png")

    img_dict = {}
    img_dict['img'] = img
    img_dict['lines'] = lines
    img_dict['lanes'] = lanes
    show_img(img_dict, "lines.png")

    img_dict = {}
    img_dict['img'] = img
    img_dict[ 'lanes'] = lanes
    img_dict['res'] = res
    show_img(img_dict, "result.png")



if __name__ == '__main__':
    main()
