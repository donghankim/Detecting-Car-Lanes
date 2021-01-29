# Detecting-Lane-Lines
This is a small project for detecting lane lines using computer vision.

## Overview
This project explores a simple approach to identifying car lanes from a dashcam video.

![Result GIF](media/output_gif.gif)

While most use sophisticated deep learning algorithms to identify car lanes, as part of the Udacity self-driving car engineer nano degree program, I was able to use simple computer vision alrogithms to detect car lanes from a dashcam video. This process was achieved in 4 steps: <strong>edge detection, ROI bounding box, Hough transform and line interpolation.</strong> Each step is explained in detail below.

## Installation
In order to run my code, you will first have to install all of the dependancies:
```python3
pip install -r requirements.txt
```
I highly recomend using a virtual enviornment.

## Edge Detection
There are many different algorithms in computer vision for detecting edges. For this mini project, I used the Canny edge detection algorithm to first identify edges in a video frame. Essentially this algorithm works by finding the derivative of pixel intensity, and identifies the pixels that have a sharp change in intensity. As a result, to use this algorihtm, I first converted the image into a gray scale image, so that all pixel values lie in the range of 0 - 255 (255 represents white, and 0 represents black). Here is an example of what the output from the Canny edge detection algorithm looks like:

<img src="media/edges.png" alt="Canny edge result." style="text-align:center" />

left image: original image.
middle image: grayscale image (after Gaussian blur).
right image: Canny edge detection output.

## ROI (Bounding Box)
As you can see from the Canny edge output, we have a lot of edges that we are not interested in. The goal of this project is to find the car lanes, so we are not interested in any edges that do not represent the car lanes. Therefore, we can apply a mask to capture the region of the image that we are interested in. This makes sense as the location of the car lanes will not change from frame to frame. The vertecies corresponding to the bounding box are hard coded. Here is what the ouput looks like after the bounding box has been applied:

<img src = "media/roi.png" alt = "ROI output." style="text-align:center" />

## Hough Transfrom
We now need to convert these edges to lines, after all, we want to detect lines not edges. The Hough transform is perfect for converting edges into lines. Without getting into too much detail, the Hough transform basically tranforms all edge coordinates in cartesian space to polar coordinates. If you are not familiar with Hough transform, I recommend you check this medium article [here.](https://towardsdatascience.com/lines-detection-with-hough-transform-84020b3b1549) Essentially we have used the output from the Canny edge detection algorithm into lines using the Hough transform.

<img src = "media/lines.png" alt = "Hough transform output." style = "test-align:center" />

## Interpolation
The final step is to identify two lines (one for the left lane and another for the right lane) from the Hough transfrom output. In my opinion, this is the part that matters the most, since until now we've just been using "out-of-the-box" algorithms.


