import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from filter import gaussian_blur , selectWhiteAndYellow, grayscale, canny, region_of_interest, outlierCleaner, findTwoPoints, regress_a_lane, draw_lines, hough_lines, weighted_img
def drawLanesPipeline(image):
    """ Process one image, detect two lanes and highlight them with solid color lines
    (1) apply the gaussian blur
    (2) convert bgr to hsv and segment while and yellow, because it is easier in HSV space than RGB
    (3) Canny edge detection
    (4) apply the designed mask to the image to obtian the region of interest
    (5) apply hough transform to get lines
    (6) augmented the lanes on the original image

    :param image: input image
    :return: an augmented image with two lane highlighted
    """

    #---------set parameters----------#
    # gaussian_blur para
    kernel_size = 5
    # canny edege detection para
    low_threshold = 50
    high_threshold = 150
    # region_of_interest para
    height, width, _ = image.shape
    scale_w = 7 / 16
    scale_h = 11 / 18
    left_bottom = [0, height - 1]
    right_bottom = [width - 1, height - 1]
    left_up = [scale_w * width, scale_h * height]
    right_up = [(1 - scale_w) * width, scale_h * height]
    # hough_line para
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15  # minimum number of pixels making up a line
    max_line_gap = 40  # maximum gap in pixels between connectable line segments
    #---------------------------------#

    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussian_blur(image, kernel_size)

    # convert image from bgr to hsv
    hsv_img = cv2.cvtColor(blur_gray, cv2.COLOR_BGR2HSV)

    # filter out the white and yellow segments (lanes are either white or yellow in this case)
    filtered_hsv = selectWhiteAndYellow(hsv_img)

    # Apply Canny edge detection
    edges = canny(filtered_hsv, low_threshold, high_threshold)

    # create a masked edges image
    vertices = np.array([[left_bottom, left_up, right_up, right_bottom]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the edge image
    output = weighted_img(line_image, image)

    return output

def detect_lane(img):

    # draw two lanes on the image
    image_augmented = drawLanesPipeline(img)


    return image_augmented
