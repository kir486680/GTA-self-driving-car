#importing some useful packages
import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression

# in order to get stable lane, buffer N frames' slopes and intercepts
pre_l_slopes = []
pre_l_inters = []
pre_r_slopes = []
pre_r_inters = []




def selectWhiteAndYellow(img):
    """
    selec the white and yellow component in the hsv space.
    (1) set the yellow/white lower and upper bound
    (2) apply the mask to the hsv space image
    """
    lower_yellow = np.array([65, 100, 100], np.uint8)
    upper_yellow = np.array([105, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(img, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([179, 20, 255])  # range for H is 0:179
    white_mask = cv2.inRange(img, lower_white, upper_white)

    img = cv2.bitwise_and(img, img, mask=cv2.bitwise_or(yellow_mask, white_mask))
    return img

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def outlierCleaner(predictions, x, y, inlier_percent=0.9):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual y values).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (x, y, error).
    """
    threshold = 10  # if the biggest error is greater than 5 pixels, we perform outliers remove
    errs = np.fabs(y - predictions)
    max_err = max(errs)[0]
    if max_err > threshold:  # if the biggest error is greater than 5 pixels, we remove the outliers
        k = math.ceil(errs.size * inlier_percent)
        survived_idx = np.argsort(errs, axis=0)[:k]  # find the number of k min errs, and return their index
        if survived_idx.size > 0:
            x = np.take(x, survived_idx)
            y = np.take(y, survived_idx)

    return (x, y)

def findTwoPoints(slope, inter, side, height):
    """
    In order to get two stable lanes,
    (1) average the slope and itercept values in the buffers,
    (2) fix the y coordinate of the top points
    (3) then use the averaged slope and inter to locate the two end points of a line

    :param slope: current slope from the regressor for current frame
    :param inter: current intercept from the regressor for current frame
    :param side:  'l': left, 'r': right lane
    :param height: hight of the image
    :return tow points locations, which are the two ends of a lane
    """
    number_buffer_frames = 3
    scale_y = 0.65
    top_y = int(float(height) * scale_y)  # fix the y coordinates of the top point, so that the line is more stable


    if side == 'l':
        if len(pre_l_slopes) == number_buffer_frames:  # reach the max
            pre_l_slopes.pop(0)  # remove the oldest frame
            pre_l_inters.pop(0)

        pre_l_slopes.append(slope)
        pre_l_inters.append(inter)
        slope = sum(pre_l_slopes) / len(pre_l_slopes)
        inter = sum(pre_l_inters) / len(pre_l_inters)

        p1_y = height-1
        p1_x = int((float(p1_y)-inter)/slope)
        p2_y = top_y
        p2_x = int((float(p2_y)-inter)/slope)
    else:  # 'r'
        if len(pre_r_slopes) == number_buffer_frames:  # reach the max
            pre_r_slopes.pop(0)  # remove the oldest frame
            pre_r_inters.pop(0)

        pre_r_slopes.append(slope)
        pre_r_inters.append(inter)
        slope = sum(pre_r_slopes) / len(pre_r_slopes)
        inter = sum(pre_r_inters) / len(pre_r_inters)

        p1_y = top_y
        p1_x = int((float(p1_y)-inter)/slope)
        p2_y = height-1
        p2_x = int((float(p2_y)-inter)/slope)

    return (p1_x, p1_y, p2_x, p2_y)


def regress_a_lane(img, x, y, color=[255, 0, 0], thickness=10):
    """ regress a line from x, y and add it to img
    (1) use a linear regressor to fit the data (x,y)
    (2) remove outlier, and then fit the cleaned data again to get slope and intercept
    (3) find the two ends of the desired line by using slope and intercept

    :param img: input image
    :param x: x coordinate
    :param y: y coordinate
    :param color: line color
    :param thickness: thickness of the line
    """
    reg = LinearRegression()
    reg.fit(x, y)

    # identify and remove outliers
    cleaned_data = []
    try:
        predictions = reg.predict(x)
        cleaned_data = outlierCleaner(predictions, x, y)
    except NameError:
        print("err in regression prediction")

    if len(cleaned_data) > 0:
        x, y = cleaned_data
        # refit cleaned data!
        try:
            reg.fit(x, y)
        except NameError:
            print("err in reg.fit for cleaned data")
    else:
        print("outlierCleaner() is returning an empty list, no refitting to be done")

    height = img.shape[0]
    slope = reg.coef_
    inter = reg.intercept_

    # find the two end points of the line by using slope and iter, and then visulize the line
    if reg.coef_ < 0:  # left lane
        p1_x, p1_y, p2_x, p2_y = findTwoPoints(slope, inter, 'l', height)
        cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), color, thickness)
    else:  # right lane
        p1_x, p1_y, p2_x, p2_y = findTwoPoints(slope, inter, 'r', height)
        cv2.line(img, (p1_x, p1_y), (p2_x, p2_y), color, thickness)


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    (1) remove some horizontal lines with given threshold_angle
    (2) seperate the points belongs to the left and right lane by computing line slope
    (3) handle the left/right lane points to a linear regressor to fit the line, with additional
        steps to remove the outliers for getting a better fit.
    """
    threshold_angle = 25  # if the line angle is between -25 to 25 degrees, lines are discarded
    threshold_slope = math.tan(threshold_angle / 180 * math.pi)
    left_lane_x = []
    left_lane_y = []
    right_lane_x = []
    right_lane_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 != x1:
                slope = float(y2 - y1) / float(x2 - x1)
                if abs(slope) < threshold_slope:  # remove the horizontal lines
                    continue
                elif slope < 0:  # left lane, note the origin is on the left-up corner of the image
                    left_lane_x.append([x1])
                    left_lane_y.append([y1])
                    left_lane_x.append([x2])
                    left_lane_y.append([y2])
                else:
                    right_lane_x.append([x1])
                    right_lane_y.append([y1])
                    right_lane_x.append([x2])
                    right_lane_y.append([y2])

    # get left and right solid lanes with regression
    if len(left_lane_x) > 0:  # if there are no enough points at the current frame
        regress_a_lane(img, left_lane_x, left_lane_y)
    if len(right_lane_x) > 0:
        regress_a_lane(img, right_lane_x, right_lane_y)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
