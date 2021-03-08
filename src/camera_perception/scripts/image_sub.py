#!/usr/bin/env python
# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import matplotlib.pyplot as plt
import numpy as np

low_H = 0
low_S = 86
low_V = 0
high_H = 255
high_S = 155
high_V = 255

def skeletonize(img):
    # Threshold the image
    ret,img = cv2.threshold(img, 127, 255, 0)

    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
            break
    return skel

def gradient(img):
    # grayscale the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(img, 15, 100, 100)
    canny_image = cv2.Canny(bilateral,15,40)
    kernel = np.ones((5, 5))
    dilate_image = cv2.dilate(canny_image, kernel, iterations=1)
    erode_image = cv2.erode(dilate_image, kernel, iterations=1)
    skel = skeletonize(erode_image)
    # # gaussian blur of image with a 5x5 kernel
    # gauss = cv2.GaussianBlur(gray,(3,3),0)
    # v = np.median(gray)
    # sigma = 0.75
    # lower_thresh = int(max(0, (1.0 - sigma) * v))
    # upper_thresh = int(min(255, (1.0 + sigma) * v))
    # Return the canny of the image
    return skel

def region_of_interest(img):
    # Height of image (number of rows)
    height = img.shape[0]
    # Width of the image (number of columns)
    width = img.shape[1]
    # Create an array of polygons to use for the masking of the canny image
    polygons = np.array([
    [(100,height), (100,100), (700,100), (700,height)]
    ])
    # Create the mask image's background (black color)
    mask_bg = np.zeros_like(img)
    # Create the mask image (image with black background an white region of interest)
    mask = cv2.fillPoly(mask_bg, polygons, 255)
    # Isolate the area of interest using the bitwise operator of the mask and canny image
    masked_image = cv2.bitwise_and(img,cv2.fillPoly(mask_bg, polygons, 255))
    # Return the updated image
    return masked_image

def make_coordinates(img, line_parameters):
    # Extract the average slope and intercept of the line
    slope, intercept = line_parameters
    # Coordinate y(1) of the calculated line
    y1 = img.shape[0]
    # Coordinate y(2) of the calculated line
    y2 = int(y1*0.5)
    # Coordinate x(1) of the calculated line
    x1 = int((y1-intercept)/slope)
    # Coordinate x(2) of the calculated line
    x2 = int((y2-intercept)/slope)
    # Return the coordinates of the average line
    return np.array([x1,y1,x2,y2])

def average_slope_intercep(img,lines):
    # Create an empty list containing the coordinates of the detected line
    left_fit = []
    right_fit = []
    # Loop through all the detected lines
    for line in lines:
        # Store the coordinates of the detected lines into an 1D array of 4 elements
        x1,y1,x2,y2 = line.reshape(4)
        # Create a line y = mx+b based on the coordinates
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        # Extract the slope m
        slope = parameters[0]
        # Extract the intercept b
        intercept = parameters[1]
        # Check slope of line
        if slope < 0:
            # Add elements on the list
            left_fit.append((x1,y1,x2,y2))
        else:
            # Add elements on the list
            right_fit.append((x1,y1,x2,y2))
    # Calculate the average of the line fit parameters list
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    # Extract the coordinates of the calculated line
    left_line = make_coordinates(img,left_fit_average)
    right_line = make_coordinates(img,right_fit_average)
    return np.array([left_line, right_line])


def display_lines(img,lines):
    # Create a mask image that will have the drawn lines
    line_image = np.zeros_like(img)
    # If no lines were detected
    if lines is not None:
        # Loop through all the lines
	    for line in lines:
            # Store the coordinates of the first and last point of the lines into 1D arrays
		    x1, y1, x2, y2 = line.reshape(4)
            # Draw the lines on the image with blue color and thicknes of 10
		    cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    # Return the mask image with the drawn lines
    return line_image

def find_lane_pixels(image, original):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((image, image, image)) * 255
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 24
    margin = 15
    minpix = 10

    window_height = np.int(image.shape[0] // nwindows)

    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []
    left_center = []
    right_center = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 4)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 4)

        # Find the center point of the windows
        win_x_center_left = (win_xleft_low + win_xleft_high)/2
        win_y_center_left = (win_y_low + win_y_high)/2
        win_x_center_right = (win_xright_low + win_xright_high)/2
        win_y_center_right = (win_y_low + win_y_high)/2

        # Draw the center point of the windows
        cv2.circle(out_img,(win_x_center_left,win_y_center_left),radius = 0,color = (0,0,255),thickness = 5)
        cv2.circle(out_img,(win_x_center_right,win_y_center_right),radius = 0,color = (0,0,255),thickness = 5)
        cv2.circle(original,(win_x_center_left,win_y_center_left),radius = 0,color = (0,0,255),thickness = 25)
        cv2.circle(original,(win_x_center_right,win_y_center_right),radius = 0,color = (0,0,255),thickness = 25)

        # Store the center points in lists
        left_center.append((win_x_center_left,win_y_center_left))
        right_center.append((win_x_center_right,win_y_center_right))

        # Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return left_center, right_center, out_img
    # return leftx, lefty, rightx, righty, out_img

# Fit a poly to perform a directed search in well known areas
def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each with np.polyfit()
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty

def search_around_poly(image):
    margin = 100

    # Grab activated pixels
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx, lefty, rightx, righty, out_img = find_lane_pixels(image)
    if ((len(leftx) == 0) or (len(rightx) == 0) or (len(righty) == 0) or (len(lefty) == 0)):
        out_img = np.dstack((image, image, image)) * 255
        left_curverad = 0
        right_curverad = 0
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                             left_fit[1] * nonzeroy + left_fit[
                                                                                 2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                               right_fit[1] * nonzeroy + right_fit[
                                                                                   2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = fit_poly(image.shape, leftx, lefty, rightx, righty)

        ym_per_pix = 1 / 800  # meters per pixel in y dimension
        xm_per_pix = 0.5 / 800  # meters per pixel in x dimension
        print('ploty',ploty)
        print('left_fitx',left_fitx)
        print('right_fitx',right_fitx)

        # Calculate the curvature
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 4)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((image, image, image)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate and draw a poly to illustrate the lane area
        left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        points = np.hstack((left, right))
        out_img = cv2.fillPoly(out_img, np.int_(points), (0, 200, 255))

    return out_img#, left_curverad, right_curverad

def all_equal(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)

def get_path(image,x_left,y_left,x_right,y_right):
    x_path = []
    y_path = []
    for i in range(0,len(x_left)):
        x_path.append((x_left[i] + x_right[i])/2)
        y_path.append((y_left[i] + y_right[i])/2)
        cv2.circle(image,(x_path[i],y_path[i]),radius = 0,color = (0,255,0),thickness = 25)
    if((all_equal(x_left) and x_left[0]<=image.shape[0]//2) ):
        print('We are not aligned, towards the left. Heading: turn right')
    elif((all_equal(x_right) and x_right[0]>=image.shape[0]//2)):
        print('We are not aligned, towards the right. Heading: turn left')
    else:
        print('Heading is good. Heading: straight')

    # heading = 
    return #path, heading

def image_callback(msg):
    # print("Received an image!")
    # Instantiate CvBridge
    bridge = CvBridge()
    try:
        # Convert your ROS Image message to OpenCV2
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
	    # Copy of the original frame
	    frame_copy = np.copy(frame)
	    frame_HSV = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
	    frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
	    # out_img = search_around_poly(frame_threshold)
	    # leftx, lefty, rightx, righty, out_img = find_lane_pixels(frame_threshold)
	    left_center, right_center, out_img = find_lane_pixels(frame_threshold, frame_copy)
	    x_left = []
	    y_left = []
	    x_right = []
	    y_right = []
	    for i in range(0,len(left_center)):
	        a = left_center[i]
	        b = right_center[i]
	        x_left.append(a[0])
	        y_left.append(a[1])
	        x_right.append(b[0])
	        y_right.append(b[1])
	    get_path(frame_copy,x_left,y_left,x_right,y_right)
        # heading_angle, path = get_path(frame_copy,x_left,y_left,x_right,y_right)
	    # z_left = np.polyfit(x_left,y_left,9)
	    # f = np.poly1d(z)
	    # lspace = np.linspace(0,800,100)
	    # draw_x = lspace
	    # draw_y = np.polyval(z,draw_x)
	    # draw_points = (np.asarray([draw_x,draw_y]).T).astype(np.int32)
	    # cv2.polylines(frame_copy,[draw_points],False,(0,255,0))
        # # Canny of image
	    # canny_frame = gradient(frame_copy)
	    # # Apply mask in region of interest
	    # cropped_image = region_of_interest(canny_frame)
	    # # Apply Hough Transform on the region of interest
	    # lines = cv2.HoughLinesP(frame_threshold,10,np.pi/180,1,np.array([]),minLineLength=10,maxLineGap=0)
        # # Calculate the average slope of the detected lines
	    # averaged_lines = average_slope_intercep(frame_copy,lines)
        # # Create a mask image with the drawn lines
	    # line_image = display_lines(frame_copy,lines)
        # # Plot lines on the camera feed frame
	    # combo_image = cv2.addWeighted(frame_copy,0.8,line_image,1,1)
	    # Show manipulated image feed
	    cv2.imshow("Original",frame_copy)
	    # cv2.imshow("Result feed",out_img)
	    # cv2.imshow("Preprocessed",frame_threshold)
	    cv2.waitKey(1)
	    # plt.imshow(canny_frame)
	    # plt.show()

def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "rover/camera1/image_raw"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
