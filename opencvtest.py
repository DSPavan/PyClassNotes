# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:12:46 2019

@author: pavan
"""

# import the necessary packages
import imutils
import cv2



# load the input image and show its dimensions, keeping in mind that
# images are represented as a multi-dimensional NumPy array with
# shape no. rows (height) x no. columns (width) x no. channels (depth)
image = cv2.imread("C:\\Users\\pavan\\Downloads\\open\\jp.jpg")
image
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))


#Extracting “regions of interest” (ROIs) is an important skill for image processing.

roi = image[60:160, 320:420]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

# resize the image to 200x200px, ignoring aspect ratio
resized = cv2.resize(image, (200, 200))
cv2.imshow("Fixed Resizing", resized)
cv2.waitKey(0)

##image[startY:endY, startX:endX]

# console IS NOT CLOSING


# fixed resizing and distort aspect ratio so let's resize the width
# to be 300px but compute the new height based on the aspect ratio
r = 300.0 / w
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
cv2.imshow("Aspect Ratio Resize", resized)
cv2.waitKey(0)


# manually computing the aspect ratio can be a pain so let's use the
# imutils library instead
resized = imutils.resize(image, width=300)
cv2.imshow("Imutils Resize", resized)
cv2.waitKey(0)
# display the image to our screen -- we will need to click the window
# open by OpenCV and press a key on our keyboard to continue execution
#cv2.imshow("Image", image)
#cv2.waitKey(0)

# let's rotate an image 45 degrees clockwise using OpenCV by first
# computing the image center, then constructing the rotation matrix,
# and then finally applying the affine warp
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("OpenCV Rotation", rotated)
cv2.waitKey(0)


# rotation can also be easily accomplished via imutils with less code
rotated = imutils.rotate(image, -45)
cv2.imshow("Imutils Rotation", rotated)
cv2.waitKey(0)


# OpenCV doesn't "care" if our rotated image is clipped after rotation
# so we can instead use another imutils convenience function to help
# us out
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)


# apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
# useful when reducing high frequency noise
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# draw a 2px thick red rectangle surrounding the face
output = image.copy()
cv2.rectangle(output, (320, 60), (420, 160), (0, 0, 255), 2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)


# draw a blue 20px (filled in) circle on the image centered at
# x=300,y=150
output = image.copy()
cv2.circle(output, (300, 150), 20, (255, 0, 0), -1)
cv2.imshow("Circle", output)
cv2.waitKey(0)



# draw a 5px thick red line from x=60,y=20 to x=400,y=200
output = image.copy()
cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
cv2.imshow("Line", output)
cv2.waitKey(0)


# draw green text on the image
output = image.copy()
cv2.putText(output, "OpenCV + Jurassic Park!!!", (10, 25), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Text", output)
cv2.waitKey(0)




#img : The destination image to draw upon. We’re drawing on output .
#pt1 : Our starting pixel coordinate which is the top-left. In our case, the top-left is (320, 60) .
#pt2 : The ending pixel — bottom-right. The bottom-right pixel is located at  (420, 160) .
#color : BGR tuple. To represent red, I’ve supplied (0 , 0, 255) .
#thickness : Line thickness (a negative value will make a solid rectangle). I’ve supplied a thickness of 2 .


######################
# This function should be followed by waitKey function which displays the image
# for specified milliseconds. Otherwise, it won’t display the image. For example, 
# waitKey(0) will display the window infinitely until any keypress
# (it is suitable for image display). waitKey(25) will display a frame for 25 ms,
# after which display will be automatically closed. 
# (If you put it in a loop to read videos, it will display the video frame-by-frame)

#I call image.shape  to extract the height, width, and depth.
#
#It may seem confusing that the height comes before the width, but think of it this way:
#
#We describe matrices by # of rows x # of columns
#The number of rows is our height
#And the number of columns is our width
#Therefore, the dimensions of an image represented as a NumPy array are actually represented as (height, width, depth).
#
#Depth is the number of channels — in our case this is three since we’re working with 3 color channels: Blue, Green, and Red.
#


# A 640 x 480 image has 640 columns (the width) and 480 rows (the height).
# There are 640 * 480 = 307200  pixels in an image with those dimensions.
#
#Each pixel in a grayscale image has a value representing the shade of gray

#In OpenCV color images in the RGB (Red, Green, Blue) color space have a 3-tuple associated with each pixel: (B, G, R) .

# access the RGB pixel located at x=50, y=100, keepind in mind that
# OpenCV stores images in BGR order rather than RGB
(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))


      

