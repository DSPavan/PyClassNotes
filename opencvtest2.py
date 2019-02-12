# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 22:24:09 2019

@author: pavan
"""

#count the number of Tetris blocks in the following image:

#Learning how to convert images to grayscale with OpenCV
#Performing edge detection
#Thresholding a grayscale image
#Finding, counting, and drawing contours
#Conducting erosion and dilation
#Masking an image

# import the necessary packages
##import argparse
import imutils
import cv2
import sys
image = cv2.imread("C:\\Users\\pavan\\Downloads\\tetris.png")

(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))
#ap.parse_args()

# load the input image (whose path was supplied via command line
# argument) and display the image to our screen
#image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

# convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
cv2.waitKey(0)


#Edge
# applying edge detection we can find the outlines of objects in
# images
edged = cv2.Canny(gray, 30, 150)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

# Using the popular Canny algorithm (developed by John F. Canny in 1986), we can find the edges in the image.
#We provide three parameters to the cv2.Canny  function:
#
#img : The gray  image.
#minVal : A minimum threshold, in our case 30 .
#maxVal : The maximum threshold which is 150  in our example.
#aperture_size : The Sobel kernel size. By default this value is 3  and hence is not shown 



# threshold the image by setting all pixel values less than 225
# to 255 (white; foreground) and all pixel values >= 225 to 255
# (black; background), thereby segmenting the image
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)

# find contours (i.e., outlines) of the foreground objects in the
# thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

# loop over the contours
for c in cnts:
	# draw each contour on the output image with a 3px thick purple
	# outline, then display the output contours one at a time
	cv2.drawContours(output, [c], -1, (240, 0, 159), 3)
	cv2.imshow("Contours", output)
	cv2.waitKey(0)
    
    
# draw the total number of contours found in purple
text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(240, 0, 159), 2)
cv2.imshow("Contours", output)
cv2.waitKey(0)

#Erosions and dilations
#Erosions and dilations are typically used to reduce noise in binary images (a side effect of thresholding).
#
#To reduce the size of foreground objects we can erode away pixels given a number of iterations

# we apply erosions to reduce the size of foreground objects
mask = thresh.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

# similarly, dilations can increase the size of the ground objects
mask = thresh.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

# a typical operation we may want to apply is to take our mask and
# apply a bitwise AND to our input image, keeping only the masked
# regions
mask = thresh.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Output", output)
cv2.waitKey(0)











