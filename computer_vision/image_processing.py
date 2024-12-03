#Install OPENCV FIRST

import cv2
import matplotlib.pyplot as plt
import numpy as np

#read image
# image_path = "./test_img/free-nature-images.jpg"
# image = cv2.imread(image_path)

# im= image.shape
# print(f"size of image is : {im}")

# print(f"hieght:{im[0]}, width:{im[1]}")


#blur image
# blur = cv2.GaussianBlur(image,(15,15),cv2.BORDER_DEFAULT)
# cv2.imshow("blurred image",blur)



# cropping image
# cropped_image = image[80:280, 150:330]
 
# # Display cropped image
# cv2.imshow("cropped", cropped_image)



#changing color
# gray_img= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# cv2.imshow('grayimage',gray_img)

#resizing image
# half = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
# cv2.imshow('resizedimage',half)


#edge cascade
# canny= cv2.Canny(image,125,175)
# cv2.imshow("canny",canny)

#resizing window
# cv2.namedWindow("Resized_Window",cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Resized_Window",700,700)
# cv2.imshow("Resized_Window",image)



# cv2.imshow("nature",image)
# cv2.waitKey(0)

# blank = np.zeros((400, 400), dtype='uint8')

# rectangle = cv2.rectangle(blank.copy(), (50, 50), (350, 350), 255, -1)
# cv2.imshow("rectangle", rectangle)

# circle = cv2.circle(blank.copy(), (200, 200), 200, 255, -1)
# cv2.imshow("circle", circle)

# #bitwise AND --> intersecting region
# bitwise_and = cv2.bitwise_and(rectangle, circle)
# cv2.imshow("bitwise_and", bitwise_and)

# #bitwise OR -- both intersecting and non intersecting region
# bitwise_or = cv2.bitwise_or(rectangle, circle)
# cv2.imshow("bitwise_or", bitwise_or)

# #bitwise XOR --> non intersecting region
# bitwise_xor = cv2.bitwise_xor(rectangle, circle)
# cv2.imshow('bitwise_xor', bitwise_xor)

# #bitwise NOT --> inversing color
# bitwise_not = cv2.bitwise_not(circle)
# cv2.imshow("bitwise_not", bitwise_not)

# cv2.waitKey(0)

img = cv2.imread('./test_img/free-nature-images.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshold, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("threshold image", thresh)

threshold, thresh_inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("inversed thresh image", thresh_inv)

#adeptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 5)
cv2.imshow("adaptive threshold img", adaptive_thresh)


adaptive_thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 5)
cv2.imshow("adaptive threshold img inverse", adaptive_thresh_inv)
cv2.waitKey(0)