import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])
cv2.imshow('org', img)
org = img.copy()
img=cv2.GaussianBlur(img, (5,5), 0)
img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

blue_lower=np.array([130,130,110],np.uint8)
blue_upper=np.array([200,200,255],np.uint8)
blue=cv2.inRange(img,blue_lower,blue_upper)

cv2.imshow('img',blue)
cv2.waitKey(0)

im2, contours, hierarchy = cv2.findContours(blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

height, width = blue.shape
min_x, min_y = width, height
max_x = max_y = 0

for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    min_x, max_x = min(x, min_x), max(x+w, max_x)
    min_y, max_y = min(y, min_y), max(y+h, max_y)
    #cv2.rectangle(org, (x,y), (x+w,y+h), (255, 0, 0), 2)

if max_x - min_x > 0 and max_y - min_y > 0:
    cv2.rectangle(org, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

#cv2.imshow('org', org)
crop_img = org[min_y:max_y, min_x:max_x]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

cv2.imwrite("crop_img.jpg", crop_img)
