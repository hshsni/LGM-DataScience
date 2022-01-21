import cv2
import numpy as np

image = cv2.imread("download.jpg")
cv2.imshow("Dog", image)
cv2.waitKey(0)

#converting to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscaled Dog", gray)
cv2.waitKey(0)

#invert the grayscale to enhance details
invertedgray = 255-gray
cv2.imshow("Inverted Dog", invertedgray)
cv2.waitKey(0)

#blurring the image so as to sharpen the edges of the image, then smoothen them 
blurimg = cv2.GaussianBlur(invertedgray, (21, 21), 0)

invertblur = 255 - blurimg
pencil_sketch = cv2.divide(gray, invertblur, scale=256.0)
cv2.imshow("Sketch", pencil_sketch)
cv2.waitKey(0)