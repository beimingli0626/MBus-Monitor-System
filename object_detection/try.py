import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("/home/pi/test_images/frame30.jpg")
cv2.imshow('i', image)

cv2.waitKey(0)

