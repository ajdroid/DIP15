import numpy as np 
np.set_printoptions(suppress=True, precision=3) 
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from matplotlib import pyplot as plt
import math

filt1 = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
filt2 = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
print filt2
winimg = cv2.imread('../window.jpg',0)
filt1win = cv2.filter2D(winimg,-1, filt1)
filt2win = cv2.filter2D(winimg,-1, filt2)
cv2.imshow('Window image', winimg)
cv2.imshow('Window filt1 image', filt1win)
cv2.imshow('Window filt2 image', filt2win)	
cv2.imshow('Thresh image', filt1win+filt2win)
for i in [200]:
	th1, threshimg = cv2.threshold((filt1win+filt2win), i, 255, cv2.THRESH_BINARY)
	cv2.imshow('Thresh1 image'+str(i), threshimg)	
cv2.waitKey(0)