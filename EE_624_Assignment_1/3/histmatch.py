import numpy as np 
np.set_printoptions(suppress=True, precision=3) 

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from matplotlib import pyplot as plt
import math

givimg = cv2.imread('../givenhist.jpg',0)
spimg = cv2.imread('../sphist.jpg',0)
#Calculate both histograms
#Calculate equalization transforms (CDF)
#Inverse mapping for match	

cv2.imshow('Given image', givimg)
cv2.imshow('Specified hist image', spimg)

cv2.waitKey(0)