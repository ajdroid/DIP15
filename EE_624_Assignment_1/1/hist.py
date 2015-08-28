import numpy as np 
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('../flower.jpg',0)
cv2.imshow('lala',img)
plt.hist(img.ravel(),8,[0,256])
plt.show()

