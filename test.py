import numpy as np 
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from matplotlib import pyplot as plt

# def histplot(img, bins, intensityRange = [0,256]):
# 	flatimg = img.ravel()
# 	count = np.zeros(8, dtype=int)
# 	idx = 0
# 	binsize = (intensityRange[1]-intensityRange[0])/bins
# 	for pixel in flatimg:
# 		idx = np.floor(pixel/binsize)
# 		pixel=idx
# 		count[idx]=count[idx]+1
# 	print count
# 	print flatimg[5]
# 	cv2.imshow('New image', np.reshape(flatimg, img.shape))




# img = cv2.imread('EE_624_Assignment_1/flower.jpg',0)
# flatimg = img.ravel()
# cv2.imshow('image',img)



# plt.hist(flatimg,256,[0,256])
# plt.show()

img = cv2.imread('EE_624_Assignment_1/flower.jpg',0)