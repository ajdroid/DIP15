# sizes of 3/5/7 work best
import numpy as np 
np.set_printoptions(suppress=True, precision=3) 
import itertools as it
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from matplotlib import pyplot as plt
import math

def medianFilter(src, size):
	
	src_padded = cv2.copyMakeBorder(src,int(size/2),int(size/2),int(size/2),int(size/2),cv2.BORDER_REPLICATE)
	for i,j in it.product(xrange(src.shape[0]), xrange(src.shape[1])):
		src[i][j] = np.median(src_padded[i:i+size, j:j+size]) 
					
	return src


def bilateralFilter(src, sigmad, sigmar, size = 5):
	src_padded = cv2.copyMakeBorder(src,int(size/2),int(size/2),int(size/2),int(size/2),cv2.BORDER_REPLICATE)
	# Added int(size/2) to index for borders
	# Compute domain weights
	Wd = np.array([[	m*m + n*n		\
		for m in range(-int(size/2),int(size/2)+1)] \
		for n in range(-int(size/2),int(size/2)+1)], dtype=float)	
	Wd = np.square(Wd)
	Wd = Wd/(sigmad)
	Wd = np.exp(-Wd)
	for i,j in it.product(xrange(src.shape[0]), xrange(src.shape[1])):
			# Compute range weights
			Wr = np.array(src_padded[i:i+size, j:j+size], dtype=float)
			Wr = np.square(Wr-src_padded[i,j])/(sigmar)
			Wr = np.exp(-Wr)
			try:
				src[i][j] = np.sum(Wd*Wr*src_padded[i:i+size, j:j+size])/np.sum(Wd*Wr)
			except:
				if np.sum(Wd*Wr)==0:
					print Wd
					print Wr
					
	print Wr
				

	return src


# if __name__ == "main":	
spnoisy = cv2.imread('../spnoisy.jpg',0)
unifnoisy = cv2.imread('../unifnoisy.jpg',0)
spunifnoisy = cv2.imread('../spunifnoisy.jpg',0)

for i in [10]:
	dst = bilateralFilter(unifnoisy,i*i,4*i*i)
# 	cv2.imshow('unifnoise cleared ' + str(i), dst)
	cv2.imshow('unifnoise cleared cvb' + str(i)+'.jpg', cv2.adaptiveBilateralFilter(unifnoisy,(5,5),i))
	cv2.imshow('spunifnoise cleared cvb' + str(i)+'.jpg', cv2.adaptiveBilateralFilter(spunifnoisy,(5,5),i))
	cv2.imshow('spnoise cleared cvb' + str(i)+'.jpg', cv2.adaptiveBilateralFilter(spnoisy,(5,5),i))
# cv2.imshow('spnoisy', spnoisy)
for i in [3,5]:
	dst = medianFilter(spnoisy,i)
	# cv2.imshow('spnoise cleared ' + str(i), dst)
	cv2.imshow('spnoise cleared cv m' + str(i)+'.jpg', cv2.medianBlur(spnoisy,i))
	cv2.imshow('spunifnoise cleared cv m' + str(i)+'.jpg', cv2.medianBlur(spunifnoisy,i))
	cv2.imshow('uninoise cleared cv m' + str(i)+'.jpg', cv2.medianBlur(unifnoisy,i))

cv2.waitKey(0)



# dst = cv2.filter2D()
