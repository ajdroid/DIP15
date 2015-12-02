import numpy as np 
# np.set_printoptions(suppress=True, precision=3) 
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import math
import numpy.linalg as LA
import itertools
givimg = cv2.imread('../givenhist.jpg',0)
spimg = cv2.imread('../sphist.jpg',0)


def harrisCorners( src, winSize, blockSize, alpha = 0.04): 
	# Preprocess
	src = cv2.GaussianBlur(src,(5,5),0)
	pad = winSize + blockSize
	size = 2*winSize + 1
	# src_padded = cv2.copyMakeBorder(src,pad,pad,pad,pad,cv2.BORDER_REPLICATE) # DO I need this?
	windower = np.ones((size,size))
	norm = np.sum(windower)
	sc = float(2**(winSize-1))*blockSize
	sc = sc*255
	gradX = cv2.Sobel(src,cv2.CV_64F,1,0,ksize=blockSize)
	gradY = cv2.Sobel(src,cv2.CV_64F,0,1,ksize=blockSize)
	print gradX[gradX.nonzero()]
	print gradY[gradY.nonzero()]
	cv2.imshow('gradX', gradX)
	cv2.imshow('gradY', gradY)
	# prC = cv2.preCornerDetect(src, blockSize)
	# print prC.shape
	# return prC
	eigs = cv2.cornerEigenValsAndVecs(src, winSize, blockSize)
	eigs = eigs[:,:,0:2]
	R = np.zeros(src.shape)
	structTensor = np.zeros((2,2), dtype=np.float64)

	Lambda1 = eigs[:,:,0]
	Lambda2 = eigs[:,:,1]

	R = (Lambda1*Lambda2) - alpha*(Lambda1+Lambda2)**2
				
		


	# perform non-maximal suppression on R
	return R

def nonmaxSupress( resp, winLen, G ):
	dst = np.zeros(resp.shape, dtype=bool)
	size = 2*winLen
	src_padded = cv2.copyMakeBorder(resp,winLen,winLen,winLen,winLen,cv2.BORDER_REPLICATE)
	for i,j in itertools.product(xrange(resp.shape[0]), xrange(resp.shape[1])):
		if 	G[i,j]:
			if resp[i,j] == np.max(src_padded[i:i+size, j:j+size]):
				dst[i,j] = True
			else:
				dst[i,j] = False

	return dst


# Main

img = cv2.imread('../IITG.jpg', 1)
imgBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
candidates = harrisCorners(imgBW, 2, 3)
dsp = cv2.dilate(candidates, None)


G = dsp>0.0005*dsp.max()
resp = np.full(imgBW.shape, dsp.min())
resp[G] = dsp[G]
print resp.min()
print resp.max()
print resp.shape

indexer = nonmaxSupress(resp, 1, G)
img[indexer]=[0,0,255]

print dsp.min()
print dsp.max()

cv2.imshow('Harris Corners (in red)', img)
cv2.imwrite('HC.jpg', img)
cv2.waitKey(0)