import numpy as np 
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from matplotlib import pyplot as plt
import math

def medianfilt(src, size):
	
	src_padded = cv2.copyMakeBorder(src,int(size/2),int(size/2),int(size/2),int(size/2),cv2.BORDER_REPLICATE)
	for i in range(src.shape[0]):
		for j in range(src.shape[1]):
				src[i][j] = np.median([[src_padded[int(size/2)+i+m][int(size/2)+j+n] \
				for m in range(-int(size/2),int(size/2)+1)] \
				for n in range(-int(size/2),int(size/2)+1)])
	print range(-int(size/2),int(size/2)+1)
	print src.size			
	return src


# if __name__ == "main":	
spnoisy = cv2.imread('../spnoisy.jpg',0)
# unifnoisy = cv2.imread('../unifnoisy.jpg',0)
# spunifnoisy = cv2.imread('../spunifnoisy.jpg',0)
# cv2.imshow('spnoisy', spnoisy)
for i in [3,5]:
	dst = medianfilt(spnoisy,i)
	cv2.imshow('spnoise cleared ' + str(i), dst)
	# cv2.imshow('spnoise cleared cv ' + str(i), cv2.medianBlur(spnoisy,i))
# cv2.waitKey(0)

# dst = cv2.filter2D()

# plt.hist(img.ravel(),8,[0,256])
# plt.show()
# cv2.waitKey(4000)

