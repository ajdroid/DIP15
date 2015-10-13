import numpy as np 
np.set_printoptions(suppress=True, precision=3) 
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
import math

def harrisCorners( src, winSize, blockSize, alpha ):
	src_padded = cv2.copyMakeBorder(src,pad,pad,pad,pad,cv2.BORDER_REPLICATE) # DO I need this?
	image = src.copy() 
	for pixel in src[pad:-pad, pad:-pad]:
		imagepixel[]
