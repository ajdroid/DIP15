import numpy as np 
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2
from matplotlib import pyplot as plt
import math

def lloydmax(img, L, uniform = False ,intensityRange = [0,255]):
	# construct pdf
	flatimg = img.flatten()
	# no of pixels per value
	probcount = np.zeros(intensityRange[1]-intensityRange[0]+1 , dtype=int) 
	r = np.zeros(L , dtype=float)
	t = np.zeros(L+1, dtype=float)
	# set initial values
	t[0] = intensityRange[0]
	t[L] = intensityRange[1]
	q = (intensityRange[1]-intensityRange[0]+1)/L
	idx = 0	
	# Do histogram calculation
	for pixel in flatimg:
		idx = pixel-intensityRange[0]
		probcount[idx]=probcount[idx]+1
	MSER = 0.0
	lastMSER = -0.1
	# Init with uniform quantizer
	for k in range(1,L+1):
		# Apply thresholds
		t[k] = t[k-1]+q
		r[k-1] = (t[k]+t[k-1])/2
	# Iterate
	run = 0
	if (uniform==False):
		while (MSER>lastMSER and run<11):
			lastMSER = MSER
			SER = 0.0
			numerator = np.zeros(L, dtype=int) 
			denominator = np.zeros(L, dtype=int) 
			for k in range(1,L):
				# Apply thresholds
				t[k] = (r[k-1] + r[k])/2
			for k in range(0,L):	
				# Get numerators + denoms
				for pixel in range(int(math.ceil(t[k])), int(math.ceil(t[k+1]))):
					print run, pixel, int(math.ceil(t[k])), int(math.ceil(t[k+1]))
					numerator[k] = numerator[k] + pixel*probcount[pixel-intensityRange[0]]
					denominator[k] = denominator[k] + probcount[pixel-intensityRange[0]]
				# Get representatives
				r[k] = numerator[k]/denominator[k]
				r[k]=round(r[k])
				for pixel in range(int(math.ceil(t[k])), int(math.floor(t[k+1]))):
					SER = SER + (pixel-r[k])*(pixel-r[k])*probcount[pixel-intensityRange[0]]
				# Apply MSE minimization

			# Calc MSER
			MSER = math.sqrt(SER)
			run = run + 1
		
	# Apply quantization
	for index, pixel in enumerate(flatimg):
		for k in range(1,L+1):
			if pixel >= t[k-1] and pixel <= t[k]:
				flatimg[index] = r[k-1]
	MSER = math.sqrt(sum((flatimg-img.ravel())*(flatimg-img.ravel())))
	print r
	print t
	print run
	print MSER
	cv2.imshow('Quantized image (' +('Uniform' if uniform else 'Lloyd-Max')+ ')',\
				np.reshape(flatimg, img.shape))
	hist = cv2.calcHist([flatimg], [0], None, [256], [0, 256])
	plt.figure()
	plt.title("Grayscale Histogram")
	plt.xlabel("Bins")
	plt.ylabel("# of Pixels")
	plt.plot(hist)
	plt.xlim(intensityRange)

if __name__ == "main":
	img = cv2.imread('../flower.jpg',0)
	cv2.imshow('flower',img)
	lloydmax(img, 8)
	lloydmax(img, 8, True)

	# plt.hist(img.ravel(),8,[0,256])
	plt.show()

