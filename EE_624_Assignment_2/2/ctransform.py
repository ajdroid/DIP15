from __future__ import division
from math import sqrt
import numpy as np
import cv2
from matplotlib import pyplot as plt

def topfilt (block, f=0):
	M, N = block.shape[:2]
	dct = cv2.dct(block)
	A = dct.flatten()

	check = A[sorted(range(len(A)), key=lambda i: A[i])[-16:][0]]
	if f:
		print A
		print check
		print dct
	dct[dct<check]=0
	block = np.uint8(cv2.idct(dct)*255)
	return block



img = cv2.imread('../sunflower.jpg',0)
img_float32 = np.float32(img)/255.0
dct = cv2.dct(img_float32)
img_back = np.zeros(img.shape, dtype=float)
for i in range(0,img.shape[0],8):
	for j in range(0,img.shape[1],8):
		img_back[i:i+8,j:j+8] = topfilt(img_float32[i:i+8,j:j+8])
print topfilt(img_float32[:8,:8],1)
# for i in range(0,img.shape[0],8):
# 	for j in range(0,img.shape[1],8):
# 		dct_comp[i:i+8,j:j+8] = topfilt(img_float32[i:i+8,j:j+8])

imgn_back = np.uint8(cv2.idct(dct)*255)
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Reconstructed from blocks'), plt.xticks([]), plt.yticks([])
plt.show()