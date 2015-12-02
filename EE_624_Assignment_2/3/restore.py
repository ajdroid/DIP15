from __future__ import division
from math import sqrt
import numpy as np
import cv2
from matplotlib import pyplot as plt

pi = np.pi
def DFT_matrix(N):

	i, j = np.meshgrid(np.arange(N), np.arange(N))
	A = (i*j) # Restrict to -0.2*pi to 0.2*pi
	omega = np.exp( - 2 * pi * 1j / N )
	W = np.power( omega, A ) 

	return W

def DFT(img):

	M,N = img.shape[:2]
	# result = np.empty(img.shape[:-1], dtype=complex)
	A = DFT_matrix(M)
	B = DFT_matrix(N)
	result = A.dot(img).dot(B) 
	return result

img1 = cv2.imread('../degraded.tif',0)
imgf1 = np.float32(img1)


dft1 = cv2.dft(imgf1, flags = cv2.DFT_COMPLEX_OUTPUT)

# dftc1 = dft1
# dftc1[:,:,0] = dftc.real
# dftc1[:,:,1] = dftc.imag

print dft1.shape
dft_shift1 = np.fft.fftshift(dft1)
dftmag1, dftphase1 = cv2.cartToPolar(dft_shift1[:,:,0], dft_shift1[:,:,1])
invdftmag = np.ones(dftmag1.shape)
invdftphase = np.zeros(dftphase1.shape)


for i in range(-240,241):
	for j in range(-240,241):
		if np.exp(-0.0025*(i*i+j*j)**(5/6)) >= 0.01:
			invdftmag[i,j] = np.exp(0.0025*(i*i+j*j)**(5/6))
			
print invdftmag
dftmag2 = invdftmag*dftmag1
dftphase2 = dftphase1 - invdftphase

dft_shift1[:,:,0], dft_shift1[:,:,1] = cv2.polarToCart(dftmag2, dftphase2)

f_ishift = np.fft.ifftshift(dft_shift1)
img2 = cv2.idft(f_ishift)
img2 = cv2.magnitude(img2[:,:,0],img2[:,:,1])
print img1
print img2/np.max(img2)*255

plt.subplot(121),plt.imshow(img1, cmap = 'gray')
plt.title('Input'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(np.uint8(img2/np.max(img2)*255), cmap = 'gray')
plt.title('Restored'), plt.xticks([]), plt.yticks([])

plt.show()
