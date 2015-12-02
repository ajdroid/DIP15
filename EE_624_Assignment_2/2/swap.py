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

img1 = cv2.imread('../hand.JPG',0)
img2 = cv2.imread('../calender.JPG',0)
imgf1 = np.float32(img1)
imgf2 = np.float32(img2)

dft1 = cv2.dft(imgf1, flags = cv2.DFT_COMPLEX_OUTPUT)
dft2 = cv2.dft(imgf2, flags = cv2.DFT_COMPLEX_OUTPUT)

# dftc1 = dft1
# dftc1[:,:,0] = dftc.real
# dftc1[:,:,1] = dftc.imag
dftmag1, dftphase1 = cv2.cartToPolar(dft1[:,:,0], dft1[:,:,1])
dftmag2, dftphase2 = cv2.cartToPolar(dft2[:,:,0], dft2[:,:,1])

dft1[:,:,0], dft1[:,:,1] = cv2.polarToCart(dftmag1, dftphase2)
dft2[:,:,0], dft2[:,:,1] = cv2.polarToCart(dftmag2, dftphase1)

# print dft1.shape

# dft_shift1 = np.fft.fftshift(dft1)
# dft_shift2 = np.fft.fftshift(dft2)

# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
# phase_spectrum = (cv2.phase(dft_shift[:,:,0],dft_shift[:,:,1]))

			
# dft_shift = np.fft.fftshift(dft1)
# magnitude_spectrum1 = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
# phase_spectrum1 = (cv2.phase(dft_shift[:,:,0],dft_shift[:,:,1]))

# f_ishift = np.fft.ifftshift(dft_shift)

img_back1 = cv2.idft(dft1)
img_back1 = cv2.magnitude(img_back1[:,:,0],img_back1[:,:,1])
img_back2 = cv2.idft(dft2)
img_back2 = cv2.magnitude(img_back2[:,:,0],img_back2[:,:,1])


plt.subplot(221),plt.imshow(img1, cmap = 'gray')
plt.title('Input 1'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(img2, cmap = 'gray')
plt.title('Input 2'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img_back1, cmap = 'gray')
plt.title('Modified 1'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(img_back2, cmap = 'gray')
plt.title('Modified 2'), plt.xticks([]), plt.yticks([])
plt.show()
