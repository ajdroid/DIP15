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

img = cv2.imread('../sunflower.jpg',0)
# img = cv2.imread('../filt.jpg',0)
img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
dftc = DFT(img)
dft1 = np.zeros(dft.shape)
dft1[:,:,0] = dftc.real
dft1[:,:,1] = dftc.imag
# print dft1.shape

dft_shift = np.fft.fftshift(dft1)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
phase_spectrum = (cv2.phase(dft_shift[:,:,0],dft_shift[:,:,1]))

for i in range(phase_spectrum.shape[0]):
	for j in range(phase_spectrum.shape[1]):
		if phase_spectrum[i,j] < 0.2*np.pi or phase_spectrum[i,j] > 1.8*np.pi:
			dft1[i,j,0] = 0
			dft1[i,j,1] = 0
			
dft_shift = np.fft.fftshift(dft1)
magnitude_spectrum1 = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
phase_spectrum1 = (cv2.phase(dft_shift[:,:,0],dft_shift[:,:,1]))

f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(phase_spectrum, cmap = 'gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
plt.figure()
plt.subplot(131),plt.imshow(img_back, cmap = 'gray')
plt.title('Reconstructed Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum1, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(phase_spectrum1, cmap = 'gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
