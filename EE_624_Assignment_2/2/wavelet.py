from __future__ import division
from math import sqrt
import numpy as np
import cv2,  pywt
from matplotlib import pyplot as plt

img = cv2.imread('../lena.jpg',0)
print img.shape
coeffs = pywt.dwt2(data=img, wavelet='haar')

img1, (img2, img3, img4) = coeffs
imgb = pywt.idwt2(coeffs, wavelet='haar')

plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Given Image (512x512)'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(img1, cmap = 'gray')
plt.title('Approximation (256x256)'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(img2, cmap = 'gray')
plt.title('Horizontal (256x256)'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(img3, cmap = 'gray')
plt.title('Vertical (256x256)'), plt.xticks([]), plt.yticks([])
plt.figure()
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(imgb, cmap = 'gray')
plt.title('Reconstructed'), plt.xticks([]), plt.yticks([])

plt.show()
