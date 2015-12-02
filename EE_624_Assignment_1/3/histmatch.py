from __future__ import division
from scipy.misc import imsave, imread
import numpy as np
import cv2
from matplotlib import pyplot as plt

imsrc =  cv2.imread('../givenhist.jpg',0)
imtint = cv2.imread('../sphist.jpg',0)

nbr_bins=255
if len(imsrc.shape) < 3:
    imsrc = imsrc[:,:,np.newaxis]
    imtint = imtint[:,:,np.newaxis]

imres = imsrc.copy()
for d in range(imsrc.shape[2]):
    imhist,bins = np.histogram(imsrc[:,:,d].flatten(),nbr_bins,normed=True)
    tinthist,bins = np.histogram(imtint[:,:,d].flatten(),nbr_bins,normed=True)

    cdfsrc = imhist.cumsum() #cumulative distribution function
    cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8) #normalize

    cdftint = tinthist.cumsum() #cumulative distribution function
    cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8) #normalize


    im2 = np.interp(imsrc[:,:,d].flatten(), bins[:-1], cdfsrc)
    im3 = np.interp(im2, cdftint, bins[:-1])

    imres[:,:,d] = im3.reshape((imsrc.shape[0],imsrc.shape[1] ))
    reshist,bins = np.histogram(imtint[:,:,d].flatten(),nbr_bins,normed=True)

plt.figure()
plt.subplot(131),plt.plot(imhist)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.plot(tinthist)
plt.title('Target Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.plot(reshist)
plt.title('Target Spectrum'), plt.xticks([]), plt.yticks([])
for i in range(50,150):
	for j in range(50,400):
		if imres[i,j,0] < 2:
			imres[i,j,0] = 185

print imres.shape
cv2.imshow("hist1", imsrc)
cv2.imshow("hist2", imtint)
plt.figure()
plt.imshow(imres[:,:,0], cmap='gray')
cv2.imshow("histnormresultjpg", imres)
plt.show()
cv2.waitKey(0)