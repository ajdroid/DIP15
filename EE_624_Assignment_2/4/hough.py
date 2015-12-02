import cv2
import numpy as np

img = cv2.imread('../building.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
r = 30
cv2.imshow('original',img)
lines = cv2.HoughLines(edges,r,np.pi/180,100)

for rho,theta in lines[:,0,:]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1,0)
cv2.imshow('houghlines'+str(r)+'.jpg',img)
cv2.imwrite('houghlines'+str(r)+'.jpg',img)
cv2.waitKey(0)