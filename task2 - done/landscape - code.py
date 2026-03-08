import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('noisy.jpg', 1)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
med = cv.medianBlur(img_rgb, 3)
dst = cv.fastNlMeansDenoisingColored(med, None, 7, 7, 5, 21)
cv.imwrite('landscape_denoised.jpg', cv.cvtColor(dst, cv.COLOR_RGB2BGR))

plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(dst) 
plt.title("nl")
plt.axis("off")

plt.show()