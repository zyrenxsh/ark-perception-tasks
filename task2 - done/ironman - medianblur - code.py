import cv2
import numpy as np
import matplotlib.pyplot as plt

inputImg = cv2.imread('./iron_man_noisy.jpg', 0) #given, a binary image

#i am using median blur to remove the noise

#larger kernel size remove more noise but also removes the sketch pixels as the line-width is of that range
# outputImg = cv2.medianBlur(inputImg, 5) 
outputImg = cv2.medianBlur(inputImg, 3)
outputImg = cv2.medianBlur(outputImg, 3)   
#i have applied median blur twice with a kernel size of 3, as it retains the sketch pixels.
#this is the best i could do, it removes most of the noise while keeping the sketch pixels intact

#median blur works as it takes median of the pixel values in the kernel, so where there are noise only one or two pixels are 255 so medium gives 0 only, but where there are sketch pixels, more pixels are 255 so median gives 255 only, thus removing the noise while keeping the sketch pixels intact.

plt.subplot(1, 2, 1)
plt.title('Input Image')    
plt.imshow(inputImg, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Output Image')
plt.imshow(outputImg, cmap='gray')
plt.axis('off')
plt.show()
