import matplotlib.pyplot as plt
import cv2

# Load the image
img = cv2.imread('/media/saptarshi/6234-E21C/inputVideo/90001.jpeg')

# Show the result
plt.imshow(img)
plt.grid()
plt.show()