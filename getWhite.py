import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

plt.ion()
image = mpimg.imread('test.jpg')
print("This image is: ", type(image), 
	"With dimensions:", image.shape)

ySize = image.shape[0]
xSize = image.shape[1]

color_select = np.copy(image)

#color select criteria
red_thresh = 200
green_thresh = 200
blue_thresh = 200

rgb_thresh = [red_thresh,green_thresh,blue_thresh]

#identify pixels below threshold
thresholds = (image[:,:,0] < rgb_thresh[0]) | (image[:,:,1] < rgb_thresh[1]) | (image[:,:,2] < rgb_thresh[2])

#img with pixels below thresholds blacked out

color_select[thresholds] = [0,0,0]

plt.imshow(color_select)