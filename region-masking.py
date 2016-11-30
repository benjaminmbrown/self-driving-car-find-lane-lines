import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

plt.ion()

def importImage():
    image = mpimg.imread('test.jpg')
    return np.copy(image)


def getRegionThresholds (image):
  
    degree = 1

    fit_line_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), degree)
    fit_line_right = np.polyfit((right_bottom[0],apex[0]),(right_bottom[1], apex[1]), degree);
    fit_line_bottom = np.polyfit((left_bottom[0],right_bottom[0]), (left_bottom[1], right_bottom[1]), degree)

    XX, YY = np.meshgrid(np.arange(0, xSize), np.arange(0, ySize))

    region_thresholds = (YY > (XX*fit_line_left[0] + fit_line_left[1])) & \
    (YY > (XX*fit_line_right[0] + fit_line_right[1])) & \
    (YY < (XX*fit_line_bottom[0] + fit_line_bottom[1]))
  
    return region_thresholds

def getColorThresholds(image):
    #color select criteria
    red_thresh = 200
    green_thresh = 200
    blue_thresh = 200

    rgb_thresh = [red_thresh,green_thresh,blue_thresh]

    #identify pixels below threshold
    color_thresholds = (image[:,:,0] < rgb_thresh[0]) | (image[:,:,1] < rgb_thresh[1]) | (image[:,:,2] < rgb_thresh[2])
    
    return color_thresholds
    
def getColorSelectedImage(image):
    color_select = np.copy(image)
    color_thresholds = getColorThresholds(image)
    color_select[color_thresholds] = [0,0,0]
    
    return color_select
    
    
def getRegionMaskedImage(image):
    region_select = np.copy(image)
    region_thresholds = getRegionThresholds(region_select)
    region_select[~region_thresholds] = [255, 0, 0]
    
    return region_select

def getColoredImageInRegion(image):
    image_select = np.copy(image)
    color_select = np.copy(image)
    
    color_thresholds = getColorThresholds(color_select)
    region_thresholds = getRegionThresholds(region_select)
    
    #color_select[color_thresholds] = [0,0,0]
    image_select[~color_thresholds & region_thresholds] = [255,0,0]
    
    #plt.imshow(color_select);
    
    return image_select


if __name__ == "__main__":
    
    image = importImage()
    
    ySize = image.shape[0]
    xSize = image.shape[1]
    left_bottom = [0, ySize]
    right_bottom = [xSize, ySize]
    #TODO find horizon and base apex y val on horizon y value
    apex = [xSize/2, ySize/2] 
    

    #plt.imshow(getColorSelectedImage(image));
    #plt.imshow(getRegionMaskedImage(image));
    plt.imshow(getColoredImageInRegion(image));
    