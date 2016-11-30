
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

plt.ion()
plt.interactive(False)

def importImage():
    #image = mpimg.imread('exit-ramp.png')#.jpg
    image =(mpimg.imread('exit-ramp.png')*255).astype('uint8')
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
    red_thresh = 200
    green_thresh = 200
    blue_thresh = 200

    rgb_thresh = [red_thresh,green_thresh,blue_thresh]
    color_thresholds = (image[:,:,0] < rgb_thresh[0]) | (image[:,:,1] < rgb_thresh[1]) | (image[:,:,2] < rgb_thresh[2])
    
    return color_thresholds
    
def getColorSelectedImage(image):
    color_select = np.copy(image)
    color_thresholds = getColorThresholds(image)
    color_select[color_thresholds] = [0,0,0]
    
    return color_select
 
def getGrayScaledImage(image):
	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);
	return gray
    
def getRegionMaskedImage(image):
    region_select = np.copy(image)
    region_thresholds = getRegionThresholds(region_select)
    region_select[region_thresholds] = [255, 0, 0]
    
    return region_select

def getColoredImageInRegion(image):
    image_select = np.copy(image)
    color_select = np.copy(image)
    
    color_thresholds = getColorThresholds(color_select)
    region_thresholds = getRegionThresholds(image_select)
    
    #color_select[color_thresholds] = [0,0,0]
    image_select[~color_thresholds & region_thresholds] = [255,0,0]
    
    #plt.imshow(color_select);
    
    return image_select

def getCannyEdgeImage(image):
    kernel_size = 5
    blurred_image =  cv2.GaussianBlur(image,(kernel_size,kernel_size),0)
    low_threshold = 110
    high_threshold = 160
    edged_image = cv2.Canny(blurred_image, low_threshold,high_threshold);
    return edged_image

def getHoughTransform(image):
    rho = 1
    theta = np.pi/180
    threshold = 1
    min_line_length = 15
    max_line_gap  = 5
    line_image = np.copy(image)*0

    lines = cv2.HoughLinesP(image,rho,theta,threshold,np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2 ),(255,0,0),10)

    #color_edges = np.dstack((image,image,image))
    color_edges = np.copy(image);

    print(color_edges.shape)
    print(image.shape)
    print(line_image.shape)

    hough_transformed_image = cv2.addWeighted(color_edges,0.8,line_image,1,0)

    return hough_transformed_image

if __name__ == "__main__":
    
    image = importImage()
    
    ySize = image.shape[0]
    xSize = image.shape[1]
    left_bottom = [0, ySize]
    right_bottom = [xSize, ySize]
    #TODO find horizon and base apex y val on horizon y value
    apex = [xSize/2, ySize/2] 
    
    grayed = getGrayScaledImage(image);
    edged_image = getCannyEdgeImage(grayed);
    line_detected_image = getHoughTransform(edged_image);


    plt.imshow(line_detected_image)
    #plt.imshow(edged_image, cmap='gray');

    #plt.imshow(grayed, cmap='gray');

    #plt.imshow(getColorSelectedImage(image));
    #plt.imshow(getRegionMaskedImage(image));
    #plt.imshow(getColoredImageInRegion(image));


    plt.show()
    