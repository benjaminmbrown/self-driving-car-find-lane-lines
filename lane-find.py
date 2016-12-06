
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

plt.ion()
plt.interactive(False)

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
	return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
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
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15 #minimum number of pixels making up a line
    max_line_gap  = 5 # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0

    lines = cv2.HoughLinesP(image,rho,theta,threshold,np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2 ),(255,0,0),10)

    color_edges = np.copy(image);

    hough_transformed_image = cv2.addWeighted(color_edges,0.8,line_image,1,0)

    return hough_transformed_image

def region_of_interest(img, vertices):

    mask = np.zeros_like(img)   
  
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho=1, theta=np.pi/80, threshold=2, min_line_len=10, max_line_gap=20):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    

    line_img = np.zeros((*img.shape, 4), dtype=np.uint8)

    draw_lines(line_img, lines)
    return line_img

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def adaptive_bilateral_filter(img, kernel_size,):
    """(src, ksize, sigmaSpace[, dst[, maxSigmaColor[, anchor[, borderType]]]]) → ds"""
    return  cv2.adaptiveBilateralFilter(img,9,75,75)

def bilateral_filter(img, diameter, sigmaColor, sigmaSpace):
    """(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) → dst¶"""
    return cv2.bilateralFilter(img,diameter,sigmaColor,sigmaSpace)
    
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    ySize = image.shape[0]
    xSize = image.shape[1]
    left_bottom = [80, ySize]
    right_bottom = [xSize-70, ySize]
    midline = [[xSize/2,0],[xSize/2,ySize]]
    #TODO find horizon and base apex y val on horizon y value
    apex = [xSize/2, ySize/2+30] 

    kernel_size = 3

    vertices = np.array( [[left_bottom,apex,apex,right_bottom]], dtype=np.int32 )

    processed_image = getGrayScaledImage(image);
    processed_image = cv2.GaussianBlur(processed_image,(kernel_size, kernel_size), 0)
    processed_image = getColorSelectedImage(image);
    processed_image = canny(processed_image, 100,150);#blurs and edg
    processed_image = region_of_interest(processed_image, vertices)
    #processed_image = getColorSelectedImage(image);

    processed_image = hough_lines(processed_image);
    #processed_image = getHoughTransform(processed_image);
    return processed_image


def bilateral_filter(img, diameter, sigmaColor, sigmaSpace):
    """(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) → dst¶"""
    return cv2.bilateralFilter(img,diameter,sigmaColor,sigmaSpace)
    
if __name__ == "__main__":

    imgArr = os.listdir("test_images/")

    for img in imgArr:

        image = (mpimg.imread('test_images/'+img)*255).astype('uint8')
        #processed_image = process_image(image);
        grayed = getGrayScaledImage(image);
        edged_image = getCannyEdgeImage(grayed);
        line_detected_image = getHoughTransform(edged_image);   
        plt.imshow(line_detected_image)
        #cv2.imshow('t',processed_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        mpimg.imsave('test_images_results/'+img, line_detected_image);

    plt.show()

    