
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

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
	#return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);
    
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
    
    image_select[~color_thresholds & region_thresholds] = [255,0,0]
    
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
    threshold = 3 # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15 #minimum number of pixels making up a line
    max_line_gap  = 5 # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0

    lines = cv2.HoughLinesP(image,rho,theta,threshold,np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2 ),(255,0,0),10)

    #color_edges = np.dstack((image,image,image))
    color_edges = np.copy(image);

    hough_transformed_image = cv2.addWeighted(color_edges,0.8,line_image,1,0)

    return hough_transformed_image

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


def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
  
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    #cv2.imshow('ti',masked_image)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho=1, theta=np.pi/180, threshold=1, min_line_len=10, max_line_gap=15):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    for line in lines:
        if line[0]
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)

   # draw_lines(line_img, lines)
    return line_img

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

    vertices = np.array( [[left_bottom,apex,apex,right_bottom]], dtype=np.int32 )

    blurred = bilateral_filter(image, 15,100,100);
    masked_image = region_of_interest(blurred, vertices)
    edged_image = canny(masked_image,110,150);
    line_detected_image = hough_lines(edged_image);

    weighted_lined = weighted_img(line_detected_image,image)

    return weighted_lined

def process_video(video):
    white_output = 'white.mp4'
    clip1 = VideoFileClip("solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)

if __name__ == "__main__":
    
    image = importImage()

   # processed_image = process_image(image);
    #mpimg.imsave('test33.png', processed_image);
   
    imgArr = os.listdir("test_images/")

    for img in imgArr:
        img = (mpimg.imread('test_images/'+img)*255).astype('uint8')
        processed_image = process_image(img);
       
        #cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('img',processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #mpimg.imsave('test_images_results/'+img, processed_image);

    plt.show()
    