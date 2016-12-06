# self-driving-car-lane-selection-by-rgb-color
Using color to find white lines (lane lines) in images.
We import an image, create thresholds, then use bitwise OR to create a new image where any units below the threshold are blacked out. RGB values of 200,200,200 seem to do the trick for filtering white lines

#Video processed try 1
<img src="https://media.giphy.com/media/l0HlEyyssLqfPlqc8/giphy.gif"/>
#Color Selection:
<img src="https://media.giphy.com/media/l2JhGj6HM7dmjVde0/giphy.gif"/>
#Region Masking
<img src="https://media.giphy.com/media/3o7TKKcBgjMd7QRUSk/giphy.gif"/>
#Using region masking & selection to highlight lanes
<img src="https://media.giphy.com/media/l2JhuQFoDs8gdD9du/giphy.gif"/>

#1st attempt at canny edge detection
<img src="https://media.giphy.com/media/l2JhnPEc2tLPrUUGA/giphy.gif"/>

#Improved thresholds on canny detection
<img src="https://media.giphy.com/media/l3vR7ICI0lK3KzzcA/giphy.gif"/>

#Hough tranform to detect line sets
<img src="https://media.giphy.com/media/3o6Zt94A6nRVf9U6WI/giphy.gif"/>

#Adding mask to greyscaling, gaussian smoothing, and Hough transform to detect lanes
<img src ="https://media.giphy.com/media/3oriO5J7MAjjua7NKM/giphy.gif"/>

#Applying bilateral filter blurring to blur image while retaining edges, effectively enhancing them
<img src = "https://media.giphy.com/media/3oz8xLLpQJyIjqYVMs/giphy.gif"/>

#Modified bilateral blur 
<img src="https://media.giphy.com/media/3o6Ztg2cziwxzEiHqU/giphy.gif"/>