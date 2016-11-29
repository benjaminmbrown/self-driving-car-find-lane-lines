# self-driving-car-lane-selection-by-rgb-color
Using color to find white lines (lane lines) in images.
We import an image, create thresholds, then use bitwise OR to create a new image where any units below the threshold are blacked out. RGB values of 200,200,200 seem to do the trick for filtering white lines


Here's the resulting image after filtering:
<img src="https://media.giphy.com/media/l2JhGj6HM7dmjVde0/giphy.gif"/>
