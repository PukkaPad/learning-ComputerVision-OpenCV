
# coding: utf-8

# # Transformation
# 
# Image transformations can be divided into **Affine** and **Non-Affine**.
# 
# **1) Affine transformation**
# 
# Is a generalization of Eucliden transformation (preserves distances, (more [here](https://en.wikipedia.org/wiki/Rigid_transformation))). The affine transformation preserve ratios of distances between points lying on a straight line. . It is used to correct distortion. The affine transformation is used for scaling, skewing and rotation.
# 
# The general representation of affine matrix is:
# 
# $\begin{bmatrix}a1 & a2 & b1\\a3 & a4 & b2\\c1 & c2 & 1\end{bmatrix}$
# 
# Where:
# 
# $\begin{bmatrix}a1 & a2 \\a3 & a4 \end{bmatrix}$ is the rotation matrix, which defines the transformation that will be performed (scaling, skewing and rotation).
# 
# $\begin{bmatrix}b1\\b2\end{bmatrix}$ is the translation vector, which moves the points. $b1$ represents the shift along the x-axis and $b2$ along the vertical axis.
# 
# $\begin{bmatrix}c1 & c2 \end{bmatrix}$ is the translation vector, which is = 0 for affice transformations
# 
# Given $x$ and $y$ as coordinates of a point, the transformamed coordinates $x'$ and $y'$ can be achieved by
# 
# $\begin{bmatrix}a1 & a2 & b1\\a3 & a4 & b2\\c1 & c2 & 1\end{bmatrix}$ $\times$ $\begin{bmatrix}x\\y\\1\end{bmatrix}$ = $\begin{bmatrix}x'\\y'\\1\end{bmatrix}$
# 
# 
# 1.1) Translation
# 
# All points are translated to new positions by adding offsets. $cv2.warpAffine$ to implement translations.
# 
# T= $\begin{bmatrix}1 & 0 & Tx\\0 & 1 & Ty\end{bmatrix}$ is the translation matrix
# 
# 1.2) Rotation
# 
# All points in the plane are rotated about apoint (such as the center) through the counterclockwise angle $\theta$. $cv2.getRotationMatrix2D$ to implement 2D rotations (and also scale).
# 
# M= $\begin{bmatrix}cos\theta & -sin\theta\\sin\theta & cos\theta\end{bmatrix}$ is the rotation matrix
# 
# 1.3) Resizing/scaling and interpolation
# 
# Points are scaled by applying scale factors to coordnates. Enlargements = + scale factors that are larger than unity. Reductions = + scale factors that are smaller than unity. Negative scale factors = mirrored image.
# 
# Interpolation: method of constructing new data points whthin a range of known data points.
# 
# $cv2.resize$
# 
# 
# **2) Non-affine transformation (aka Projective transform or Homography):**
# 
# It is transformation that maps lines to lines (but does not necessarily preserve parallelism). It does preserve co-linearity and incidence. Non-affine transforms result when the third row of the transform matrix is set to values other than 0, 0, and 1. It is generated by different camera angles.

# In[11]:


import cv2
import urllib
from skimage import io, util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython.display import display
get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (10.0, 10.0)

import warnings
warnings.filterwarnings('ignore')


# ## Translation

# In[2]:


image = cv2.imread("./images/out_3.png")


# In[3]:


# store heigh and width of image
height, width = image.shape[:2]


# In[4]:


# I will move the image 1/4 of the heigh and width in y and y
quarter_height, quarter_width = height/4, width/4


# In[5]:


# create the translation matrix
T = np.float32([[1,0, quarter_width], [0, 1, quarter_height]])


# In[6]:


# translate the image
img_translation = cv2.warpAffine(image, T, (width, height))


# In[7]:


# Because I am viewing here:
#cv2.imshow('Translation', img_translation)
#cv2.waitKey()
#cv2.destroyAllWindows()
RGB_image = cv2.cvtColor(img_translation, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# In[8]:


# Check the translation matrix
print(T)


# In[9]:


print(height, width)


# ## Rotation

# In[10]:


# Create rotation matrix 
# rotating around the centre. 90 degree. scale = 1
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)


# In[11]:


rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

#cv2.imshow('Rotated Image', rotated_image)
#cv2.waitKey()
#cv2.destroyAllWindows()
RGB_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# In[12]:


# need to remove the dark area: cv2.transpose
# it only does at 90degrees and clockwise
rotated_image=cv2.transpose(image)
RGB_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# ## Scaling/resizing and interpolation

# Note that when keeping the proportions of the image, it will be impossible to notice any change.

# In[13]:


# fx and fy: scaling 3/4 of the original size.
image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75) # fx and fy: scaling 3/4 of the original size.
# default interpolation is linear
#cv2.imshow('Scaling - Linear Interpolation', image_scaled) 
#cv2.waitKey()

RGB_image = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# In[14]:


# change only x
image_scaled = cv2.resize(image, None, fx=0.75, fy=1) # fx 3/4 of the original size.
# default interpolation is linear

#cv2.imshow('Scaling - Linear Interpolation', image_scaled) 
#cv2.waitKey()

RGB_image = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# In[15]:


# Double the size
img_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
#cv2.imshow('Scaling - Cubic Interpolation', img_scaled)
#cv2.waitKey()

RGB_image = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# In[16]:


# Skew the re-sizing by setting exact dimensions
img_scaled = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA)
#cv2.imshow('Scaling - Skewed Size', img_scaled) 
#cv2.waitKey()

#cv2.destroyAllWindows()
RGB_image = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# ### Image pyramid

# Refers to upscaling (enlarging) or downscaling(shrinking).
# 
# Scaling down redudes height and width by half.

# In[17]:


smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(smaller)


# In[18]:


#cv2.imshow('Original', image )

#cv2.imshow('Smaller ', smaller )
#cv2.imshow('Larger ', larger )
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# In[19]:


RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()
print(image.shape)


# In[20]:


RGB_image = cv2.cvtColor(smaller, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()
print(smaller.shape)


# In[21]:


# quality is lost
RGB_image = cv2.cvtColor(larger, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()
print(larger.shape)


# In[22]:


# foor loop making it smaller and smaller :)
G = image.copy()
gpA = [G]
for i in range(7):
    G = cv2.pyrDown(G)
    gpA.append(G)
    RGB_image = cv2.cvtColor(G, cv2.COLOR_BGR2RGB)
    io.imshow(RGB_image)
    io.show()
    print(G.shape)


# ## Cropping

# In[23]:


height, width = image.shape[:2]

# get the starting pixel coordiantes (top  left of cropping rectangle)
start_row, start_col = int(height * .25), int(width * .25)

# get the ending pixel coordinates (bottom right)
end_row, end_col = int(height * .75), int(width * .75)

# indexing to crop out the rectangle desired
cropped = image[start_row:end_row , start_col:end_col]


# In[24]:


RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
RGB_image_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
n_col = 2
n_row = 1
plt.figure(figsize=(5 * n_col, 10 * n_row))
plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

plt.subplot(1, 2, 1)
plt.imshow(RGB_image)

plt.subplot(1, 2, 2)
plt.imshow(RGB_image_crop)


# In[25]:


height, width = image.shape[:2]

# indexing to crop out the rectangle desired
cropped = image[25:200 , 330:700]

RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
RGB_image_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
n_col = 2
n_row = 1
plt.figure(figsize=(5 * n_col, 10 * n_row))
plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

plt.subplot(1, 2, 1)
plt.imshow(RGB_image)

plt.subplot(1, 2, 2)
plt.imshow(RGB_image_crop)


# ## Arithmetic Operations
# Add or subtract colour intensity.
# 
# Has the effect of increasing/decreasing brightness

# In[26]:


# Create a matrix of ones, with same dimesions of the image and multiply by a scalar
M = np.ones(image.shape, dtype = "uint8") * 100

# add this matrix M, to the image: increase in brightness
added = cv2.add(image, M)
#cv2.imshow("Added", added)
RGB_image = cv2.cvtColor(added, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()




# In[27]:


# subtracting will cause decrease in brightness
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted", subtracted)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

RGB_image = cv2.cvtColor(subtracted, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# ## Bitwise Operations and Masking

# Creat some geometrical images so the bitwise operations will be performed

# In[28]:


# Making a sqare
plt.figure(figsize=(5, 5))
square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
io.imshow(square)
io.show()


# In[29]:


# Making a half ellipse
plt.figure(figsize=(5, 5))
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
io.imshow(ellipse)
io.show()


# ### Bitwise operations

# In[30]:


# 'And': Shows only where they intersect
plt.figure(figsize=(5, 5))
And = cv2.bitwise_and(square, ellipse)
io.imshow(And)
io.show()


# In[31]:


# 'Or': Shows either square or ellipse is. Will show both.
plt.figure(figsize=(5, 5))
Or = cv2.bitwise_or(square, ellipse)
io.imshow(Or)
io.show()


# In[32]:


# 'Xor': Shows where either exist by itself.
# Where is 'and' goes back to zero (black).
# White remains where the images existed as the 'or' statement
plt.figure(figsize=(5, 5))
Xor = cv2.bitwise_xor(square, ellipse)
io.imshow(Xor)
io.show()


# In[33]:


# Not: Shows everything that isn't part of the square
# inverts the colours!
plt.figure(figsize=(5, 5))
Not_sq = cv2.bitwise_not(square)
io.imshow(Not_sq)
io.show()


# ## Convolutions and Bluring
# 
# And image has width( number of columns) and a height (number of rows). An image also have a depth, which are the number of channels in the image. For a RGB image, the depth is 3, one for each channel (R, G, B). So an image is a "big matrix" and a kernel is a "tiny matrix". 
# 
# The kernel sits on top og the "big matrix" and slides from left to right,  top to bottom, applying a mathematical operation (convolution) at each coordinate.
# 
# A convolution requires 3 components:
# 1) image
# 2) kernel matrix
# 3) output image
# 
# Output_Image = Input_Image * Function_kernel_size

# In[34]:


image = cv2.imread("./images/out_2.png")
RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
io.imshow(RGB)
io.show


# #### Create a 3x3 and a 7x7 kernels

# In[35]:


kernel3x3=np.ones((3,3), np.float32)/9 # it's multiplied by 1/9 to normalise, i.e, to sum to 1
kernel7x7=np.ones((7,7), np.float32)/49


# ### Convolve

# In[36]:


blurred = cv2.filter2D(image, -1, kernel3x3)
RGB = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
io.imshow(RGB)
io.show


# In[37]:


blurred = cv2.filter2D(image, -1, kernel7x7)
RGB = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
io.imshow(RGB)
io.show


# ### Other commonly used blurring methods in OpenCV

# #### Averaging
# Takes the average of all the pixels under kernel area and replace the central element. 

# In[38]:


blur = cv2.blur(image, (7,7))
RGB_image = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# #### Gaussian
# 
# https://en.wikipedia.org/wiki/Gaussian_blur
# 
# This blurring performs a weighted average of surrounding pixels based on the Gaussian distribution. 

# In[39]:


Gaussian = cv2.GaussianBlur(image, (7,7), 0)
RGB_image = cv2.cvtColor(Gaussian, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# #### Median
# Takes median of all the pixes under kernel area and entral element is replaced with this median value

# In[40]:


median = cv2.medianBlur(image, 7)
RGB_image = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# #### Bilateral
# Removes noise and keep edges sharp. Preserves horizontal and vertical lines.

# In[41]:


bilateral = cv2.bilateralFilter(image, 9, 75, 75)
RGB_image = cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# ## Image De-noising - Non-Local Means Denoising

# In[42]:


dst = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
RGB_image = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# ## Sharpening

# Opposite of blurring. It strenghs or emphasizes edges.

# In[48]:


image = cv2.imread('./images/out_1.png')
RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
io.imshow(RGB)
io.show()


# In[50]:


# The kernel is different from blurring, but it still sums to 1
# So there's no need to normalize it
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])


# In[51]:


# applying different kernels to the input image
sharpened = cv2.filter2D(image, -1, kernel_sharpening)


# In[52]:


RGB_image = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# ## Thresholding (Binarization)

# Thresholding is converting image to its binary form. All thresholding uses greyscale. It goes from 0 (black) to 255 (white)

# In[53]:


# download a new image to use later
image = io.imread("https://data.whicdn.com/images/16084973/large.jpg")
cv2.startWindowThread()
cv2.imwrite("./images/Alice.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# In[55]:


# Load a grayscale image for binarization
plt.figure(figsize=(5, 5))
image = cv2.imread('./images/gradient.jpg',0)
io.imshow(image)
io.show()


# ### Binarization 
# vaues below threshold become zero and values above it becomes 255

# In[56]:


plt.figure(figsize=(5, 5))
ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
io.imshow(thresh1)
io.show()


# ### Reverse binarization
# values below threshold become 255 and values above threshold become 0 

# In[57]:


plt.figure(figsize=(5, 5))
ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
io.imshow(thresh1)
io.show()


# ### Trucated
# Everythin higher than threshold is kept at threshold 

# In[60]:


plt.figure(figsize=(5, 5))
ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
io.imshow(thresh1)
io.show()


# ### Threshsold zero
# values above threshold are unchanged, values below threshold become 0

# In[62]:


plt.figure(figsize=(5, 5))
ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
io.imshow(thresh1)
io.show()


# ### Threshsold zero inversed
# inverse of the above

# In[64]:


plt.figure(figsize=(5, 5))
ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
io.imshow(thresh1)
io.show()


# ### Adaptive threshold
# Better/smarter way of thresholding

# In[72]:


plt.figure(figsize=(5, 5))
image = cv2.imread('./images/Origin_of_Species.jpg', 0)
cv2.imshow('Original', image)
io.imshow(image)


# In[71]:


plt.figure(figsize=(5, 5))
image2 = cv2.imread('./images/Alice.png', 0)
cv2.imshow('Original', image2)
io.imshow(image)


# In[73]:


# Binarization 


# In[75]:


plt.figure(figsize=(5, 5))
ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', thresh1)
io.imshow(thresh1)


# In[76]:


plt.figure(figsize=(5, 5))
ret,thresh1 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary', thresh1)
io.imshow(thresh1)


# In[79]:


# remove noise by bluring
plt.figure(figsize=(5, 5))
image = cv2.GaussianBlur(image, (3, 3), 0)
# Using adaptiveThreshold
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 3, 5) 
io.imshow(thresh)


# In[81]:


plt.figure(figsize=(5, 5))
image = cv2.GaussianBlur(image2, (3, 3), 0)
thresh2 = cv2.adaptiveThreshold(image2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                               cv2.THRESH_BINARY, 3, 5) 

io.imshow(thresh2)


# by the way: https://en.wikipedia.org/wiki/Image_noise

# In[83]:


plt.figure(figsize=(5, 5))
_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
io.imshow(thresh)


# In[84]:


plt.figure(figsize=(5, 5))
_, th2 = cv2.threshold(image2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
io.imshow(thresh)


# In[86]:


# Otsu's thresholding after Gaussian filtering
plt.figure(figsize=(5, 5))
blur = cv2.GaussianBlur(image, (5,5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Guassian Otsu's Thresholding", thresh) 
io.imshow(thresh)


# ## Dilation, Erosion, Opening and Closing
# 
# More info [here](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)
# 
# Dilation: adds pixels to the boundaries of objects in a image
# 
# Erosion: Removes pixels at the boundaries. It removes small white noise, detachs connected objects, etc.
# 
# Opening: Erosion followed by dilation. Useful for removing noise
# 
# Closing: Reverse of Opening - Dilation folloed by erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object.

# In[4]:


plt.rcParams['figure.figsize'] = (10.0, 10.0)
image = cv2.imread('./images/opencv_inv.png', 0)
io.imshow(image)
io.show()


# In[12]:


inverted_img = util.invert(image)
io.imshow(inverted_img)
io.show()


# In[5]:


# kernel
kernel = np.ones((5,5), np.uint8)


# In[6]:


# Erode
erosion = cv2.erode(image, kernel, iterations = 1) # in most case, it seem sI wont need to iterate more than once.
io.imshow(erosion)
io.show()


# In[13]:


erosion = cv2.erode(inverted_img, kernel, iterations = 1)
io.imshow(erosion)
io.show()


# Note that Erode removes white. See both images above. OpenCV understands the white as being the object itself. So when the background is white, erosion will ave the thickening effect of dilation. 

# In[7]:


# Dilate
dilation = cv2.dilate(image, kernel, iterations = 1)
io.imshow(dilation)
io.show()


# In[14]:


dilation = cv2.dilate(inverted_img, kernel, iterations = 1)
io.imshow(dilation)
io.show()


# Note that on a white background, dilation will have the thinning effect. See both images above.

# In[8]:


# Opening
# good for removing noise
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
io.imshow(opening)
io.show()


# In[9]:


# Closing - 
# good for removing noise
# looks very close to the original image
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
io.imshow(closing)
io.show()


# ## Edge Detection and Image Gradients
# 
# The 3 main types are:
# 
# Sobel: for vertical and horizontal edges
# 
# Laplacian: all orientations
# 
# Canny: optimal. Well defined edges and accurate detection.

# In[16]:


image = cv2.imread('./images/out_1.png', 0)
io.imshow(image)
io.show()


# In[17]:


height, width = image.shape


# In[27]:


# Extract Sobel Edges
# Extract vertical and horizontal edges
sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5) 
sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) 


# In[29]:


# I'm not sure why it shows the edges as RGB. I tried using cmp='gray' but the edges do not look as sharp
io.imshow(sobel_x)
#io.imshow(sobel_x, cmap = 'gray')
io.show()


# In[30]:


io.imshow(sobel_y)
io.show()


# In[31]:


# Combine x and y edges
sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
io.imshow(sobel_OR)
io.show()


# In[32]:


laplacian = cv2.Laplacian(image, cv2.CV_64F)
io.imshow(laplacian)
io.show()


# Canny needs two thresholds. Any value below threshold 1 is not going to be considered an edge and values above threshold 2 are edges. Values in between are either edges or non-edges. This is decided based on how their 
# intensities are connected.

# In[33]:


canny = cv2.Canny(image, 50, 120)
io.imshow(canny)
io.show()


# In[34]:


canny = cv2.Canny(image, 10, 120)
io.imshow(canny)
io.show()


# In[35]:


canny = cv2.Canny(image, 10, 50)
io.imshow(canny)
io.show()


# ## Perspective and Affine Transforms

# ### Perspective Transform

# In[42]:


image = cv2.imread('./images/scan.jpg')
io.imshow(image)
io.show()
# resolution is not good. I tried to solve but I had no success


# In[43]:


# Cordinates of the 4 corners of the original image
points_A = np.float32([[320,15], [700,215], [85,610], [530,780]])

# Cordinates of the 4 corners of the desired output, for a A4 paper
points_B = np.float32([[0,0], [420,0], [0,594], [420,594]])
 
# Use the two sets of four points to compute the Perspective Transformation matrix, M    
M = cv2.getPerspectiveTransform(points_A, points_B)
 
# Generate warped image    
warped = cv2.warpPerspective(image, M, (420,594)) # final size is 420,594
 
io.imshow(warped)
io.show()


# ### Affine Transform
# Only need 3 coordinates

# In[37]:


image = cv2.imread('./images/ex2.jpg')
rows,cols,ch = image.shape
io.imshow(image)
io.show()


# In[39]:


# Cordinates of the 4 corners of the original image
points_A = np.float32([[320,15], [700,215], [85,610]])

# Cordinates of the 4 corners of the desired output
# using a ratio of an A4 Paper 1 : 1.41
points_B = np.float32([[0,0], [420,0], [0,594]])


# In[40]:


# Use the two sets of four points to compute the Perspective Transformation matrix, M    
M = cv2.getAffineTransform(points_A, points_B)

warped = cv2.warpAffine(image, M, (cols, rows))

io.imshow(warped)
io.show()

