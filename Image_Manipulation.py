
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

# In[42]:


import cv2
import urllib
from skimage import io
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

# In[4]:


image = cv2.imread("./images/out_3.png")


# In[5]:


# store heigh and width of image
height, width = image.shape[:2]


# In[6]:


# I will move the image 1/4 of the heigh and width in y and y
quarter_height, quarter_width = height/4, width/4


# In[9]:


# create the translation matrix
T = np.float32([[1,0, quarter_width], [0, 1, quarter_height]])


# In[10]:


# translate the image
img_translation = cv2.warpAffine(image, T, (width, height))


# In[11]:


# Because I am viewing here:
#cv2.imshow('Translation', img_translation)
#cv2.waitKey()
#cv2.destroyAllWindows()
RGB_image = cv2.cvtColor(img_translation, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# In[12]:


# Check the translation matrix
print(T)


# In[14]:


print(height, width)


# ## Rotation

# In[16]:


# Create rotation matrix 
# rotating around the centre. 90 degree. scale = 1
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)


# In[17]:


rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

#cv2.imshow('Rotated Image', rotated_image)
#cv2.waitKey()
#cv2.destroyAllWindows()
RGB_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# In[22]:


# need to remove the dark area: cv2.transpose
# it only does at 90degrees and clockwise
rotated_image=cv2.transpose(image)
RGB_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# ## Scaling/resizing and interpolation

# Note that when keeping the proportions of the image, it will be impossible to notice any change.

# In[23]:


# fx and fy: scaling 3/4 of the original size.
image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75) # fx and fy: scaling 3/4 of the original size.
# default interpolation is linear
#cv2.imshow('Scaling - Linear Interpolation', image_scaled) 
#cv2.waitKey()

RGB_image = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# In[24]:


# change only x
image_scaled = cv2.resize(image, None, fx=0.75, fy=1) # fx 3/4 of the original size.
# default interpolation is linear

#cv2.imshow('Scaling - Linear Interpolation', image_scaled) 
#cv2.waitKey()

RGB_image = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# In[25]:


# Double the size
img_scaled = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
#cv2.imshow('Scaling - Cubic Interpolation', img_scaled)
#cv2.waitKey()

RGB_image = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# In[26]:


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

# In[29]:


smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(smaller)


# In[ ]:


#cv2.imshow('Original', image )

#cv2.imshow('Smaller ', smaller )
#cv2.imshow('Larger ', larger )
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# In[27]:


RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()
print(image.shape)


# In[31]:


RGB_image = cv2.cvtColor(smaller, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()
print(smaller.shape)


# In[32]:


# quality is lost
RGB_image = cv2.cvtColor(larger, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()
print(larger.shape)


# In[38]:


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

# In[44]:


height, width = image.shape[:2]

# get the starting pixel coordiantes (top  left of cropping rectangle)
start_row, start_col = int(height * .25), int(width * .25)

# get the ending pixel coordinates (bottom right)
end_row, end_col = int(height * .75), int(width * .75)

# indexing to crop out the rectangle desired
cropped = image[start_row:end_row , start_col:end_col]


# In[53]:


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


# In[89]:


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

# In[98]:


# Create a matrix of ones, with same dimesions of the image and multiply by a scalar
M = np.ones(image.shape, dtype = "uint8") * 100

# add this matrix M, to the image: increase in brightness
added = cv2.add(image, M)
#cv2.imshow("Added", added)
RGB_image = cv2.cvtColor(added, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()




# In[99]:


# subtracting will cause decrease in brightness
subtracted = cv2.subtract(image, M)
cv2.imshow("Subtracted", subtracted)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

RGB_image = cv2.cvtColor(subtracted, cv2.COLOR_BGR2RGB)
io.imshow(RGB_image)
io.show()


# ## Bitwise Operations and Masking

# In[ ]:


Creat some geometrical images so the bitwise operations will be performed


# In[101]:


# Making a sqare
plt.figure(figsize=(5, 5))
square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
io.imshow(square)
io.show()


# In[103]:


# Making a half ellipse
plt.figure(figsize=(5, 5))
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
io.imshow(ellipse)
io.show()


# ### Bitwise operations

# In[106]:


# 'And': Shows only where they intersect
plt.figure(figsize=(5, 5))
And = cv2.bitwise_and(square, ellipse)
io.imshow(And)
io.show()


# In[107]:


# 'Or': Shows either square or ellipse is. Will show both.
plt.figure(figsize=(5, 5))
Or = cv2.bitwise_or(square, ellipse)
io.imshow(Or)
io.show()


# In[108]:


# 'Xor': Shows where either exist by itself.
# Where is 'and' goes back to zero (black).
# White remains where the images existed as the 'or' statement
plt.figure(figsize=(5, 5))
Xor = cv2.bitwise_xor(square, ellipse)
io.imshow(Xor)
io.show()


# In[110]:


# Not: Shows everything that isn't part of the square
# inverts the colours!
plt.figure(figsize=(5, 5))
Not_sq = cv2.bitwise_not(square)
io.imshow(Not_sq)
io.show()

