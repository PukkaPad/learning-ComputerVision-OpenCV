
# coding: utf-8

# **Image segmentation** refers to the partition of image into a set of regions that represent meaningful areas of the image. It has two objectives. (1) Decomposition of images into parts for further analysis; (2) perform change of a representation.

# In[1]:


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


# # Countours
# 
# Continuos lines or curves. They are important for **object detection** and **shape analysis**

# In[2]:


#image = cv2.imread('./images/keyboard.jpg', 0)
image = cv2.imread('./images/keyboard.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17) # blur the image (remove noise)
io.imshow(gray)
io.show()


# In[3]:


canny = cv2.Canny(gray, 30, 200)
io.imshow(canny)
io.show()


# In[4]:


# find the contours

#_, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
_, contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cv2.imshow('Canny Edges After Contouring', edged)
#cv2.waitKey(0)

print(("Number of Contours found = " + str(len(contours))))


# In[5]:


# Draw all contours
# Use '-1' as the 3rd parameter to draw all countours
cv2.drawContours(image, contours, -1, (0,255,0), 3)

#cv2.imshow('Contours', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

io.imshow(image)
io.show()


# In[6]:


# Find contour of my TI-84


# In[7]:


image = cv2.imread('./images/TI84.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17) # blur the image (remove noise)
io.imshow(gray)
io.show()


# In[8]:


canny = cv2.Canny(gray, 30, 200)
io.imshow(canny)
io.show()


# In[9]:


# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
_, contours, hierarchy = cv2.findContours(canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(("Number of Contours found = " + str(len(contours))))


# In[10]:


# Draw all contours
# Use '-1' as the 3rd parameter to draw all countours
cv2.drawContours(image, contours, -1, (0,255,0), 3)

#cv2.imshow('Contours', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

io.imshow(image)
io.show()

