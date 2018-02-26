
# coding: utf-8

# # Greyscaling, Colour, Histogram and Drawing

# **Greyscaling:** Images are composed exclusively of shades of grey, with the intensity varying from black to white. According to [this](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029740) article, not all color-to-grayscale algorithms work equally well (but I do not need to worry about this now).
# 
# **Colour of images:** RGB = red, green, blue; HSV = hue, saturation, value; CMYK = cyan, magenta, yellow and black
# 
# **Drawing:** see [here](https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html)

# ---
# • Considerations:
# 
# 1 - OpenCV follows BGR order, while matplotlib likely follows RGB order. So an image loaded in OpenCV using matplotlib functions might need to convert it into RGB mode.
# 
# 2 - Each colour shade (Hue) has a value. Saturation refers to intensity, Brightness/Value refers to lightness. 

# In[1]:


import cv2
import urllib
from skimage import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
get_ipython().magic(u'matplotlib inline')
plt.rcParams['figure.figsize'] = (20.0, 10.0)

print ("opencv version",( cv2.__version__))


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# ## Convert local image to greyscale

# In[3]:


feather = io.imread("./images/pena_pavao.jpg")
io.imshow(feather)
io.show()


# In[4]:


feather = io.imread("./images/pena_pavao.jpg", as_grey=True)
print(type(feather), feather.shape, feather.dtype)
io.imshow(feather)
io.show()


# In[5]:


image = cv2.imread("./images/pena_pavao.jpg", 0)
cv2.imshow('Grayscale', image)
#cv2.waitKey(0) dont use this one on jupyter notebook
cv2.startWindowThread()
cv2.destroyAllWindows()

# save image
cv2.imwrite('./images/pavao_grey.jpg', image)


# ## Convert web image to greyscale

# In[6]:


urls = [
    "https://goo.gl/VTM6eJ",
    "https://goo.gl/26dtHN",
    "https://goo.gl/cnWBz4",
    "https://goo.gl/1Dz4Yw"
]


# In[7]:


# If I want to save them as they are
count = 0
for url in urls:
    print ("downloading{0}".format(url))
    image = io.imread(url)
    cv2.startWindowThread()
    cv2.imwrite("./images/out_" + str(count) + ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    count +=1


# In[8]:


bird = io.imread("https://goo.gl/1Dz4Yw")
print(type(bird), bird.shape, bird.dtype)
io.imshow(bird)
io.show()


# In[9]:


count = 0
for url in urls:
    print ("downloading{0}".format(url))
    image = io.imread(url)
    cv2.startWindowThread()
    cv2.imwrite("./images/gray_out_" + str(count) + ".png", cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    count +=1


# In[10]:


bird = io.imread("https://goo.gl/1Dz4Yw", as_grey=True)
print(type(bird), bird.shape, bird.dtype)
io.imshow(bird)
io.show()


# ## Colour spec

# In[11]:


imageG = cv2.imread('./images/pavao_grey.jpg')
image = io.imread("./images/pena_pavao.jpg")


# ### Check individual colour level for some pixels (RGB)

# Red, Green and Blue range from 0 to 255

# In[12]:


# First pixel
B, G, R = image[0,0]
print(B, G, R)
print(image.shape)


# In[13]:


B, G, R = image[10, 50] 
print(B, G, R)
print(image.shape)


# #### What's the RGB for the greyscale image?

# In[14]:


gray_img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray_img[0, 0])
print(gray_img.shape) 


# Note that this shape means (640, 960, 1), whereas for the clour image I had (640, 960, 3). 3 channel for colour (RGB) and I for greyscaled image.
# 
# Also, now each pixel has only one value(previously 3) with a range from 0 to 255.

# ### HSV

# H: 0 - 18
# 
# S: 0 - 255
# 
# V: 0 - 255

# In[15]:


#image = cv2.imread("./images/pena_pavao.jpg")
RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
plt.imshow(hsv_image)


# In[16]:


plt.figure(figsize=(10, 3))

plt.subplot(1, 2, 1)
plt.imshow(RGB_image)

plt.subplot(1, 2, 2)
plt.imshow(hsv_image)


# #### Individual channels in a RGB image

# Each R, G, B channel will be in greyscale. This happens because I am using a single channel to display each (R, G, B).
# 
# To display the R channel as red, the B as blue and the G as green, I have to create a **3-channels image**, for each of them, and keep the other channels remain at 0.

# In[17]:


imageBGR = cv2.imread("./images/pena_pavao.jpg")
#RGB_image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
B, G, R = cv2.split(imageBGR)
# or: B = img[:,:,0]

print(B.shape)
print(G.shape)
print(R.shape)

cv2.imwrite("./images/pavao_red.jpg", R)
cv2.imwrite("./images/pavao_blue.jpg", B)
cv2.imwrite("./images/pavao_green.jpg", G)

# merge colours again
merged = cv2.merge([B, G, R]) 
cv2.imwrite("./images/pavao_merged.jpg", merged)


# In[18]:


zeros = np.zeros(image.shape[:2], dtype = "uint8")

Rmerged = cv2.merge([zeros, zeros, R]) 
Gmerged = cv2.merge([zeros, G, zeros])
Bmerged = cv2.merge([B, zeros, zeros])

cv2.imwrite("./images/pavao_RedRed.jpg", Rmerged)
cv2.imwrite("./images/pavao_GreenGreen.jpg", Gmerged)
cv2.imwrite("./images/pavao_BlueBlue.jpg", Bmerged)


# In[43]:


def show(image):
    # Figure size in inches
    plt.figure(figsize=(8, 8))
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Show image, with nearest neighbour interpolation
    plt.imshow(RGB_image, interpolation='nearest')
    #plt.imshow(RGB_image)


# In[20]:


# Show Blue/Green/Red
images = []
for i in [0, 1, 2]:
    colour = image.copy()
    if i != 0: colour[:,:,0] = 0 #B
    if i != 1: colour[:,:,1] = 0 #G
    if i != 2: colour[:,:,2] = 0 #R
    images.append(colour)

show(np.vstack(images))


# In[21]:


# What if I want to amplify the B colour
merged = cv2.merge([B+100, G, R])
cv2.imwrite('./images/pavao_B_Amplified.jpg', merged)
#cv2.imshow("Merged with Blue Amplified", merged) 
RGB_merged = cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)
plt.imshow(RGB_merged)


# #### Individuals channels in a HSV image

# Hue is the colour (mainly red, yellow, green, cyan, blue or magenta).
# 
# Saturation is the intensity of the color between gray (low saturation) and pure color (high saturation). It's basically the amount of grey in the colour. It shows the dominance of hue in the colour.
# 
# Value is the brightness of the color, between black (low value) and average saturation (maximum value).

# In[22]:


# Convert from RGB to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

images = []
for i in [0, 1, 2]:
    colour = hsv.copy()
    if i != 0: colour[:,:,0] = 0 # hue
    if i != 1: colour[:,:,1] = 255 # saturation
    if i != 2: colour[:,:,2] = 255 # value
    images.append(colour)

hsv_stack = np.vstack(images)
rgb_stack = cv2.cvtColor(hsv_stack, cv2.COLOR_HSV2RGB)
show(rgb_stack)



# Hue: good separation in green, blue and red

# ## Histograms

# They are good way to visualise individual colour components.

# In[23]:


histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# ravel() flatens the image array 
plt.hist(image.ravel(), 256, [0, 256]);
plt.xlabel('Intensity of color', fontsize = 20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.show()
# Viewing Separate Color Channels
color = ('b', 'g', 'r')


# In[25]:


# Viewing Separate Color Channels
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color = col)
    plt.xlim([0,260])
    plt.xlabel('Intensity of color', fontsize = 20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


# In[26]:


def show_rgb_hist(image):
    colours = ('r','g','b')
    for i, c in enumerate(colours):
        plt.figure(figsize=(20, 4))
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])

        if c == 'r': colours = [((i/256, 0, 0)) for i in range(0, 256)]
        if c == 'g': colours = [((0, i/256, 0)) for i in range(0, 256)]
        if c == 'b': colours = [((0, 0, i/256)) for i in range(0, 256)]

        plt.bar(range(0, 256), histr, color=colours, edgecolor=colours, width=1)
        plt.xlabel('Intensity of color', fontsize = 20)
        plt.show()

show_rgb_hist(RGB_image)


# The brightest colours are green and blue. The bulk of the image is dark.

# In[27]:


def show_hsv_hist(image):
    # Hue
    plt.figure(figsize=(20, 3))
    histr = cv2.calcHist([image], [0], None, [180], [0, 180])
    plt.xlim([0, 180])
    colours = [colors.hsv_to_rgb((i/180, 1, 0.9)) for i in range(0, 180)]
    plt.bar(range(0, 180), histr, color=colours, edgecolor=colours, width=1)
    plt.title('Hue')

    # Saturation
    plt.figure(figsize=(20, 3))
    histr = cv2.calcHist([image], [1], None, [256], [0, 256])
    plt.xlim([0, 256])

    colours = [colors.hsv_to_rgb((0, i/256, 1)) for i in range(0, 256)]
    plt.bar(range(0, 256), histr, color=colours, edgecolor=colours, width=1)
    plt.title('Saturation')

    # Value
    plt.figure(figsize=(20, 3))
    histr = cv2.calcHist([image], [2], None, [256], [0, 256])
    plt.xlim([0, 256])

    colours = [colors.hsv_to_rgb((0, 1, i/256)) for i in range(0, 256)]
    plt.bar(range(0, 256), histr, color=colours, edgecolor=colours, width=1)
    plt.title('Value')

show_hsv_hist(hsv)


# In[28]:


im = cv2.imread("./images/pena_pavao.jpg")
show(im)


# ## Drawing

# ### Create a black square

# In[50]:


im_sq = np.zeros((512, 512, 3), np.uint8) # 512 by 512 with 3 layers (BRG)
show(im_sq)


# ### Draw a line over it

# In[51]:


cv2.line(im_sq, (0,0), (511,511), (255,127,0), 5); #(start), (end), (BGR), (thickness)
show(im_sq)


# ### Draw rectangle over it

# In[52]:


cv2.rectangle(im_sq, (100, 100), (299, 250), (127, 50, 127), -1); # (start), (end), (BGR), (-1 = filled)
show(im_sq)


# ### Draw circle

# In[61]:


im_sq = np.zeros((512, 512, 3), np.uint8)
cv2.circle(im_sq, (350, 350), 100, (10,200,50), -1) # center and radius
show(im_sq)


# ### Draw polygon

# In[63]:


# 4 points
pts = np.array( [[10,50], [400,50], [90,200], [50,500]], np.int32)
# polylines requires the points to be stored in the format performed by `reshape`
pts = pts.reshape((-1,1,2))

cv2.polylines(im_sq, [pts], True, (0,0,255), 3)
show(im_sq)


# ### Add text

# In[67]:


cv2.putText(im_sq, 'Hello World!', (75,290), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 3);
show(im_sq)

