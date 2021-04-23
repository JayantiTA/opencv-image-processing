# Ichiro - OpenCV Image Processing
This project was created to fulfill Ichiro's second internship assignment - Open Source Computer Vision (OpenCV) Image Processing

# Changing Colospaces
Know how to change color space to another: BGR -> Gray, BGR -> HSV, etc. We can change color space to extract a colored object. Like, when we want to convert BGR to. So, the steps are:
1. Use library `cv2` and `numpy`
2. Take image's frame from video capture
3. Convert BGR to HSV using `cv.cvtColor()` with parameters `frame` and `cv.COLOR_BGR2HSV`
4. Define upper and lower color in HSV (in this example bloe color)
5. Threshold HSV color using `mask` to get only colors that we want (blue color)
6. Use `cv.bitwise_and()` to merge `mask` and original image: `frame`
7. If we press escape, looping will be ended
```
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while (1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask= mask)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)

    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
    
cv.destroyAllWindows()
```

# Geometric Transformation
## Scaling
By using function `cv.resize()`, we can resize the image. In this function, there are 3 kinds of interpolation: `cv.inter_AREA` (for shrinking), `cv.inter_CUBIC` and `cv.inter_LINEAR` (for zooming). And the steps are:
1. Use library `cv2` and `numpy`
2. Load an image
3. Use `cv.resize()` to resize an image. There are two choices to use `cv.resize`:
    * Define parameters `fx` and `fy`
    * Define scale of `height` and `width` first, and then pass it into parameter `dsize`
4. Show an image after being resized
5. If we press escape, looping will be ended
```
import numpy as np
import cv2 as cv

# load an image
image = cv.imread('oggy.jpg')

# we should define `fx` and `fy` (scale of width and height that we want after resize)
result = cv.resize(image, None, fx = 2, fy = 2, interpolation = cv.INTER_CUBIC)

# OR

# we define scale of `height` and `width` first
height, width = image.shape[:2]
# pass it into parameter `dsize`
result = cv.resize(image, (2*width, 2*height), interpolation = cv.INTER_CUBIC)

# show an image after being resized
cv.imshow('result', result)

k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
```

## Translation
Translation is about object/image location. If we want to shift an image, we can put the direction (x,y) into Matrix 2x3. The steps are:
1. Use library `cv2` and `numpy`
2. Load an image
3. Get rows and columns from image.shape
4. Create matrix 2x3 and make it into Numpy array `np.float32`
5. Pass matrix into `cv.warpAffine()`
6. Show an image after translation
7. If we press escape, looping will be ended
```
import numpy as np
import cv2 as cv

# load an image
image = cv.imread('oggy.jpg')

# get rows and columns
rows, cols = image.shape[:2]

# put the direction into matrix 2x3
M = np.float32([[1, 0, 150], [0, 1, 150]])
dst = cv.warpAffine(image, M, (cols, rows))

# show an image after translation
cv.imshow('dst', dst)

k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
```

## Rotation
Rotation of an image with theta angle can represent by Matrix 2x2. We can use `cv.getRotationMatrix2D()`. The steps are:
1. Use library `cv2` and `numpy`
2. Load an image
3. Get rows and columns from `image.shape`
4. Create matrix 2x3 and put the return from `cv.getRotation()` into matrix. Function `cv.getRotation()` has parameters: center, angle, and scale. If we want to rotate with angle of origin 0, scale must be -1. And if we want to rotate with angle of origin 180, scale must be 1.
5. Pass matrix into `cv.warpAffine()`
6. Show an image after rotation
7. If we press escape, looping will be ended

```
import cv2 as cv
import numpy as np

# load an image
image = cv.imread('minion.jpg')

# get rows and columns
rows, cols = image.shape[:2]

# put the return from getRotation into matrix 2x3 
M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), 0, -1)
dst = cv.warpAffine(image, M, (cols, rows))

# show an image after rotation
cv.imshow('dst', dst)

k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
```

## Affine Transformation
We need three points from input and their corresponding locations in the output image. The steps are:
1. Use library `cv2`, `numpy`, and `matplotlib`
2. Load an image
3. Get rows, columns, and color from `image.shape`
4. Define three points from input (variable `pts1`)
5. Define three points from corresponding locations in the output (variable `pts2`)
6. Function `cv.getAffineTransform()` will create matrix 2x3 that stores points from variable `pts1` and `pts2`
7. Then pass that matrix into `cv.warpAffine()`
8. Show input and output image
```
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load an image
image = cv.imread('minion.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# get rows, columns, and color
rows, cols, ch = image.shape

# three points from input
pts1 = np.float32([[50,50], [200,50], [50,200]])

# three points from corresponding locations in the output
pts2 = np.float32([[10,100], [200,50], [50,350]])

# put points into matrix 2x3
M = cv.getAffineTransform(pts1, pts2)
dst = cv.warpAffine(image, M, (cols, rows))

# show an image from input
plt.subplot(121)
plt.imshow(image)
plt.title('Input')

# show an output image
plt.subplot(122)
plt.imshow(dst)
plt.title('Output')

plt.show()
```

## Perspective Transformation
In perspective transformation, we need 3x3 transformation matrix and four points from input and their corresponding points n the output image.
1. Use library `cv2`, `numpy`, and `matplotlib`
2. Load an image
3. Get rows, columns, and color from `image.shape`
4. Define four points from input (variable `pts1`)
5. Define four points from corresponding locations in the output (variable `pts2`)
6. Function `cv.getPerspectiveTransform()` will create matrix 3x3 that stores points from variable `pts1` and `pts2`
7. Then pass that matrix into `cv.warpPerspective()`
8. Show input and output image
```
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# load an image
img = cv.imread('minion.jpg', 1)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# get rows, columns, and color
rows, cols, ch = img.shape

# four points from input
pts1 = np.float32([[56,65], [368,52], [28,387], [389,390]])

# four points from corresponding locations in the output
pts2 = np.float32([[0,0], [300,0], [0,300], [300,300]])

# put points into matrix 3x3 
M = cv.getPerspectiveTransform(pts1, pts2)
dst = cv.warpPerspective(img, M, (300,300))

# show an image from input
plt.subplot(121)
plt.imshow(img)
plt.title('Input')

# show an output image
plt.subplot(122)
plt.imshow(dst)
plt.title('Output')

plt.show()
```

# Image Thresholding
## Simple Thresholding
Thresholding is method to assign pixel value in relation to the threshold value that provided.

From this implementation code, the steps are:
1. Use library `cv2`, `numpy`, and `matplotlib`
2. Load an image
3. Use `cv.threshold()` with parameters: 
    * source image -> should be a grayscale image
    * threshold value
    * maximum value -> assigned to pixel value that exceeding threshold
    * types of threshold. There are 5 types of simple thresholding:
        1. `cv.THRESH_BINARY`
        2. `cv.THRESH_BINARY_INV`
        3. `cv.THRESH_TRUNC`
        4. `cv.THRESH_TOZERO`
        5. `cv.THRESH_TOZERO_INV`
4. Plot all the images with their titles
5. Show an original image and the output images with thresholding. Use `matplotlib.pyplot`:
    * `plt.subplot()` to show and compare those 5 types of threshold
    * `plt.imshow()` and `plt.title()` to show images and titles
    * `plt.xticks()` and `plt.yticks()` to custom coordinate lines
```
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# load an image
img = cv.imread('gradient.jpg',0)

# simple thresholding
ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(img, 127, 255, cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(img, 127, 255, cv.THRESH_TOZERO_INV)

# plot all the images with their titles
titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# show the original image and five images with thresholding. 
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
```

## Adaptive Thresholding
Adaptive thresholding is used for image that has different lighting conditions in different areas. 

Implementation steps of adaptive thresholding:
1. Use library `cv2`, `numpy`, and `matplotlib`
2. Load an image which has different lighting conditions in different areas.
3. Use `cv.medianBlur()` to blur an image and reduce noise before thresholding
4. Use `cv.threshold()` to compare global thresholding with adaptive thresholding
5. Use `cv.adaptiveThresholding()` with parameters: 
    * source image
    * maximum value
    * adaptive method. There are 2 methods of adaptive thresholding:
        1. `cv.ADAPTIVE_THRESH_MEAN_C`: the threshold value is the mean of neighbourhood area minus the constant C
        2. `cv.ADAPTIVE_THRESH_GAUSSIAN_C`: the threshold value is a gaussian-weighted sum of neighbourhood area minus the constant C
    * threshold type (there are 5 types in simple thresholding)
    * block size
    * constant C.
6. Plot all the images with their titles
7. Show the output images. Use `matplotlib.pyplot`:
    * `plt.subplot()` to show and compare an image with grayscale, an image with global thresholding, and two images with adaptive thresholding
    * `plt.imshow()` and `plt.title()` to show images and titles
    * `plt.xticks()` and `plt.yticks()` to custom coordinate lines
```
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# load an image
img = cv.imread('spongebob3D.jpg', 0)

# blur and reduce noise
img = cv.medianBlur(img, 5)

# global thresholding v = 127
ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# adaptive mean thresholding
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,\
      cv.THRESH_BINARY, 11, 2)

# adaptive gaussian thresholding
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
      cv.THRESH_BINARY, 11, 2)

# plot all the images with their titles
titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

# show the output images
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
```

## Otsu's Binarization
In global thresholding, threshold value is provided and chosen. But in Otsu's binarization, can determine threshold value automatically and avoid being chosen. The steps in implementation Otsu's Binarization are:
1. Use library `cv2`, `numpy`, and `matplotlib`
2. Load an image which has noise
3. Use `cv.threshold()` to compare global thresholding (v = 127) with Otsu's thresholding
4. Use `cv.threshold()` with threshold types `cv.THRESH_BINARY + cv.THRESH_OTSU` for Otsu's thresholding
5. Use `cv.GaussianBlur()` to remove noise. And then same as fourth point, for Otsu's thresholding with Gaussian blur
6. Plot all the images and their histograms
7. Show the output images. Use `matplotlib.pyplot`:
    * `plt.subplot()` to show and compare an image with global thresholding, an image after Otsu's thresholding being applied, and an image with Gaussian blur and Otsu's thresholding
    * `plt.imshow()` and `plt.title()` to show images and titles
    * `plt.xticks()` and `plt.yticks()` to custom coordinate lines
```
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# load an image
img = cv.imread('noisy2.png',0)

# global thresholding
ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

# Otsu's thresholding
ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img, (11,11), 0)
ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

# show the output images
for i in range(3):
    plt.subplot(3,3,i*3+1)
    plt.imshow(images[i*3], 'gray')
    plt.title(titles[i*3])
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3,3,i*3+2)
    plt.hist(images[i*3].ravel(), 256)
    plt.title(titles[i*3+1])
    plt.xticks([])
    plt.yticks([])

    plt.subplot(3,3,i*3+3)
    plt.imshow(images[i*3+2], 'gray')
    plt.title(titles[i*3+2])
    plt.xticks([])
    plt.yticks([])

plt.show()
```

# Smoothing Images
## 2D Convolution (Image Filtering)
If we want to try an averaging filter on an image, we can use `cv.filter2D()`. Implementation's steps are:
1. Use library `cv2`, `numpy`, and `matplotlib`
2. Load an image
3. Define the kernel (small matrix that is used to apply effects on an image) matrix as return from `np.ones`. Size of kernel = 7x7
4. Pass the kernel into `cv.filter2D()` and define the result as output image `dst`. Function `cv.filer2D()` has parameters:
    * source image
    * ddepth or desire depth of destination image
    * kernel
5. Show an original image and the result image after being filtered
```
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# load an image
img = cv.imread('noisy2.png')

# define kernel matrix
kernel = np.ones((7,7), np.float32) / 49

# result after being filtered
dst = cv.filter2D(img, -1, kernel)

# show an original image
plt.subplot(121)
plt.imshow(img)
plt.title('Original')
plt.xticks([])
plt.yticks([])

# show an image after being filtered
plt.subplot(122)
plt.imshow(dst)
plt.title('Averaging')
plt.xticks([])
plt.yticks([])

plt.show()
```

## Image Blurring (Image Smoothing)
Image blurring is used to remove the noises and the other high frequency content. There are 4 types of blurring techniques:
1. Averaging
    
    Averaging is blurring technique that takes average of pixels under the kernel area and then replaces the central elements with average value. We can use `cv.blur()` with parameters image source and kernel size.
2. Gaussian Blurring
    
    In gaussian blurring technique, we can use `cv.GaussianBlur()` with parameters image source, kernel size, and standard deviation in X and Y direction. We should specify the width and the height of the kernel which should be positive and odd number. We also should specify the standard deviation in X and Y direction.
3. Median blurring
    
    Median blurring technique takes the median of all pixels under the kernel and replaces the central element with the median value. We can use `cv.medianBlur()` with parameters image source and kernel size. The kernel size should be positive and odd number.
4. Bilateral Filtering 
    
    Bilateral filtering technique is used to remove noise but still keep the edges sharp. We can use `cv.bilateralFilter()` with parameters image source, diameter of each pixel neighbourhood (that is used during filtering), sigma color, and sigma space.

Implementation in program:
1. Use library `cv2`, `numpy`, and `matplotlib`
2. Load an image which has noise or the other high frequency content (ex: edges)
3. Convert color of image to RGB
4. Use four types of image blurring:
    * Averaging: `cv.blur` with 5x5 kernel size
    * Gaussian blurring: `cv.GaussianBlur` with 9x9 kernel size and standard deviation = 0
    * Median blurring: `cv.medianBlur` with kernel size = 7
    * Bilateral blurring: `cv.bilateralFilter` with diameter = 9, sigma color = 75, and sigma space = 75
5. Plot all the images with their titles
6. Show the output images. Use `matplotlib.pyplot`:
    * `plt.subplot()` to show and compare an original image, and four images from image blurring
    * `plt.imshow()` and `plt.title()` to show images and titles
    * `plt.xticks()` and `plt.yticks()` to custom coordinate lines
```
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# load an image
img = cv.imread('oggy.jpg')

# convert color to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# use four types of image blurring
blur = cv.blur(img,(5,5))
gaussian_blur = cv.GaussianBlur(img, (9,9), 0)
median_blur = cv.medianBlur(img, 7)
bilateral_blur = cv.bilateralFilter(img, 9, 75, 75)

# plot all the images with their titles
titles = ['Original', 'Blurred', 'Gaussian Blur',
          'Median Blur', 'Bilateral Blur']
images = [img, blur, gaussian_blur, median_blur, bilateral_blur]

# show the output images
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
```

# Morphological Transformations
Some simple operations based on image shape (normally performed on binary image). The inputs are image source and kernel.
1. Erosion

    Erosion is used to decrease the thickness of the foreground object. It is useful for removing small noise. We can use `cv.erode()` with parameters image source, kernel, and iterations (number of times erosion applied).
2. Dilation
    
    Dilation is used to increase the thickness of the foreground object. After we erosion and remove noise of an object area, we can use dilation to increase object area. We can use `cv.dilate()` with parameters image source, kernel, and iterations (number of times dilation applied).
3. Opening
    
    Opening is another name of **erosion followed by dilation**. It is useful to remove noise. We can use `cv.morphologyEx()` with parameters image source, `cv.MORPH_OPEN`, and kernel.
4. Closing 
    
    Closing is reversed of opening **dilation followed by erosion**. It is useful to remove small noise inside foreground objects. We can use `cv.morphologyEx` with parameters image source, `cv.MORPH_CLOSE`, and kernel.
5. Morphological Gradient
    
    The difference between dilation and erosion, it will look like outline of the object. We can use `cv.morphologyEx` with parameters image source, `cv.MORPH_GRADIENT`, and kernel.
6. Top Hat
    
    The difference between input image and opening. `cv.morphologyEx` with parameters image source, `cv.MORPH_TOPHAT`, and kernel.
7. Black Hat
    
    The difference between input image and closing. `cv.morphologyEx` with parameters image source, `cv.MORPH_BLACKHAT`, and kernel.

Implementation erosion, dilation, gradient, tophat, and blackhat in program:
1. Use library `cv2`, `numpy`, and `matplotlib`
2. Load an image
3. Define the kernel (small matrix that is used to apply effects on an image) matrix as return from `np.ones`. Size of kernel = 5x5
4. Use some morphological transformations:
    * Erosion: `cv.erode()` with iteration = 1
    * Dilation: `cv.dilate()` with iteration = 1
    * Morphological gradient: `cv.morphologyEx()` with morphological type = `cv.MORPH_GRADIENT`
    * Tophat: `cv.morphologyEx()` with morphological type = `cv.MORPH_TOPHAT`
    * Blackhat: `cv.morphologyEx()` with morphological type = `cv.MORPH_BLACKHAT`
5. Plot all the images with their titles
6. Show the output images. Use `matplotlib.pyplot`:
    * `plt.subplot()` to show and compare an original image, and five images from morphological transformations
    * `plt.imshow()` and `plt.title()` to show images and titles
    * `plt.xticks()` and `plt.yticks()` to custom coordinate lines
```
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# load an image
img = cv.imread('j-morphology.png', 1)

# define kernel matrix
kernel = np.ones((5,5), np.uint8)

# morphological transformation functions
erosion = cv.erode(img, kernel, iterations = 1)
dilation = cv.dilate(img, kernel, iterations = 1)
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)

# plot all the images and titles
titles = ['Original Image', 'Erosion', 'Dilation',
          'Gradient', 'Tophat', 'Blackhat']
images = [img, erosion, dilation, gradient, tophat, blackhat]

# show the output images
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])

plt.show()
```

Besides, implementation program opening and closing:
1. Use library `cv2`, `numpy`, and `matplotlib`
2. Load an image1 for opening and an image2 for closing
3. Define kernel with size 5x5
4. Use some morphological transformations:
    * Opening: `cv.morphologyEx()` with morphological type = `cv.MORPH_OPEN`
    * Closing: `cv.morphologyEx()` with morphological type = `cv.MORPH_CLOSE`
5. Plot all the images and their titles
6. Show the output images. Use `matplotlib.pyplot`:
    * `plt.subplot()` to show and compare images before opening-after opening and before closing-after closing
    * `plt.imshow()` and `plt.title()` to show images and titles
    * `plt.xticks()` and `plt.yticks()` to custom coordinate lines
```
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# load images
img1 = cv.imread('j-opening.png', 1)
img2 = cv.imread('j-closing.png', 1)

# define kernel matrix
kernel = np.ones((5,5), np.uint8)

# morphological transformation functions
opening = cv.morphologyEx(img1, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(img2, cv.MORPH_CLOSE, kernel)

# plot all the images with their titles
titles = ['Before Opening', 'After Opening',
          'Befor Closing', 'After Closing']
images = [img1, opening, img2, closing]

# show the output images
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
```

# Image Gradients
1. Sobel and Scharr Derivatives
    
    Sobel operators is a joint gaussian smoothing and differentiation operation. We can specify the direction of derivatives that is taken (vertical or horizontal). we can use `cv.Sobel()` with parameters image source, output image depth, order of derivative x, order of derivative y, and kernel size. If kernel size = -1, 3x3 Scharr filter is used which give better results.
2. Laplacian Derivatives
    
    It calculates the Laplacian of the image where each derivative is found using Sobel derivatives. We use `cv.Laplacian()` with parameters image source and desired depth of the destination image.

Implementation in program:
1. Use library `cv2`, `numpy`, and `matplotlib`
2. Load an image
3. Use image gradient functions:
    * Laplacian: `cv.Laplacian()` with ddepth `cv.CV_64F`
    * Sobel x: `cv.Sobel()` with ddepth `cv.CV_64F`, order of derivative x = 1, order of derivative y = 0, kernel size = 5
    * Sobel y: `cv.Sobel()` with ddepth `cv.CV_64F`, order of derivative x = 0, order of derivative y = 1, kernel size = 5
4. Show the output images. Use `matplotlib.pyplot`:
    * `plt.subplot()` to show and compare an original image, laplacian, sobel x, and sobel y
    * `plt.imshow()` and `plt.title()` to show images and titles
    * `plt.xticks()` and `plt.yticks()` to custom coordinate lines
```
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# load an image
img = cv.imread('sudoku.png')

# image gradient functions
laplacian = cv.Laplacian(img, cv.CV_64F)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

# show the output images
plt.subplot(2,2,1)
plt.imshow(img, cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,2)
plt.imshow(laplacian, cmap = 'gray')
plt.title('Laplacian')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,3)
plt.imshow(sobelx, cmap = 'gray')
plt.title('Sobel X')
plt.xticks([])
plt.yticks([])

plt.subplot(2,2,4)
plt.imshow(sobely, cmap = 'gray')
plt.title('Sobel Y')
plt.xticks([])
plt.yticks([])

plt.show()
```

# Canny Edge Detection
In function `cv.canny()`, we can reduce noise with gaussian blur, and then find intensity gradient of the image with sorbel kernel, then every pixel is checked if it is a local maximum in its neighbourhood in the direction of gradient, and the last make sure the edges and non-edges. 
Implementation in program:
1. Use library `cv2`, `numpy`, and `matplotlib`
2. Load an image
3. Use `cv.canny()` to get edges with parameters image source, minimun value = 100, and maximum value = 200. The aperture size (size of sobel kernel) is set default = 3.
4. Show the output images. Use `matplotlib.pyplot`:
    * `plt.subplot()` to show and compare an original image with an edge image
    * `plt.imshow()` and `plt.title()` to show images and titles
    * `plt.xticks()` and `plt.yticks()` to custom coordinate lines
```
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# load an image
img = cv.imread('oggy.jpg', 0)

# get edges
edges = cv.Canny(img, 100, 200)

# show teh output images
plt.subplot(121)
plt.imshow(img, cmap = 'gray')
plt.title('Original Image')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(edges, cmap = 'gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])

plt.show()
```

# Image Pyramids
## Image Blending using Pyramids
Image blending is one apaplication of pyramids. Image blending with pyramids gives us seamless blending without leaving much data in the images. The steps are:
1. Use library `cv2` and `numpy`
2. Load two images apple and orange
3. Generate gaussian pyramid for apple and orange
4. Generate laplacian pyramid for apple and orange
5. Join the left half of apple and right half of orange in each levels of Laplacian pyramids.
6. Reconstruct joint image
```
import cv2 as cv
import numpy as np,sys

A = cv.imread('apple.jpg')
B = cv.imread('orange.jpg')

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv.pyrUp(gpA[i])
    L = cv.subtract(gpA[i-1], GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv.pyrUp(gpB[i])
    L = cv.subtract(gpB[i-1], GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv.pyrUp(ls_)
    ls_ = cv.add(ls_, LS[i])
    
# image with direct connecting each half
real = np.hstack((A[:,:cols//2],B[:,cols//2:]))
cv.imwrite('Pyramid_blending2.jpg', ls_)
cv.imwrite('Direct_blending.jpg', real)
```

# Contours in OpenCV
Contours can be explained as a curve joining all the continous points. Contours are useful in object detection and recognition.

## Find Contours
Implementation 'how to find contours in binary image':
1. Use library `cv2` and `numpy`
2. Load an image
3. Convert image color to gray
4. Use `cv.threshold()` to create threshold of an image
5. Find contours in image with `cv.findContours()`, parameters:

    * source image
    * contour retrieval mode
    * contour approximation method
```
import numpy as np
import cv2 as cv

# load an image
im = cv.imread('test.jpg')

# convert image color to gray
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

# create threshold of an image
ret, thresh = cv.threshold(imgray, 127, 255, 0)

# find contours
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
```

## Draw The Contours
To draw the contours, we use `cv.drawContours()` with parameters:
    
    * source image
    * contours which should be passed
    * index of contours
    * color, thickness, etc.
1. To draw all the contours in an image:
```
cv.drawContours(img, contours, -1, (0,255,0), 3)
```
2. To draw an individual contour
```
cv.drawContours(img, contours, 3, (0,255,0), 3)
```
3. Another method
```
cnt = contours[4]
cv.drawContours(img, [cnt], 0, (0,255,0), 3)
```

## Contour Approximation Method
Contour approximation method is the third argument in function `cv.findContours()`. There are 2 methods:
1. `cv.CHAIN_APPROX_NONE` : stores all the boundary points
2. `cv.CHAIN_APPROX_SIMPLE` : just need two end points of each lines

# Documentation Color Detection - Optional Assignment
This program can detect object inside background area. Implementation:
1. Use library `cv2` and `numpy`
```
import cv2 as cv
import numpy as np
```
2. Define empty function `nothing` as an optional argument for trackbar parameter
```
def nothing():
    pass
```
3. Capture video from webcam
```
webcam = cv.VideoCapture(0)
```
4. Create new windows for background and object trackbars
```
cv.namedWindow('Colorbars Background')
cv.namedWindow('Colorbars Object')
```
5. Assign strings for title of each trackbar
```
hue_high    = 'Hue High'
hue_low     = 'Hue Low'
sat_high    = 'Saturation High'
sat_low     = 'Saturation Low'
val_high    = 'Value High'
val_low     = 'Value Low'
area_min    = 'Area Minimum'
window      = 'Colorbars Background'
window2     = 'Colorbars Object'
```
6. Begin creating trackbars for background and object
```
# for background
cv.createTrackbar(hue_low, window, 0, 179, nothing)
cv.createTrackbar(hue_high, window, 0, 179, nothing)
cv.createTrackbar(sat_low, window, 0, 255, nothing)
cv.createTrackbar(sat_high, window, 0, 255, nothing)
cv.createTrackbar(val_low, window, 0, 255, nothing)
cv.createTrackbar(val_high, window, 0, 255, nothing)
cv.createTrackbar(area_min, window, 0, 5000, nothing)

# for object
cv.createTrackbar(hue_low, window2, 0, 179, nothing)
cv.createTrackbar(hue_high, window2, 0, 179, nothing)
cv.createTrackbar(sat_low, window2, 0, 255, nothing)
cv.createTrackbar(sat_high, window2, 0, 255, nothing)
cv.createTrackbar(val_low, window2, 0, 255, nothing)
cv.createTrackbar(val_high, window2, 0, 255, nothing)
cv.createTrackbar(area_min, window2, 0, 5000, nothing)
```
7. Start a while loop:
    
    1. Read the video capture
    2. Convert video color to HSV
    ```
    while True:
        _, image_frame = webcam.read()
        hsv_frame = cv.cvtColor(image_frame, cv.COLOR_BGR2HSV)
    ```
    3. Get trackbar positions for background and object
    ```
        # for background
        hue_low_background  = cv.getTrackbarPos(hue_low, window)
        hue_high_background = cv.getTrackbarPos(hue_high, window)
        sat_low_background  = cv.getTrackbarPos(sat_low, window)
        sat_high_background = cv.getTrackbarPos(sat_high, window)
        val_low_background  = cv.getTrackbarPos(val_low, window)
        val_high_background = cv.getTrackbarPos(val_high, window)
        area_min_background = cv.getTrackbarPos(area_min, window)

        # for object
        hue_low_object  = cv.getTrackbarPos(hue_low, window2)
        hue_high_object = cv.getTrackbarPos(hue_high, window2)
        sat_low_object  = cv.getTrackbarPos(sat_low, window2)
        sat_high_object = cv.getTrackbarPos(sat_high, window2)
        val_low_object  = cv.getTrackbarPos(val_low, window2)
        val_high_object = cv.getTrackbarPos(val_high, window2)
        area_min_object = cv.getTrackbarPos(area_min, window2)
    ```
    4. Make array from final values (trackbars background and object)
    5. Assign array to variables upper and lower
    ```
        # for background
        hsv_lower_background = np.array([hue_low_background, sat_low_background,
                                        val_low_background], np.uint8)
        hsv_upper_background = np.array([hue_high_background, sat_high_background,
                                        val_high_background], np.uint8)    

        # for object
        hsv_lower_object = np.array([hue_low_object, sat_low_object, val_low_object],
                                    np.uint8)
        hsv_upper_object = np.array([hue_high_object, sat_high_object, val_high_object],
                                    np.uint8)    
    ```
    6. Define area minimum background and object from trackbars
    7. Create masks for background and object
    ```
        # for background
        area_min_background = area_min_background
        background_mask     = cv.inRange(hsv_frame, hsv_lower_background, hsv_upper_background)

        # for object
        area_min_object = area_min_object
        object_mask     = cv.inRange(hsv_frame, hsv_lower_object, hsv_upper_object)
    ```
    8. Create new kernel with size 5x5
    ```
        kernel = np.ones((5, 5), np.uint8)
    ```
    9. Apply Closing to masks (Dilation followed by Erotion)
    ```
        # for background mask
        background_mask = cv.morphologyEx(background_mask, cv.MORPH_OPEN, kernel)
        cv.imshow("Background Mask", background_mask)

        # for object mask
        object_mask = cv.morphologyEx(object_mask, cv.MORPH_CLOSE, kernel)
        cv.imshow("Object Mask", object_mask)
    ```
    10. Create contour to track background
    ```
        background_contour, hierarchy = cv.findContours(background_mask,
                                                        cv.RETR_TREE,
                                                        cv.CHAIN_APPROX_SIMPLE)
    ```
    11. In looping for tracking background contour : create contour to track object and create looping for tracking object contour; set limit for object area inside background area
    ```
        for b_contour in background_contour:
            background_area = cv.contourArea(b_contour)
            if background_area > area_min_background:
                x, y, w, h = cv.boundingRect(b_contour)
                image_frame = cv.rectangle(image_frame, (x,y),
                                        (x + w, y + h),
                                        (0, 0, 255), 2)
                
                cv.putText(image_frame, "Background", (x, y),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))
                
                # create contour to track object
                object_contour, hierarchy = cv.findContours(object_mask,
                                                            cv.RETR_TREE,
                                                            cv.CHAIN_APPROX_SIMPLE)
            
                for o_contour in object_contour:
                    object_area = cv.contourArea(o_contour)
                    if object_area > area_min_object:
                        x2, y2, w2, h2 = cv.boundingRect(o_contour)
                        # if object inside background area
                        if x2 >= x and y2 >= y and x2 + w2 <= x + w and y2 + h2 <= y + h:
                            image_frame_2 = cv.rectangle(image_frame, (x2, y2),
                                                        (x2 + w2, y2 + h2),
                                                        (0, 255, 0), 2)
                            
                            cv.putText(image_frame_2, "Object", (x2, y2),
                                    cv.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (0, 255, 0))
    ```
    12. Show result color detection window
    ```
        cv.imshow("Color detection", image_frame)
    ```
    13. End loop when press 'q'
    ```
        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break
    ```

# References
https://docs.opencv.org/master/d2/d96/tutorial_py_table_of_contents_imgproc.html

https://www.geeksforgeeks.org/multiple-color-detection-in-real-time-using-python-opencv/

https://botforge.wordpress.com/2016/07/02/basic-color-tracker-using-opencv-python/