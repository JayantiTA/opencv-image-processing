# Ichiro - OpenCV Image Processing
This project was created to fulfill Ichiro's second internship assignment - Open Source Computer Vision (OpenCV) Image Processing

# Changing Colospaces
Know how to change color space to another: BGR -> Gray, BGR -> HSV, etc. We can change color space to extract a colored object. Like, when we want to convert BGR to. So, the steps are:
1. Use library `cv2` and `numpy`
2. Take image's frame from video capture
3. Convert BGR to HSV using `cv2.cvtColor()` with parameters `frame` and `cv.COLOR_BGR2HSV`
4. Define upper and lower color in HSV (in this example bloe color)
5. Threshold HSV color using `mask` to get only colors that we want (blue color)
6. Use `cv.bitwise_and` to merge `mask` and original image: `frame`
7. If we press escape, looping will be ended

# Geometric Transformation
## Scaling
Using function `cv2.resize()`, we can resize the image. There are 3 kinds of interpolation: `cv2.inter_AREA` (for shrinking), `cv2.inter_CUBIC` and `cv2.inter_LINEAR` (for zooming). There are 2 methods to define new size:
```
img = cv.imread('image.jpg')
res = cv.resize(img, (0,0), fx=2, fy=2, interpolation = cv.INTER_CUBIC)
```
In the first method, we should define `fx` and `fy` (scale of width and height that we want after resize).

or
```
height, width = img.shape[:2]
res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
```
In the second, we define scale of `height` and `width` first, and then pass it into parameter `dsize`.

## Translation
Translation is about object/image location. If we want to shift an image, we can put the direction (x,y) into Matrix 2x3. Then we make it into Numpy array `np.float32` and pass it into `cv.warpAffine()`.

## Rotation
Rotation of an image with theta angle can represent by Matrix 2x2. We can use `cv.getRotationMatrix2D` with parameters: center, angle, and scale. If we want to rotate with angle of origin 0, scale must be -1. And if we want to rotate with angle of origin 180, scale must be 1.

## Afiine Transformation
We need three points from input and their corresponding locations in the output image. Function `cv.getAffineTransform` will create matrix 2x3 and then pass it into `cv.warpAffine`.

## Perspective Transformation
We need 3x3 transformation matrix and four points from input and their corresponding points n the output image. Function `cv.getPerspectiveTransform` will create matrix 3x3 and then pass it into `cv.warpPerspective`.

# Image Thresholding
## Simple Thresholding
Thresholding is method to assign pixel value in relation to the threshold value that provided. We can use `cv.threshold` with parameters: source image (should be a grayscale image), threshold value, maximum value (assigned to pixel value that exceeding threshold), types of threshold. There are 5 types of simple thresholding:
1. `cv.THRESH_BINARY`
2. `cv.THRESH_BINARY_INV`
3. `cv.THRESH_TRUNC`
4. `cv.THRESH_TOZERO`
5. `cv.THRESH_TOZERO_INV`

We also use `matplotlib.pyplot`: `plt.subplot()` to show and compare those 5 types of threshold, `plt.imshow()` and `plt.title()` to show image and title of threshold, and `plt.xticks()` and `plt.yticks()` to custom coordinate lines.

## Adaptive Thresholding
Adaptive thresholding is used for image that has different lighting conditions in different areas. There are 2 methods of adaptive thresholding:
1. `cv.ADAPTIVE_THRESH_MEAN_C`: the threshold value is the mean of neighbourhood area minus the constant C
2. `cv.ADAPTIVE_THRESH_GAUSSIAN_C`: the threshold value is a gaussian-weighted sum of neighbourhood area minus the constant C
The parameters in `cv.adaptiveThresholding()` are: source image, maximum value, adaptive method, threshold type, block size, and constant C.

The input is an image which has different lighting conditions in different areas. The first output is an image with grayscale. The second, is an image with global thresholding. The third is an image with adaptive mean thresholding. And the last one is an image with adaptive gaussian thresholding.

## Otsu's Binarization
In global thresholding, threshold value is provided and chosen. But in Otsu's binarization, can determine threshold value automatically and avoid being chosen. 

The input image is noisy image. First output is an image after global thresholding with v=127 being applied. The second output is and image after otsu's thresholding being applied. The third, an image that use `cv.THRESH_BINARY + cv.THRESH_OTSU`. We also use `cv.GaussianBlur()` to remove noise.

# Smoothing Images
## 2D Convolution (Image Filtering)
If we want to try an averaging filter on an image, we can use `cv.filter2D()` with parameters: source image, desire depth of destination image, and kernel (small matrix that is used to apply effects on an image). The program keeps the kernel above the pixel, and then add 49 pixels above the kernel. After that, take the average, and replace the central pixel with the new average value.

## Image Blurring (Image Smoothing)
Image blurring is used to remove the noises and the other hight frequency content. There are 4 types of blurring techniques:
1. Averaging
Averaging is blurring technique that takes average of pixels under the kernel area and then replaces the central elements with average value. We can use `cv.blur()` with parameters image source and kernel size.
2. Gaussian Blurring
In gaussian blurring technique, we can use `cv.GaussianBlur()` with parameters image source, kernel size, and standard deviation in X and Y direction. We should specify the width and the height of the kernel which should be positive and odd number. We also should specify the standard deviation in X and Y direction.
3. Median blurring
Median blurring technique takes the median of all pixels under the kernel and replaces the central element with the median value. We can use `cv.medianBlur()` with parameters image source and kernel size. The kernel size should be positive and odd number.
4. Bilateral Filtering 
Bilateral filtering technique is used to remove noise but still keep the edges sharp. We can use `cv.bilateralFilter()` with parameters image source, diameter of each pixel neighbourhood (that is used during filtering), sigma color, and sigma space.

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

# Image Gradients
1. Sobel and Scharr Derivatives
Sobel operators is a joint gaussian smoothing and differentiation operation. We can specify the direction of derivatives that is taken (vertical or horizontal). we can use `cv.Sobel()` with parameters image source, output image depth, order of derivative x, order of derivative y, and kernel size. If kernel size = -1, 3x3 Scharr filter is used which give better results.
2. Laplacian Derivatives
It calculates the Laplacian of the image where each derivative is found using Sobel derivatives. We use `cv.Laplacian()` with parameters image source and desired depth of the destination image.

# Canny Edge Detection
In function `cv.canny()`, we can reduce noise with gaussian blur, and then find intensity gradient of the image with sorbel kernel, then every pixel is checked if it is a local maximum in its neighbourhood in the direction of gradient, and the last make sure the edges and non-edges. Function `cv.canny()` has parameters image source, minimun value and maximum value respectively, and aperture size (size of sorbel kernel).

# Image Pyramids
## Image Blending using Pyramids
Image blending is one apaplication of pyramids. Image blending with pyramids gives us seamless blending without leaving much data in the images. The steps are:
1. Load two images apple and orange
2. Generate gaussian pyramid for apple and orange
3. Generate laplacian pyramid for apple and orange
4. Join the left half of apple and right half of orange in each levels of Laplacian pyramids.

# Hough Line Transform
