

# Gui Features in OpenCV

## Getting Started with Images

Learn to load an image, display it and save it back

- open image
- write image
- show image 

## Getting Started with Videos

Learn to play videos, capture videos from Camera and write it as a video

- Capture Video from Camera
- Playing Video from file
- Saving a Video

## Drawing Functions in OpenCV

Learn to draw lines, rectangles, ellipses, circles etc with OpenCV

## Mouse as a Paint-Brush

Draw stuffs with your mouse

## Trackbar as the Color Palette

Create trackbar to control certain parameters

---- 
# Core Operations

## Basic Operations on Images
- accessing and modifying image values 
- splitting and merging image channels 
- making border for image padding 

## Arithmetic Operations on Images
- cv image operations 

## Performance Measurement and Improvement Techniques
Getting a solution is important. But getting it in the fastest way is more important. Learn to check the speed of your code, optimize the code etc.
- cv time 
- cv time optimization 

-----
# Image Processing in OpenCV

## Transformations 
- scaling
- translation 
- affine
- perspective 

## Thresholding 
- thresholding 
- adaptive thresholding 
- Otsu's thresholding 

## Filtering 
- filtering
- smoothing 
- image gradients 
- canny 
- pyrdown

--- 
# Feature Detection and Description
- Harris corners 
- Harris subpix 
- Shi-Tomasi 
- SIFT
- SURF
- FAST
- BRIEF
- ORB
- Feature matching
- Feature matching + homography 

## Understanding Features
What are the main features in an image? How can finding those features be useful to us?

## Harris Corner Detection
Okay, Corners are good features? But how do we find them?

## Shi-Tomasi Corner Detector & Good Features to Track
We will look into Shi-Tomasi corner detection

## Introduction to SIFT (Scale-Invariant Feature Transform)
Harris corner detector is not good enough when scale of image changes. Lowe developed a breakthrough method to find scale-invariant features and it is called SIFT

## Introduction to SURF (Speeded-Up Robust Features)
SIFT is really good, but not fast enough, so people came up with a speeded-up version called SURF.

## FAST Algorithm for Corner Detection
All the above feature detection methods are good in some way. But they are not fast enough to work in real-time applications like SLAM. There comes the FAST algorithm, which is really “FAST”.

## BRIEF (Binary Robust Independent Elementary Features)
SIFT uses a feature descriptor with 128 floating point numbers. Consider thousands of such features. It takes lots of memory and more time for matching. We can compress it to make it faster. But still we have to calculate it first. There comes BRIEF which gives the shortcut to find binary descriptors with less memory, faster matching, still higher recognition rate.

## ORB (Oriented FAST and Rotated BRIEF)
SIFT and SURF are good in what they do, but what if you have to pay a few dollars every year to use them in your applications? Yeah, they are patented!!! To solve that problem, OpenCV devs came up with a new “FREE” alternative to SIFT & SURF, and that is ORB.

## Feature Matching
We know a great deal about feature detectors and descriptors. It is time to learn how to match different descriptors. OpenCV provides two techniques, Brute-Force matcher and FLANN based matcher.

## Feature Matching + Homography to find Objects
Now we know about feature matching. Let’s mix it up with calib3d module to find objects in a complex image.

-------
# Camera Calibration and 3D Reconstruction

## Camera Calibration
Let’s find how good is our camera. Is there any distortion in images taken with it? If so how to correct it?

## Pose Estimation
This is a small section which will help you to create some cool 3D effects with calib module.

## Epipolar Geometry
Let’s understand epipolar geometry and epipolar constraint.

## Depth Map from Stereo Images
Extract depth information from 2D images.