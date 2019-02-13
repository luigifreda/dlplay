# ******************************************************************************
# This file is part of dlplay
# 
# Copyright (C) Luigi Freda <luigi dot freda at gmail dot com>
# 
# dlplay is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# dlplay is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with dlplay. If not, see <http://www.gnu.org/licenses/>.
# ******************************************************************************
import numpy as np
import cv2
from matplotlib import pyplot as plt

"""
Feature Matching + Homography to find Objects

So what we did in last session? We used a queryImage, found some feature points in it, we took another trainImage, 
found the features in that image too and we found the best matches among them. In short, 
we found locations of some parts of an object in another cluttered image. 
This information is sufficient to find the object exactly on the trainImage.

For that, we can use a function from calib3d module, ie cv2.findHomography(). 
If we pass the set of points from both the images, it will find the perpective transformation of that object. 
Then we can use cv2.perspectiveTransform() to find the object. It needs atleast four correct points to find the transformation.

We have seen that there can be some possible errors while matching which may affect the result. 
To solve this problem, algorithm uses RANSAC or LEAST_MEDIAN (which can be decided by the flags). 
So good matches which provide correct estimation are called inliers and remaining are called outliers. cv2.findHomography() 
returns a mask which specifies the inlier and outlier points.
"""

MIN_MATCH_COUNT = 10
img1 = cv2.imread('data/box.png',0)          # queryImage
img2 = cv2.imread('data/box_in_scene.png',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

"""
Now we set a condition that atleast 10 matches (defined by MIN_MATCH_COUNT) are to be there to find the object. 
Otherwise simply show a message saying not enough matches are present.

If enough matches are found, we extract the locations of matched keypoints in both the images. 
They are passed to find the perpective transformation. Once we get this 3x3 transformation matrix, 
we use it to transform the corners of queryImage to corresponding points in trainImage. Then we draw it.
"""        

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

"""
Finally we draw our inliers (if successfully found the object) or matching keypoints (if failed).
"""

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()    