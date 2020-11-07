# Homography-Calculation
Different homography calculations in one repo

## Steps
* Find keypoints with SIFT
* Match keypoints
* Find correspondences
* Find inliers with RANSAC and calculate homography for each random correspondence
* Find the final homography with the help of threshold
  
## Future Works
New methods to calculate homograpy will be added day by day. Their comparisons will be also here.

## Outputs
Images:
![alt text](https://github.com/nuwandda/Homography-Calculation/blob/main/image_l.jpg "Left Image")
![alt text](https://github.com/nuwandda/Homography-Calculation/blob/main/image_r.jpg "Right Image")
Matches:
![alt text](https://github.com/nuwandda/Homography-Calculation/blob/main/matches.png "Matches")
Matches with inliers:
![alt text](https://github.com/nuwandda/Homography-Calculation/blob/main/Inlier_matches.png "Inliers(Green lines)")

