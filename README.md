# ark-perception-tasks
this repo contains my solution to ARK Perception tasks using Python and MatLab. The tasks mainly on noise filtering and geomtric feature extraction.

## experience and learning
working on these tasks was interesting as well as challenging, i got learn many things like morphological tranformation, blurs, vectorisation, etc

#### PS
The instructions recommended creating documentation using the Overleaf LaTeX template. However, due to some technical issues, I was unable to complete the documentation in Overleaf. Instead, the reports have been written and submitted in PDF format.

## repo structure
each folder contains documentation on the task, the output and the code involved.

## tasks overview
### 1. vision based line follower:
The objective of this task was to process a downward-facing camera feed and compute the line angle deviation so that a drone could follow a track autonomously.
Due to time constraints, I was unable to complete this task. A significant amount of time was spent working on Task 2 and Task 3, especially implementing parts of the algorithms manually as required.
However, the folder still contains my approach.
### 2. noise-filtering:
In this task, the goal was to remove high-frequency noise from corrupted images while preserving the original features.
Approach used: Applied Median Blur to remove salt noise and Used Non-Local Means Denoising for improved noise removal while preserving edges.
The processed images are saved along with the scripts used to generate them.
### 3. medial axis detection:
This task involved detecting the medial axis (central skeleton line) of a moving object in video frames.
Approach used: 
Background subtraction to isolate the moving object
Morphological operations (erosion and dilation) to clean the silhouette
Edge detection using image derivatives
Implementation of the Hough Line Transform from scratch to detect edges
Estimation of the medial axis using the detected lines
The final output overlays the medial axis onto the original frame.





