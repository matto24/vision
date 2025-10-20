This archive contains artificial scenes used to try out stereo reconstruction. 
Each scene consist of two images and an according camera calibration file. 
The matlab file readPJ.m can be used to read these calibration files.

All the cubes in the scenes have a side length of 2 and all cubes are aligned with the coordinate system 
(i.e., sides of the cubes are parallel to the axis of the coordinate system.) 
The map.png file gives the coordinates of some of the corners of the cubes. 
You should be able to infer the coordinates of all the corners from this. 
The positions of the cubes are the same in all the scenes.

Please be aware that when computing the epipole for the rectified case you will face a division by zero. 
In a natural scene the 3rd component would usually not be perfectly zero but here it is. 
Instead of dividing by zero you can divide by a very small positive number (in this specific case).
