Structure From Motion module
============================

This module contains algorithms to perform 3d reconstruction from 2d images. The core of the module is a light version of [libmv](https://github.com/libmv/libmv) which is a Structure from Motion (SfM) library divided into different modules (image/detector/descriptor/multiview) that allow to resolve part of the SfM process.


Installation (Linux)
--------------------

In order to install the SfM module into your system, you must add the following third party libraries: Eigen, Glog and Gflag. If you are on Linux you can simply type the following command:

    sudo apt-get install libeigen3-dev libgflags-dev libgoogle-glog-dev


Ceres installation (Linux)
--------------------------

The main reconstruction API currently uses the open-source [Ceres Solver](http://ceres-solver.org/) in order to solve part of the Bundle Adjustment plus the points Intersect. If Ceres Solver is not installed on your system, the reconstruction funcionality will be disabled.

Here are the instructions on how to install the Ceres Solver on your machine (Linux).

    sudo apt-get install libceres-dev


Run
---

There are provided some samples in order to show the reconstruction functionality use.

**recon2v.cpp**

This program shows the two view reconstruction capabilities in the OpenCV Structure From Motion (SFM) module. It uses the following data from the VGG datasets at "reconv2 _pts.txt" where the first line has the number of points and each subsequent line has entries for matched points as: x1 y1 x2 y2. Finally, the script reconstructs the given set of correspondences and show the result using the OpenCV 3d visualizer (viz).

<p align="center">
  <img src="doc/pics/recon2v.jpg" width="400" height="400">
</p>

**nView_scene_reconstruction.cpp**

This program shows the n view reconstruction capabilities in the OpenCV Structure From Motion (SFM) module. It loads a file with the tracked 2d points over all the frames which are embedded into a vector of 2d points array, where each inner array represents a different frame. Every frame is composed by a list of 2d points which e.g. the first point in frame 1 is the same point in frame 2. If there is no point in a frame the assigned value will be (-1,-1).

To run this example you can type the following command in the opencv binaries directory specifying the file path in your system and the camera intrinsics (in this case obtained from the scene generation script).

    ./example_sfm_nView_scene_reconstruction /path_to_opencv_contrib/samples/data/backyard_tracks.txt 800 400 225

Finally, the script reconstructs the given set of tracked points and show the result using the OpenCV 3d visualizer (viz). On the left image the original scene from where we extracted the input file, on the right, the obtained result.

<p align="center">
  <img src="doc/pics/nView1.jpg" width="400" height="400">
  <img src="doc/pics/nView2.jpg" width="400" height="400">
</p>

**unordered_scene_reconstruction.cpp**

//TODO

**random_scene_reconstruction.cpp**

//TODO