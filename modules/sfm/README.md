Structure From Motion module
============================

This module contains algorithms to perform 3d reconstruction from 2d images. The core of the module is a light version of [libmv](http://code.google.com/p/libmv/) which is a Structure from Motion (SfM) library divided into different modules (image/detector/descriptor/multiview) that allow to resolve part of the SfM process.

The main reconstruction API currently uses the open-source [Ceres Solver](http://ceres-solver.org/) in order to solve part of the Bundle Adjustment plus the points Intersect. If Ceres Solver is not installed on your system, the reconstruction funcionality will be disabled.


Ceres installation (Linux)
--------------------------

Here are the instructions on how to install the Ceres Solver on your machine (Linux).

    sudo apt-get install libceres-dev


Run
---

There are some samples provided in order to show the reconstruction functionality use.

**recon2v.cpp**

This program shows the two view reconstruction capabilities in the OpenCV Structure From Motion (SFM) module. It uses the following data from the VGG datasets at "reconv2 _pts.txt" where the first line has the number of points and each subsequent line has entries for matched points as: x1 y1 x2 y2. Finally, the script reconstruct the given set of correspondences and show the result using the OpenCV 3d visualizer (viz).

<p align="center">
  <img src="data/recon2v.jpg">
</p>

**simple_reconstruction.cpp**

**nView_reconstruction.cpp**
