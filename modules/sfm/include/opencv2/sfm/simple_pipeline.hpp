/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef __OPENCV_SFM_SIMPLE_PIPELINE_HPP__
#define __OPENCV_SFM_SIMPLE_PIPELINE_HPP__

#include <opencv2/core.hpp>

namespace cv
{
namespace sfm
{

enum { SFM_BUNDLE_FOCAL_LENGTH    = 1,  // libmv::BUNDLE_FOCAL_LENGTH
       SFM_BUNDLE_PRINCIPAL_POINT = 2,  // libmv::BUNDLE_PRINCIPAL_POINT
       SFM_BUNDLE_RADIAL_K1       = 4,  // libmv::BUNDLE_RADIAL_K1
       SFM_BUNDLE_RADIAL_K2       = 8,  // libmv::BUNDLE_RADIAL_K2
       SFM_BUNDLE_TANGENTIAL      = 48, // libmv::BUNDLE_TANGENTIAL
};


class CV_EXPORTS SFMLibmvReconstruction
{
public:
  virtual ~SFMLibmvReconstruction() {};

  virtual void run(const std::vector<cv::Mat> &points2d, int keyframe1, int keyframe2, double focal_length,
                 double principal_x, double principal_y, double k1, double k2, double k3, int refine_intrinsics=0) = 0;

  virtual void run(const std::vector <std::string> &images, int keyframe1, int keyframe2, double focal_length,
                   double principal_x, double principal_y, double k1, double k2, double k3, int refine_intrinsics=0) = 0;

  virtual double getError() = 0;
};


class CV_EXPORTS SFMLibmvEuclideanReconstruction : public SFMLibmvReconstruction
{
public:
  virtual void run(const std::vector<cv::Mat> &points2d, int keyframe1, int keyframe2, double focal_length,
                   double principal_x, double principal_y, double k1, double k2, double k3, int refine_intrinsics=0) = 0;

  virtual void run(const std::vector <std::string> &images, int keyframe1, int keyframe2, double focal_length,
                   double principal_x, double principal_y, double k1, double k2, double k3, int refine_intrinsics=0) = 0;

  virtual double getError() = 0;

  /** @brief Creates an instance of the SFMLibmvEuclideanReconstruction class. Initializes Libmv. */
  static Ptr<SFMLibmvEuclideanReconstruction> create();
};


class CV_EXPORTS SFMLibmvProjectiveReconstruction : public SFMLibmvReconstruction
{
public:
  virtual void run(const std::vector<cv::Mat> &points2d, int keyframe1, int keyframe2, double focal_length,
                   double principal_x, double principal_y, double k1, double k2, double k3, int refine_intrinsics=0) = 0;

  virtual void run(const std::vector <std::string> &images, int keyframe1, int keyframe2, double focal_length,
                   double principal_x, double principal_y, double k1, double k2, double k3, int refine_intrinsics=0) = 0;

  virtual double getError() = 0;

  /** @brief Creates an instance of the SFMLibmvProjectiveReconstruction class. Initializes Libmv. */
  static Ptr<SFMLibmvProjectiveReconstruction> create();
};


class CV_EXPORTS SFMLibmvUncalibratedReconstruction : public SFMLibmvReconstruction
{
public:
  virtual void run(const std::vector<cv::Mat> &points2d, int keyframe1, int keyframe2, double focal_length,
                   double principal_x, double principal_y, double k1, double k2, double k3, int refine_intrinsics=0) = 0;

  virtual void run(const std::vector <std::string> &images, int keyframe1, int keyframe2, double focal_length,
                   double principal_x, double principal_y, double k1, double k2, double k3, int refine_intrinsics=0) = 0;

  virtual double getError() = 0;

  /** @brief Creates an instance of the SFMLibmvUncalibratedReconstruction class. Initializes Libmv. */
  static Ptr<SFMLibmvUncalibratedReconstruction> create();
};

} /* namespace cv */
} /* namespace sfm */

#endif

/* End of file. */
