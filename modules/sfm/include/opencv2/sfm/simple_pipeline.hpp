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

enum {
  LIBMV_DISTORTION_MODEL_POLYNOMIAL = 0,
  LIBMV_DISTORTION_MODEL_DIVISION = 1,
};

typedef struct libmv_CameraIntrinsicsOptions {
  libmv_CameraIntrinsicsOptions(const int _distortion_model=0,
                                const double _focal_length=0,
                                const double _principal_point_x=0,
                                const double _principal_point_y=0,
                                const double _polynomial_k1=0,
                                const double _polynomial_k2=0,
                                const double _polynomial_k3=0,
                                const double _polynomial_p1=0,
                                const double _polynomial_p2=0)
    : distortion_model(_distortion_model),
      image_width(2*_principal_point_x),
      image_height(2*_principal_point_y),
      focal_length(_focal_length),
      principal_point_x(_principal_point_x),
      principal_point_y(_principal_point_y),
      polynomial_k1(_polynomial_k1),
      polynomial_k2(_polynomial_k2),
      polynomial_k3(_polynomial_k3),
      division_k1(0),
      division_k2(0)
  {
    if ( _distortion_model == LIBMV_DISTORTION_MODEL_DIVISION )
    {
      division_k1 = _polynomial_k1;
      division_k2 = _polynomial_k2;
    }
  }

  // Common settings of all distortion models.
  int distortion_model;
  int image_width, image_height;
  double focal_length;
  double principal_point_x, principal_point_y;

  // Radial distortion model.
  double polynomial_k1, polynomial_k2, polynomial_k3;
  double polynomial_p1, polynomial_p2;

  // Division distortion model.
  double division_k1, division_k2;
} libmv_CameraIntrinsicsOptions;


enum { SFM_REFINE_FOCAL_LENGTH         = 1,  // libmv::BUNDLE_FOCAL_LENGTH
       SFM_REFINE_PRINCIPAL_POINT      = 2,  // libmv::BUNDLE_PRINCIPAL_POINT
       SFM_REFINE_RADIAL_DISTORTION_K1 = 4,  // libmv::BUNDLE_RADIAL_K1
       SFM_REFINE_RADIAL_DISTORTION_K2 = 8,  // libmv::BUNDLE_RADIAL_K2
};

typedef struct libmv_ReconstructionOptions {
  libmv_ReconstructionOptions(const int _keyframe1=1,
                              const int _keyframe2=2,
                              const int _refine_intrinsics=1,
                              const int _select_keyframes=1)
    : keyframe1(_keyframe1), keyframe2(_keyframe2),
      refine_intrinsics(_refine_intrinsics),
      select_keyframes(_select_keyframes) {}
  int keyframe1, keyframe2;
  int refine_intrinsics;
  int select_keyframes;
} libmv_ReconstructionOptions;


class CV_EXPORTS SFMLibmvReconstruction
{
public:
  virtual ~SFMLibmvReconstruction() {};

  virtual void run(const std::vector<cv::Mat> &points2d) = 0;
  virtual void run(const std::vector<cv::Mat> &points2d, cv::Matx33d &K, std::vector<cv::Matx33d> &Rs,
                   std::vector<cv::Vec3d> &Ts, cv::Mat &points3d) = 0;

  virtual void run(const std::vector <std::string> &images) = 0;
  virtual void run(const std::vector <std::string> &images, cv::Matx33d &K, std::vector<cv::Matx33d> &Rs,
                   std::vector<cv::Vec3d> &Ts, cv::Mat &points3d) = 0;

  virtual double getError() const = 0;
  virtual cv::Mat getPoints() const = 0;
  virtual cv::Mat getIntrinsics() const = 0;
  virtual void getCameras(std::vector<cv::Matx33d> &Rs, std::vector<cv::Vec3d> &Ts) = 0;

  virtual void
  setReconstructionOptions(const libmv_ReconstructionOptions &libmv_reconstruction_options) = 0;

  virtual void
  setCameraIntrinsicOptions(const libmv_CameraIntrinsicsOptions &libmv_camera_intrinsics_options) = 0;
};


class CV_EXPORTS SFMLibmvEuclideanReconstruction : public SFMLibmvReconstruction
{
public:
  virtual void run(const std::vector<cv::Mat> &points2d) = 0;
  virtual void run(const std::vector<cv::Mat> &points2d, cv::Matx33d &K, std::vector<cv::Matx33d> &Rs,
                   std::vector<cv::Vec3d> &Ts, cv::Mat &points3d) = 0;

  virtual void run(const std::vector <std::string> &images) = 0;
  virtual void run(const std::vector <std::string> &images, cv::Matx33d &K, std::vector<cv::Matx33d> &Rs,
                   std::vector<cv::Vec3d> &Ts, cv::Mat &points3d) = 0;

  virtual double getError() const = 0;
  virtual cv::Mat getPoints() const = 0;
  virtual cv::Mat getIntrinsics() const = 0;
  virtual void getCameras(std::vector<cv::Matx33d> &Rs, std::vector<cv::Vec3d> &Ts) = 0;

  virtual void
  setReconstructionOptions(const libmv_ReconstructionOptions &libmv_reconstruction_options) = 0;

  virtual void
  setCameraIntrinsicOptions(const libmv_CameraIntrinsicsOptions &libmv_camera_intrinsics_options) = 0;

  /** @brief Creates an instance of the SFMLibmvEuclideanReconstruction class. Initializes Libmv. */
  static Ptr<SFMLibmvEuclideanReconstruction>
  create(const libmv_CameraIntrinsicsOptions &camera_instrinsic_options=libmv_CameraIntrinsicsOptions(),
         const libmv_ReconstructionOptions &reconstruction_options=libmv_ReconstructionOptions());
};

} /* namespace cv */
} /* namespace sfm */

#endif

/* End of file. */
