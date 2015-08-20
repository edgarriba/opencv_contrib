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

#include "precomp.hpp"

#if CERES_FOUND

// Eigen
#include <Eigen/Core>

// OpenCV
#include <opencv2/sfm.hpp>

using namespace cv;
using namespace cv::sfm;
using namespace std;

namespace cv
{
  template<class T>
  void
  reconstruct_(const T &input, OutputArrayOfArrays Rs, OutputArrayOfArrays Ts, InputOutputArray K, OutputArray points3d)
  {
    // Initial reconstruction
    const int keyframe1 = 1, keyframe2 = 2;
    const int select_keyframes = 1; // enable automatic keyframes selection

    // Refinement parameters
    const int refine_intrinsics = SFM_REFINE_FOCAL_LENGTH | SFM_REFINE_PRINCIPAL_POINT | SFM_REFINE_RADIAL_DISTORTION_K1 | SFM_REFINE_RADIAL_DISTORTION_K2;

    // Camera data
    Matx33d Ka = K.getMat();
    const double focal_length = Ka(0,0);
    const double principal_x = Ka(0,2), principal_y = Ka(1,2), k1 = 0, k2 = 0, k3 = 0;

    // Set reconstruction options
    libmv_ReconstructionOptions reconstruction_options(keyframe1, keyframe2, refine_intrinsics, select_keyframes);

    libmv_CameraIntrinsicsOptions camera_instrinsic_options =
          libmv_CameraIntrinsicsOptions(LIBMV_DISTORTION_MODEL_POLYNOMIAL,
                                        focal_length, principal_x, principal_y,
                                        k1, k2, k3);

    Ptr<SFMLibmvReconstruction> reconstruction =
        SFMLibmvEuclideanReconstruction::create(camera_instrinsic_options, reconstruction_options);

    //-- Run reconstruction pipeline
    Matx33d K_est;
    Mat points3d_est;
    vector<Matx33d> Rs_est;
    vector<Vec3d> Ts_est;
    reconstruction->run(input, K_est, Rs_est, Ts_est, points3d_est);

    //-- Create output
    const int nviews_est = Rs_est.size();
    const int depth = Mat(Ka).depth();
    Rs.create(nviews_est, 1, depth);
    Ts.create(nviews_est, 1, depth);

    for (size_t i = 0; i < nviews_est; ++i)
    {
      Mat(Rs_est[i]).copyTo(Rs.getMatRef(i));
      Mat(Ts_est[i]).copyTo(Ts.getMatRef(i));
    }

    // Extract reconstructed points
    points3d.create(3, points3d_est.cols, CV_64F);
    points3d_est.copyTo(points3d);

    // Extract refined intrinsic parameters
    Mat(K_est).copyTo(K.getMat());
  }


  //  Reconstruction function for API
  void
  reconstruct(const InputArrayOfArrays points2d, OutputArrayOfArrays projection_matrices, OutputArray points3d,
    bool is_projective, bool has_outliers, bool is_sequence)
  {
    const int nviews = points2d.total();
    CV_Assert( nviews >= 2 );

    Matx33d F;

    // OpenCV data types
    std::vector<Mat> pts2d;
    points2d.getMatVector(pts2d);
    const int depth = pts2d[0].depth();

    // Projective reconstruction

    if (is_projective)
    {

      // Two view reconstruction

      if (nviews == 2)
      {

        // Get fundamental matrix
        if ( has_outliers )
        {
          double max_error = 0.1;
          std::vector<int> inliers;
          fundamentalFromCorrespondences8PointRobust(pts2d[0], pts2d[1], max_error, F, inliers);
        }
        else
        {
          normalizedEightPointSolver(pts2d[0], pts2d[1], F);
        }

        // Get Projection matrices
        Matx34d P, Pp;
        projectionsFromFundamental(F, P, Pp);
        projection_matrices.create(2, 1, depth);
        Mat(P).copyTo(projection_matrices.getMatRef(0));
        Mat(Pp).copyTo(projection_matrices.getMatRef(1));

        // Triangulate and find 3D points using inliers
        triangulatePoints(points2d, projection_matrices, points3d);
      }

    }


    // Affine reconstruction

    else
    {

      // Two view reconstruction

      if (nviews == 2)
      {

      }
      else
      {

      }

    }

  }


  void
  reconstruct(InputArrayOfArrays points2d, OutputArrayOfArrays Rs, OutputArrayOfArrays Ts, InputOutputArray K,
              OutputArray points3d, bool is_projective, bool has_outliers, bool is_sequence)
  {
    const int nviews = points2d.total();
    CV_Assert( nviews >= 2 );

    std::vector<Mat> pts2d;
    points2d.getMatVector(pts2d);

    const int depth = pts2d[0].depth();

    Matx33d Ka, R;
    Vec3d t;

    if (nviews == 2)
    {
      std::vector < Mat > Ps_estimated;
      reconstruct(points2d, Ps_estimated, points3d, is_projective, has_outliers, is_sequence);

      const int nviews_est = Ps_estimated.size();
      Rs.create(nviews_est, 1, depth);
      Ts.create(nviews_est, 1, depth);

      for(unsigned int i = 0; i < nviews_est; ++i)
      {
        KRt_From_P(Ps_estimated[i], Ka, R, t);
        Mat(R).copyTo(Rs.getMatRef(i));
        Mat(t).copyTo(Ts.getMatRef(i));
      }
      Mat(Ka).copyTo(K.getMat());
    }
    else if (nviews > 2)
    {
      Ka = K.getMat();
      CV_Assert( Ka(0,0) > 0 && Ka(1,1) > 0);

      reconstruct_(pts2d, Rs, Ts, K, points3d);
    }

  }


  void
  reconstruct(const std::vector<string> images, OutputArrayOfArrays projection_matrices, OutputArray points3d, InputOutputArray K)
  {
    Matx33d Ka = K.getMat();
    CV_Assert( Ka(0,0) > 0 && Ka(1,1) > 0);
    CV_Assert( images.size() >= (unsigned)2 );

    std::vector<Mat> Rs, Ts;
    reconstruct(images, Rs, Ts, Ka, points3d, false);

    // From Rs and Ts, extract Ps

    const int nviews = Rs.size();
    const int depth = Mat(Ka).depth();
    projection_matrices.create(nviews, 1, depth);

    Matx34d P;
    for (size_t i = 0; i < nviews; ++i)
    {
      P_From_KRt(Ka, Rs[i], Vec3d(Ts[i]), P);
      Mat(P).copyTo(projection_matrices.getMatRef(i));
    }

    Mat(Ka).copyTo(K.getMat());
  }


  void
  reconstruct(const std::vector<std::string> images, OutputArrayOfArrays Rs, OutputArrayOfArrays Ts,
              InputOutputArray K, OutputArray points3d, bool is_projective)
  {
    Matx33d Ka = K.getMat();
    CV_Assert( Ka(0,0) > 0 && Ka(1,1) > 0);
    CV_Assert( images.size() >= (unsigned)2 );

    if ( is_projective )
    {
//      std::vector < Mat > Ps_estimated;
//      reconstruct(images, Ps_estimated, points3d, Ka);
//
//      const int depth = K.getMat().depth();
//      const unsigned nviews = Ps_estimated.size();
//
//      Rs.create(nviews, 1, depth);
//      Ts.create(nviews, 1, depth);
//
//      Matx33d R; Vec3d t;
//      for(unsigned int i = 0; i < nviews; ++i)
//      {
//        KRt_From_P(Ps_estimated[i], Ka, R, t);
//        Mat(R).copyTo(Rs.getMatRef(i));
//        Mat(t).copyTo(Ts.getMatRef(i));
//      }

    }
    else
    {
      reconstruct_(images, Rs, Ts, K, points3d);
    }

  }

} // namespace cv

#endif /* HAVE_CERES */