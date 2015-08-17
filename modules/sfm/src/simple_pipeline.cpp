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

#include <opencv2/core/eigen.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/sfm/simple_pipeline.hpp>

#include "libmv/correspondence/feature.h"
#include "libmv/correspondence/feature_matching.h"
#include "libmv/correspondence/matches.h"
#include "libmv/correspondence/nRobustViewMatching.h"

#include "libmv/simple_pipeline/pipeline.h"
#include "libmv/simple_pipeline/camera_intrinsics.h"
#include "libmv/simple_pipeline/bundle.h"
#include "libmv/simple_pipeline/initialize_reconstruction.h"
#include "libmv/simple_pipeline/reconstruction_scale.h"
//#include "libmv/simple_pipeline/uncalibrated_reconstructor.h"
#include "libmv/simple_pipeline/tracks.h"

using namespace cv;
using namespace std;
using namespace libmv;

namespace cv
{
namespace sfm
{

enum {
  LIBMV_DISTORTION_MODEL_POLYNOMIAL = 0,
  LIBMV_DISTORTION_MODEL_DIVISION = 1,
};

typedef struct libmv_CameraIntrinsicsOptions {
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

typedef struct libmv_ReconstructionOptions {
  int select_keyframes;
  int keyframe1, keyframe2;
  int refine_intrinsics;
} libmv_ReconstructionOptions;

struct libmv_Reconstruction {
  EuclideanReconstruction reconstruction;

  /* Used for per-track average error calculation after reconstruction */
  Tracks tracks;
  CameraIntrinsics *intrinsics;

  double error;
  bool is_valid;
};


// Based on the 'selectTwoKeyframesBasedOnGRICAndVariance()' function from 'libmv_capi' (blender API)

bool selectTwoKeyframesBasedOnGRICAndVariance(
    Tracks& tracks,
    Tracks& normalized_tracks,
    CameraIntrinsics& camera_intrinsics,
    int& keyframe1,
    int& keyframe2) {

  // TODO: implement me (check blender api)

  return true;
}


// Based on the 'libmv_cameraIntrinsicsFillFromOptions()' function from 'libmv_capi' (blender API)

static void libmv_cameraIntrinsicsFillFromOptions(
    const libmv_CameraIntrinsicsOptions* camera_intrinsics_options,
    CameraIntrinsics* camera_intrinsics) {
  camera_intrinsics->SetFocalLength(camera_intrinsics_options->focal_length,
                                    camera_intrinsics_options->focal_length);

  camera_intrinsics->SetPrincipalPoint(
      camera_intrinsics_options->principal_point_x,
      camera_intrinsics_options->principal_point_y);

  camera_intrinsics->SetImageSize(camera_intrinsics_options->image_width,
      camera_intrinsics_options->image_height);

  switch (camera_intrinsics_options->distortion_model) {
    case LIBMV_DISTORTION_MODEL_POLYNOMIAL:
      {
        PolynomialCameraIntrinsics *polynomial_intrinsics =
          static_cast<PolynomialCameraIntrinsics*>(camera_intrinsics);

        polynomial_intrinsics->SetRadialDistortion(
            camera_intrinsics_options->polynomial_k1,
            camera_intrinsics_options->polynomial_k2,
            camera_intrinsics_options->polynomial_k3);

        break;
      }

    case LIBMV_DISTORTION_MODEL_DIVISION:
      {
        DivisionCameraIntrinsics *division_intrinsics =
          static_cast<DivisionCameraIntrinsics*>(camera_intrinsics);

        division_intrinsics->SetDistortion(
            camera_intrinsics_options->division_k1,
            camera_intrinsics_options->division_k2);
        break;
      }

    default:
      assert(!"Unknown distortion model");
  }
}


// Based on the 'libmv_cameraIntrinsicsCreateFromOptions()' function from 'libmv_capi' (blender API)

CameraIntrinsics* libmv_cameraIntrinsicsCreateFromOptions(
    const libmv_CameraIntrinsicsOptions* camera_intrinsics_options) {
  CameraIntrinsics *camera_intrinsics = NULL;
  switch (camera_intrinsics_options->distortion_model) {
    case LIBMV_DISTORTION_MODEL_POLYNOMIAL:
      //camera_intrinsics = LIBMV_OBJECT_NEW(PolynomialCameraIntrinsics);
      camera_intrinsics = new PolynomialCameraIntrinsics();
      break;
    case LIBMV_DISTORTION_MODEL_DIVISION:
      //camera_intrinsics = LIBMV_OBJECT_NEW(DivisionCameraIntrinsics);
      camera_intrinsics = new DivisionCameraIntrinsics();
      break;
    default:
      assert(!"Unknown distortion model");
  }
  libmv_cameraIntrinsicsFillFromOptions(camera_intrinsics_options,
                                        camera_intrinsics);
  return camera_intrinsics;
}


// Based on the 'libmv_getNormalizedTracks()' function from 'libmv_capi' (blender API)

void
libmv_getNormalizedTracks(const libmv::Tracks &tracks,
                          const libmv::CameraIntrinsics &camera_intrinsics,
                          libmv::Tracks *normalized_tracks) {
  libmv::vector<libmv::Marker> markers = tracks.AllMarkers();
  for (int i = 0; i < markers.size(); ++i) {
    libmv::Marker &marker = markers[i];
    camera_intrinsics.InvertIntrinsics(marker.x, marker.y,
                                       &marker.x, &marker.y);
    normalized_tracks->Insert(marker.image,
                              marker.track,
                              marker.x, marker.y,
                              marker.weight);
  }
}


// Based on the 'libmv_solveRefineIntrinsics()' function from 'libmv_capi' (blender API)

void libmv_solveRefineIntrinsics(
    const Tracks &tracks,
    const int refine_intrinsics,
    const int bundle_constraints,
    /*reconstruct_progress_update_cb progress_update_callback,
    void* callback_customdata,*/
    EuclideanReconstruction* reconstruction,
    CameraIntrinsics* intrinsics) {
  /* only a few combinations are supported but trust the caller/ */
  int bundle_intrinsics = 0;

  if (refine_intrinsics & SFM_REFINE_FOCAL_LENGTH) {
    bundle_intrinsics |= libmv::BUNDLE_FOCAL_LENGTH;
  }
  if (refine_intrinsics & SFM_REFINE_PRINCIPAL_POINT) {
    bundle_intrinsics |= libmv::BUNDLE_PRINCIPAL_POINT;
  }
  if (refine_intrinsics & SFM_REFINE_RADIAL_DISTORTION_K1) {
    bundle_intrinsics |= libmv::BUNDLE_RADIAL_K1;
  }
  if (refine_intrinsics & SFM_REFINE_RADIAL_DISTORTION_K2) {
    bundle_intrinsics |= libmv::BUNDLE_RADIAL_K2;
  }

  //progress_update_callback(callback_customdata, 1.0, "Refining solution");

  EuclideanBundleCommonIntrinsics(tracks,
                                  bundle_intrinsics,
                                  bundle_constraints,
                                  reconstruction,
                                  intrinsics);
}


// Based on the 'finishReconstruction()' function from 'libmv_capi' (blender API)

void finishReconstruction(
    const Tracks &tracks,
    const CameraIntrinsics &camera_intrinsics,
    libmv_Reconstruction *libmv_reconstruction/*,
    reconstruct_progress_update_cb progress_update_callback,
    void *callback_customdata*/) {
  EuclideanReconstruction &reconstruction =
    libmv_reconstruction->reconstruction;

  /* Reprojection error calculation. */
  //progress_update_callback(callback_customdata, 1.0, "Finishing solution");
  libmv_reconstruction->tracks = tracks;
  libmv_reconstruction->error = EuclideanReprojectionError(tracks,
                                                           reconstruction,
                                                           camera_intrinsics);
}


// Based on the 'libmv_solveReconstruction()' function from 'libmv_capi' (blender API)

libmv_Reconstruction *libmv_solveReconstruction(
    const Tracks &libmv_tracks,
    const libmv_CameraIntrinsicsOptions* libmv_camera_intrinsics_options,
    libmv_ReconstructionOptions* libmv_reconstruction_options,
    libmv_Reconstruction *libmv_reconstruction)
{

  Tracks tracks = libmv_tracks;
  EuclideanReconstruction &reconstruction =
    libmv_reconstruction->reconstruction;

//  ReconstructUpdateCallback update_callback =
//    ReconstructUpdateCallback(progress_update_callback,
//                              callback_customdata);

  /* Retrieve reconstruction options from C-API to libmv API. */
  CameraIntrinsics *camera_intrinsics;
  camera_intrinsics = libmv_reconstruction->intrinsics =
    libmv_cameraIntrinsicsCreateFromOptions(libmv_camera_intrinsics_options);

  /* Invert the camera intrinsics/ */
  Tracks normalized_tracks;
  libmv_getNormalizedTracks(tracks, *camera_intrinsics, &normalized_tracks);

  /* keyframe selection. */
  int keyframe1 = libmv_reconstruction_options->keyframe1,
      keyframe2 = libmv_reconstruction_options->keyframe2;

  if (libmv_reconstruction_options->select_keyframes) {
    LG << "Using automatic keyframe selection";

    //update_callback.invoke(0, "Selecting keyframes");

    selectTwoKeyframesBasedOnGRICAndVariance(tracks,
                                             normalized_tracks,
                                             *camera_intrinsics,
                                             keyframe1,
                                             keyframe2);

    /* so keyframes in the interface would be updated */
    libmv_reconstruction_options->keyframe1 = keyframe1;
    libmv_reconstruction_options->keyframe2 = keyframe2;
  }

  /* Actual reconstruction. */
  LG << "frames to init from: " << keyframe1 << " " << keyframe2;

  libmv::vector<Marker> keyframe_markers =
    normalized_tracks.MarkersForTracksInBothImages(keyframe1, keyframe2);

  LG << "number of markers for init: " << keyframe_markers.size();

  if (keyframe_markers.size() < 8) {
    LG << "No enough markers to initialize from";
    libmv_reconstruction->is_valid = false;
    return libmv_reconstruction;
  }

  //update_callback.invoke(0, "Initial reconstruction");

  //libmv::EuclideanReconstructTwoFrames(keyframe_markers, &reconstruction);
  //libmv::EuclideanBundle(normalized_tracks, &reconstruction);
  //libmv::EuclideanCompleteReconstruction(normalized_tracks,
  //                                       &reconstruction,
  //                                       &update_callback);

  libmv::EuclideanReconstructTwoFrames(keyframe_markers, &reconstruction);
  libmv::EuclideanBundle(normalized_tracks, &reconstruction);
  libmv::EuclideanCompleteReconstruction(libmv::ReconstructionOptions(),
                                         normalized_tracks,
                                         &reconstruction);

  /* Refinement/ */
  if (libmv_reconstruction_options->refine_intrinsics) {
//    libmv_solveRefineIntrinsics(
//                                tracks,
//                                libmv_reconstruction_options->refine_intrinsics,
//                                libmv::BUNDLE_NO_CONSTRAINTS,
//                                progress_update_callback,
//                                callback_customdata,
//                                &reconstruction,
//                                camera_intrinsics);

    libmv_solveRefineIntrinsics(
                                tracks,
                                libmv_reconstruction_options->refine_intrinsics,
                                libmv::BUNDLE_NO_CONSTRAINTS,
                                &reconstruction,
                                camera_intrinsics);
  }

  /* Set reconstruction scale to unity. */
  EuclideanScaleToUnity(&reconstruction);

  /* Finish reconstruction. */
//  finishReconstruction(tracks,
//                       *camera_intrinsics,
//                       libmv_reconstruction,
//                       progress_update_callback,
//                       callback_customdata);
  finishReconstruction(tracks,
                       *camera_intrinsics,
                       libmv_reconstruction);

  libmv_reconstruction->is_valid = true;
  return (libmv_Reconstruction *) libmv_reconstruction;
}


void
parser_2D_tracks( const std::vector<Mat> &points2d, libmv::Tracks &tracks )
{
  const int nframes = static_cast<int>(points2d.size());

  for (int frame = 1; frame <= nframes; ++frame)
  {
    const int ntracks = points2d[frame-1].cols;

    for (int track = 1; track <= ntracks; ++track)
    {
      const Vec2d track_pt = points2d[frame-1].col(track-1);
      if ( ( track_pt[0] != 0 && track_pt[1] != 0 ) &&
           ( track_pt[0] != -1 && track_pt[1] != -1 ) )
      {
        //cout << frame << " " << track << " " << track_pt << endl;
        tracks.Insert(frame, track, track_pt[0], track_pt[1]);
      }

    }

  }
}


void
parser_2D_tracks( const libmv::Matches &matches, libmv::Tracks &tracks )
{
  std::set<Matches::ImageID>::const_iterator iter_image =
      matches.get_images().begin();

  bool is_first_time = true;

  for (; iter_image != matches.get_images().end(); ++iter_image) {
    // Exports points
    Matches::Features<PointFeature> pfeatures =
        matches.InImage<PointFeature>(*iter_image);

    while(pfeatures) {

      double x = pfeatures.feature()->x(),
             y = pfeatures.feature()->y();

      // valid marker
      if ( x > 0 && y > 0 )
      {
          tracks.Insert(*iter_image+1, pfeatures.track()+1, x, y);

          if ( is_first_time )
              is_first_time = false;
      }

      // lost track
      else if ( x < 0 && y < 0 )
      {
          is_first_time = true;
      }

      pfeatures.operator++();
    }
  }
}


template <class T>
void
libmv_solveReconstructionImpl( const std::vector<std::string> &images,
                               int keyframe1, int keyframe2,
                               double focal_length,
                               double principal_x, double principal_y,
                               double k1, double k2, double k3,
                               T &libmv_reconstruction,
                               int refine_intrinsics)
{
  Ptr<Feature2D> edetector = ORB::create(10000);
  Ptr<Feature2D> edescriber = xfeatures2d::DAISY::create();
  //Ptr<Feature2D> edescriber = xfeatures2d::LATCH::create(64, true, 4);

  cout << "Initialize nViewMatcher ... ";
  libmv::correspondence::nRobustViewMatching nViewMatcher(edetector, edescriber);

  cout << "OK" << endl << "Performing Cross Matching ... ";
  nViewMatcher.computeCrossMatch(images); cout << "OK" << endl;

  // Building tracks
  libmv::Tracks tracks;
  libmv::Matches matches = nViewMatcher.getMatches();
  parser_2D_tracks( matches, tracks );

  // Perform reconstruction
  libmv_solveReconstruction( tracks, keyframe1, keyframe2,
                             focal_length, principal_x, principal_y, k1, k2, k3,
                             libmv_reconstruction, refine_intrinsics );
}

template <class T>
class SFMLibmvReconstructionImpl : public T
{
public:

  virtual void run(const std::vector<Mat> &points2d, int keyframe1, int keyframe2, double focal_length,
                   double principal_x, double principal_y, double k1, double k2, double k3, int refine_intrinsics=0)
  {
    // Parse 2d points to Tracks
    Tracks tracks;
    parser_2D_tracks(points2d, tracks);

    // Initialize camera options
    libmv_camera_intrinsics_options_.distortion_model = LIBMV_DISTORTION_MODEL_POLYNOMIAL;
    libmv_camera_intrinsics_options_.image_width = 2*principal_x;
    libmv_camera_intrinsics_options_.image_height = 2*principal_y;
    libmv_camera_intrinsics_options_.focal_length = focal_length;
    libmv_camera_intrinsics_options_.principal_point_x = principal_x;
    libmv_camera_intrinsics_options_.principal_point_y = principal_y;
    libmv_camera_intrinsics_options_.polynomial_k1 = k1;
    libmv_camera_intrinsics_options_.polynomial_k2 = k2;
    libmv_camera_intrinsics_options_.polynomial_k3 = k3;

    // Initialize reconstruction options
    libmv_reconstruction_options_.select_keyframes = 0;
    libmv_reconstruction_options_.keyframe1 = keyframe1;
    libmv_reconstruction_options_.keyframe2 = keyframe2;
    libmv_reconstruction_options_.refine_intrinsics = 1;

    // Perform reconstruction
    libmv_Reconstruction *libmv_reconstruction =
      libmv_solveReconstruction(tracks,
                                &libmv_camera_intrinsics_options_,
                                &libmv_reconstruction_options_,
                                &libmv_reconstruction_);

    if ( libmv_reconstruction->is_valid )
      libmv_reconstruction_ = *libmv_reconstruction;
  }

  virtual void run(const std::vector <std::string> &images, int keyframe1, int keyframe2, double focal_length,
                   double principal_x, double principal_y, double k1, double k2, double k3, int refine_intrinsics=0)
  {
    //libmv_solveReconstructionImpl(images, keyframe1, keyframe2, focal_length, principal_x, principal_y, k1, k2, k3,
    //                              libmv_reconstruction_, refine_intrinsics);
  }

  virtual double getError() { return libmv_reconstruction_.error; }

  virtual Mat getPoints()
  {
    const size_t n_points =
      libmv_reconstruction_.reconstruction.AllPoints().size();

    Mat points3d(3, n_points, CV_64F);

    for ( size_t i = 0; i < n_points; ++i )
      for ( int j = 0; j < 3; ++j )
        points3d.at<double>(j, i) =
          libmv_reconstruction_.reconstruction.AllPoints()[i].X[j];

    return points3d;
  }

  virtual cv::Mat getIntrinsics()
  {
    Mat K;
    eigen2cv(libmv_reconstruction_.intrinsics->K(), K);
    return K;
  }

  virtual std::vector<std::pair<Matx33d,Vec3d> > getCameras()
  {
    const size_t n_views =
      libmv_reconstruction_.reconstruction.AllCameras().size();

    Matx33d R;
    Vec3d t;
    std::vector<std::pair<Matx33d,Vec3d> > cameras;

    for(size_t i = 0; i < n_views; ++i)
    {
      eigen2cv(libmv_reconstruction_.reconstruction.AllCameras()[i].R, R);
      eigen2cv(libmv_reconstruction_.reconstruction.AllCameras()[i].t, t);
      cameras.push_back(std::make_pair(R,t));
    }
    return cameras;
  }

private:
  libmv_Reconstruction libmv_reconstruction_;
  libmv_ReconstructionOptions libmv_reconstruction_options_;
  libmv_CameraIntrinsicsOptions libmv_camera_intrinsics_options_;
};


Ptr<SFMLibmvEuclideanReconstruction> SFMLibmvEuclideanReconstruction::create()
{
  return makePtr<SFMLibmvReconstructionImpl<SFMLibmvEuclideanReconstruction> >();
}

} /* namespace cv */
} /* namespace sfm */

#endif

/* End of file. */