#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <iostream>

using namespace std;
using namespace cv;

static void help() {
  cout
      << "\n----------------------------------------------------------------------------------\n"
      << " This program shows the multiview reconstruction capabilities in the \n"
      << " OpenCV Structure From Motion (SFM) module.\n"
      << " It generates a scene with synthetic data to then reconstruct it \n"
      << " from the 2D correspondences.\n"
      << " Usage:\n"
      << "       example_sfm_simple_reconstruction [<nCameras>] [<nPoints>] \n"
      << " where nCameras is the number of cameras to generate in the scene (default: 20) \n"
      << "       nPoints is the number of 3D points to generate in the scene (default: 500) \n"
      << "----------------------------------------------------------------------------------\n\n"
      << endl;
}

void
generateScene(const size_t n_views, const size_t n_points, Matx33d & K, vector<Matx33d> & R,
              vector<Vec3d> & t, vector<Matx34d> & P, Mat_<double> & points3d,
              vector<Mat_<double> > & points2d );

void
parser_2D_tracks( const vector<Mat_<double> > &points2d, libmv::Tracks &tracks );

int main(int argc, char* argv[])
{
  int nviews = 20;
  int npoints = 500;

  // read input parameters
  if ( argc > 1 )
  {
    if ( string(argv[1]).compare("-h") == 0 ||
         string(argv[1]).compare("--help") == 0 )
    {
      help();
      exit(0);
    }
    else
      nviews = atoi(argv[1]);

    if ( argc > 2 )
      npoints = atoi(argv[2]);
  }

  vector< Mat_<double> > points2d;
  vector< Matx33d > Rs;
  vector< Vec3d > ts;
  vector< Matx34d > Ps;
  Matx33d K;
  Mat_<double> points3d;

  /// Generate ground truth scene
  generateScene(nviews, npoints, K, Rs, ts, Ps, points3d, points2d);

  /// Reconstruct the scene using the 2d correspondences
  Matx33d K_ = K;
  vector<Mat> Rs_est, ts_est;
  Mat_<double> points3d_estimated;
  const bool is_projective = true;
  const bool has_outliers = false;
  const bool is_sequence = true;
  reconstruct(points2d, Rs_est, ts_est, K_, points3d_estimated, is_projective, has_outliers, is_sequence);


  // Print output

  cout << "\n----------------------------\n" << endl;
  cout << "Generated Scene: " << endl;
  cout << "============================" << endl;
  cout << "Generated 3D points: " << npoints << endl;
  cout << "Generated cameras: " << nviews << endl;
  cout << "Initial intrinsics: " << endl << K << endl << endl;
  cout << "Reconstruction: " << endl;
  cout << "============================" << endl;
  cout << "Estimated 3D points: " << points3d_estimated.cols << endl;
  cout << "Estimated cameras: " << Rs_est.size() << endl;
  cout << "Refined intrinsics: " << endl << K_ << endl << endl;


  /// Compute the orientation and the scale between pointclouds
  Matx33d R_rel;
  Vec3d t_rel;
  double s_rel;
  //computeOrientation(points3d, points3d_estimated, R_rel, t_rel, s_rel);


  cout << "3D Visualization: " << endl;
  cout << "============================" << endl;

  /// Create 3D windows

  viz::Viz3d window_gt("Ground Truth Coordinate Frame");
             window_gt.setWindowSize(Size(500,500));
             window_gt.setWindowPosition(Point(150,150));

  viz::Viz3d window_est("Estimation Coordinate Frame");
             window_est.setWindowSize(Size(500,500));
             window_est.setWindowPosition(Point(750,150));


  /// Add coordinate axes
  window_gt.showWidget("Ground Truth Coordinate Widget", viz::WCoordinateSystem());
  window_est.showWidget("Estimation Coordinate Widget", viz::WCoordinateSystem());

  // Create the pointcloud
  cout << "Recovering points  ... ";

  vector<Vec3f> point_cloud;
  for (int i = 0; i < points3d.cols; ++i) {
    // recover ground truth points3d
    Vec3f point3d((float) points3d(0, i),
                  (float) points3d(1, i),
                  (float) points3d(2, i));
    point_cloud.push_back(point3d);
  }

  vector<Vec3f> point_cloud_est;
  for (int i = 0; i < points3d_estimated.cols; ++i) {

    // recover estimated points3d
    Vec3f point3d_est((float) points3d_estimated(0, i),
                      (float) points3d_estimated(1, i),
                      (float) points3d_estimated(2, i));
    point_cloud_est.push_back(point3d_est);
  }

  cout << "OK" << endl;


  /// Recovering cameras
  cout << "Recovering cameras ... ";

  std::vector<Affine3d> path_gt;
  for (int i = 0, j = 1; i < nviews; ++i, ++j)
    path_gt.push_back(Affine3d(Rs[i],ts[i]));
  path_gt.push_back(Affine3d(Rs[0],ts[0]));

  std::vector<Affine3d> path_est;
  for (size_t i = 0, j = 1; i < Rs_est.size(); ++i, ++j)
    path_est.push_back(Affine3d(Rs_est[i],ts_est[i]));
  path_est.push_back(Affine3d(Rs_est[0],ts_est[0]));

  cout << "OK" << endl;


  /// Add the pointcloud
  if ( !point_cloud.empty() && !point_cloud_est.empty() )
  {
    cout << "Rendering points   ... ";

    viz::WCloud cloud_widget(point_cloud, viz::Color::green());
    viz::WCloud cloud_est_widget(point_cloud_est, viz::Color::red());
    window_gt.showWidget("point_cloud", cloud_widget);
    window_est.showWidget("point_cloud_est", cloud_est_widget);

    cout << "OK" << endl;
  }
  else
  {
    cout << "Cannot render points: Empty pointcloud" << endl;
  }


  /// Add cameras
  if ( !path_gt.empty() && !path_est.empty() )
  {
    cout << "Rendering Cameras  ... ";

    window_gt.showWidget("cameras_frames_and_lines_gt", viz::WTrajectory(path_gt, viz::WTrajectory::BOTH, 0.2, viz::Color::green()));
    window_gt.showWidget("cameras_frustums_gt", viz::WTrajectoryFrustums(path_gt, K, 0.3, viz::Color::yellow()));
    window_est.showWidget("cameras_frames_and_lines_est", viz::WTrajectory(path_est, viz::WTrajectory::BOTH, 0.2, viz::Color::green()));
    window_est.showWidget("cameras_frustums_est", viz::WTrajectoryFrustums(path_est, K, 0.3, viz::Color::yellow()));

    cout << "OK" << endl;
  }
  else
  {
    cout << "Cannot render the cameras: Empty path" << endl;
  }

  /// Wait for key 'q' to close the window
  cout << endl << "Press 'q' to close each windows ... " << endl;

  window_gt.spinOnce(); window_est.spinOnce();
  window_gt.spin(); window_est.spin();

  cout
       << "\n-----------------------------------------------------------\n"
       << "Closing Application.\n"
       << "Now you know a little bit more about Structure from Motion.\n"
       << "Thanks to follow this tutorial. The OpenCV team.\n"
       << "-----------------------------------------------------------\n"
       << endl;

  return 0;
}


void
generateScene(const size_t n_views, const size_t n_points, Matx33d & K, vector<Matx33d> & R,
              vector<Vec3d> & t, vector<Matx34d> & P, Mat_<double> & points3d,
              vector<Mat_<double> > & points2d)
{
  R.resize(n_views);
  t.resize(n_views);

  cv::RNG rng;

  const float size_scene = 10.0f;
  const float offset_scene = 0.0f;

  // Generate a bunch of random 3d points in a 0, 1 cube
  points3d.create(3, n_points);
  rng.fill(points3d, cv::RNG::UNIFORM, 0, size_scene);

  // Generate random intrinsics
  K = Matx33d(500,   0, 320,
                0, 500, 240,
                0,   0,   1);

  float r = size_scene;
  float cx = r/2.0f + offset_scene;
  float cy = r/2.0f + offset_scene;
  float cz = r/2.0f + offset_scene;
  int num_segments = n_views;
  cx = cy = cz = 0;

  for(int ii = 0; ii < num_segments; ii++)
  {
    float theta = 2.0f * CV_PI * float(ii) / float(num_segments);//get the current angle

    float x = r * cosf(theta);//calculate the x component
    float y = r * sinf(theta);//calculate the y component

    // set camera position
    t[ii] = cv::Vec3d(x + cx, y + cy, cz);//output vertex

    // set rotation around x and y axis
    Vec3d vecx(-CV_PI/2, 0, 0);
    Vec3d vecy(0, -CV_PI/2-theta, 0);
    Vec3d vecz(0, 0, 0);

    Matx33d Rx, Ry, Rz;
    Rodrigues(vecx, Rx);
    Rodrigues(vecy, Ry);
    Rodrigues(vecz, Rz);

    // apply ordered rotations
    R[ii] = Rx * Ry * Rz;
  }

  // Compute projection matrices
  P.resize(n_views);
  for (size_t i = 0; i < n_views; ++i)
  {
    Matx33d K3 = K, R3 = R[i];
    Vec3d t3 = t[i];
    P_From_KRt(K3, R3, t3, P[i]);
    //cout << P[i] << endl;
  }

  // Compute homogeneous 3d points
  Mat_<double> points3d_homogeneous(4, n_points);
  points3d.copyTo(points3d_homogeneous.rowRange(0, 3));
  points3d_homogeneous.row(3).setTo(1);

  // Project those points for every view
  points2d.resize(n_views);
  for (size_t i = 0; i < n_views; ++i)
  {
    Mat_<double> points2d_tmp = Mat(P[i]) * points3d_homogeneous;
    points2d[i].create(2, n_points);
    for (unsigned char j = 0; j < 2; ++j)
      Mat(points2d_tmp.row(j) / points2d_tmp.row(2)).copyTo(points2d[i].row(j));
  }

// TODO: remove a certain number of points per view
// TODO: add a certain number of outliers per view

}