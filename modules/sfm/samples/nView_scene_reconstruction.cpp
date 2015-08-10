#include <opencv2/core.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>

#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

static void help() {
  cout
      << "\n------------------------------------------------------------------\n"
      << " This program shows the n view reconstruction capabilities in the \n"
      << " OpenCV Structure From Motion (SFM) module.\n"
      << " \n"
      << " Usage:\n"
      << "        example_sfm_nView_scene_reconstruction <path_to_tracks_file> <f> <cx> <cy>\n"
      << " where: is the tracks file absolute path into your system. \n"
      << " \n"
      << "        The file must have the following format: \n"
      << "        row1 : x1 y1 x2 y2 ... x36 y36 for track 1\n"
      << "        row2 : x1 y1 x2 y2 ... x36 y36 for track 2\n"
      << "        etc\n"
      << " \n"
      << "        i.e. a row gives the 2D measured position of a point as it is tracked\n"
      << "        through frames 1 to 36.  If there is no match found in a view then x\n"
      << "        and y are -1.\n"
      << " \n"
      << "        Each row corresponds to a different point.\n"
      << " \n"
      << "        f  is the focal lenght in pixels. \n"
      << "        cx is the image principal point x coordinates in pixels. \n"
      << "        cy is the image principal point y coordinates in pixels. \n"
      << "------------------------------------------------------------------\n\n"
      << endl;
}

void
parser_2D_tracks(const string &_filename, std::vector < Mat_<double> > &points2d );

int main(int argc, char** argv)
{
  // Read input parameters

  if ( argc != 5 )
  {
    help();
    exit(0);
  }

  // Do projective reconstruction
  bool is_projective = true;

  // Assume noise free
  bool has_outliers = false;

  // Assume the input images are a sequence
  bool is_sequence = true;

  // Read 2D points from text file
  std::vector < Mat_<double> > points2d;
  parser_2D_tracks( argv[1], points2d );

  // Set the camera calibration matrix
  double f = atof(argv[2]), cx = atof(argv[3]), cy = atof(argv[4]);

  Matx33d K = Matx33d(f, 0, cx,
                      0, f, cy,
                      0,  0, 1);

  // Perform reconstruction
  std::vector<cv::Mat> Rs_est, ts_est;
  Mat_<double> points3d_estimated;
  std::vector < cv::Mat > Ps_estimated;
  reconstruct(points2d, Rs_est, ts_est, K, points3d_estimated, is_projective, has_outliers, is_sequence);


  // Print output

  cout << "\n----------------------------\n" << endl;
  cout << "Reconstruction: " << endl;
  cout << "============================" << endl;
  cout << "Estimated 3D points: " << points3d_estimated.cols << endl;
  cout << "Estimated cameras: " << Rs_est.size() << endl;
  cout << "Refined intrinsics: " << endl << K << endl << endl;

  cout << "3D Visualization: " << endl;
  cout << "============================" << endl;


  /// Create 3D windows
  viz::Viz3d window_est("Estimation Coordinate Frame");

  /// Add coordinate axes
  window_est.showWidget("Estimation Coordinate Widget", viz::WCoordinateSystem());

  // Create the pointcloud
  cout << "Recovering points  ... ";

  std::vector<cv::Vec3f> point_cloud_est;
  for (int i = 0; i < points3d_estimated.cols; ++i) {

    // recover estimated points3d
    cv::Vec3f point3d_est((float) points3d_estimated(0, i),
                          (float) points3d_estimated(1, i),
                          (float) points3d_estimated(2, i));
    point_cloud_est.push_back(point3d_est);
  }

  cout << "[DONE]" << endl;


  /// Recovering cameras
  cout << "Recovering cameras ... ";

  std::vector<Affine3d> path_est;
  for (size_t i = 0; i < Rs_est.size(); ++i)
    path_est.push_back(Affine3d(Rs_est[i],ts_est[i]));

  cout << "[DONE]" << endl;


  /// Add the pointcloud
  cout << "Rendering points   ... ";

  if ( point_cloud_est.size() > 0 )
  {
    viz::WCloud cloud_est_widget(point_cloud_est, viz::Color::red());
    window_est.showWidget("point_cloud_est", cloud_est_widget);
  }

  cout << "[DONE]" << endl;


  /// Add cameras
  cout << "Rendering Cameras  ... ";

  if ( path_est.size() > 0 )
  {
    window_est.showWidget("cameras_frames_and_lines_est", viz::WTrajectory(path_est, viz::WTrajectory::BOTH, 0.2, viz::Color::green()));
    window_est.showWidget("cameras_frustums_est", viz::WTrajectoryFrustums(path_est, K, 0.3, viz::Color::yellow()));
  }

  cout << "[DONE]" << endl;


  /// Wait for key 'q' to close the window
  cout << endl << "Press 'q' to close each windows ... " << endl;

  window_est.spin();

  return 0;
}


void
parser_2D_tracks(const string &_filename, std::vector < Mat_<double> > &points2d )
{
  ifstream myfile(_filename.c_str());

  if (!myfile.is_open())
  {
    cout << "Unable to read file: " << _filename << endl;
    exit(0);

  } else {

    double x, y;
    string line_str;
    int n_frames = 0, n_tracks = 0, track = 0;

    while ( getline(myfile, line_str) )
    {
      istringstream line(line_str);

      if ( track > n_tracks )
      {
        n_tracks = track;

        for (int i = 0; i < n_frames; ++i)
          cv::hconcat(points2d[i], Mat_<double>(2,1,-1), points2d[i]);
      }

      for (int frame = 1; line >> x >> y; ++frame)
      {
        if ( frame > n_frames )
        {
          n_frames = frame;
          points2d.push_back(Mat_<double>(2,1,-1));
        }

        points2d[frame-1](0,track) = x;
        points2d[frame-1](1,track) = y;
      }

      ++track;
    }

    myfile.close();
  }

}
