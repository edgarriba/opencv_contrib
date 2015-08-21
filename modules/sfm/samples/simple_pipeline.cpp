#include <opencv2/core.hpp>
#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>

#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::sfm;

static void help() {
  cout
      << "\n------------------------------------------------------------------\n"
      << " This program shows the camera path reconstruction capabilities in the \n"
      << " OpenCV Structure From Motion (SFM) module.\n"
      << " \n"
      << " Usage:\n"
      << "        example_sfm_simple_pipeline <path_to_tracks_file> <f> <cx> <cy>\n"
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


/* Build the following structure data
 *
 *            frame1           frame2           frameN
 *  track1 | (x11,y11) | -> | (x12,y12) | -> | (x1N,y1N) |
 *  track2 | (x21,y11) | -> | (x22,y22) | -> | (x2N,y2N) |
 *  trackN | (xN1,yN1) | -> | (xN2,yN2) | -> | (xNN,yNN) |
 *
 *
 *  In case a marker (x,y) does not appear in a frame its
 *  values will be (-1,-1).
 */

void
parser_2D_tracks(const string &_filename, std::vector<Mat> &points2d )
{
  ifstream myfile(_filename.c_str());

  if (!myfile.is_open())
  {
    cout << "Unable to read file: " << _filename << endl;
    exit(0);

  } else {

    double x, y;
    string line_str;
    Mat nan_mat = Mat(2, 1 , CV_64F, -1);
    int n_frames = 0, n_tracks = 0, track = 0;

    while ( getline(myfile, line_str) )
    {
      istringstream line(line_str);

      if ( track > n_tracks )
      {
        n_tracks = track;

        for (int i = 0; i < n_frames; ++i)
          cv::hconcat(points2d[i], nan_mat, points2d[i]);
      }

      for (int frame = 1; line >> x >> y; ++frame)
      {
        if ( frame > n_frames )
        {
          n_frames = frame;
          points2d.push_back(nan_mat);
        }

        points2d[frame-1].at<double>(0,track) = x;
        points2d[frame-1].at<double>(1,track) = y;
      }

      ++track;
    }

    myfile.close();
  }

}


/* Sample main code
 */

int main(int argc, char** argv)
{
  // Read input parameters

  if ( argc != 5 )
  {
    help();
    exit(0);
  }

  // Read 2D points from text file
  std::vector<Mat> points2d;
  parser_2D_tracks( argv[1], points2d );

  // Set the camera calibration matrix
  const double f  = atof(argv[2]),
               cx = atof(argv[3]), cy = atof(argv[4]);

  Matx33d K = Matx33d( f, 0, cx,
                       0, f, cy,
                       0, 0,  1);

  /// Reconstruct the scene using the 2d correspondences

  vector<Mat> Rs_est, ts_est;
  Mat_<double> points3d_estimated;
  bool is_projective = true, has_outliers = false;
  reconstruct(points2d, Rs_est, ts_est, K, points3d_estimated, is_projective, has_outliers);


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
             window_est.setBackgroundColor(); // black by default

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
  cout << "Rendering Trajectory  ... ";

  /// Wait for key 'q' to close the window
  cout << endl << "Press 'q' to close each windows ... " << endl;

  if ( path_est.size() > 0 )
  {
    // render complete trajectory
    window_est.showWidget("cameras_frames_and_lines_est", viz::WTrajectory(path_est, viz::WTrajectory::PATH, 0.2, viz::Color::green()));

    // animated trajectory
    int idx = 0, forw = -1, n = static_cast<int>(path_est.size());

    while(!window_est.wasStopped())
    {
      viz::WCameraPosition cpw(0.25); // Coordinate axes
      viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599), 0.3, viz::Color::yellow()); // Camera frustum
      window_est.showWidget("CPW", cpw, path_est[idx]);
      window_est.showWidget("CPW_FRUSTUM", cpw_frustum, path_est[idx]);

      // update trajectory index (spring effect)
      forw *= (idx==n || idx==0) ? -1: 1; idx += forw;

      // frame rate 1s
      window_est.spinOnce(1, true);
    }

  }

  return 0;
}
