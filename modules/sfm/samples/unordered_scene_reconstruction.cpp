#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

static void help() {
  cout
      << "\n------------------------------------------------------------------------------------\n"
      << " This program shows the multiview reconstruction capabilities in the \n"
      << " OpenCV Structure From Motion (SFM) module.\n"
      << " It reconstruct a scene from a set of unordered 2D images \n"
      << " Usage:\n"
      << "        example_sfm_unordered_scene_reconstruction <path_to_file> <f> <cx> <cy>\n"
      << " where: path_to_file is the file absolute path into your system which contains\n"
      << "        the list of images to use for reconstruction. \n"
      << "        f  is the focal lenght in pixels. \n"
      << "        cx is the image principal point x coordinates in pixels. \n"
      << "        cy is the image principal point y coordinates in pixels. \n"
      << "------------------------------------------------------------------------------------\n\n"
      << endl;
}


int getdir(const string _filename, vector<string> &files)
{
  ifstream myfile(_filename.c_str());
  if (!myfile.is_open()) {
    cout << "Unable to read file: " << _filename << endl;
    exit(0);
  } else {
    string line_str;
    while ( getline(myfile, line_str) )
      files.push_back(line_str);
  }
  return 1;
}


int main(int argc, char* argv[])
{
  // Read input parameters

  if ( argc != 5 )
  {
    help();
    exit(0);
  }

  // Parse the image paths

  vector<string> images_paths;
  getdir( argv[1], images_paths );


  // Build instrinsics

  float f  = atof(argv[2]),
        cx = atof(argv[3]), cy = atof(argv[4]);

  Matx33d K = Matx33d( f, 0, cx,
                       0, f, cy,
                       0, 0,  1);


  /// Reconstruct the scene using the 2d images

  vector<Mat> Rs_est, ts_est;
  Mat_<double> points3d_estimated;
  bool is_projective = true, has_outliers = false;
  reconstruct(images_paths, Rs_est, ts_est, K, points3d_estimated, is_projective, has_outliers);


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

  viz::Viz3d window("Coordinate Frame");
             window.setWindowSize(Size(500,500));
             window.setWindowPosition(Point(150,150));
             window.setBackgroundColor(); // black by default

  // Recovering the pointcloud
  cout << "Recovering points  ... ";

  std::vector<cv::Vec3f> point_cloud;
  for (int i = 0; i < points3d_estimated.cols; ++i) {
    cv::Vec3f point3d((float) points3d_estimated(0, i),
                      (float) points3d_estimated(1, i),
                      (float) points3d_estimated(2, i));
    point_cloud.push_back(point3d);
  }

  cout << "OK" << endl;

  /// Recovering cameras
  cout << "Recovering cameras ... ";

  vector<Affine3d> path;
  for (size_t i = 0; i < Rs_est.size(); ++i)
    path.push_back(Affine3d(Rs_est[i],ts_est[i]));

  cout << "OK" << endl;


  /// Add the pointcloud
  if ( !point_cloud.empty() )
  {
    cout << "Rendering points   ... ";

    viz::WCloud cloud_widget(point_cloud, viz::Color::green());
    //window.showWidget("point_cloud", cloud_widget);

    cout << "OK" << endl;
  }
  else
  {
    cout << "Cannot render points: Empty pointcloud" << endl;
  }


  /// Add cameras
  if ( !path.empty() )
  {
    cout << "Rendering Cameras  ... ";

    window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
    window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path, K, 0.1, viz::Color::yellow()));

    cout << "OK" << endl;
  }
  else
  {
    cout << "Cannot render the cameras: Empty path" << endl;
  }

  /// Wait for key 'q' to close the window
  cout << endl << "Press 'q' to close each windows ... " << endl;

  window.spinOnce();
  window.spin();

  return 0;
}