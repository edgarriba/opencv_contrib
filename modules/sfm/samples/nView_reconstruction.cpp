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


int main(int argc, char* argv[])
{
  // read input parameters
  if ( argc > 1 )
  {
    if ( string(argv[1]).compare("-h") == 0 ||
         string(argv[1]).compare("--help") == 0 )
    {
      help();
      exit(0);
    }
  }


  /// Reconstruct the scene using the 2d correspondences

  std::vector<std::string> images_paths;
  images_paths.push_back( string(TEST_DATA_DIR) + "resized_IMG_2889.jpg" );
  images_paths.push_back( string(TEST_DATA_DIR) + "resized_IMG_2890.jpg" );
  images_paths.push_back( string(TEST_DATA_DIR) + "resized_IMG_2891.jpg" );
  images_paths.push_back( string(TEST_DATA_DIR) + "resized_IMG_2892.jpg" );
  images_paths.push_back( string(TEST_DATA_DIR) + "resized_IMG_2893.jpg" );
  images_paths.push_back( string(TEST_DATA_DIR) + "resized_IMG_2894.jpg" );
  images_paths.push_back( string(TEST_DATA_DIR) + "resized_IMG_2895.jpg" );
  images_paths.push_back( string(TEST_DATA_DIR) + "resized_IMG_2896.jpg" );

  Matx33d K = Matx33d(19,   0, 240,
                        0, 19, 359,
                        0,   0,  1);

  vector<Mat> Rs_est, ts_est;
  Mat_<double> points3d_estimated;
  reconstruct(images_paths, Rs_est, ts_est, K, points3d_estimated);


  return 0;
}