#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <iostream>

//#include "../test/scene.cpp"

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
      << "       example_sfm_random_scene_reconstruction [<nCameras>] [<nPoints>] \n"
      << " where nCameras is the number of cameras to generate in the scene (default: 20) \n"
      << "       nPoints is the number of 3D points to generate in the scene (default: 500) \n"
      << "----------------------------------------------------------------------------------\n\n"
      << endl;
}

void
generateScene(const size_t n_views, const size_t n_points, Matx33d & K, std::vector<Matx33d> & R,
              std::vector<Vec3d> & t, std::vector<Matx34d> & P, Mat_<double> & points3d,
              std::vector<Mat_<double> > & points2d );

bool
intersection(Vec3d &bmin, Vec3d &bmax, Vec3d &orig, Vec3d &end);


int main(int argc, char* argv[])
{
  int nviews = 20;
  int npoints = 50;

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


  /// Generate ground truth scene
  std::vector< Mat_<double> > points2d;
  std::vector< Matx33d > Rs;
  std::vector< Vec3d > ts;
  std::vector< Matx34d > Ps;
  Matx33d K;
  Mat_<double> points3d;
  const bool is_projective = true;
  generateScene(nviews, npoints, K, Rs, ts, Ps, points3d, points2d);


  /// Reconstruct the scene using the 2d correspondences
  Matx33d K_ = K;
  std::vector<cv::Mat> Rs_est, ts_est;
  Mat_<double> points3d_estimated;
  const bool has_outliers = false;
  reconstruct(points2d, Rs_est, ts_est, K_, points3d_estimated, is_projective, has_outliers);


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
  //Matx33d R_rel;
  //Vec3d t_rel;
  //double s_rel;
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

  std::vector<cv::Vec3f> point_cloud;
  for (int i = 0; i < points3d.cols; ++i) {
    // recover ground truth points3d
    cv::Vec3f point3d((float) points3d(0, i),
                      (float) points3d(1, i),
                      (float) points3d(2, i));
    point_cloud.push_back(point3d);
  }

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

  std::vector<Affine3d> path_gt;
  for (size_t i = 0; i < Rs.size(); ++i)
    path_gt.push_back(Affine3d(Rs[i],ts[i]));
  if ( Rs.size() > 0 )
    path_gt.push_back(Affine3d(Rs[0],ts[0]));

  std::vector<Affine3d> path_est;
  for (size_t i = 0; i < Rs_est.size(); ++i)
      path_est.push_back(Affine3d(Rs_est[i],ts_est[i]));
  if ( Rs_est.size() > 0 )
    path_est.push_back(Affine3d(Rs_est[0],ts_est[0]));

  cout << "[DONE]" << endl;


  /// Add the pointcloud
  cout << "Rendering points   ... ";

  if ( point_cloud.size() > 0 )
  {
    viz::WCloud cloud_widget(point_cloud, viz::Color::green());
    window_gt.showWidget("point_cloud", cloud_widget);
  }

  if ( point_cloud_est.size() > 0 )
  {
    viz::WCloud cloud_est_widget(point_cloud_est, viz::Color::red());
    window_est.showWidget("point_cloud_est", cloud_est_widget);

  }

  cout << "[DONE]" << endl;


  /// Add cameras
  cout << "Rendering Cameras  ... ";

  if ( path_gt.size() > 0 )
  {
    window_gt.showWidget("cameras_frames_and_lines_gt", viz::WTrajectory(path_gt, viz::WTrajectory::BOTH, 0.2, viz::Color::green()));
    window_gt.showWidget("cameras_frustums_gt", viz::WTrajectoryFrustums(path_gt, K, 2.0, viz::Color::yellow()));
  }

  if ( path_est.size() > 0 )
  {
    window_est.showWidget("cameras_frames_and_lines_est", viz::WTrajectory(path_est, viz::WTrajectory::BOTH, 0.2, viz::Color::green()));
    window_est.showWidget("cameras_frustums_est", viz::WTrajectoryFrustums(path_est, K, 0.3, viz::Color::yellow()));
  }

  cout << "[DONE]" << endl;


  /// Wait for key 'q' to close the window
  cout << endl << "Press 'q' to close each windows ... " << endl;

  window_gt.spinOnce(); window_est.spinOnce();
  window_gt.spin(); window_est.spin();

  return 0;
}


void
generateScene(const size_t n_views, const size_t n_points, Matx33d & K, std::vector<Matx33d> & R,
              std::vector<Vec3d> & t, std::vector<Matx34d> & P, Mat_<double> & points3d,
              std::vector<Mat_<double> > & points2d)
{
  R.resize(n_views);
  t.resize(n_views);

  cv::RNG rng;

  const double size_scene = 10.0f, offset_scene = 50.0f;
  const double x0 = offset_scene - size_scene/2, y0 = x0, z0 = x0;

  // Generate a bunch of random 3d points in a cube
  points3d.create(3, n_points);
  //rng.fill(points3d, cv::RNG::UNIFORM, -size_scene/2, size_scene/2);

  // Generate a bunch of random 3d points in a cube surface
  for (size_t i = 0; i < n_points; ++i)
  {
    //int face = rng.uniform(1, 7);
    int face = rng.uniform(1, 5);

    double u = rng.uniform( (double)0, size_scene),
           v = rng.uniform( (double)0, size_scene);

    double x = 0, y = 0, z = 0;

    if ( face == 1 )
      x = u, y = 0, z = v;
    else if ( face == 2 )
      x = u, y = size_scene, z = v;
    else if ( face == 3 )
      x = 0, y = u, z = v;
    else if ( face == 4 )
      x = size_scene, y = u, z = v;
    //else if ( face == 5 )
    //  x = u, y = v, z = 0;
    //else if ( face == 6 )
    //  x = u, y = v, z = size_scene;

    points3d.at<double>(0, i) = x + x0;
    points3d.at<double>(1, i) = y + y0;
    points3d.at<double>(2, i) = z + y0;
  }

  // Generate camera intrinsics
  const double f = 500;           // focal length in pixels
  const double img_width  = 640;  // image width  in pixels
  const double img_height = 480;  // image heigth in pixels
  const double cx = img_width/2;  // optical center in x direction
  const double cy = img_height/2; // optical center in y direction

  K = Matx33d(f, 0, cx,
              0, f, cy,
              0,  0, 1);

  // Generate cameras
  const float r = 4*size_scene; // camera distance in front of the object

  for (size_t i = 0; i < n_views; ++i)
  {
    // get the current angle
    const float theta = 2.0f * CV_PI * float(i) / float(n_views);

    // set rotation around x and y axis and apply a 90deg rotation
    const Vec3d vecx(CV_PI/2, 0, 0),
                vecy(0, 3*CV_PI/2+theta, 0),
                vecz(0, 0, 0);

    Matx33d Rx, Ry, Rz;
    Rodrigues(vecx, Rx);
    Rodrigues(vecy, Ry);
    Rodrigues(vecz, Rz);

    // compute rotation matrix
    R[i] = Rx * Ry * Rz;

    const float x = r * cos(theta), // calculate the x component
                y = r * sin(theta); // calculate the y component

    // compute translation vector
    t[i] = cv::Vec3d(x + offset_scene, y + offset_scene, offset_scene); //output vertex
  }

  // Compute projection matrices
  P.resize(n_views);
  for (size_t i = 0; i < n_views; ++i)
  {
    const Matx33d K3 = K, R3 = R[i];
    const Vec3d t3 = t[i];
    P_From_KRt(K3, R3, t3, P[i]);
  }

  // Compute homogeneous 3d points
  Mat_<double> points3d_homogeneous(4, n_points);
  points3d.copyTo(points3d_homogeneous.rowRange(0, 3));
  points3d_homogeneous.row(3).setTo(1);

  // Project those points for every view
  points2d.resize(n_views);
  for (size_t i = 0; i < n_views; ++i)
  {
    //cout << "**************" << endl;
    //cout << "Cam " << i << endl;

    Mat_<double> points2d_tmp = cv::Mat(P[i]) * points3d_homogeneous;
    points2d[i].create(2, n_points);

    // convert points from homogeneous
    for (unsigned char j = 0; j < 2; ++j)
      cv::Mat(points2d_tmp.row(j) / (points2d_tmp.row(2))).copyTo(points2d[i].row(j));

    // check which planes see this camera
    const Matx33d R3 = R[i];
    const Vec3d t3 = t[i];

    // check all points for this view
    for (int j = 0; j < n_points; ++j)
    {
      const Vec2d point2d = points2d[i].col(j);
      const Vec4d point3d = points3d_homogeneous.col(j);

      // check if point is seen

      Vec3d box_min = Vec3d(x0, y0, z0),
            box_max = box_min + Vec3d(size_scene, size_scene, size_scene);
      Vec3d origin = t3,
            direction = Vec3d(point3d[0], point3d[1], point3d[2]);

      bool is_seen = intersection(box_min, box_max, origin, direction);
      if (is_seen)
      {
        //cout << "Is seen ";
        //cout << point3d << endl;
      }
      else
      {
        //cout << "Is not seen ";
        //cout << point3d << endl;
      }
      //cout << "*****";

      // check if point is out of this view and set as NaN
      if ( point2d[0] < 0 || point2d[0] > img_width ||
           point2d[1] < 0 || point2d[1] > img_height )
      {
        //cout << pt << endl;
        //cv::Mat(Vec2d(-1,-1)).copyTo(points2d[i].col(j));
        //cv::Mat_<double>(2,1,-1).copyTo(points2d[i].col(j));
      }
      else
      {
        //cout << "----> " << pt << endl;
      }

    }

  }

// TODO: remove a certain number of points per view
// TODO: add a certain number of outliers per view

}


// https://github.com/hpicgs/cgsee/wiki/Ray-Box-Intersection-on-the-GPU

bool
intersection(Vec3d &bmin, Vec3d &bmax, Vec3d &orig, Vec3d &end)
{
    Vec3d bounds[2] = { bmin, bmax };

    double tmin = std::numeric_limits<double>::min(),
           tmax = std::numeric_limits<double>::max();

    double tymin, tymax, tzmin, tzmax;

    double mag = sqrt( end[0]*end[0] + end[1]*end[1] + end[2]*end[2] );

    Vec3d direction;
    direction[0] = acos( end[0] / mag );
    direction[1] = acos( end[1] / mag );
    direction[2] = acos( end[2] / mag );

    Vec3d inv_direction;
    inv_direction[0] = 1. / direction[0];
    inv_direction[1] = 1. / direction[1];
    inv_direction[2] = 1. / direction[2];

    int sign[3];
    sign[0] = ( inv_direction[0] < 0 );
    sign[1] = ( inv_direction[1] < 0 );
    sign[2] = ( inv_direction[2] < 0 );

    tmin = ( bounds[sign[0]][0] - orig[0] ) / inv_direction[0];
    tmax = ( bounds[1-sign[0]][0] - orig[0] ) / inv_direction[0];

    tymin = ( bounds[sign[1]][1] - orig[1] ) / inv_direction[1];
    tymax = ( bounds[1-sign[1]][1] - orig[1] ) / inv_direction[1];

    tzmin = ( bounds[sign[2]][2] - orig[2] ) / inv_direction[2];
    tzmax = ( bounds[1-sign[2]][2] - orig[2] ) / inv_direction[2];

    tmin = std::max(std::max(tmin, tymin), tzmin);
    tmax = std::min(std::min(tmax, tymax), tzmax);

    // post condition:
    // if tmin > tmax (in the code above this is represented by a return value of INFINITY)
    //     no intersection
    // else
    //     front intersection point = ray.origin + ray.direction * tmin (normally only this point matters)
    //     back intersection point  = ray.origin + ray.direction * tmax

    Vec3d front = orig + direction * tmin;
    Vec3d back  = orig + direction * tmax;

    bool verbose = false;
    if ( verbose )
    {
      cout << "origin -> " << orig << endl;
      cout << "end    -> " << end << endl;
      cout << "dir    -> " << direction << endl;
      cout << "front  -> " << front << endl;
      cout << "back   -> " << back << endl;
    }

    if ( tmin > tmax )
    {
      if ( verbose ) cout << "NO INTERSECTION" << endl;
      return false;
    }
    else
    {
      if ( verbose ) cout << "INTERSECTION" << endl;
      if ( front == end ) return true;
      else return false;
    }

}