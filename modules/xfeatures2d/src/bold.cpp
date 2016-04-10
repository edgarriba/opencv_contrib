#include "precomp.hpp"
#include <fstream>



// constants
#define NTESTS 768
#define DIMS 512
#define NROTS 3

namespace cv
{
namespace xfeatures2d
{

/*
 !BOLD implementation
 */
class BOLD_Impl : public BOLD
{
public:
    /** Constructor
     */
    explicit BOLD_Impl();

    virtual ~BOLD_Impl();

    /**
     * @param image image to extract descriptors
     * @param keypoints of interest within image
     * @param descriptors resulted descriptors array
     */
    virtual void compute(InputArray image,
                         std::vector<KeyPoint>& keypoints,
                         OutputArray descriptors);

private:

    /*
     * BOLD functions
     */

    inline void rectifyPatch(const Mat& image, const KeyPoint& kp,
                             const int& patchSize, Mat& patch);

    inline void compute_patch(const Mat& img,
                              Mat& descr, Mat& masks);

    inline int hampopmaskedLR(uchar *a,uchar *ma,uchar *b,uchar *mb);
    inline int hampop(uchar *a,uchar *b);

    /*
     * BOLD arrays
     */

    int **bin_tests;
    int rotations[2];

};  // END BOLD_Impl CLASS

// constructor
/* load tests and init 2 rotations
   for fast affine aprox. (example -20,20) */
BOLD_Impl::BOLD_Impl(void)
{
  bin_tests = (int**) malloc(NROTS * sizeof(int *));
  for (int i = 0; i < NROTS; i++){
    bin_tests[i] = (int*)malloc(NTESTS*2 * sizeof(int));
  }
  std::ifstream file;
  file.open("bold.descr");
  /* read original tests and set them to rotation 0 */
  for(int j = 0; j < NTESTS*2; j++ )
    {
      file >> bin_tests[0][j];
    }
  file.close();
  rotations[0] = 20;
  rotations[1] = -20;
  /* compute the rotations offline */
  for (int i = 0; i < NTESTS*2; i+=2) {
    int x1 = bin_tests[0][i] % 32;
    int y1 = bin_tests[0][i] / 32;
    int x2 = bin_tests[0][i+1] % 32;
    int y2 = bin_tests[0][i+1] / 32;
    for (int a = 1; a < NROTS; a++) {
      float angdeg = rotations[a-1];
      float angle = angdeg*(float)(CV_PI/180.f);
      float ca = (float)cos(angle);
      float sa = (float)sin(angle);
      int rotx1 = (x1-15)*ca - (y1-15)*sa + 16;
      int roty1 = (x1-15)*sa + (y1-15)*ca + 16;
      int rotx2 = (x2-15)*ca - (y2-15)*sa + 16;
      int roty2 = (x2-15)*sa + (y2-15)*ca + 16;
      bin_tests[a][i] = rotx1 + 32*roty1;
      bin_tests[a][i+1] = rotx2 + 32*roty2;
    }
  }
}

// destructor
BOLD_Impl::~BOLD_Impl(void)
{
  /* free the tests */
  for (int i = 0; i < NROTS; i++){
    free(bin_tests[i]);
  }
  free(bin_tests);
}

// -------------------------------------------------
/* BOLD interface implementation */

// keypoint scope
void BOLD_Impl::compute(InputArray _image,
                        std::vector<KeyPoint>& keypoints,
                        OutputArray _descriptors) {
  // do nothing if no image
  const Mat img = _image.getMat();
  if (img.empty()) return;

  // allocate array
  const int m_descriptor_size = static_cast<int>(DIMS);
  const int num_kpts = static_cast<int>(keypoints.size());
  _descriptors.create(num_kpts, m_descriptor_size, CV_8U);

  // prepare descriptors
  Mat descriptors = _descriptors.getMat();
  descriptors.setTo(Scalar(0));

  // prepare masks
  // TODO: let's see how to interface that
  Mat masks = descriptors.clone();

  Mat tmp_patch, desc, mask;
  for (int i = 0; i < num_kpts; ++i) {
    // extract patch
    // TODO: check patch size, by now set to 1
    rectifyPatch(img, keypoints[i], 1, tmp_patch);
    // compute descriptor
    desc = descriptors.row(i);
    mask = masks.row(i);
    compute_patch(tmp_patch, desc, mask);
  }
}

// -------------------------------------------------
/* BOLD computation routines */

inline void BOLD_Impl::rectifyPatch(const Mat& image,
                                    const KeyPoint& kp,
                                    const int& patchSize,
                                    Mat& patch) {
    float s = 1.5f * (float) kp.size / (float) patchSize;

    float cosine = (kp.angle>=0) ? cos(kp.angle*M_PI/180) : 1.f;
    float sine   = (kp.angle>=0) ? sin(kp.angle*M_PI/180) : 0.f;

    float M_[] = {
        s*cosine, -s*sine,   (-s*cosine + s*sine  ) * patchSize/2.0f + kp.pt.x,
        s*sine,   s*cosine,  (-s*sine   - s*cosine) * patchSize/2.0f + kp.pt.y
    };

    warpAffine(image, patch, Mat(2,3,CV_32FC1,M_), Size(patchSize, patchSize),
              WARP_INVERSE_MAP + INTER_CUBIC + WARP_FILL_OUTLIERS);
}

inline void BOLD_Impl::compute_patch(const Mat& patch, Mat& descr, Mat& masks) {
  /* init cv mats */
  /*int nkeypoints = 1;
  descr.create(nkeypoints, DIMS/8, CV_8U);
  masks.create(nkeypoints, DIMS/8, CV_8U);*/

  /* apply box filter */
  /*cv::Mat patch;
  boxFilter(img, patch, img.depth(), cv::Size(5,5),
	    cv::Point(-1,-1), true, cv::BORDER_REFLECT);*/

  /* get test and mask results  */
  int k = 0;
  uchar* dsc = descr.ptr<uchar>(k);
  uchar* msk = masks.ptr<uchar>(k);
  uchar *smoothed = patch.data;
  int* tests = bin_tests[0];
  int* r0 = bin_tests[1];
  int* r1 = bin_tests[2];
  unsigned int val = 0;
  unsigned int var = 0;
  int idx = 0;
  int j = 0;
  int bit;
  int tdes,tvar;
  for (int i = 0; i < DIMS; i++, j+=2) {
      bit = i % 8;
      int temp_var = 0;
      tdes = (smoothed[tests[j]] < smoothed[tests[j+1]]);
      temp_var += (smoothed[r0[j]] < smoothed[r0[j+1]])^tdes;
      temp_var += (smoothed[r1[j]] < smoothed[r1[j+1]])^tdes;
      /* tvar-> 0 not stable --------  tvar-> 1 stable */
      tvar = (temp_var == 0);
      if (bit==0) {
          val = tdes;
          var = tvar;
      } else {
          val |= tdes << bit;
          var |= tvar << bit;
      }
      if (bit==7) {
          dsc[idx] = val;
          msk[idx] = var;
          val = 0;
          var = 0;
          idx++;
      }
  }
}

/* masked distance  */
inline int BOLD_Impl::hampopmaskedLR(uchar *a,uchar *ma,uchar *b,uchar *mb)
{
  int distL = 0;
  int distR = 0;
  int nL = 0;
  int nR = 0;
  for (int i = 0; i < 64; i++) {
    int axorb = a[i] ^ b[i];
    int xormaskedL = axorb & ma[i]  ;
    int xormaskedR = axorb & mb[i]  ;
    nL += ma[i];
    nR += mb[i];
    distL += __builtin_popcount(xormaskedL);
    distR += __builtin_popcount(xormaskedR);
  }
  float n = nL + nR;
  float wL = nL / n;
  float wR = nR / n;
  return distL*wL + distR*wR;
}

/* hamming distance  */
inline int BOLD_Impl::hampop(uchar *a,uchar *b)
{
  int distL = 0;
  for (int i = 0; i < 64; i++) {
    int axorb = a[i] ^ b[i];
    distL += __builtin_popcount(axorb);
  }
  return distL;
}

Ptr<BOLD> BOLD::create() {
    return makePtr<BOLD_Impl>();
}


} // END NAMESPACE XFEATURES2D
} // END NAMESPACE CV