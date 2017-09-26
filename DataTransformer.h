//
// Created by Michal Faber on 14/09/2017.
//

#ifndef DATA_TRANSFORMER_DATATRANSFORMER_H
#define DATA_TRANSFORMER_DATATRANSFORMER_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"

using namespace cv;
using namespace std;


struct TransformationParameter {
  // For data pre-processing, we can do simple scaling and subtracting the
  // data mean, if provided. Note that the mean subtraction is always carried
  // out before scaling.
  float scale = 1;
  // Specify if we want to randomly mirror data.
  bool mirror = false;
  // Specify if we would like to randomly crop an image.
  int crop_size = 0;
  // mean_file and mean_value cannot be specified at the same time
  string mean_file;
  // if specified can be repeated once (would substract it from all the channels)
  // or can be repeated the same number of times as channels
  // (would subtract them from the corresponding channel)
  float mean_value; // should be array
  int stride = 4;
  float scale_cvg = 0.5;
  int max_cvg_len = 50;
  int min_cvg_len = 50;
  bool opaque_coverage = true;
  string coverage = "gridbox_max";
  float flip_prob = 0.5;
  float max_rotate_degree = 5.0;
  bool visualize = false;
  int crop_size_x = 368;
  int crop_size_y = 368;
  float scale_prob = 0.5;
  float scale_min = 0.9;
  float scale_max = 1.1;
  float bbox_norm_factor = 300;
  string img_header = ".";
  // Force the decoded image to have 3 color channels.
  bool force_color = false;
  // Force the decoded image to have 1 color channels.
  bool force_gray = false;
  float target_dist = 1.0;
  float center_perterb_max = 10.0;
  float sigma = 7.0;
  float sigma_center = 21.0;
  float clahe_tile_size = 8.0;
  float clahe_clip_limit = 4.0;
  bool do_clahe = false;
  int num_parts = 14;
  int num_total_augs = 82;
  string aug_way = "rand";
  int gray = 0;
  int np_in_lmdb = 16;
  bool transform_body_joint = true;
};

class CPMDataTransformer {
public:

  explicit CPMDataTransformer(const TransformationParameter& param);

  struct AugmentSelection {
    bool flip;
    float degree;
    Size crop;
    float scale;
  };

  struct Joints {
    vector<Point2f> joints;
    vector<float> isVisible;
  };

  struct MetaData {
    string dataset;
    Size img_size;
    bool isValidation;
    int numOtherPeople;
    int people_index;
    int annolist_index;
    int write_number;
    int total_write_number;
    int epoch;
    Point2f objpos; //objpos_x(float), objpos_y (float)
    float scale_self;
    Joints joint_self; //(3*16)

    vector<Point2f> objpos_other; //length is numOtherPeople
    vector<float> scale_other; //length is numOtherPeople
    vector<Joints> joint_others; //length is numOtherPeople
  };

  /**
    * @brief Initialize the Random number generations if needed by the
    *    transformation.
    */
  void InitRand();

  /**
  * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
  *
  * @param n
  *    The upperbound (exclusive) value of the random number.
  * @return
  *    A uniformly random integer value from ({0, 1, ..., n-1}).
  */
  int Rand(int n);

  void TransformMetaJoints(MetaData& meta);
  void TransformJoints(Joints& joints);



  bool onPlane(Point p, Size img_size);

  bool augmentation_flip(Mat& img, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode);

  float augmentation_rotate(Mat& img_src, Mat& img_aug, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode);

  float augmentation_scale(Mat& img, Mat& img_temp, Mat& mask_miss, Mat& mask_all, MetaData& meta, int mode);

  Size augmentation_croppad(Mat& img_temp, Mat& img_aug, Mat& mask_miss, Mat& mask_miss_aug, Mat& mask_all, Mat& mask_all_aug, MetaData& meta, int mode);


  void generateLabelMap(double*, Mat&, MetaData meta);
  void putGaussianMaps(double* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma);
  void putVecMaps(double* entryX, double* entryY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre);
  void putVecPeaks(double* entryX, double* entryY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre);
  void clahe(Mat& img, int, int);

  void dumpEverything(double* transformed_data, double* transformed_label, MetaData meta);
  void ReadMetaData(MetaData& meta, const uchar *data, size_t offset3, size_t offset1);
  void Transform_nv(const uchar *data, const int datum_channels, const int datum_height, const int datum_width, uchar* transformed_data, double* transformed_label);

  //void Transform();

  void swapLeftRight(Joints& j);
  void SetAugTable(int numData);
  void RotatePoint(Point2f& p, Mat R);
protected:

  // Tranformation parameters
  TransformationParameter param_;

  boost::shared_ptr<RNGen::RNG> rng_;

  vector<vector<float> > aug_degs;

  vector<vector<int> > aug_flips;

  int np;

  int np_in_lmdb;

  bool is_table_set;
};


#endif //DATA_TRANSFORMER_DATATRANSFORMER_H
