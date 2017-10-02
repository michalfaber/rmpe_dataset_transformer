#include <iostream>
#include <vector>
#include "DataTransformer.h"
#include "H5Cpp.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;

struct DatumMetadata {
  uchar* datum;
  hsize_t shape[3];
};

int getKeys(std::vector<string> *keys, const H5::Group *group) {
  hsize_t  num_obj;
  H5Gget_num_objs(group->getId(), &num_obj);
  char buff[10];

  for (int i=0; i < num_obj; i++) {
    snprintf(buff, sizeof(buff), "%07d", i);
    keys->push_back(string(buff));
  }
  return 0;
}

int getDatum(const string key, const H5::Group *group, DatumMetadata &meta) {

  H5::DataSet *dataset = new H5::DataSet( group->openDataSet( key.data()));
  H5::DataSpace dataspace = dataset->getSpace();
  int rank = dataspace.getSimpleExtentNdims();
  dataspace.getSimpleExtentDims( meta.shape, NULL);
  H5::DataSpace mspace1(rank, meta.shape);

  meta.datum = new uchar[meta.shape[0] * meta.shape[1] * meta.shape[2]];

  dataset->read(meta.datum, H5::PredType::NATIVE_UCHAR, mspace1, dataspace);

  dataset->close();
  delete dataset;

  return 0;
}

int saveTransformed(
    uchar *transformed_data, double *transformed_label,
    const char *key,
    const int rows_data, const int cols_data,
    const int rows_label,
    const int cols_label,
    const int np,
    const int start_label_data,
    const H5::Group *data_group,
    const H5::Group *mask_group,
    const H5::Group *label_group) {

  // prepare data

  Eigen::MatrixXd weights = Eigen::Map<Eigen::MatrixXd>(
      transformed_label, rows_label * np, cols_label);

  Eigen::MatrixXd vec = Eigen::Map<Eigen::MatrixXd>(
      transformed_label + start_label_data, rows_label * np, cols_label);

  Eigen::MatrixXd label = vec.cwiseProduct(weights);

  Eigen::MatrixXd mask = Eigen::Map<Eigen::MatrixXd>(
      transformed_label, rows_label, cols_label);

  // save label
  const hsize_t rows(rows_label * np);
  const hsize_t cols(cols_label);
  hsize_t fdim[] = {rows, cols}; // dim sizes of ds (on disk)
  H5::DataSpace fspace( 2, fdim );
  int fillvalue = 0;   /* Fill value for the dataset */
  H5::DSetCreatPropList plist;
  plist.setFillValue(H5::PredType::NATIVE_DOUBLE, &fillvalue);

  H5::DataSet* dataset = new H5::DataSet(label_group->createDataSet(
      key, H5::PredType::NATIVE_DOUBLE, fspace, plist));

  dataset->write( label.data() , H5::PredType::NATIVE_DOUBLE);
  dataset->close();
  delete dataset;

  // save data
  const hsize_t rows2(rows_data);
  const hsize_t cols2(cols_data);
  hsize_t fdim_data[] = {rows2, cols2}; // dim sizes of ds (on disk)
  H5::DataSpace fspace_data( 2, fdim_data );

  dataset = new H5::DataSet(data_group->createDataSet(
      key, H5::PredType::NATIVE_UCHAR, fspace_data, plist));
  dataset->write( transformed_data , H5::PredType::NATIVE_UCHAR);
  dataset->close();
  delete dataset;

  // save mask
  const hsize_t rows_mask(rows_label);
  const hsize_t cols_mask(cols_label);
  hsize_t fdim_mask[] = {rows_mask, cols_mask}; // dim sizes of ds (on disk)
  H5::DataSpace fspace_mask( 2, fdim_mask );
  dataset = new H5::DataSet(mask_group->createDataSet(
      key, H5::PredType::NATIVE_DOUBLE, fspace_mask, plist));
  dataset->write( mask.data() , H5::PredType::NATIVE_DOUBLE);
  dataset->close();
  delete dataset;

  return 0;
}

int main(int argc, char* argv[]) {

  // Check the number of parameters
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <INPUT DATASET>" << " <OUTPUT DATASET>" << std::endl;
    return 1;
  }

  const string in_dataset = argv[1];
  const string out_dataset = argv[2];

  // Initialize params

  TransformationParameter params;

  params.stride=8;
  params.crop_size_x=368;
  params.crop_size_y=368;
  params.target_dist=0.6;
  params.scale_prob=1;
  params.scale_min=0.5;
  params.scale_max=1.1;
  params.max_rotate_degree=40;
  params.center_perterb_max=40;
  params.do_clahe=false;
  params.visualize=false;
  params.np_in_lmdb=17;
  params.num_parts=56;
  params.mirror = true;
  params.visualize = 0;

  CPMDataTransformer* cpmDataTransformer = new CPMDataTransformer(params);
  cpmDataTransformer->InitRand();

  const int np = 2*(params.num_parts+1);
  const int stride = params.stride;
  const int grid_x = params.crop_size_x / stride;
  const int grid_y = params.crop_size_y / stride;
  const int channelOffset = grid_y * grid_x;
  const int vec_channels = 38;
  const int heat_channels = 19;
  const int ch = vec_channels + heat_channels;
  const int start_label_data = (params.num_parts+1) * channelOffset;

  uchar* transformed_data = new uchar[params.crop_size_x * params.crop_size_y * 3];
  double* transformed_label = new double[grid_x * grid_y * np];

  // read all samples ids
  std::vector<string> keys;
  H5::H5File *f_in = new H5::H5File( in_dataset.c_str(), H5F_ACC_RDONLY );
  H5::Group* datum = new H5::Group( f_in->openGroup( "datum" ));

  getKeys(&keys, datum);

  H5::H5File *f_out = new H5::H5File( out_dataset.c_str(), H5F_ACC_TRUNC );

  H5::Group* data_group = new H5::Group( f_out->createGroup( "/data" ));
  H5::Group* mask_group = new H5::Group( f_out->createGroup( "/mask" ));
  H5::Group* label_group = new H5::Group( f_out->createGroup( "/label" ));

  // process all samples

  for (int i=0; i<keys.size(); i++) {
    string key = keys[i];

    cout << "Transforming sample " << i << "/" << keys.size() << endl;

    // read sample

    DatumMetadata meta;
    getDatum(key, datum, meta);

    // transform sample

    int channels = meta.shape[0];
    int height = meta.shape[1];
    int width = meta.shape[2];

    cpmDataTransformer->Transform_nv(meta.datum, channels, height, width,
                                     transformed_data, transformed_label);

    saveTransformed(
        transformed_data, transformed_label,
        key.c_str(),
        params.crop_size_y * 3, params.crop_size_x,
        grid_y, grid_x, ch, start_label_data, data_group, mask_group, label_group);

    delete [] meta.datum;

  }

  // cleanup

  datum->close();
  label_group->close();
  data_group->close();
  mask_group->close();
  delete datum;
  delete [] transformed_data;
  delete [] transformed_label;
  delete cpmDataTransformer;

  cout << "Done !!!" << endl;

  return 0;
}



