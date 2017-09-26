//
// Created by Michal Faber on 14/09/2017.
//

#ifndef DATA_TRANSFORMER_UTILS_H
#define DATA_TRANSFORMER_UTILS_H
#include <boost/random.hpp>
#include "RNGen.h"

typedef boost::mt19937 rng_t;
inline rng_t* caffe_rng() {
  return static_cast<rng_t*>(RNGen::rng_stream().generator());
}

#endif //DATA_TRANSFORMER_UTILS_H
