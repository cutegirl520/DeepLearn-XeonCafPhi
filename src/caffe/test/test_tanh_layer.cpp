#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

double tanh_naive(double x) {
  if (x < -40) {
    // avoid negative overflow
    return -1;
  } else if (x > 40) {
    // avoid positive overflow
    return 1;
  } else {
    // exact expression for tanh, which is unstable for large x
    double exp2x = exp(2 * x);
    return (exp2x - 1.0) / (exp2x + 1.0);
  }
}

template <typename TypeParam>
class TanHLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TanHLayerTest()
    