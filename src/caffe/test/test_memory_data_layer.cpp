#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class MemoryDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MemoryDataLayerTest()
    : data_(new Blob<Dtype>()),
      labels_(new Blob<Dtype>()),
      data_blob_(new Blob<Dtype>()),
      label_blob_(new Blob<Dtype>()) {}
  virtual