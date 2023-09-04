#include <cstring>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/insert_splits.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SplitLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SplitLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_a_(new Blob<Dtype>()),
        blob_top_b_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_a_);
    blob_top_vec_.push_back(blob_top_b_);
  }
  virtual ~SplitLayerTest() {
    delete blob_bottom_;
    delete blob_top_a_;
    delete blob_top_b_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_a_;
  Blob<Dtype>* const blob_top_b_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SplitLayerTest, TestDtypesAndDevices);

TYPED_TEST(SplitLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SplitLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_a_->num(), 2);
  EXPECT_EQ(this->blob_top_a_->channels(), 3);
  EXPECT_EQ(this->blob_top_a_->height(), 6);
  EXPECT_EQ(this->blob_top_a_->width(), 5);
  EXPECT_EQ(this->blob_top_b_->num(), 2);
  EXPECT_EQ(this->blob_top_b_->channels(), 3);
  EXPECT_EQ(this->blob_top_b_->height(), 6);
  EXPECT_EQ(this->blob_top_b_->width(), 5);
}

TYPED_TEST(SplitLayerTest, Test) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SplitLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    Dtype bottom_value = this->blob_bottom_->cpu_data()[i];
    EXPECT_EQ(bottom_value, this->blob_top_a_->cpu_data()[i]);
    EXPECT_EQ(bottom_value, this->blob_top_b_->cpu_data()[i]);
  }
}

TYPED_TEST(SplitLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SplitLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}


class SplitLayerInsertionTest : public ::testing::Test {
 protected:
  void RunInsertionTest(
      const string& input_param_string, const string& output_param_string) {
    // Test that InsertSplits called on the proto specified by
    // input_param_string results in the proto specified by
    // output_param_string.
    NetParameter input_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        input_param_string, &input_param));
    NetParameter expected_output_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        output_param_string, &expected_output_param));
    NetParameter actual_output_param;
    InsertSplits(input_param, &actual_output_param);
    EXPECT_EQ(expected_output_param.DebugString(),
        actual_output_param.DebugString());
    // Also test idempotence.
    NetParameter double_split_insert_param;
    InsertSplits(actual_output_param, &double_split_insert_param);
    EXPECT_EQ(actual_output_param.DebugString(),
       double_split_insert_param.DebugString());
  }
};

TEST_F(SplitLayerInsertionTest, TestNoInsertion1) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'innerprod' "
      "  type: 'InnerProduct' "
      "  bottom: 'data' "
      "  top: 'innerprod' "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'SoftmaxWithLoss' "
      "  bottom: 'innerprod' "
      "  bottom: 'label' "
      "} ";
  this->RunInsertionTest(input_proto, input_proto);
}

TEST_F(SplitLayerInsertionTest, TestNoInsertion2) {
  const string& input_proto =
      "name: 'TestNetwork' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layer { "
      "  name: 'data_split' "
      "  type: 'Split' "
      "  bottom: 'data' "
      "  top: 'data_split_0' "
      "  top: 'data_split_1' "
      "} "
      "layer { "
      "  name: 'innerprod1' "
      "  type: 'InnerProduct' "
      "  bottom: 'data_split_0' "
      "  top: 'innerprod1' "
      "} "
      "layer { "
      "  name: 'innerprod2' "
      "  type: 'InnerProduct' "
      "  bottom: 'data_split_1' "
      "  top: 'innerprod2' "
      "} "
      "layer { "
      "  name: 'loss' "
      "  type: 'EuclideanLoss' "
      "  bottom: 'innerprod1' "
      "  bottom: 'innerprod2' "
      "} ";
  this->RunInsertionTest(input_proto, input_proto);
}

TEST_F(SplitLayerInsertionTest, TestNoInsertionImageNet) {
  const string& input_proto =
      "name: 'CaffeNet' "
      "layer { "
      "  name: 'data' "
      "  type: 'Data' "
      "  data_param { "
      "    source: '/home/jiayq/Data/ILSVRC12/train-leveldb' "
      "    batch_size: 256 "
      "