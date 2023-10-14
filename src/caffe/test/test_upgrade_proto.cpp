#include <cstring>
#include <string>
#include <vector>

#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/upgrade_proto.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class PaddingLayerUpgradeTest : public ::testing::Test {
 protected:
  void RunPaddingUpgradeTest(
      const string& input_param_string, const string& output_param_string) {
    // Test that UpgradeV0PaddingLayers called on the proto specified by
    // input_param_string results in the proto specified by
    // output_param_string.
    NetParameter input_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        input_param_string, &input_param));
    NetParameter expected_output_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        output_param_string, &expected_output_param));
    NetParameter actual_output_param;
    UpgradeV0PaddingLayers(input_param, &actual_output_param);
    EXPECT_EQ(expected_output_param.DebugString(),
        actual_output_param.DebugString());
    // Also test idempotence.
    NetParameter double_pad_upgrade_param;
    UpgradeV0PaddingLayers(actual_output_param, &double_pad_upgrade_param);
    EXPECT_EQ(actual_output_param.DebugString(),
       double_pad_upgrade_param.DebugString());
  }
};

TEST_F(PaddingLayerUpgradeTest, TestSimple) {
  const string& input_proto =
      "name: 'CaffeNet' "
      "layers { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "    source: '/home/jiayq/Data/ILSVRC12/train-leveldb' "
      "    meanfile: '/home/jiayq/Data/ILSVRC12/image_mean.binaryproto' "
      "    batchsize: 256 "
      "    cropsize: 227 "
      "    mirror: true "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pad1' "
      "    type: 'padding' "
      "    pad: 2 "
      "  } "
      "  bottom: 'data' "
      "  top: 'pad1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv1' "
      "    type: 'conv' "
      "    num_output: 96 "
      "    kernelsize: 11 "
      "    stride: 4 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pad1' "
      "  top: 'conv1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'fc8' "
      "    type: 'innerproduct' "
      "    num_output: 1000 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0 "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'conv1' "
      "  top: 'fc8' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'loss' "
      "    type: 'softmax_loss' "
      "  } "
      "  bottom: 'fc8' "
      "  bottom: 'label' "
      "} ";
  const string& expected_output_proto =
      "name: 'CaffeNet' "
      "layers { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "    source: '/home/jiayq/Data/ILSVRC12/train-leveldb' "
      "    meanfile: '/home/jiayq/Data/ILSVRC12/image_mean.binaryproto' "
      "    batchsize: 256 "
      "    cropsize: 227 "
      "    mirror: true "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv1' "
      "    type: 'conv' "
      "    num_output: 96 "
      "    kernelsize: 11 "
      "    stride: 4 "
      "    pad: 2 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'data' "
      "  top: 'conv1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'fc8' "
      "    type: 'innerproduct' "
      "    num_output: 1000 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0 "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'conv1' "
      "  top: 'fc8' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'loss' "
      "    type: 'softmax_loss' "
      "  } "
      "  bottom: 'fc8' "
      "  bottom: 'label' "
      "} ";
  this->RunPaddingUpgradeTest(input_proto, expected_output_proto);
}

TEST_F(PaddingLayerUpgradeTest, TestTwoTops) {
  const string& input_proto =
      "name: 'CaffeNet' "
      "layers { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "    source: '/home/jiayq/Data/ILSVRC12/train-leveldb' "
      "    meanfile: '/home/jiayq/Data/ILSVRC12/image_mean.binaryproto' "
      "    batchsize: 256 "
      "    cropsize: 227 "
      "    mirror: true "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pad1' "
      "    type: 'padding' "
      "    pad: 2 "
      "  } "
      "  bottom: 'data' "
      "  top: 'pad1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv1' "
      "    type: 'conv' "
      "    num_output: 96 "
      "    kernelsize: 11 "
      "    stride: 4 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pad1' "
      "  top: 'conv1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'fc8' "
      "    type: 'innerproduct' "
      "    num_output: 1000 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0 "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'conv1' "
      "  top: 'fc8' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv2' "
      "    type: 'conv' "
      "    num_output: 96 "
      "    kernelsize: 11 "
      "    stride: 4 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pad1' "
      "  top: 'conv2' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'loss' "
      "    type: 'softmax_loss' "
      "  } "
      "  bottom: 'fc8' "
      "  bottom: 'label' "
      "} ";
  const string& expected_output_proto =
      "name: 'CaffeNet' "
      "layers { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "    source: '/home/jiayq/Data/ILSVRC12/train-leveldb' "
      "    meanfile: '/home/jiayq/Data/ILSVRC12/image_mean.binaryproto' "
      "    batchsize: 256 "
      "    cropsize: 227 "
      "    mirror: true "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv1' "
      "    type: 'conv' "
      "    num_output: 96 "
      "    kernelsize: 11 "
      "    stride: 4 "
      "    pad: 2 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'data' "
      "  top: 'conv1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'fc8' "
      "    type: 'innerproduct' "
      "    num_output: 1000 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0 "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'conv1' "
      "  top: 'fc8' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv2' "
      "    type: 'conv' "
      "    num_output: 96 "
      "    kernelsize: 11 "
      "    stride: 4 "
      "    pad: 2 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'data' "
      "  top: 'conv2' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'loss' "
      "    type: 'softmax_loss' "
      "  } "
      "  bottom: 'fc8' "
      "  bottom: 'label' "
      "} ";
  this->RunPaddingUpgradeTest(input_proto, expected_output_proto);
}

TEST_F(PaddingLayerUpgradeTest, TestImageNet) {
  const string& input_proto =
      "name: 'CaffeNet' "
      "layers { "
      "  layer { "
      "    name: 'data' "
      "    type: 'data' "
      "    source: '/home/jiayq/Data/ILSVRC12/train-leveldb' "
      "    meanfile: '/home/jiayq/Data/ILSVRC12/image_mean.binaryproto' "
      "    batchsize: 256 "
      "    cropsize: 227 "
      "    mirror: true "
      "  } "
      "  top: 'data' "
      "  top: 'label' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv1' "
      "    type: 'conv' "
      "    num_output: 96 "
      "    kernelsize: 11 "
      "    stride: 4 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'data' "
      "  top: 'conv1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu1' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'conv1' "
      "  top: 'conv1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pool1' "
      "    type: 'pool' "
      "    pool: MAX "
      "    kernelsize: 3 "
      "    stride: 2 "
      "  } "
      "  bottom: 'conv1' "
      "  top: 'pool1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'norm1' "
      "    type: 'lrn' "
      "    local_size: 5 "
      "    alpha: 0.0001 "
      "    beta: 0.75 "
      "  } "
      "  bottom: 'pool1' "
      "  top: 'norm1' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pad2' "
      "    type: 'padding' "
      "    pad: 2 "
      "  } "
      "  bottom: 'norm1' "
      "  top: 'pad2' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv2' "
      "    type: 'conv' "
      "    num_output: 256 "
      "    group: 2 "
      "    kernelsize: 5 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pad2' "
      "  top: 'conv2' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu2' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'conv2' "
      "  top: 'conv2' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pool2' "
      "    type: 'pool' "
      "    pool: MAX "
      "    kernelsize: 3 "
      "    stride: 2 "
      "  } "
      "  bottom: 'conv2' "
      "  top: 'pool2' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'norm2' "
      "    type: 'lrn' "
      "    local_size: 5 "
      "    alpha: 0.0001 "
      "    beta: 0.75 "
      "  } "
      "  bottom: 'pool2' "
      "  top: 'norm2' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pad3' "
      "    type: 'padding' "
      "    pad: 1 "
      "  } "
      "  bottom: 'norm2' "
      "  top: 'pad3' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv3' "
      "    type: 'conv' "
      "    num_output: 384 "
      "    kernelsize: 3 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 0. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pad3' "
      "  top: 'conv3' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu3' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'conv3' "
      "  top: 'conv3' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pad4' "
      "    type: 'padding' "
      "    pad: 1 "
      "  } "
      "  bottom: 'conv3' "
      "  top: 'pad4' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv4' "
      "    type: 'conv' "
      "    num_output: 384 "
      "    group: 2 "
      "    kernelsize: 3 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pad4' "
      "  top: 'conv4' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu4' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'conv4' "
      "  top: 'conv4' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pad5' "
      "    type: 'padding' "
      "    pad: 1 "
      "  } "
      "  bottom: 'conv4' "
      "  top: 'pad5' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'conv5' "
      "    type: 'conv' "
      "    num_output: 256 "
      "    group: 2 "
      "    kernelsize: 3 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.01 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pad5' "
      "  top: 'conv5' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu5' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'conv5' "
      "  top: 'conv5' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'pool5' "
      "    type: 'pool' "
      "    kernelsize: 3 "
      "    pool: MAX "
      "    stride: 2 "
      "  } "
      "  bottom: 'conv5' "
      "  top: 'pool5' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'fc6' "
      "    type: 'innerproduct' "
      "    num_output: 4096 "
      "    weight_filler { "
      "      type: 'gaussian' "
      "      std: 0.005 "
      "    } "
      "    bias_filler { "
      "      type: 'constant' "
      "      value: 1. "
      "    } "
      "    blobs_lr: 1. "
      "    blobs_lr: 2. "
      "    weight_decay: 1. "
      "    weight_decay: 0. "
      "  } "
      "  bottom: 'pool5' "
      "  top: 'fc6' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'relu6' "
      "    type: 'relu' "
      "  } "
      "  bottom: 'fc6' "
      "  top: 'fc6' "
      "} "
      "layers { "
      "  layer { "
      "    name: 'drop6' "
      "    type: 'dropout' "
      "    dropout_ratio: 0.5 "
      "  } "
      "  bottom: 'fc6' "
  