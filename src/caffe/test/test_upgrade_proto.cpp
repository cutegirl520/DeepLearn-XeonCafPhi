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
      