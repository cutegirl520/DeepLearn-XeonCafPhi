#ifndef CAFFE_VISION_LAYERS_HPP_
#define CAFFE_VISION_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype> {
 public:
  explicit BaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);


  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, int n, bool skip_im2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias, int n);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, int n);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights, int n);
  void backward_cpu_bias(Dtype* bias, const Dtype* input, int n);








#ifdef XEON_PHI
  void forward_convolution(const Dtype* input, const Dtype* weight,
      Dtype* output, const Dtype* bias);
  void forward_bias(Dtype* output, const Dtype* bias);
#endif

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int num_;
  int channels_;
  int pad_h_, pad_w_;
  int height_, width_;
  int group_;
  int num_output_;
  int height_out_, width_out_;
  bool bias_term_;
  bool is_1x1_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    im2col_cpu(data, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_buff);
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {
    col2im_cpu(col_buff, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, data);
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    im2col_gpu(data, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, col_buff);
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    col2im_gpu(col_buff, conv_in_channels_, conv_in_height_, conv_in_width_,
        kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_, data);
  }
#endif

  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int conv_in_height_;
  int conv_in_width_;
  int kernel_dim_;
  int weight_offset_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
};

/**
 * @brief Convolves the input image with a bank of learned filters,
 *        and (optionally) adds biases.
 *
 *   Caffe convolves by reduction to matrix multiplication. This achieves
 *   high-throughput and generality of input and filter dimensions but comes at
 *   the cost of memory for matrices. This makes use of efficiency in BLAS.
 *
 *   The input is "im2col" transformed to a channel K' x H x W data matrix
 *   for multiplication with the N x K' x H x W filter matrix to yield a
 *   N' x H x W output matrix that is then "col2im" restored. K' is the
 *   input channel * kernel height * kernel width dimension of the unrolled
 *   inputs so that the im2col matrix has a column for each input region to
 *   be filtered. col2im restores the output spatial structure by rolling up
 *   the output channel N' columns of the output matrix.
 */
template <typename Dtype>
class ConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines.
   */
  explicit ConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "Convolution"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void 