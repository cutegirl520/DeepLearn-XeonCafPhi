
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"


#include <pmmintrin.h>
#include <immintrin.h>
#include <cilk/cilk.h>
#include <cilk/reducer.h>

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  this->height_out_ = (this->height_ + 2 * this->pad_h_ - this->kernel_h_)
      / this->stride_h_ + 1;
  this->width_out_ = (this->width_ + 2 * this->pad_w_ - this->kernel_w_)
      / this->stride_w_ + 1;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
#ifdef XEON_PHI_ESSENTIAL_DEBUG
  LOG(INFO) << "XEON conv_layer.cpp: Forward_cpu";
#endif
  const Dtype* weight = this->blobs_[0]->cpu_data();
#ifdef XEON_PHI
  for(int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

    cilk_for(int n = 0; n < this->num_; ++n) {
      if(this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        /* Forward convolution */
        this->forward_convolution(bottom_data + bottom[i]->offset(n),
                                  weight,
                                  top_data + top[i]->offset(n),
                                  bias);
      }
      else{
        this->forward_convolution(bottom_data + bottom[i]->offset(n),
                                  weight,
                                  top_data + top[i]->offset(n),
                                  NULL);
      }

 //      if(this->bias_term_) {
        // const Dtype* bias = this->blobs_[1]->cpu_data();
        // this->forward_bias(top_data + top[i]->offset(n), bias);
 //      }
    }
  }
#else
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n), n);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias, n);
      }
    }
  }
#endif
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  if (this->param_propagate_down_[0]) {
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    caffe_set(this->blobs_[1]->count(), Dtype(0),
        this->blobs_[1]->mutable_cpu_diff());
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n), n);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {

        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          cilk_for (int n = 0; n < this->num_; ++n)
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff, n);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          cilk_for (int n = 0; n < this->num_; ++n)
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n), n);
        }

    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe