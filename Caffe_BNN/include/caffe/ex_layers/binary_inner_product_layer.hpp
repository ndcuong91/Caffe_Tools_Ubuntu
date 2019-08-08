#ifndef CAFFE_BINARY_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_BINARY_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Also known as a "fully-connected" layer, computes an inner product
 *        with a set of learned weights, and (optionally) adds biases.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BinaryInnerProductLayer : public Layer<Dtype> {
 public:
  explicit BinaryInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BinaryInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  // For Binary 
  shared_ptr<Blob<Dtype> >  W_b;
  shared_ptr<Blob<Dtype> >  W_buffer;
  bool normalized_weights;
  //Blob<Dtype> rand_vec_;
  virtual inline Dtype hard_sigmoid(const Dtype value) {
    return std::max(Dtype(0), std::min(Dtype(1), Dtype((value+1)/2)));
  }
  virtual inline void hard_sigmoid(const shared_ptr<Blob<Dtype> > weights) {
    for (int index = 0; index < weights->count(); index++)
      weights->mutable_cpu_data()[index] = hard_sigmoid(weights->cpu_data()[index]);
  }
  virtual void binarizeCPUTo(const shared_ptr<Blob<Dtype> > weights, const shared_ptr<Blob<Dtype> > wb);
  virtual inline void copyCPUTo(const shared_ptr<Blob<Dtype> > ori, const shared_ptr<Blob<Dtype> > buf) {
    CHECK_EQ(ori->count(), buf->count());
    caffe_copy(ori->count(), ori->cpu_data(), buf->mutable_cpu_data());
  }
#ifdef USE_CUDNN
  virtual void binarizeGPUTo(const shared_ptr<Blob<Dtype> > weights, const shared_ptr<Blob<Dtype> > wb);
  virtual inline void copyGPUTo(const shared_ptr<Blob<Dtype> > ori, const shared_ptr<Blob<Dtype> > buf) {
    CHECK_EQ(ori->count(), buf->count());
    cudaMemcpy(buf->mutable_gpu_data(), ori->gpu_data(), sizeof(Dtype)*ori->count(), cudaMemcpyDefault);
  }
#endif
  // Pre
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights
};

}  // namespace caffe

#endif  // CAFFE_BINARY_INNER_PRODUCT_LAYER_HPP_
