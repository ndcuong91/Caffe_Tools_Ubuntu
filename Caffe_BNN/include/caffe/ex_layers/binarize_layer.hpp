#ifndef CAFFE_BINARIZE_LAYER_HPP_
#define CAFFE_BINARIZE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Tests whether the input exceeds a threshold: outputs 1 for inputs
 *        above threshold; 0 otherwise.
 */
template <typename Dtype>
class BinarizeLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ThresholdParameter threshold_param,
   *     with BinarizeLayer options:
   *   - threshold (\b optional, default 0).
   *     the threshold value @f$ t @f$ to which the input values are compared.
   */
  explicit BinarizeLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Binarize"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *       y = \left\{
   *       \begin{array}{lr}
   *         -1 & \mathrm{if} \; x < t \\
   *         1 & \mathrm{if} \; x \ge t
   *       \end{array} \right.
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype threshold_;
  Dtype binaryMode;
};

}  // namespace caffe

#endif  // CAFFE_BINARIZE_LAYER_HPP_
