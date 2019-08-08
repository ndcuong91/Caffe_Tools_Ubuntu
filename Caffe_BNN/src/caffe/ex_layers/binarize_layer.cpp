#include <vector>

#include "caffe/ex_layers/binarize_layer.hpp"

namespace caffe {

template <typename Dtype>
void BinarizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  //threshold_ = this->layer_param_.threshold_param().threshold();
  binaryMode=lssBNNSign_NP;
  if(this->layer_param_.phase()==lssBNNSign_0P || this->layer_param_.phase()==lssBNNSign_NP|| this->layer_param_.phase()==lssBNNSign_N0P)
        binaryMode = this->layer_param_.phase();
}

template <typename Dtype>
void BinarizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  // clipping
  for (int i = 0; i < count; ++i) {
    switch ((int) (binaryMode))
    {
        case lssBNNSign_0P:
           top_data[i] = (bottom_data[i] > Dtype(0)) ? Dtype(1) : Dtype(0);
           break;       
        case lssBNNSign_NP:
           top_data[i] = (bottom_data[i] >= Dtype(0)) ? Dtype(1) : Dtype(-1);
           break;           
        case lssBNNSign_N0P:
           //Tensorflow sign mode
//            SoundCmd_BNN_N  1e-9
//            Tensorflow Positive, Zero,Negative:
//            (1987, 226, 1883)
//            Caffe Positive, Zero,Negative:
//            (1996, 219, 1881)
            if( bottom_data[i] - Dtype(0) ==0)
            { 
                top_data[i] = Dtype(0); 
            }
            else 
            {
                if (bottom_data[i]>Dtype(0))
                { 
                    top_data[i] = Dtype(1); 
                }
                else
                {
                    top_data[i] = Dtype(-1); 
                }
            }
           break;   
                 
    }
    
  }
}

template <typename Dtype>
void BinarizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const int count = bottom[0]->count();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_abs(count, bottom[0]->cpu_data(), bottom_diff);
  for (int i = 0; i < count; ++i) {
    bottom_diff[i] = bottom_diff[i] > Dtype(1) ? Dtype(0) : Dtype(1);   // 1 if x <= [-1:1] else 0
  }
  caffe_mul(count, bottom_diff, top_diff, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(BinarizeLayer);
#endif

INSTANTIATE_CLASS(BinarizeLayer);
REGISTER_LAYER_CLASS(Binarize);

}  // namespace caffe
