#include <algorithm>
#include <vector>

#include "caffe/ex_layers/quantrelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void QuantReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  
  num_bit         = this->layer_param_.quantize_param().num_bit();
  quant_min       = this->layer_param_.quantize_param().min();
  quant_max       = this->layer_param_.quantize_param().max();
  quant_resolution= this->layer_param_.quantize_param().resolution();
}

template <typename Dtype>
void QuantReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  //const int num_bit = this->layer_param_.quantize_param().num_bit();
  for (int i = 0; i < count; ++i) {
    if(num_bit>2)
    { 
            
              Dtype min_rng=quant_min;
              Dtype max_rng=quant_max ;
              Dtype resolution=quant_resolution;
              Dtype range=max_rng-min_rng;
              //TODO rint  (256 or 0)
              Dtype min_clip=round(min_rng*resolution/range) ;
              Dtype max_clip=round(max_rng*resolution/range)-1; 
              Dtype wq= round(resolution * bottom_data[i] /range) ;
              if(wq>max_clip) wq=max_clip;
              if(wq<min_clip) wq=min_clip;
              
              wq= wq / resolution *range;
              
              Dtype wclip=bottom_data[i];
              if(wclip>max_rng) wclip=max_rng;
              if(wclip<min_rng) wclip=min_rng;
              
              top_data[i]=wclip+(wq-wclip); 
    }  
    else if(num_bit==2)
          top_data[i] = bottom_data[i] >= 3.0 ? Dtype(4) :
                        bottom_data[i] >= 1.5 ? Dtype(2) :
                        bottom_data[i] >= 0.5 ? Dtype(1) :
                        Dtype(0); 
    else  if(num_bit==1)
          top_data[i] = bottom_data[i] >= 0.5 ? Dtype(1) :
                        Dtype(0);
         
    
     
  }
}

template <typename Dtype>
void QuantReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    //const int num_bit = this->layer_param_.quantize_param().num_bit();
    
    Dtype maxval ;
    Dtype minval ;
   
    if(num_bit>2)
    {
        maxval =  quant_max;
        minval =  quant_min;
    }
    else if(num_bit==2)
    {
        maxval =  4; 
        minval =  0;
    }
    else if(num_bit==1)
    {
        maxval =  1; 
        minval =  0;
    }
 
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (bottom_data[i] > minval && bottom_data[i] <= maxval);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(QuantReLULayer);
#endif

INSTANTIATE_CLASS(QuantReLULayer);
REGISTER_LAYER_CLASS(QuantReLU);

}  // namespace caffe
