"""
  Transpose
"""
import numpy as np

import caffe


class Transpose(caffe.Layer):

    """
    layer {
        type: 'Python'
        name: 'tranpose1'
        bottom: 'conv1'
        top: 'tpos1'
        python_param {
            module: 'caffe.layers.speech_detect.transpose'
            layer: 'Transpose'
            param_str: '0,3,1,2'
        }
    }
    """
    def setup(self, bottom, top):
        try:
            self.plist = list(map(int, self.param_str.split(',')))
        except ValueError:
            raise ValueError("Parameter string missing or data type is wrong!")
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape( bottom[0].shape[ self.plist[0] ],
                        bottom[0].shape[ self.plist[1] ],
                        bottom[0].shape[ self.plist[2] ],
                        bottom[0].shape[ self.plist[3] ] )


    def reshape(self, bottom, top):
        """ No dynamic reshape needed """
        pass


    def forward(self, bottom, top):
        top[0].data[...] = np.transpose(bottom[0].data, tuple(self.plist) )

    def backward(self, top, propagate_down, bottom):
        pass

