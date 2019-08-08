"""
  Transpose
"""
import numpy as np
import cv2

import caffe


class ColorAdjust(caffe.Layer):

    """
    layer {
        type: 'Python'
        name: 'cadj1'
        bottom: 'data'
        top: 'cadj'
        python_param {
            module: 'caffe.layers.facedet.coloradjust'
            layer: 'ColorAdjust'
            param_str: '-15,5'
        }
    }
    """
    def setup(self, bottom, top):
        try:
            plist = map(int, self.param_str.split(','))
            self.bright_min = int(plist[0])
            self.bright_max = int(plist[1])
        except ValueError:
            raise ValueError("Parameter string missing or data type is wrong!")
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        #top[0].reshape( bottom[0].shape[0], bottom[0].shape[1], bottom[0].shape[2],  bottom[0].shape[3] )
        top[0].reshape( bottom[0].shape )


    def reshape(self, bottom, top):
        """ No dynamic reshape needed """
        pass


    def forward(self, bottom, top):
        batch = bottom[0].shape[0]
        br_val = np.random.randint(self.bright_min, self.bright_max, size=batch)
        for i in xrange(batch):
            im = np.transpose(bottom[0].data[i], (1,2,0))   # CHW -> HWC
            im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            v = im[:,:,2]
            v = np.where(v > 255-br_val[i], 255,
                np.where(v < -br_val[i], 0, v+br_val[i]) )
            im[:,:,2] = v
            im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
            top[0].data[i, ...] = np.transpose(bottom[0].data, (2,0,1)) # HWC -> CHW

    def backward(self, top, propagate_down, bottom):
        pass

