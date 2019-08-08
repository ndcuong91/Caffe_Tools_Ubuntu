"""
  Random crop rectagle shape
"""
import numpy as np

import caffe


class CropRectangle(caffe.Layer):

    """
    layer {
        type: 'Python'
        name: 'cropdata'
        bottom: 'exdata'
        top: 'data'
        python_param {
            module: 'caffe.layers.tollgate.croprect'
            layer: 'CropRectangle'
            param_str: '16,72'
        }
        exclude: {
            phase: TEST
            stage: 'val'
        }
    }
    """
    def setup(self, bottom, top):
        try:
            plist = self.param_str.split(',')
            self.crop_height = int(plist[0])
            self.crop_width = int(plist[1])
        except ValueError:
            raise ValueError("Parameter string missing or data type is wrong!")
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape( bottom[0].shape[0], bottom[0].shape[1], self.crop_height, self.crop_width )
        self.max_cut = (bottom[0].shape[2]-self.crop_height, bottom[0].shape[3]-self.crop_width)


    def reshape(self, bottom, top):
        """ No dynamic reshape needed """
        pass


    def forward(self, bottom, top):
        off_xlist = np.random.randint(0, self.max_cut[1], size=bottom[0].shape[0])
        off_ylist = np.random.randint(0, self.max_cut[0], size=bottom[0].shape[0])
        for itt in range(bottom[0].shape[0]):
            off_x = off_xlist[itt]
            off_y = off_ylist[itt]
            crop_img = bottom[0].data[itt, :, off_y:(off_y+self.crop_height), off_x:(off_x+self.crop_width)]
            top[0].data[itt, ...] = crop_img

    def backward(self, top, propagate_down, bottom):
        pass

