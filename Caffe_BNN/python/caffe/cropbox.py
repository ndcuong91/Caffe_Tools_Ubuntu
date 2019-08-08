import numpy as np

import caffe


class CropFaceBox(caffe.Layer):

    """
    layer {
        type: 'Python'
        name: 'imgcrop'
        bottom: 'exdata'
        bottom: 'exbbox'
        top: 'data'
        top: 'label'
        python_param {
            module: 'caffe.layers.facedet.cropbox'
            layer: 'CropFaceBox'
            param_str: '90,1'
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
            self.crop_size = int(plist[0])
            self.mirror = int(plist[1]) > 0
        except ValueError:
            raise ValueError("Parameter string missing or data type is wrong!")
        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape( bottom[0].shape[0], bottom[0].shape[1], self.crop_size, self.crop_size )
        top[1].reshape( bottom[1].shape[0], bottom[1].shape[1], 1, 1 )
        self.max_cut = (bottom[0].shape[2]-self.crop_size, bottom[0].shape[3]-self.crop_size)


    def reshape(self, bottom, top):
        """ No dynamic reshape needed """
        pass


    def forward(self, bottom, top):
        off_xlist = np.random.randint(0, self.max_cut[1], size=bottom[0].shape[0])
        off_ylist = np.random.randint(0, self.max_cut[0], size=bottom[0].shape[0])
        do_mirror = np.random.randint(0, 1, size=bottom[0].shape[0]) if self.mirror else np.zeros(bottom[0].shape[0])
        for itt in range(bottom[0].shape[0]):
            off_x = off_xlist[itt]
            off_y = off_ylist[itt]
            crop_img = bottom[0].data[itt, :, off_y:(off_y+self.crop_size), off_x:(off_x+self.crop_size)]
            crop_bb = bottom[1].data[itt, ...].squeeze() - np.array([off_x, off_y, 0, 0])
            if do_mirror[itt]:
                crop_bb[0] = self.crop_size - crop_bb[0]
                top[0].data[itt, ...] = crop_img[..., ::-1] 
            else:
                top[0].data[itt, ...] = crop_img
            top[1].data[itt, ...] = crop_bb.reshape(4, 1, 1)

    def backward(self, top, propagate_down, bottom):
        pass

