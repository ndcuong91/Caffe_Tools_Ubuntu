import caffe, os, cv2
import numpy as np

folder='/home/atsg/PycharmProjects/gvh205/Lattice_BNN/source_code/Lattice_Caffe/caffe_source/examples/bnn'

caffe_proto=os.path.join(folder,'facedet3_bbn.proto') #facedet3_bbn HumanPresence
caffe_params=os.path.join(folder,'facedet_iter_2500.caffemodel')
caffe_net_with_pretrained=caffe.Net(caffe_proto, caffe_params, caffe.TEST)
caffe.set_mode_cpu()
#caffe_net = caffe.Net(caffe_proto, caffe.TEST)
img_file='/home/atsg/lscc/ml/2.0/examples/test_img/man.jpg'

mean = [128,128,128]
#mean = [0,0,0]
std = [1.,1.,1.]
iput_size = 32


def get_list_file_in_folder(dir, ext='jpg'):
    included_extensions = [ext]
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir

def save_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def preprocess(img_path):
    src = cv2.imread(img_path)
    input_sz = (iput_size, iput_size)
    #img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    img = cv2.resize(src, input_sz)
    img = img.astype(np.float32)
    img -= mean
    img /= std
    img = img.transpose((2, 0, 1))
    return img

def preprocess2(img_path):
    src = cv2.imread(img_path)
    #input_sz = (iput_size, iput_size)
    #img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    img = src[14:14+iput_size,14:14+iput_size]
    img = img.astype(np.float32)
    img -= mean
    img /= std
    img = img.transpose((2, 0, 1))
    return img

def preprocess_gray(img_path):
    src = cv2.imread(img_path)
    input_sz = (iput_size, iput_size)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, input_sz)
    img = img.astype(np.float32)
    img -= mean
    img /= std
    img = img.transpose((2, 0, 1))
    return img

def classify_img(net, img_path):
    img = preprocess2(img_path)

    net.blobs['data'].data[...] = img
    #layer_name='conv1'
    #W = net.params[layer_name][0].data[...]
    #b = net.params[layer_name][1].data[...]

    out = net.forward()
    out['prob'][0][0]=int(out['prob'][0][0])
    out['prob'][0][1]=int(out['prob'][0][1])
    # pool2=net.blobs['pool2'].data[...]
    # conv3=net.blobs['conv3'].data[...]
    # pool3=net.blobs['pool3'].data[...]
    # pool4=net.blobs['pool4'].data[...]
    # pool5=net.blobs['maxpool5'].data[...]
    # conv6=net.blobs['maxconv6'].data[...]
    #idx=np.argmax(out['conv12/convolution'][0])
    # print label[idx]
    # print out
    from mxnet import nd
    softmax=(nd.softmax(nd.array(out['prob']), axis=1)).asnumpy()
    # if(softmax[0][0]>softmax[0][1]):
    #     print "No human face"
    # else:
    #     print "Human face"
    return softmax

def classsify_dir(caffe_net, test_dir):

    list_dir=get_list_dir_in_folder(test_dir)

    true_pred={}
    true_pred['0']=0
    true_pred['1']=0
    total_pred={}
    total_pred['0']=0
    total_pred['1']=0

    for sub_dir in list_dir:
        list_img = get_list_file_in_folder(os.path.join(test_dir, sub_dir))
        total_pred[sub_dir]=len(list_img)
        for img in list_img:
            output=classify_img(caffe_net,os.path.join(test_dir, sub_dir, img))
            if (output[0][0] > output[0][1] and sub_dir=='0'):
                true_pred[sub_dir]+=1
            if (output[0][0] <= output[0][1] and sub_dir=='1'):
                true_pred[sub_dir]+=1
        accuracy=(float)(true_pred[sub_dir])/(float)(total_pred[sub_dir])
        print 'Class:',sub_dir,', Accuracy:',accuracy

    print 'Final accuracy:',(float)(true_pred['0']+true_pred['1'])/(float)(total_pred['0']+total_pred['1'])

if __name__ == "__main__":
    #classify_img(caffe_net_with_pretrained,img_file)
    classsify_dir(caffe_net_with_pretrained,'/home/atsg/PycharmProjects/gvh205/Object_detection_Caffe/data/Face_custom/val')

kk=1
