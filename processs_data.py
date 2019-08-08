import os, cv2
import random,shutil

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

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print 'Finish make dir:',dir

def create_train_val_detection(shuffle=True, train_ratio=0.8):
    dataset_dir='/home/atsg/PycharmProjects/gvh205/Caffe_Tools_Ubuntu/data/OpenImages/validation'
    img_dir = os.path.join(dataset_dir, 'Person/JPEGImages')
    ano_dir = os.path.join(dataset_dir, 'Person/Annotations')
    txt_dir=os.path.join(dataset_dir, 'Person/ImageSets/Main')
    create_dir(txt_dir)
    list_img = get_list_file_in_folder(img_dir)
    if(shuffle):
        random.shuffle(list_img)

    total_img=len(list_img)
    num_train_img=int(train_ratio*total_img)

    train_val_txt=''
    test_txt=''
    for i in range(total_img):
        if(i<num_train_img):
            train_val_txt+=os.path.join(img_dir,list_img[i])+' '+os.path.join(ano_dir,list_img[i].replace('jpg','xml'))+'\n'
        else:
            test_txt+=os.path.join(img_dir,list_img[i])+' '+os.path.join(ano_dir,list_img[i].replace('jpg','xml'))+'\n'

    save_file(os.path.join(txt_dir,'trainval.txt'),train_val_txt)
    save_file(os.path.join(txt_dir,'test.txt'),test_txt)


def create_dataset(num_img=355):
    count=0
    src_dir='/media/atsg/Data/datasets/face_recognition/lfw'
    dst_dir='/home/atsg/PycharmProjects/gvh205/Caffe_Tools_Ubuntu/data/Face_custom/original/1'
    list_dir=get_list_dir_in_folder(src_dir)
    for sub_dir in list_dir:
        list_img = get_list_file_in_folder(os.path.join(src_dir, sub_dir))
        for img in list_img:
            if (count > num_img):
                continue
            src_file = os.path.join(src_dir, sub_dir, img)
            dst_file = os.path.join(dst_dir, img)
            image = cv2.imread(src_file)
            image = cv2.resize(image, (60, 60))
            cv2.imwrite(dst_file, image)
            # shutil.copy(src_file,dst_file)
            count += 1

    count=0
    src_dir='/home/atsg/PycharmProjects/gvh205/arm_proj/to_customer/GVH205_ARM_project_training_environment/dataset/getty_dataset2_resize300/train/dirty'
    dst_dir='/home/atsg/PycharmProjects/gvh205/Caffe_Tools_Ubuntu/data/Face_custom/original/0'
    list_img = get_list_file_in_folder(os.path.join(src_dir))
    for img in list_img:
        src_file = os.path.join(src_dir, img)
        dst_file = os.path.join(dst_dir, img)
        image = cv2.imread(src_file)
        image = cv2.resize(image, (60, 60))
        cv2.imwrite(dst_file, image)
        # shutil.copy(src_file,dst_file)
        count += 1
        if(count>num_img):
            break

    print 'Finish copy',num_img,'images'


def create_train_val_classification(shuffle=True, train_ratio=0.8):
    print 'Create train val folder from original folder with train_ratio:',train_ratio
    dataset_dir='/home/atsg/PycharmProjects/gvh205/Caffe_Tools_Ubuntu/data/classification/Face_custom'
    img_dir=os.path.join(dataset_dir,'original')
    train_dir=os.path.join(dataset_dir,'train')
    val_dir=os.path.join(dataset_dir,'val')
    lmdb_dir=os.path.join(dataset_dir,'lmdb')

    create_dir(train_dir)
    create_dir(val_dir)
    create_dir(lmdb_dir)

    classes=get_list_dir_in_folder(img_dir) #get class

    train_txt=''
    val_txt=''
    for dir in classes:
        print dir
        create_dir(os.path.join(train_dir, dir))
        create_dir(os.path.join(val_dir, dir))
        list_file=get_list_file_in_folder(os.path.join(img_dir, dir))
        total_img=len(list_file)
        num_train_img=int(train_ratio*total_img)
        count=0
        if(shuffle):
            random.shuffle(list_file)
        for file in list_file:
            src_file=os.path.join(img_dir, dir,file)
            if(count<num_train_img):
                dst_file=os.path.join(train_dir, dir,file)
                train_txt+=os.path.join(dir,file)+' '+dir+'\n'
            else:
                dst_file=os.path.join(val_dir, dir,file)
                val_txt+=os.path.join(dir,file)+' '+dir+'\n'
            shutil.move(src_file,dst_file)
            count+=1

    save_file(os.path.join(dataset_dir,'train.txt'),train_txt)
    save_file(os.path.join(dataset_dir,'val.txt'),val_txt)
    print 'Done.'

def merge_train_val_classification():
    print 'Move file from train val folder to original folder'
    dataset_dir = '/home/atsg/PycharmProjects/gvh205/Caffe_Tools_Ubuntu/data/classification/Face_custom'
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(dataset_dir, 'original')

    create_dir(img_dir)

    classes=get_list_dir_in_folder(train_dir) #get class

    for dir in classes:
        print dir
        create_dir(os.path.join(img_dir, dir))
        list_file_train=get_list_file_in_folder(os.path.join(train_dir, dir))
        list_file_val=get_list_file_in_folder(os.path.join(val_dir, dir))

        for file in list_file_train:
            src_file=os.path.join(train_dir, dir,file)
            dst_file=os.path.join(img_dir, dir,file)
            shutil.move(src_file,dst_file)
        for file in list_file_val:
            src_file=os.path.join(val_dir, dir,file)
            dst_file=os.path.join(img_dir, dir,file)
            shutil.move(src_file,dst_file)
    print 'Done.'


if __name__ == "__main__":

    #create_train_val_txt(img_dir,ano_dir)
    #create_train_val_detection()
    create_train_val_classification()
    #merge_train_val_classification()
    #create_dataset()
    print('Finish')