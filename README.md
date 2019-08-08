# Caffe Tools for Ubuntu
All tools to convert, prepare dataset and training for classification and object detection by Caffe in Ubuntu
## Environment
Ubuntu 16.04, Cuda 9.0

## Classification
### Step 1
Create a dataset folder (e.g. Face_custom) in data/classification folder with follow architecture:

Face_custom
└── original
    ├── 0
    │   ├── 1.jpg
    │   └── ...
    ├── 1
    │   ├── d.jpg
    │   └── ...
    └── ...
### Step 2
Separate train val folder and create train.txt, val.txt by using function *create_train_val_classification* in **process_data.py** script, you can also merge train val folder to original forder with *merge_train_val_classification*
### Step 3
Use **create_classification_dataset** file to generate lmdb database by modify those parameters:
- TOOLS: folder contain convert_imageset tool from Caffe
- DATA: dataset dir
- TRAIN_DATA_ROOT: train dir
- VAL_DATA_ROOT: val dir 
- LMDB_DIR: lmdb dir
- RESIZE_HEIGHT
- RESIZE_WIDTH
### Step 4
use **train_classification** to train with above lmdb dataset
