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

Final dataset should look like

Face_custom
├── lmdb
│   ├── train_lmdb
│   │   ├── data.mdb
│   │   └── lock.mdb
│   └── val_lmdb
│       ├── data.mdb
│       └── lock.mdb
├── train
│   ├── 0
│   │   ├── 3453-000030.jpg
│   │   └── 6026-000344.jpg
│   └── 1
│       ├── Adam_Scott_0001.jpg
│       └── Adam_Scott_0002.jpg
├── train.txt
├── val
│   ├── 0
│   │   ├── 3477-000022.jpg
│   │   └── 3522-000079.jpg
│   └── 1
│       ├── Adel_Al-Jubeir_0002.jpg
│       └── Adrian_McPherson_0002.jpg
└── val.txt


### Step 4
use **train_classification** to train with above lmdb dataset

## Detection

### Step 1
Create a dataset folder (e.g. Person) in data/detection folder with follow architecture:

Person
├── Annotations
│   ├── 0a1aee5d7701ce5c.xml
│   ├── 0a1bd356f90aaab6.xml
│   └── 0a3c01759e77a02d.xml
├── JPEGImages
│   ├── 0a1aee5d7701ce5c.jpg
│   ├── 0a1bd356f90aaab6.jpg
│   └── 0a3c01759e77a02d.jpg
└── lmdb

### Step 2

Create train val txt file by using function *create_train_val_detection* in script **process_data.py** 

Create file "labelmap_Person.txt" in Person folder, this file will add aditional "background" class like below:

*item {
  name: "none_of_the_above"
  label: 0
  display_name: "background"
}
item {
  name: "Person"
  label: 1
  display_name: "Person"
}
...*


### Step 3
Use **create_classification_dataset** file to generate lmdb database by modify those parameters:
- tool_dir: folder contain tools from Caffe
- data_dir: detection dataset dataset dir
- dataset_name: name of dataset
- mapfile: path of labelmap.prototxt create from step 2
- txt_dir: path of train.txt and val.txt from step 2
- lmdb_dir: path for lmdb file
- width: 
- height: 

### Step 4
use **train_detection** to train with above lmdb dataset
























