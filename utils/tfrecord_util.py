import os
import sys
sys.path = ["C:\\Users\\infinigru\\Anaconda3\\envs\\prac\\lib\\site-packages"] + sys.path
sys.path.append("..")
import numpy as np
import cv2

from skimage.transform import rescale, resize, downscale_local_mean

import tensorflow as tf

from config import *
from utils.util import normalize_img, progress

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def parse_tfrecord(record):

    keys_to_features = {
        'height': tf.FixedLenFeature((), tf.int64),
        'width': tf.FixedLenFeature((), tf.int64),
        'image/raw': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.string),
    }

    features = tf.parse_single_example(record, features=keys_to_features)
    height = tf.cast(features['height'], tf.int64)
    width = tf.cast(features['width'], tf.int64)
    image = tf.cast(features['image/raw'], tf.string)
    label = tf.cast(features['label'], tf.string)

    image = tf.decode_raw(image, tf.float64)
    image = tf.reshape(image, shape=[height, width, -1])
    
    label = tf.decode_raw(label, tf.int32)
    label = tf.reshape(label, shape=[10,])

    return image, label

def create_tfrecord(image_dir, output_file):

    def image_and_label(image_files):
        image_list = image_files
        ## get lebel from file name 
        images_label = []
        for i,names in enumerate(image_list):
            images_label.append(names.split("_")[3].split(".")[0])
        
        for image_name, label_list in zip(image_list, images_label):
            image = cv2.imread(os.path.join(image_dir,image_name), cv2.IMREAD_COLOR)
            label = LABEL[label_list]
            yield image, label

    writer = tf.python_io.TFRecordWriter(output_file)
    
    image_files = os.listdir(image_dir)
    
    pr = progress()
    total = len(image_files)
    count = 0
    for image, label in image_and_label(image_files):
        ## image 사이즈를 균일하게 맞춤
        image = resize(image,(32,32,3))
        ## image 정규화
        image = normalize_img(image)
        
        height = image.shape[0]
        width = image.shape[1]
        example = tf.train.Example(features=tf.train.Features(feature={
                        'height': int64_feature(height),
                        'width': int64_feature(width),
                        'image/raw': bytes_feature(image.tobytes()),
                        'label': bytes_feature(np.array(label).tobytes())
                    }))
        writer.write(example.SerializeToString())
        count += 1
        pr.print_progress(1,total,count)
#         print('\r{0} done'.format(count), end='')
    print('\nfinish')
    writer.close()
    
    ## step_per_epoch 계산을위해 데이터의 length를 따로 저장 
    with open(output_file.split('.')[0] + '.length' , 'w') as f:
        f.write(str(count))
    

    

if __name__ == "__main__":
    
#     if os.path.isfile(TRAIN_FILE):
#         os.popen(f"del {TRAIN_FILE}")
    
#     if os.path.isfile(TEST_FILE):
#         os.popen(f"del {TEST_FILE}")
    
    create_tfrecord(TRAIN_IMAGE,TRAIN_FILE)
    create_tfrecord(TEST_IMAGE,TEST_FILE)