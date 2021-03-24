#this is the that takes in a trt frozen pb file to make inference. Parameters that needed to be changed are the output_names and input_names, the directory that loads pb file and the directory ( or method) that loads images. Note. Images need to be changed to the size accoording to the setting in the training phase to get best result.

output_names = ['fc9/Sigmoid']
input_names = ['input_1']

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import csv
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import pandas as pd
from time import time
from datetime import datetime
import os
import numpy as np


OUTPUT_DIRECTORY = "./outputs/"
LABEL_DIRECTORY = "./labels/"
IMG_DIRECTORY = "./images_resize/"
TEST_DIRECTORY = "./test2/"

CCLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
CLASS_NAMES = ['Amaranthus_spinosus',
            'Brachypodium_sylvaticum',
            'Cirsium_arvense',
            'Cynodon_dactylon',
            'Dandelion',
            'Lambsquarters',
            'Negative',
            'Nutsedge',
            'Plantago_major',
            'Setaria_faberi',
            'Sonchus_arvensis_L',
            'Verdolagas']


def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph('/home/zhangguofeng/trt_mixed_mobilenetv2.pb')

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')


# Get graph input size
for node in trt_graph.node:
    if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print("image_size: {}".format(image_size))


# input and output tensor names.
input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(
    input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)

timestamp = datetime.fromtimestamp(time()).strftime('%Y%m%d-%H%M%S')
output_directory = "{}{}/".format(OUTPUT_DIRECTORY, timestamp)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

preprocessing_times = []
inference_times = []
all_pred = [] # adding this to store the predictions from inference
full_pred = []
pics_names = os.listdir("/home/zhangguofeng/temp/plant1")
for i in range(len(pics_names)):
    pics_dir = "/home/zhangguofeng/temp/plant1/" + pics_names[i]
    #print(pics_dir)
    start_time = time()
    img = image.load_img(pics_dir,target_size=(384,224))
    # Map to batc
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # Scale from int to float
    preprocessing_time = time() - start_time
    start_time = time()
    # Predict label
    feed_dict = { input_tensor_name: x}
    predictions = tf_sess.run(output_tensor, feed_dict)
    inference_time = time() - start_time
    for prediction in predictions:
        print(prediction)
        new_list = [0,0,0,0,0,0,0,0,0,0,0,0]
        negative_class_flag = True
        for index in range(len(prediction)):
            if (index != 6) and (prediction[index] > 0.25):
                new_list[index] = 1
                negative_class_flag = False
        if negative_class_flag == True:
            new_list[6] = 1
        print(new_list)
        all_pred.append(new_list)
    # Append times to lists
    preprocessing_times.append(preprocessing_time)
    inference_times.append(inference_time)
all_pred = list(map(str, all_pred))
# Save inference times to csv
with open(output_directory + "tf_inference_times.csv", 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['Preprocessing time (ms)', 'Inference time (ms)'])
    for i in range(len(all_pred)):
        writer.writerow([preprocessing_times[i] * 1000, inference_times[i] * 1000])
