import argparse
import os
from zipfile import ZipFile
from urllib.request import urlopen
import shutil
import pandas as pd
from time import time
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adam
import csv
from tensorflow.keras.models import Model, load_model
import numpy as np
np.set_printoptions(threshold = np.inf)
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from tensorflow.keras import backend as K
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import applications
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from collections import Counter # want to check frequencies in lists with Counter()
import sys
from sklearn.utils import shuffle


# Global paths
OUTPUT_DIRECTORY = "./outputs/"
IMG_DIRECTORY = "./img"


# Global variables
RAW_IMG_SIZE_1 = (1280, 720)
RAW_IMG_SIZE_2 = (1080, 1920)
RAW_IMG_SIZE = (384, 224)

#IMG_SIZE = (255, 255)
#IMG_SIZE = (224, 224)
IMG_SIZE = (384, 224)
INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
MAX_EPOCH = 16 # originally 2
BATCH_SIZE = 32 # originally 32
FOLDS = 1
STOPPING_PATIENCE = 32
LR_PATIENCE = 16
INITIAL_LR = 0.0001

#CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10','11']
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

CLASS_NAMES =['Amaranthus_spinosus',
            'Brachypodium_sylvaticum',
            'Cirsium_arvense',
            'Dandelion',
            'Lambsquarters',
            'Negative',
            'Nutsedge',
            'Plantago_major',
            'Setaria_faberi',
            'Sonchus_arvensis_L',
            'Verdolagas']

# function to remove the label to_remove from df
def remove_label(df, to_remove):
    return df[df.Label != to_remove].reset_index(drop=True)

def remove_some_label(df, to_remove, amount):
    t = remove_label(df, to_remove)
    return(t.append(df[df.Label == to_remove].sample(amount)).reset_index(drop=True))

def get_images_counts(dir):
    count = 0
    all_dir = os.listdir(dir)
    for i in range(len(all_dir)):
        image_dir = os.path.join(dir, all_dir[i])
        count = count + len(os.listdir(image_dir))
    return count

def parse_args():
    parser = argparse.ArgumentParser(description='Train and test ResNet50, InceptionV3, or custom model on DeepWeeds.')
    parser.add_argument("command", default='train', help="'cross_validate' or 'inference'")
    parser.add_argument('--model', default='resnet', help="'resnet', 'inception', or path to .hdf5 file.")
    args = parser.parse_args()
    return args.command, args.model

def check_img_exist(df):
    filenames = df['Filename']
    validate_images = os.listdir("./img")
    for name in filenames:
        if name not in validate_images:
            print(name)

def cross_validate(model_name):

    # Create new output directory for individual folds from timestamp
    timestamp = datetime.fromtimestamp(time()).strftime('%Y%m%d-%H%M%S')
    output_directory = "{}{}/".format(OUTPUT_DIRECTORY, timestamp)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Prepare training, validation and testing labels for kth fold
    # train_image_count = get_images_counts(TRAIN_DATA_DIR)
    # val_image_count = get_images_counts(VAL_DATA_DIR)
    # test_image_count = get_images_counts(TEST_DATA_DIR)
    # print('train image count: {0}'.format(train_image_count))
    # print('val image count: {0}'.format(val_image_count))
    # print('test image count {0}'.format(test_image_count))

    # Training image augmentation to avoid drastical over-fitting
    train_data_generator = ImageDataGenerator(
        rescale = None,
        fill_mode="constant",
        shear_range=0.2,
        zoom_range=(0.5, 1),#scale both vertically and horizontally
        horizontal_flip=True,
        rotation_range=360,
        channel_shift_range=25, #account for the illumination variance, pixle intensity was randomly shifted within [-25, 25] range
        brightness_range=(0.75, 1.25)
        ) #pixel intensity

    # Validation image augmentation
    val_data_generator = ImageDataGenerator(
        rescale = None,
        fill_mode="constant",
        shear_range=0.2,
        zoom_range=(0.5, 1),
        horizontal_flip=True,
        rotation_range=360,
        channel_shift_range=25,
        brightness_range=(0.75, 1.25)
        )

    test_data_generator = ImageDataGenerator(rescale=None)


    single_label_df = pd.read_csv('single_label_no_cyn_set.csv')
    multi_label_df = pd.read_csv('multi_label_no_cyn_set.csv')
    negative_df = pd.read_csv('negative_label_no_cyn_set.csv')
    negative_df = shuffle(negative_df)
    all_df = single_label_df.append(negative_df[1:])
    all_df = all_df.append(multi_label_df[1:])
    all_df = shuffle(all_df)
    check_img_exist(all_df)
    #for data in single_label_df:
        #for i in range(len(data['Filename'])):
            #if data['Filename'][i] == '(':
                #single_label_df.remove(data['Filename'])
    #print(single_label_df)

    # Load train images in batches from directory and apply augmentations
    train_data_generator = train_data_generator.flow_from_dataframe(
            all_df[1:5600],
            IMG_DIRECTORY,
            x_col='Filename',
            y_col=CLASS_NAMES,
            target_size=RAW_IMG_SIZE,
            batch_size=BATCH_SIZE,
            #has_ext=True,
            #classes=CLASSES,
            class_mode='raw')
    #print("I am after train")
    # Load validation images in batches from directory and apply rescaling
    val_data_generator = val_data_generator.flow_from_dataframe(
            all_df[5601:7466],
            IMG_DIRECTORY,
            x_col="Filename",
            y_col=CLASS_NAMES,
            target_size=RAW_IMG_SIZE,
            batch_size=BATCH_SIZE,
            #has_ext=True,
            #classes=CLASSES,
            class_mode='raw')
    #print("I am after val")

    # Load test images in batches from directory and apply rescaling
    test_data_generator = test_data_generator.flow_from_dataframe(
            all_df[7467:],
            IMG_DIRECTORY,
            x_col="Filename",
            y_col=CLASS_NAMES,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            #has_ext=True,
            shuffle=False,
            #classes=CLASSES,
            class_mode='raw')
    #print("I am after test")
    # Load ImageNet pre-trained model with no top, either InceptionV3 or ResNet50
    if model_name == "resnet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == "inceptionv3":
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == "mobilenetv2":
        base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == "nasnetmobile":
        base_model = applications.NASNetMobile(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == "efficientnetb0":
        base_model = applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == "vgg19":
        base_model = applications.VGG19(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == "xception":
        base_model = applications.Xception(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    elif model_name == "densenet121":
        base_model = applications.DenseNet121(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

    x = base_model.output

    # Add a global average pooling layer
    x = GlobalAveragePooling2D(name='avg_pool')(x)

    # Add fully connected output layer with sigmoid activation for multi label classification
    outputs = Dense(len(CLASS_NAMES), activation='sigmoid', name='fc9')(x)

    # Assemble the modified model 
    model = Model(inputs=base_model.input, outputs=outputs)

    #print(model.summary())



    # Checkpoints for training
    model_checkpoint = ModelCheckpoint(output_directory + "lastbest-0.hdf5", verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(patience=STOPPING_PATIENCE, restore_best_weights=True)
    tensorboard = TensorBoard(log_dir=output_directory, histogram_freq=0, write_graph=True, write_images=False)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=LR_PATIENCE, min_lr=0.000003125)
    # together with changing to softmax, loss changes to sparse_categorical_crossentropy, which originally is binary_crossentropy
    #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=INITIAL_LR), metrics=['categorical_accuracy'])
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=INITIAL_LR), metrics=['accuracy'])
    csv_logger = CSVLogger(output_directory + "training_metrics.csv", append = True)

    # Train model until MAX_EPOCH, restarting after each early stop when learning has plateaued
    global_epoch = 0
    restarts = 0
    last_best_losses = []
    last_best_epochs = []
    while global_epoch < MAX_EPOCH:
        history = model.fit_generator(
            generator=train_data_generator,
            steps_per_epoch= int(5600/32),
            epochs=MAX_EPOCH - global_epoch,
            validation_data=val_data_generator,
            validation_steps= int(1867/32),
            callbacks=[tensorboard, model_checkpoint, early_stopping, reduce_lr, csv_logger],
            shuffle=False)
        print("this is history.history['val_loss']: {0}".format(history.history['val_loss']))
        last_best_losses.append(min(history.history['val_loss']))
        last_best_local_epoch = history.history['val_loss'].index(min(history.history['val_loss']))
        last_best_epochs.append(global_epoch + last_best_local_epoch)
        if early_stopping.stopped_epoch == 0:
            print("Completed training after {} epochs.".format(MAX_EPOCH))
            break
        else:
            global_epoch = global_epoch + early_stopping.stopped_epoch - STOPPING_PATIENCE + 1
            print("Early stopping triggered after local epoch {} (global epoch {}).".format(
                early_stopping.stopped_epoch, global_epoch))
            print("Restarting from last best val_loss at local epoch {} (global epoch {}).".format(
                early_stopping.stopped_epoch - STOPPING_PATIENCE, global_epoch - STOPPING_PATIENCE))
            restarts = restarts + 1
            # together with changing to softmax, loss changes to sparse_categorical_crossentropy, which originally is binary_crossentropy
            #model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=INITIAL_LR / 2 ** restarts),metrics=['categorical_accuracy'])
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=INITIAL_LR / 2 ** restarts),metrics=['accuracy'])
            model_checkpoint = ModelCheckpoint(output_directory + "lastbest-{}.hdf5".format(restarts),
                                                monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    print("this is last_best_losses: {0}".format(last_best_losses))
    print("this is last_best_epochs: {0}".format(last_best_epochs))

    # Evaluate model on test subset for kth fold
    predictions = model.predict_generator(test_data_generator)
    #print("predicted: {}".format(predictions))
    #print("actual: {}".format(df[1621:]))

    #y_true = test_data_generator.classes
    #print(y_true)
    print(predictions)
    y_true = all_df[7467:]
    y_true.drop(['Filename'], axis=1, inplace=True)
    y_true = (np.array(y_true)).tolist()
    y_pred = []
    for prediction in predictions:
        new_list = [0,0,0,0,0,0,0,0,0,0,0,0]
        negative_class_flag = True
        for index in range(len(prediction)):
            if (index != 5) and (prediction[index] > 0.25):
                new_list[index] = 1
                negative_class_flag = False
        if negative_class_flag == True:
            new_list[5] = 1
        y_pred.append(new_list)
    #print("y_true before mapping to string: {0}".format(y_true))

    #y_true = list(map(str,y_true))
    #y_pred = list(map(str,y_pred))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    #print("y_true: {0}".format(y_true))
    #print("y_pred: {0}".format(y_pred))
    print("y_true type: {0}".format(type(y_true)))
    print("y_pred type: {0}".format(type(y_pred)))
    print("y_true element type: {0}".format(type(y_true[0])))
    print("y_pred element type: {0}".format(type(y_pred[0])))
    print("y_true element element type: {0}".format(type(y_true[0][0])))
    print("y_true element element type: {0}".format(type(y_true[0][0])))

    # Generate and print classification metrics and confusion matrix
    print(classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES, zero_division='warn'))
    #report = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES, output_dict=True)
    #with open(output_directory + 'classification_report.csv', 'w') as f:
        #for key in report.keys():
            #f.write("%s,%s\n" % (key, report[key]))
    conf_arr = multilabel_confusion_matrix(y_true, y_pred, labels=CLASSES)
    print(conf_arr)
    #np.savetxt(output_directory + "confusion_matrix.csv", conf_arr, delimiter=",")

    #print("model: {0}".format(model))
    #print("{} epochs".format(MAX_EPOCH))
    #print("{0} with {1} epochs".format(model_name, MAX_EPOCH))

    # Clear model from GPU after each iteration
    print("Finished testing fold")



def inference2(model):
    # Create new output directory for saving inference times

    timestamp = datetime.fromtimestamp(time()).strftime('%Y%m%d-%H%M%S')
    output_directory = "{}{}/".format(OUTPUT_DIRECTORY, timestamp)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load DeepWeeds dataframe
    dataframe = pd.read_csv(LABEL_DIRECTORY + "all_infer.csv", dtype=str)
    all_true = list(dataframe['Label'])
    image_count = dataframe.shape[0]
    filenames = dataframe.Filename

    preprocessing_times = []
    inference_times = []
    all_pred = []  # adding this to store the predictions from inference
    for i in range(image_count):
        # Load image
        start_time = time()
        img = image.load_img(IMG_DIRECTORY + filenames[i], target_size=(384, 224))
        # Map to batch
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        # Scale from int to float
        preprocessing_time = time() - start_time
        start_time = time()
        # Predict label
        prediction = model.predict(img, batch_size=1, verbose=0)
        y_pred = np.argmax(prediction, axis=1)
        y_pred[np.max(prediction, axis=1) < 1 / 8] = 4
        all_pred.append(int(y_pred))  # adding y_pred in all_pred
        inference_time = time() - start_time
        # Append times to lists
        preprocessing_times.append(preprocessing_time)
        inference_times.append(inference_time)

    all_true = list(map(str, all_true))  # converts y_true into list of strings ie ['2', '3', '1']
    all_pred = list(map(str, all_pred))
    print(Counter(all_true))
    print(Counter(all_pred))
    print(classification_report(all_true, all_pred, labels=CLASSES, target_names=CLASS_NAMES))
    # Save inference times to csv
    with open(output_directory + "tf_inference_times.csv", 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Filename', 'Preprocessing time (ms)', 'Inference time (ms)'])
        for i in range(image_count):
            writer.writerow([filenames[i], preprocessing_times[i] * 1000, inference_times[i] * 1000])


def inference():
    model=load_model("/home/sci06/ydu/MultiOutput/outputs/20210104-004016/lastbest-0.hdf5")
    timestamp = datetime.fromtimestamp(time()).strftime('%Y%m%d-%H%M%S')
    output_directory = "{}{}/".format(OUTPUT_DIRECTORY, timestamp)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    test_data_generator = ImageDataGenerator(rescale=None)
    multi_label_df = pd.read_csv('multi_label.csv')
    negative_df = pd.read_csv('negative_label.csv')
    negative_df = shuffle(negative_df)
    multi_label_df = multi_label_df.append(negative_df[1:300])
    multi_label_df = shuffle(multi_label_df)
    test_data_generator = test_data_generator.flow_from_dataframe(
            multi_label_df[1:],
            IMG_DIRECTORY,
            x_col="Filename",
            y_col=CLASS_NAMES,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            #has_ext=True,
            shuffle=False,
            #classes=CLASSES,
            class_mode='raw')
    predictions = model.predict_generator(test_data_generator)
    y_true = multi_label_df[1:]
    y_true.drop(['Filename'], axis=1, inplace=True)
    y_true = (np.array(y_true)).tolist()
    y_pred = []
    for prediction in predictions:
        new_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        negative_class_flag = True
        for index in range(len(prediction)):
            if (index != 6) and (prediction[index] > 0.25):
                new_list[index] = 1
                negative_class_flag = False
        if negative_class_flag == True:
            new_list[6] = 1
        y_pred.append(new_list)
    # print("y_true before mapping to string: {0}".format(y_true))

    # y_true = list(map(str,y_true))
    # y_pred = list(map(str,y_pred))
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Generate and print classification metrics and confusion matrix
    print(classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES))
    report = classification_report(y_true, y_pred, labels=CLASSES, target_names=CLASS_NAMES, output_dict=True)
    with open(output_directory + 'classification_report.csv', 'w') as f:
        for key in report.keys():
            f.write("%s,%s\n" % (key, report[key]))
    conf_arr = multilabel_confusion_matrix(y_true, y_pred, labels=CLASSES)
    priny(conf_arr)

    np.savetxt(output_directory + "confusion_matrix.csv", conf_arr, delimiter=",")


if __name__ == '__main__':
    # Parse command line arguments
    (command, model) = parse_args()

    # Download images and models (if necessary)
    
    if command == "cross_validate":
        if not model == "resnet50" and not model == "inceptionv3" and not model == "mobilenetv2" and not model == "nasnetmobile" and not model == "efficientnetb0" and not \
                model == "vgg19" and not model == "xception" and not model == "densenet121":
            print("Error: You must ask for one of resnet, inception and mobilenet")
        else:
            # Train and test model on DeepWeeds with 5 fold cross validation
            cross_validate(model)
    else:
        inference()
