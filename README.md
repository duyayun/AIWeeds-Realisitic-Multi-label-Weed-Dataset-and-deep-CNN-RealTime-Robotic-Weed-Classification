# AIWeeds: a large realistic weed dataset and multi-label classification between plants and weeds #
This repository makes available the source code and public dataset for our work, "Deep-CNN based Real-Time Robotic Multi-Class Weed Identification" submitted to Internation Conference on Robots and Automation (ICRA) 2021. 

Here, our first contribution is the first adequately large realistic image dataset AIWeeds (one/multiple kinds of weeds in one image), a library of around  10,000 annotated images taken from 20 different locations, including flax and the 14 most common weeds in (flaxseeds) fields and gardens in North Dakota, and California and Central China. Second, we provide a thorough pipeline from training these models with maximum efficiency to deploying the TensorRT-optimized model onto a single board computer. Based on AIWeeds dataset and the pipeline, we present a baseline for classification performance using five benchmark deep learning models: DenseNet121, InceptionV3, MobileNetV2, ResNet50, and Xception. Among them, MobileNetV2, with both the shortest inference time and lowest memory consumption, is the most competitive candidate for real-time applications. Finally, we deploy MobileNetV2 onto our own miniaturized autonomous mobile robot \textit{SAMBot} for real-time weed detection.
The 90% test accuracy realized in previously unseen scenes in flaxseeds fields (with a row spacing of 0.2-0.3 m), with crops and weeds, distortion, blur, and shadows, is a milestone towards precision weed control in the real world.

## Download the dataset images and our trained models ##
Single and negative class_image: https://drive.google.com/drive/folders/1rmvun2wcleDq5fFYR8nA83j6hiHWG9Jz?usp=sharing. 

Multiple_label_image: https://drive.google.com/drive/folders/16cZXH28BAW3lrDxtvSNPjo_z7UZvAYoA?usp=sharing.

Please git clone this repo, and create a directory called img here. 
After downloading these two groups of images, please take all images out of each subdirectory, and move all these images together to the img directory we just created. Note that img directory should contain only images. Also, please make sure all images exist in the img to correspond to csv labels.

## Environment prepartion ##
In this part, we provide detailed instructions on how to settle environment on both Nvidia platform using command line and on linux or windows using anaconda. 
Please check the requirement.txt. If you have satisfied all environments listed, you can skip this part.

### Environment installation (TensorRT supported) on Nvidia platform ###
Firstly, if the embedded system has not been flashed yet, please flash the system based on images on Nvidia. Please make sure the flashed images support the environment listed.
Particularly, for Nvidia Jetson platform, please flash with Jetpack4.3.
We can follow the following command to install the environment needed on most Nvidia platforms. 

```bash
sudo apt-get install python3-pip
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev
sudo pip3 install -U pip
sudo pip3 install protobuf==3.3.0
sudo pip3 install -U numpy
sudo pip3 install -U grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast astor termcolor protobuf keras-preprocessing wrapt google-pasta
sudo pip3 install -U h5py
sudo pip3 install -U keras-applications
sudo pip3 install -U future
sudo pip3 install -U scipy
sudo pip3 install -U setuptools testresources
pip3 list
```

Then check if we meet the requirement to install tensorflow-gpu=1.15.0. The current installed version should be at least as follows:
numpy==1.16.1, future==0.17.1, mock==3.0.5, h5py==2.9.0, gast==0.2.2, keras_preprocessing==1.0.5, keras_applications==1.0.8, scipy==1.4.1
If it satisfies the requirement, we can continue to install the tensorflow-gpu=1.15.0. 
Firstly, please find the corresponding tensorflow-gpu nvidia version on Nvidia website and download it.
Then simply follows these commands to install the remained environments:

```bash
sudo pip3 install -U (the wheel document of downloaded tensorflow-gpu)
sudo pip3 install -U python3-matplotlib
sudo pip3 install -U pillow
sudo apt-get install python3-keras
sudo pip3 install pycuda
```

Then we have all environments needed. In this process, some wheels might build for too long and fail. Please find online resources to build from raw in this case.

### Environment installation on linux or Windows ###
Firstly, please create conda environment with python version greater or equal to 3.6.6. Then you can follow these commands to install the requirements needed. 
```bash  
conda install pandas
conda install keras
conda install tensorflow-gpu>=1.11.0
conda install numpy
conda install scipy
conda install scikit-learn
conda install scikit-image
```
Then it should work fine.

## Training and Results ##
In order to train the model please follow the example command:
```bash
python3 deepweeds_dataframe_multioutput_test_2nd_version.py cross_validate --model resnet50
```
This is a command that will train based on resnet50 model. We also provide the training based on other models including inceptionv3, mobilenetv2, nannetmobile, efficientnetb0, vgg19, xception, and densenet121. Just just change to option after --model to the name of the model you want to train.
Also, if you want to change the epoches or batchs, please modify the parameter.

Then, if you want to run inference, please create the csv in the similar format as we provided, and change the directory to load the inference image in the code function.
To measure inference time, following command will be helpful:
```bash
python3 deepweeds_dataframe_multioutput_test_2nd_version.py inference --model models/resnet.hdf5
```
Where the item after option --model is the directory where the trained model is saved.
This will generate the inference results and timing.

## TensorRT speed up ##
In order to speed up the inference on Nvidia platform, we can use TensorRT. The basic steps are as followed:
1. Freeze the model to pb image file 
2. Transfer the pb image generated in step1 to the Nvidia platform
3. Get nodes name by running program
4. Run inference
Now let me discuss each step in detail

### Freeze model ###
In the file freeze_model.py, please change parameter save_pb_dir as the directory where you are about to save the first-level pb file, model_fname as the directory where the hdf5 file that is about to be transformed saved, parameter save_pb_name as the name of the first-level pb file to be saved, and at the last line, please specify the directory and name of the tensorRT pb file to save. Then use
```bash
python3 freeze_model.py
```
Then please transfer the tensorRT pb file to the Nvidia platform.

### Get nodes name and run inference on Nvidia platform ###
Please modify the directory in get_nodes.py and run it to get the name of the nodes.
Then modify the inference_on_trt3.py. Change two parameters input_names and output_names as the first and last nodes printed out in last step. Then change the directory in get_frozen_graph function to read tensorRT graph, and lastly, directory to read inference picture in code 
```bash
pics_names = os.listdir(" ")
pics_dir = "" + pics_names[i]
``` 
and then run the inference


