# Magic Eraser

This project currently take a video, asks the user which class(es) of objects they want to be removed, and creates a video with these objects removed.

## Setup

This project was tested on Ubuntu Studio 16.04.3 LTS with various third party applications already installed. If you do not have ImageMagick, MPlayer, and FFmpeg, you will need to install them. The version of TensorFlow used requires a CPU that supports AVX instructions. 32 GB of memory is recommended, but memory usage depends on the number and size of frames in your video.

Put erasersetup.sh in your home directory or other desired directory. This will create a virtual environment using virtualenv, install dependencies for TensorFlow, clone repositories for using Mask RCNN, and set up directories for image processing.

Run erasersetup.sh
```
bash erasersetup.sh
```

Transfer a video to the MagicEraser directory. Objects that the program can detect are listed in the runmask.py file.

Enter the virtual environment while you're in the home directory or other directory that erasersetup.sh was run.
```
source ./venv/bin/activate
```

Enter the MagicEraser directory and run eraser.py. It will ask the name of the input video and the desired name for the output video. It is recommended that the input video is mp4 format and the name of the output video ends in .mp4.
```
python eraser.py
```
After detecting objects, a video will be shown that highlights which objects are detected, along with each object's id and class name. After you close out of the video, the program will ask whether ids or classes should be removed, or if all detected objects should be removed. When selecting ids, it will ask for integer values. When selecting classes, it will ask for the names of the classes. Multiple inputs are separated with a space. After inputting ids or classes, it will ask for any additional inputs, or to continue.

The output file will be in the working directory or the directory specified. The video directory contains videos demonstrating the steps of the program, where fourvideo.mp4 shows four videos side by side.

## GPU Setup (WIP)

To improve speed, TensorFlow can be set up for GPU support. To use TensorFlow for GPU, an NVIDIA card with CUDA support is required. The CPU and GPU versions of TensorFlow may conflict, so uninstall TensorFlow first if necessary. Magic Eraser has not been tested with GPU, but it should work the same as CPU.

Go to any directory and download and install the Miniconda installer for Python 3.7 64 bit. Anaconda may be used instead, but Minoconda is more lightweight. Miniconda is a type of virtual environment and replaces Virtualenv in this setup.
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create a conda environment named tf_gpu that installs tensorflow-gpu and various other components.
```
conda create --name tf_gpu tensorflow-gpu
```

Enter the conda environment.
```
activate tf_gpu
```

To test if tensorflow is working, enter a python shell (type python or python3), and enter the following. This should tell you what GPU you’re using and other information.
```
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

In order to install components through pip, you will need to go to the directory within miniconda containing pip, which should be located under your home directory.
```
cd #go to home directory
cd miniconda3/bin
```

In the bin directory you can install components listed in erasersetup.sh that use pip, for example:
```
pip install runipy
```

The setup procedures in erasersetup.sh are the same, except for the lines installing TensorFlow and Virtualenv. The line for installing Keras needs to be run after TensorFlow is installed and working (Line 38). Keras can be force reinstalled with --force-reinstall added to the command.

## Known Issues

The frame rate of the output does not exactly resemble input for some videos. The program extracts images at 30 fps and recreates the video at 30 fps, but a side by side comparison shows lag in the output video.

The object tracking does not work for all cases, as multiple objects moving in and out of detection causes ids to be incorrectly rearranged.

## Future Development

Option to create a green screen effect

Save images and data from previous program runs and allow user to select different objects to remove

Allow user to select which part of the video to modify

Disregard objects that have low confidence scores in object tracking
