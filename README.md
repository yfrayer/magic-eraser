# Magic Eraser

This project currently take a video, asks the user which class(es) of objects they want to be removed, and creates a video with these objects removed.

## Setup

This project was tested on Ubuntu Studio 16.04.3 LTS with various third party applications already installed. If you do not have ImageMagick, MPlayer, and FFmpeg, you will need to install them. The version of TensorFlow used requires a CPU that supports AVX instructions. 32 GB of memory is recommended.

Put erasersetup.sh in your home directory or other desired directory. This will create a virtual environment using virtualenv, install dependencies for TensorFlow, clone repositories for using Mask RCNN, and set up directories for image processing.

Run erasersetup.sh
```
bash erasersetup.sh
```

Transfer a video to the MagicEraser directory. Objects that the program can detect are listed in the runmask.py file. Enter the MagicEraser directory and run eraser.py. It will ask the name of the input video and the desired name for the output video. It is recommended that the input video is mp4 format and the name of the output video ends in .mp4.
```
python eraser.py
```
After detecting objects, a video will be shown that highlights which objects are detected, along with each object's id and class name. After you close out of the video, the program will ask whether ids or classes should be removed, or if all detected objects should be removed. When selecting ids, it will ask for integer values. When selecting classes, it will ask for the names of the classes. Multiple inputs are separated with a space. After inputting ids or classes, it will ask for any additional inputs, or to continue.

The output file will be in the working directory or the directory specified.

## To Do

Things that have yet to be added:

Option to create a green screen effect

Provide option to restore data from previous run if same video
