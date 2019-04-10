# Magic Eraser

This project currently take a video, asks the user which class(es) of objects they want to be removed, and creates a video with these objects removed.

## Setup

This project was tested on Ubuntu Studio 16.04.3 LTS with various third party applications already installed. If you do not have ImageMagick and FFmpeg, you will need to install them. The version of TensorFlow used requires a CPU that supports AVX instructions. 32 GB of memory is recommended.

Put erasersetup.sh in your home directory or other desired directory. This will create a virtual environment using virtualenv, install dependencies for TensorFlow, clone repositories for using Mask RCNN, and set up directories for image processing.

Run erasersetup.sh
```
bash erasersetup.sh
```

Put bo6.py and visualize.py in the mask/Mask_RCNN directory, overwriting the original visualize.py. The modified visualize.py applies opaque masks on a black background rather than apply half opaque masks over the image.

Put process.py in the mask directory. Also put a video of your choice in this directory. When running process.py, it will ask the name of your video file. Then it will ask the desired name for the output video (recommended to add .mp4 to end of file name). After initial processing, it will ask which class you want to be removed by typing in a number, i.e. 1 for person. You can input more than one class by separating the numbers with spaces. You can find a list of objects that can be detected in bo6.py.

Run process.py
```
python process.py
```
The output file will be in the mask directory.

## To Do

Things that have yet to be added:

Process the video with half opaque masks and automatically show to the user, give a list of classes/objects/indices detected, and then ask the user what they want to remove

Allow the user to select specific objects, not just all objects that match a class

Option to create a green screen effect

Simplify setup process

Remove images from folders before processing a new video

Provide option to restore data from previous run if same video
