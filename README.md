# Magic Eraser

## Setup

This project was tested on Ubuntu Studio 16.04.3 LTS with various third party applications already installed. If you do not have ImageMagick and FFmpeg, you will need to install them. The version of TensorFlow used requires a CPU that supports AVX instructions.

Put erasersetup.sh in your home directory or other desired directory. This will create a virtual environment using virtualenv, install dependencies for TensorFlow, clone repositories for using Mask RCNN, and set up directories for image processing.

Run erasersetup.sh
```
bash erasersetup.sh
```

Put bo6.py and visualize.py in the mask/Mask_RCNN directory, overwriting the original visualize.py. The modified visualize.py applies opaque masks on a black background rather than apply half opaque masks over the image.
