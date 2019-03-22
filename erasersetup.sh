#!/bin/bash

#for CPU use of tensorflow
sudo apt update || exit 1
sudo apt install -y python3-dev python3-pip || exit 1
pip install --upgrade pip || exit 1
sudo pip3 install -U virtualenv || exit 1
virtualenv --system-site-packages -p python3 ./venv || exit 1
source ./venv/bin/activate || exit 1
pip3 install --upgrade tensorflow || exit 1
python3 -c 'import tensorflow as tf; print(tf.__version__)' || exit 1
python3 -m pip install jupyter || exit 1
pip3 install numpy || exit 1
pip3 install scipy || exit 1
pip3 install scikit-image || exit 1
pip3 install Cython || exit 1
pip3 install opencv-python || exit 1
pip3 install runipy || exit 1
sudo apt-get install -y git-core || exit 1
mkdir mask || exit 1
cd mask || exit 1
git clone https://github.com/waleedka/coco || exit 1
git clone https://github.com/karolmajek/Mask_RCNN || exit 1
cd coco/PythonAPI || exit 1
make || exit 1
python setup.py build_ext install || exit 1
python3 setup.py install || exit 1
cd .. || exit 1
cd .. || exit 1
cd Mask_RCNN || exit 1
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 || exit 1
pip install 'keras==2.1.6' || exit 1
cd .. || exit 1
mkdir video || exit 1
cd video || exit 1
mkdir mask orig output resize trim || exit 1
