import os
import sys
import random
import math
import numpy as np
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "../video/orig")

classes = list(map(int, input("Select which class to inpaint: ").split()))

class InferenceConfig(coco.CocoConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

config = InferenceConfig()
config.print()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
           	'bus', 'train', 'truck', 'boat', 'traffic light',
           	'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
           	'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
           	'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
           	'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
           	'kite', 'baseball bat', 'baseball glove', 'skateboard',
           	'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
           	'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           	'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
           	'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
           	'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
           	'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
           	'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           	'teddy bear', 'hair drier', 'toothbrush']

# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
#image = scipy.misc.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection
#results = model.detect([image], verbose=1)

# Visualize results
#r = results[0]
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                        	class_names, r['scores'])
# Loop images
#num = 0;
#for file in os.listdir(IMAGE_DIR):
#       if file.endswith(".png"):
#              num += 1
#print("number of images: " + str(num))


i = 1
for imagefile in sorted(os.listdir(IMAGE_DIR)):
       image = scipy.misc.imread(os.path.join(IMAGE_DIR, imagefile))
       results = model.detect([image], verbose=1)
       r = results[0]
       selection = r['class_ids'].copy()
       empty = True;
       for j in range(len(selection)):
           if selection[j] in classes:
               selection[j] = 1
               empty = False;
           else:
               selection[j] = 0
       visualize.display_instances(selection, empty, image, r['rois'], 
                                 r['masks'], r['class_ids'], class_names, 
                                 r['scores'])
       print(i)
       print(r['class_ids'])
       print(classes)
       print(selection)
       print(empty)
       path = os.path.join(ROOT_DIR, '../video/mask/mask_%06i.png'%i)
       plt.savefig(path, bbox_inches='tight')
       plt.close()
       i += 1 

# Print variables
#print (r['class_ids'])

# Save as image
#plt.savefig('test.jpg', bbox_inches='tight')
#plt.close()
