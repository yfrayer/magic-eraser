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

#list where objects and associated classes and borders are dynamically added
objects = []
borders = []
allmasks = []
allscores = []

row = 0
i = 1
for imagefile in sorted(os.listdir(IMAGE_DIR)):
       image = scipy.misc.imread(os.path.join(IMAGE_DIR, imagefile))
       results = model.detect([image], verbose=1)
       r = results[0]
       if row == 0:
           ids = []
           detected = r['class_ids'].copy()
           detected = detected.tolist()
           for j in range(len(detected)):
               ids.append(j)
           objects.append(ids)
           objects.append(detected)
       detected = r['class_ids'].copy()
       detected = detected.tolist()
       length = len(objects[0])
       #add more unique ids if not enough
       if len(detected) > length:
           diff = len(detected) - length
           add = diff + length
           for j in range(length, add):
               objects[1].append(j)
               objects[0].append(detected[j])
       #border [y1, x1, y2, x2] from top left
       border = r['rois'].copy()
       border = border.tolist()
       borders.append(border)
       #print(objects)
       #print(row)
       prev = row - 1
       shorter = 0
       if len(borders[prev][0]) < len(borders[row][0]):
           shorter = len(borders[prev][0])
       else:
           shorter = len(borders[row][0])
       diffid = []
       #determine if current borders resemble previous borders
       if row > 0:
           for j in range(shorter):
               conflict = False
               for k in range(4):
                   diff = borders[prev][j][k] - borders[row][j][k]
                   if -10 >= diff or diff >= 10:
                       print("not within range")
                       conflict = True
                   else:
                       print("within range")
               if conflict == True:
                   diffid.append(j)
       diffid = list(dict.fromkeys(diffid))
       print(diffid)
       #swap columns for borders, classes, and masks
       masks = r['masks'].copy()
       scores = r['scores'].copy()
       masks = list(masks)
       scores = list(scores)
       #print(masks)
       #print(len(masks))
       #print(len(masks[0]))
       if len(diffid) == 2:
           temp = borders[row][diffid[0]]
           tempclass = detected[diffid[0]]
           tempscore = scores[diffid[0]]
           #tempmask = masks[diffid[0]]
           borders[row][diffid[0]] = borders[row][diffid[1]]
           detected[diffid[0]] = detected[diffid[1]]
           scores[diffid[0]] = scores[diffid[1]]
           #masks[diffid[0]] = masks[diffid[1]]
           borders[row][diffid[1]] = temp
           detected[diffid[1]] = tempclass
           scores[diffid[1]] = tempscore
           #masks[diffid[1]] = tempmask
           for j in range(len(masks)):
               for k in range(len(masks[0])):
                   if masks[j][k][diffid[0]] != masks[j][k][diffid[1]]:
                       temp = masks[j][k][diffid[0]]
                       masks[j][k][diffid[0]] = masks[j][k][diffid[1]]
                       masks[j][k][diffid[1]] = temp
           #for j in range(len(detected)):
           #    print(masks[1][j][diffid[1]])
           #for j in range(6):
               #tempmask = masks[j][diffid[0]]
               #masks[j][diffid[0]] = masks[j][diffid[1]]
               #masks[j][diffid[1]] = tempmask
               #for k in range(2):
               #    tempmask = masks[j][k][diffid[0]]
               #    masks[j][k][diffid[0]] = masks[j][k][diffid[1]]
               #    masks[j][k][diffid[1]] = tempmask
                  #for l in range(3):
                  #    print(masks[j][k][l])
                  #    tempmask = masks[j][k][l][diffid[0]]
                  #    masks[j][k][l][diffid[0]] = masks[j][k][l][diffid[1]]
                  #    masks[j][k][l][diffid[1]] = tempmask
           print("swapped")
       elif len(diffid) > 2:
           matchprev = []
           matchcurr = []
           for idcurr in diffid:
               for idprev in diffid:
                   if idprev in matchprev:
                       print("already matched so skipping")
                       continue;
                   match = True
                   for k in range(4):
                       diff = borders[row][idcurr][k] - borders[prev][idprev][k]
                       if -10 >= diff or diff >= 10:
                           match = False
                   if match == True:
                       print("match is true")
                       if not (idprev in matchprev):
                           matchprev.append(idprev)
                           matchcurr.append(idcurr)
                       #matchprev.append(idprev)
                       #matchcurr.append(idcurr)
           #matched = list(dict.fromkeys(matched))
           swap = []
           swap.append(matchprev)
           swap.append(matchcurr)
           print(swap)
           for j in range(len(swap[0])):
               temp = borders[row][swap[1][j]]
               tempclass = detected[swap[1][j]]
               borders[row][swap[1][j]] = borders[row][swap[0][j]]
               detected[swap[1][j]] = detected[swap[0][j]]
               borders[row][swap[0][j]] = temp
               detected[swap[0][j]] = tempclass
               for k in range(len(masks)):
                   for l in range(len(masks[0])):
                       if masks[k][l][swap[1][j]] != masks[k][l][swap[0][j]]:
                           temp = masks[k][l][swap[1][j]]
                           masks[k][l][swap[1][j]] = masks[k][l][swap[0][j]]
                           masks[k][l][swap[0][j]] = temp
               print("swapped greater than 2")
       boxes = np.asarray(borders[row], dtype=np.float32)
       class_ids = np.asarray(detected, dtype=np.int)
       scores = np.asarray(scores, dtype=np.float32)
       masks = np.asarray(masks, dtype=np.int)
       allmasks.append(masks)
       allscores.append(scores)
       print(class_ids)
       #print(r['masks'])
       #print(r['scores'])
       #print(borders)
       #test = visualize.display_instances(objects, image, r['rois'], 
       #                          r['masks'], r['class_ids'], class_names, 
       #                          r['scores'])
       #note, take out test return from visualize if not necessary
       test = visualize.display_instances(objects, image, boxes, 
                                 masks, class_ids, class_names, 
                                 scores)
       #print(test)
       #print(r['rois'])
       print(i)
       #print(r['class_ids'])
       path = os.path.join(ROOT_DIR, '../video/mask/mask_%06i.png'%i)
       plt.savefig(path, bbox_inches='tight')
       plt.close()
       row += 1
       i += 1 

# Print variables
#print (r['class_ids'])

# Save as image
#plt.savefig('test.jpg', bbox_inches='tight')
#plt.close()
