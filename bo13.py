import os
import sys
import random
import math
import numpy as np
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import collections
import copy

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

file_names = next(os.walk(IMAGE_DIR))[2]

objects = []
borders = []
allmasks = []
allscores = []
allclasses = []
allclasses2 = []
allids = []

row = 0
i = 1
for imagefile in sorted(os.listdir(IMAGE_DIR)):
       image = scipy.misc.imread(os.path.join(IMAGE_DIR, imagefile))
       results = model.detect([image], verbose=1)
       r = results[0]
       scores = copy.deepcopy(r['scores'])
       scores = list(scores)
       if row == 0:
           ids = []
           detected = copy.deepcopy(r['class_ids'])
           detected = detected.tolist()
           for j in range(len(detected)):
               ids.append(j)
           objects.append(ids)
           objects.append(detected)
       detected = copy.deepcopy(r['class_ids'])
       detected = detected.tolist()
       length = len(objects[0])
       #add more unique ids if not enough
       if len(detected) > length:
           diff = len(detected) - length
           add = diff + length
           for j in range(length, add):
               objects[0].append(j)
               objects[1].append(detected[j])
       #remove columns from objects if too many
       elif len(detected) < length:
           diff = length - len(detected)
           for j in range(diff):
               objects[0].pop()
               objects[1].pop()
       #border [y1, x1, y2, x2] from top left
       border = copy.deepcopy(r['rois'])
       border = border.tolist()
       borders.append(border)
       prev = row - 1
       shorter = 0
       shortercurr = ""
       if row > 0:
           prevlength = len(allclasses[prev])
           currlength = len(detected)
           #use shortercurr to improve later
           if prevlength > currlength:
               shorter = currlength
               shortercurr = True
           elif prevlength < currlength:
               shortercurr = False
               shorter = prevlength
           elif prevlength == currlength:
               shorter = currlength
       diffid = []
       #determine if current borders resemble previous borders
       if row > 0:
           for j in range(shorter):
               try:
                   #uncomment to filter out low confidence masks
                   #if scores[j] < 80:
                   #    print("score less than 80")
                   #    continue
                   conflict = False
                   for k in range(4):
                       diff = borders[prev][j][k] - borders[row][j][k]
                       if -30 >= diff or diff >= 30:
                           conflict = True
                   if conflict == True:
                       diffid.append(j)
               except IndexError:
                   print("border or score does not exist")
       diffid = list(dict.fromkeys(diffid))
       #swap columns for borders, classes, and masks
       masks = copy.deepcopy(r['masks'])
       masks = list(masks)
       if len(diffid) > 0:
           matchprev = []
           matchcurr = []
           for idcurr in diffid:
               for idprev in diffid:
                   if idprev in matchprev:
                       continue
                   match = []
                   for k in range(4):
                       diff = borders[row][idcurr][k] - borders[prev][idprev][k]
                       if -30 < diff and diff < 30:
                           match.append(True)
                   if match.count(True) != 4:
                       print("no match")
                   else:
                       print("match found")
                       if not (idprev in matchprev):
                           matchprev.append(idprev)
                           matchcurr.append(idcurr)
           move = []
           move.append(matchprev)
           move.append(matchcurr)
           print(move)
           if move[0] and move[1]:
               if len(diffid) == 2 and len(move[0]) == 2 and len(move[1]) == 2:
                   if move[0][1] == move[1][0] and move[0][0] == move[1][1]:
                       temp = detected[diffid[0]]
                       detected[diffid[0]] = detected[diffid[1]]
                       detected[diffid[1]] = temp
                       temp = scores[diffid[0]]
                       scores[diffid[0]] = scores[diffid[1]]
                       scores[diffid[1]] = temp
                       temp = borders[row][diffid[0]]
                       borders[row][diffid[0]] = borders[row][diffid[1]]
                       borders[row][diffid[1]] = temp
                       for j in range(len(masks)):
                           for k in range(len(masks[0])):
                               if masks[j][k][diffid[0]] != masks[j][k][diffid[1]]:
                                   temp = masks[j][k][diffid[0]]
                                   masks[j][k][diffid[0]] = masks[j][k][diffid[1]]
                                   masks[j][k][diffid[1]] = temp
                       print("swapped")
               else:
                   moveclass= []
                   movescore = []
                   moveborder = []
                   movemask = []
                   transfermask = copy.deepcopy(masks)
                   for j in range(len(move[1])):
                       item = move[1][j]
                       moveclass.append(detected[item])
                       moveborder.append(borders[row][item])
                       movescore.append(scores[item])
                       for k in range(len(masks)):
                           for l in range(len(masks[0])):
                               transfermask[k][l][move[0][j]] = masks[k][l][move[1][j]]
                       movemask.append(transfermask)
                   #if shortercurr == True:
                   move.append(moveclass)
                   move.append(movescore)
                   move.append(moveborder)
                   move.append(movemask)
                   for j in range(len(move[0])):
                       print("doing it")
                       item = move[0][j]
                       detected[item] = move[2][j]
                       scores[item] = move[3][j]
                       borders[row][item] = move[4][j]
                       for k in range(len(masks)):
                           for l in range(len(masks[0])):
                               masks[k][l][item] = move[5][j][k][l][item]
       boxes = np.asarray(borders[row], dtype=np.float32)
       class_ids = np.asarray(detected, dtype=np.int)
       scores = np.asarray(scores, dtype=np.float32)
       masks = np.asarray(masks, dtype=np.int)
       allmasks.append(boxes)
       allscores.append(scores)
       allclasses.append(detected)
       allclasses2.append(class_ids)
       allids.append(objects[0])
       print(class_ids)
       print(objects)
       visualize.display_instances(objects, image, boxes, 
                                 masks, class_ids, class_names, 
                                 scores)
       print(i)
       path = os.path.join(ROOT_DIR, '../video/mask/mask_%06i.png'%i)
       plt.savefig(path, bbox_inches='tight')
       plt.close()
       row += 1
       i += 1 

#save data for later use
"""
objects = []
borders = []
allmasks = []
allscores = []
allclasses2 = []
allids = []
"""
