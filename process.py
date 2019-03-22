import os
import subprocess
import numpy as np
import cv2

DIRORIG = 'video/orig'
DIRMASK = 'video/mask'
DIRTRIM = 'video/trim'
DIRSIZE = 'video/resize'
DIROUT = 'video/output'

#Prompt user for input and output videos
videoin = input("Which video to inpaint?: ")
videoout = input("Desired name for output video: ")

#Split first video into frames
filein = os.path.join(DIRORIG, 'output_%6d.png')
subprocess.run(['ffmpeg', '-i', videoin, filein]) 

#Run masking program
os.system('cd Mask_RCNN; python bo6.py; cd ..')

#Trim border off masked images
i = 0;
for imagefile in sorted(os.listdir(DIRMASK)):
    i = i + 1
    filein = os.path.join(DIRMASK, imagefile)
    fileout = os.path.join(DIRTRIM, 'trim_%06i.png'%i)
    subprocess.run(['convert', filein, '-trim', fileout])
    print(filein)
    print(fileout)

#Get the dimensions of original image
firstorig = os.path.join(DIRORIG, 'output_000001.png') 
imageorig = cv2.imread(firstorig)
height, width = imageorig.shape[:2]
size = str(width) + 'x' + str(height) + '!'
print(width)
print(height)
print(size)

#Resize all trimmed mask images
i = 0;
for imagefile in sorted(os.listdir(DIRTRIM)):
    i = i + 1
    filein = os.path.join(DIRTRIM, imagefile)
    fileout = os.path.join(DIRSIZE, 'output_%06i.png'%i)
    subprocess.run(['convert', filein, '-resize', size, fileout])
    print(filein)
    print(fileout)

#Inpaint
i = 0;
for imagefile in sorted(os.listdir(DIRORIG)):
    i = i + 1
    fileorig = os.path.join(DIRORIG, imagefile)
    filemask = os.path.join(DIRSIZE, imagefile)
    fileout = os.path.join(DIROUT, 'inpaint_%06i.png'%i)
    print(fileorig)
    print(filemask)
    print(fileout)
    img = cv2.imread(fileorig)
    mask = cv2.imread(filemask, 0)
    out = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(fileout, out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Combine inpainted images into video
filein = os.path.join(DIROUT, 'inpaint_%6d.png')
subprocess.run(['ffmpeg', '-i', filein, '-vf', 'scale=-1:240', '-acodec', 'mp3', '-vcodec', 'libx264', videoout]) 
