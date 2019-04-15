import os
import subprocess
import numpy as np
import cv2

#clear directories before adding images
def emptydir (directory):
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
        except Exception as e:
            print("no files to remove")

DIRORIG = 'video/orig'
DIROVERLAY = 'video/overlay'
DIROVEROUT = 'video/overlayvideo'
DIRMASK = 'video/mask'
DIRTRIM = 'video/trim'
DIRSIZE = 'video/resize'
DIROUT = 'video/output'

emptydir(DIRORIG)
emptydir(DIROVERLAY)
emptydir(DIRMASK)
emptydir(DIRTRIM)
emptydir(DIRSIZE)
emptydir(DIROUT)

#Prompt user for input and output videos
videoin = ""
while True:
    try:
        videoin = input("Which video to inpaint?: ")
        if not os.path.exists(videoin):
            raise ValueError
        break
    except ValueError:
        print("The file does not exist.")
videoout = input("Desired name for output video: ")
if not videoout.endswith(".mp4"):
    videoout = videoout + ".mp4"
print(videoout)

#Split input video into frames
framerate = str(30)
filein = os.path.join(DIRORIG, 'output_%6d.png')
#subprocess.run(['ffmpeg', '-i', videoin, filein]) 
subprocess.run(['ffmpeg', '-i', videoin, '-r', framerate, filein]) 

#Extract audio and frame rate from video
audioout = "video/audioout.aac"
if os.path.exists(audioout):
    os.remove(audioout)
subprocess.run(['ffmpeg', '-i', videoin, '-vn', '-acodec', 'copy', audioout])

#Run masking program
os.system('cd Mask_RCNN; python runmask.py; cd ..')

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
audioin = "video/audioout.aac"
subprocess.run(['ffmpeg', '-i', filein, '-i', audioin, '-c:a', 'libvorbis', '-r', framerate, '-c:v', 'libx264', videoout])

#Create video of black background masks
filein = os.path.join(DIRSIZE, 'output_%6d.png')
videomask = "video/maskvideo.mp4"
if os.path.exists(videomask):
    os.remove(videomask)
subprocess.run(['ffmpeg', '-i', filein, '-r', framerate, '-c:v', 'libx264', videomask])

#Resize overlay masks video
videoresize = "video/resizeoverlay.mp4"
if os.path.exists(videoresize):
    os.remove(videoresize)
videosize = 'scale=' + str(width) + ':' + str(height)
subprocess.run(['ffmpeg', '-i', 'video/overlayvideo/overlayvideo.mp4', '-vf', videosize, videoresize])

#Combine four videos
if os.path.exists('video/fourvideo.mp4'):
    os.remove('video/fourvideo.mp4')
subprocess.run(['ffmpeg', '-i', videoin, '-i', videoresize, '-i', videomask, '-i', videoout, '-filter_complex', '[0:v][1:v]hstack[t];[2:v][3:v]hstack[b];[t][b]vstack[v]; [0:a][3:a]amerge=inputs=2[a]', '-map', '[v]', '-map', '[a]', '-ac', '2', '-strict', '-2', 'video/fourvideo.mp4'])

