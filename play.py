import subprocess

videoout = '../video/overlayvideo/overlayvideo.mp4'
#for remote play
#subprocess.run(['export', 'DISPLAY=:0'])
subprocess.run(['mplayer', '-speed', '.2', '-vf', 'scale', '-zoom', '-xy', '600', '-loop', '0', videoout])
