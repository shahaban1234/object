from scipy.spatial import distance as dist
import time
from gtts import gTTS
import os
import pyglet
from time import sleep



tts = gTTS(text='please wait for the scan', lang="en")
tts.save("helloj.mp3")
filename = "/home/pi/Desktop/objectdetect/helloj.mp3"
tts.save(filename)

music = pyglet.media.load(filename, streaming = False)
music.play()
sleep(music.duration)