import numpy as np
import cv2
import scipy
from scipy.spatial import distance as dist
import time
from gtts import gTTS
import os
import pyglet
from time import sleep



thres = 0.5 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress

# Colors  >>> BGR Format(BLUE, GREEN, RED)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)

font = cv2.FONT_HERSHEY_PLAIN
fonts = cv2.FONT_HERSHEY_COMPLEX
fonts2 = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
fonts3 = cv2.FONT_HERSHEY_COMPLEX_SMALL
fonts4 = cv2.FONT_HERSHEY_TRIPLEX



# Camera Object
#cap = cv2.VideoCapture(0)  # Number According to Camera
cap = cv2.VideoCapture(0)
a=''
b=''

timeout = time.time() + 70



face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
Distance_level = 0
classNames = []
with open('coco.names','r') as f:
    classNames = f.read().splitlines()
print(classNames)
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

weightsPath = "frozen_inference_graph.pb"
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output21.mp4v', fourcc, 30.0, (640, 480))
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

tts = gTTS(text='please wait for the scan', lang="en")
tts.save("helloj.mp3")
filename = "/home/pi/Desktop/object detect/helloj.mp3"
tts.save(filename)

music = pyglet.media.load(filename, streaming = False)
music.play()
sleep(music.duration)


while True and time.time()<timeout:
    _, frame = cap.read()
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    if len(classIds) != 0:
        for i in indices:
            i = i
            box = bbox[i]
            confidence = str(round(confs[i], 2))
            color = Colors[classIds[i] - 1]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness=2)
            cv2.putText(frame, classNames[classIds[i] - 1] + " " + confidence, (x + 10, y + 20),
                        font, 1, color, 2)
            if classNames[classIds[i] - 1] == 'person':
                a = 'person'
            if classNames[classIds[i] - 1] == 'cell phone':
                b = 'cell phone'
            


    cv2.imshow('hi', frame)
    if cv2.waitKey(100) == 13:
        break

print(a,b)
if a or b in ('coco.names','r') :
    tts = gTTS(text=a + b + 'detected', lang="en")
    tts.save("hello.mp3")
    filename = "/home/pi/Desktop/object detect/hello.mp3"
    tts.save(filename)

    music = pyglet.media.load(filename, streaming = False)
    music.play()
    sleep(music.duration)

else:
    tts = gTTS('nothing detected', lang="en")
    tts.save("hello.mp3")
    filename = "/home/pi/Desktop/object detect/hello.mp3"
    tts.save(filename)

    music = pyglet.media.load(filename, streaming = False)
    music.play()
    sleep(music.duration)




cap.release()

cv2.destroyAllWindows()