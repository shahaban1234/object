import speech_recognition as sr
import face_recognition
import numpy as np
import cv2
import scipy
from scipy.spatial import distance as dist
import time
from gtts import gTTS
import os
import pyglet
from time import sleep
import datetime
import RPi.GPIO as GPIO
import cv2 as cv
import math

#welcome to blind assistnt system

filename = "/home/pi/Desktop/objectdetect/welcome.mp3"
music = pyglet.media.load(filename, streaming = False)
music.play()
sleep(music.duration)

#voice to text recognition

r = sr.Recognizer()
speech = sr.Microphone(device_index=1)
while (1):
    
    
    
    with speech as source:
        print("say something!…")
        audio = r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        recog = r.recognize_google(audio, language = 'en-US')

        print("You said: " + recog)
        if "hello" in recog:
            print("hello")
            #tts = gTTS('hello  what can i do for you', lang="en")
            #tts.save("hello1.mp3")
            filename = "/home/pi/Desktop/objectdetect/hello1.mp3"
            #tts.save(filename)

            music = pyglet.media.load(filename, streaming = False)
            music.play()
            sleep(music.duration)
            while(1):
                with speech as source:
                    print("what can i do for you…")
                    audio = r.adjust_for_ambient_noise(source)
                    audio = r.listen(source)
                try:
                    recog2 = r.recognize_google(audio, language = 'en-US')

                    print("You said: " + recog2)
                    if "identify" in recog2:
                        
                        thres = 0.5 # Threshold to detect object
                        nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress

                       
                        

                        font = cv2.FONT_HERSHEY_PLAIN
                        

                        #cap = cv2.VideoCapture(0)  # Number According to Camera
                        cap = cv2.VideoCapture(0)
                        a=''
                        b=''
                        c=''
                        d=''
                        e=''
                        f=''
                        g=''
                        

                        timeout = time.time() + 13
                        
                        classNames = []
                        with open('/home/pi/Desktop/objectdetect/coco.names','r') as f:
                            classNames = f.read().splitlines()
                        print(classNames)
                        

                        weightsPath = "/home/pi/Desktop/objectdetect/frozen_inference_graph.pb"
                        configPath = "/home/pi/Desktop/objectdetect/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

                        # Define the codec and create VideoWriter object
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter('output21.mp4v', fourcc, 30.0, (640, 480))
                        net = cv2.dnn_DetectionModel(weightsPath,configPath)
                        net.setInputSize(320,320)
                        net.setInputScale(1.0/ 127.5)
                        net.setInputMean((127.5, 127.5, 127.5))
                        net.setInputSwapRB(True)

                       
                        
                        filename = "/home/pi/Desktop/objectdetect/hello6.mp3"
                        
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
                                    
                                    x, y, w, h = box[0], box[1], box[2], box[3]
                                    
                                    cv2.putText(frame, classNames[classIds[i] - 1] + " " + confidence, (x + 10, y + 20),
                                                font, 1, 2)
                                    if classNames[classIds[i] - 1] == 'person':
                                        a = 'person'
                                    if classNames[classIds[i] - 1] == 'cell phone':
                                        b = 'cell phone'
                                    if classNames[classIds[i] - 1] == 'laptop':
                                        c = 'laptop'
                                    if classNames[classIds[i] - 1] == 'chair':
                                        d = 'chair'
                                    if classNames[classIds[i] - 1] == 'book':
                                        e = 'book'
                                    
                                    if classNames[classIds[i] - 1] == 'bottle':
                                        g = 'bottle'
                                    


                            cv2.imshow('hi', frame)
                            if cv2.waitKey(100) == 13:
                                break

                        print(a,b,c,d,e,g)
                        if a or b or c or d or e or g in ('/home/pi/Desktop/objectdetect/coco.names','r') :
                            tts = gTTS(text=str(a) + str(b) + str(c) + str(d) + str(e) + str(g) + 'detected', lang="en")
                            tts.save("hello.mp3")
                            filename = "/home/pi/Desktop/objectdetect/hello.mp3"
                            tts.save(filename)

                            music = pyglet.media.load(filename, streaming = False)
                            music.play()
                            sleep(music.duration)


                        else:
                            tts = gTTS('nothing detected', lang="en")
                            tts.save("hello.mp3")
                            filename = "/home/pi/Desktop/objectdetect/hello.mp3"
                            tts.save(filename)

                            music = pyglet.media.load(filename, streaming = False)
                            music.play()
                            sleep(music.duration)



                        cap.release()

                        cv2.destroyAllWindows()
                        break
                    if "person" in recog2:
                        
                        name1=''
                        name2=''
                        name3=''
                        name4=''
                        filename = "/home/pi/Desktop/Tutorial 8/helloj.mp3"

                        music = pyglet.media.load(filename, streaming = False)
                        music.play()
                        sleep(music.duration)

                        video_capture = cv2.VideoCapture(0)

                        # Load a sample picture and learn how to recognize it.
                        shahaban_image = face_recognition.load_image_file("/home/pi/Desktop/Tutorial 8/shahaban/shahaban.jpg")
                        shahaban_face_encoding = face_recognition.face_encodings(shahaban_image)[0]

                        # Load a second sample picture and learn how to recognize it.
                        chai_image = face_recognition.load_image_file("/home/pi/Desktop/Tutorial 8/chaithanya/chaithanya.jpg")
                        chai_face_encoding = face_recognition.face_encodings(chai_image)[0]
                        
                        manjuma_image = face_recognition.load_image_file("/home/pi/Desktop/Tutorial 8/manjuma/manjuma.jpg")
                        manjuma_face_encoding = face_recognition.face_encodings(manjuma_image)[0]
                        
                        shankar_image = face_recognition.load_image_file("/home/pi/Desktop/Tutorial 8/shankar/shankar.jpg")
                        shankar_face_encoding = face_recognition.face_encodings(shankar_image)[0]


                        # Create arrays of known face encodings and their names
                        known_face_encodings = [
                            shahaban_face_encoding,
                            chai_face_encoding,
                            manjuma_face_encoding,
                            shankar_face_encoding
                        ]
                        known_face_names = [
                            "shahaban",
                            "chaithanya",
                            "manjuma",
                            "shankar"
                        ]

                        # Initialize some variables
                        face_locations = []
                        face_encodings = []
                        face_names = []
                        process_this_frame = True

                        timeout = time.time() + 10

                        while True and time.time()<timeout:
                            
                            # Grab a single frame of video
                            ret, frame = video_capture.read()

                            # Resize frame of video to 1/4 size for faster face recognition processing
                            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                            rgb_small_frame = small_frame[:, :, ::-1]

                            # Only process every other frame of video to save time
                            if process_this_frame:
                                # Find all the faces and face encodings in the current frame of video
                                face_locations = face_recognition.face_locations(rgb_small_frame)
                                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                                face_names = []
                                for face_encoding in face_encodings:
                                    # See if the face is a match for the known face(s)
                                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                                    name = "Unknown"

                                    # # If a match was found in known_face_encodings, just use the first one.
                                    # if True in matches:
                                    #     first_match_index = matches.index(True)
                                    #     name = known_face_names[first_match_index]

                                    # Or instead, use the known face with the smallest distance to the new face
                                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                    best_match_index = np.argmin(face_distances)
                                    if matches[best_match_index]:
                                        name = known_face_names[best_match_index]

                                    face_names.append(name)

                            process_this_frame = not process_this_frame


                            # Display the results
                            for (top, right, bottom, left), name in zip(face_locations, face_names):
                                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                                top *= 4
                                right *= 4
                                bottom *= 4
                                left *= 4

                                # Draw a box around the face
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                                # Draw a label with a name below the face
                                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                                font = cv2.FONT_HERSHEY_DUPLEX
                                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                                if name == "shahaban":
                                    name1 = "shahaban"
                                if name == "chaithanya":
                                    name2= "chaithanya"
                                if name == "manjuma":
                                    name3 = "manjuma"
                                if name == "shankar":
                                    name4 = "shankar"
                                
                        
                                
                            # Display the resulting image
                            cv2.imshow('Video', frame)
                            

                            # Hit 'q' on the keyboard to quit!
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                            

                        print(name1+' '+name2+' '+name3+' '+name4 +" detected")# Release handle to the webcam
                        if name == "shahaban" or "chaithanya" or "manjuma":
                            
                            tts = gTTS(text=name1+' '+name2+' '+name3+' '+name4 + " detected", lang="en")
                            tts.save("helloj.mp3")
                            filename = "/home/pi/Desktop/Tutorial 8/hello.mp3"
                            tts.save(filename)
                            


                            music = pyglet.media.load(filename, streaming = False)
                            music.play()
                            sleep(music.duration)
                        
                        else:
                            tts = gTTS('nothing detected', lang="en")
                            tts.save("hello.mp3")
                            filename = "/home/pi/Desktop/objectdetect/hello.mp3"
                            tts.save(filename)

                            music = pyglet.media.load(filename, streaming = False)
                            music.play()
                            sleep(music.duration)
                        video_capture.release()
                        cv2.destroyAllWindows()
                        break
                    if "song" in recog2:
                        tts = gTTS(text='laa laa la laa la sa ri ga ma pa tha ni sa', lang="en")
                        tts.save("hello2.mp3")
                        filename = "/home/pi/Desktop/Tutorial 8/hello2.mp3"
                        tts.save(filename)

                        music = pyglet.media.load(filename, streaming = False)
                        music.play()
                        sleep(music.duration)
                        break
                    if "time" in recog2:
                        day = datetime.datetime.today().weekday() + 1
                        Day_dict = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday',
                                        4: 'Thursday', 5: 'Friday', 6: 'Saturday',
                                        7: 'Sunday'}
                             
                        if day in Day_dict.keys():
                            day_of_the_week = Day_dict[day]
                             
                        now= datetime.datetime.now()
                        date=now.strftime("it's" + day_of_the_week + "and the time is %H: hours and %M minutes")

                        tts = gTTS(text=date, lang="en")
                        tts.save("hello8.mp3")
                        filename = "/home/pi/Desktop/objectdetect/hello8.mp3"
                        tts.save(filename)

                        music = pyglet.media.load(filename, streaming = False)
                        music.play()
                        sleep(music.duration)
                        break
                    if "navigation" in recog2:
                        #tts = gTTS(text="entering walk mode", lang="en")
                        #tts.save("hello9.mp3")
                        filename = "/home/pi/Desktop/objectdetect/hello9.mp3"
                        #tts.save(filename)

                        music = pyglet.media.load(filename, streaming = False)
                        music.play()
                        sleep(music.duration)
                        timeout = time.time() + 60
                        if time.time()>timeout:
                            break
                        TRIG=21
                        ECHO=20
                        GPIO.setmode(GPIO.BCM)
                        while True and time.time()<timeout:
                            print("distance measurement in progress")
                            GPIO.setup(TRIG,GPIO.OUT)
                            GPIO.setup(ECHO,GPIO.IN)
                            GPIO.output(TRIG,False)
                            print("waiting for sensor to settle")
                            time.sleep(0.2)
                            GPIO.output(TRIG,True)
                            time.sleep(0.00001)
                            GPIO.output(TRIG,False)
                            while GPIO.input(ECHO)==0:
                                pulse_start=time.time()
                            while GPIO.input(ECHO)==1:
                                pulse_end=time.time()
                            pulse_duration=pulse_end-pulse_start
                            distance=pulse_duration*17150
                            distance=round(distance,2)
                            print("distance:",distance,"cm")
                            
                            
                            
                            if distance <= 50:
                                #tts = gTTS(text="careful! obstacle ahead", lang="en")
                                #tts.save("hello110.mp3")
                                filename = "/home/pi/Desktop/objectdetect/hello110.mp3"
                                #tts.save(filename)

                                music = pyglet.media.load(filename, streaming = False)
                                music.play()
                                sleep(music.duration)
                            time.sleep(2)
                        break
                    if "bottle" in recog2:
                        
                        while True:

                            KNOWN_DISTANCE = 80 #INCHES
                            PERSON_WIDTH = 19 #INCHES
                            MOBILE_WIDTH = 3.0 #INCHES
                            var=60


                            # Object detector constant 
                            CONFIDENCE_THRESHOLD = 0.4
                            NMS_THRESHOLD = 0.3

                            # colors for object detected
                            COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
                            GREEN =(0,255,0)
                            BLACK =(0,0,0)
                            # defining fonts 
                            FONTS = cv.FONT_HERSHEY_COMPLEX

                            # getting class names from classes.txt file 
                            class_names = []
                            with open("/home/pi/Desktop/objectdetect/classes.txt", "r") as f:
                                class_names = [cname.strip() for cname in f.readlines()]
                            #  setttng up opencv net
                            yoloNet = cv.dnn.readNet('/home/pi/Desktop/objectdetect/yolov4-tiny.weights', '/home/pi/Desktop/objectdetect/yolov4-tiny.cfg')

                            yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
                            yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

                            model = cv.dnn_DetectionModel(yoloNet)
                            model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

                            # object detector funciton /method
                            def object_detector(image):
                                classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
                                # creating empty list to add objects data
                                data_list =[]
                                for (classid, score, box) in zip(classes, scores, boxes):
                                    # define color of each, object based on its class id 
                                    color= COLORS[int(classid) % len(COLORS)]
                                
                                    label = "%s : %f" % (class_names[classid], score)

                                    # draw rectangle on and label on object
                                    cv.rectangle(image, box, color, 2)
                                    cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
                                
                                    # getting the data 
                                    # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
                                    #if classid ==0: # person class id 
                                    #data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
                                    if classid ==39:
                                        data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
                                    # if you want inclulde more classes then you have to simply add more [elif] statements here
                                    # returning list containing the object data. 
                                return data_list

                            def focal_length_finder (measured_distance, real_width, width_in_rf):
                                focal_length = (width_in_rf * measured_distance) / real_width

                                return focal_length

                            # distance finder function 
                            def distance_finder(focal_length, real_object_width, width_in_frmae):
                                distance = (real_object_width * focal_length) / width_in_frmae
                                return distance

                            # reading the reference image from dir 
                            #ref_person = cv.imread('ReferenceImages/image14.png')
                            ref_mobile = cv.imread('/home/pi/Desktop/objectdetect/ReferenceImages/image1.png')

                            mobile_data = object_detector(ref_mobile)
                            mobile_width_in_rf = mobile_data[0][1]

                            #person_data = object_detector(ref_person)
                            #person_width_in_rf = person_data[0][1]

                            #print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")

                            # finding focal length 
                            #focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

                            focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
                            cap = cv.VideoCapture(0)
                            timeout = time.time() + 2
                            while True and time.time()<timeout:
                                ret, frame = cap.read()

                                data = object_detector(frame) 
                                for d in data:

                                    
                                    distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
                                    x, y = d[2]
                                    
                                    

                                    #cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
                                    cv.putText(frame, f'Dis: {round(distance)} CM', (x+5,y+13), FONTS, 0.48, GREEN, 2)
                                    var= str(math.trunc(distance))
                                    
                                    
                                    
                                    
                                    
#                               #cv.imshow('frame',frame)
                                if var == 60:
                                    print("turn right")
                                    
                                    
                                    filename = ("/home/pi/Desktop/objectdetect/hello17.mp3")
                                    

                                    music = pyglet.media.load(filename, streaming = False)
                                    music.play()
                                    sleep(music.duration)
                                    
                                
                                
                                key = cv.waitKey(1)
                                if key ==ord('q'):
                                    break
                            if var != 60:
                                    print('stop the bottle is at'+var+'centi meters'+'and move forward')
                                    tts = gTTS('the bottle is at'+var+'centi meters'+'    and move forward', lang="en")
                                    tts.save("hello10.mp3")
                                    filename = ("/home/pi/Desktop/objectdetect/hello10.mp3")
                                    tts.save(filename)

                                    music = pyglet.media.load(filename, streaming = False)
                                    music.play()
                                    sleep(music.duration)
                            

                            
                            filename = ("/home/pi/Desktop/objectdetect/hello13.mp3")
                            

                            music = pyglet.media.load(filename, streaming = False)
                            music.play()
                            sleep(music.duration)
                            cv.destroyAllWindows()
                            cap.release()
                            if int(var) <= 40:
                                
                                print('you reached the destination')
                                tts = gTTS('you reached the destination', lang="en")
                                tts.save("reach.mp3")
                                filename = ("/home/pi/Desktop/objectdetect/reach.mp3")
                                tts.save(filename)

                                music = pyglet.media.load(filename, streaming = False)
                                music.play()
                                sleep(music.duration)
                                break


                        break
                    



                        
                        
                    else:
                        tts = gTTS(text='sorry can you say once more', lang="en")
                        tts.save("hello4.mp3")
                        filename = "/home/pi/Desktop/Tutorial 8/hello4.mp3"
                        tts.save(filename)

                        music = pyglet.media.load(filename, streaming = False)
                        music.play()
                        sleep(music.duration)
                                
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))       
        
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        