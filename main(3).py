#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#note - cv2 documentation - https://pypi.org/project/opencv-python/
import cv2   
import numpy

#note - cars_xml - https://gist.github.com/199995/37e1e0af2bf8965e8058a9dfa3285bc6
cars_xml = 'cars.xml'     
car_video = 'car_video.mp4'
   
def detect_cars(car_video):
    rectangles = []
    #note - cv2.CascadeClassifier - https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
    cars_cascade = cv2.CascadeClassifier(cars_xml)
    car_video_capture = cv2.VideoCapture(car_video)
    #note - extracting frames from a video using opencv and python - https://stackoverflow.com/questions/36701608/extracting-frames-from-a-video-using-opencv-and-python 
    if car_video_capture.isOpened():
        rval, frame = car_video_capture.read()
    else:
        rval = False   
        
    while rval:
        rval, frame = car_video_capture.read()
        frameHeight, frameWidth, fdepth = frame.shape
        frame = cv2.resize(frame, (500, 500))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #note - cv2.detectMultiScale - https://docs.opencv.org/3.4/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498
        #note - cv2.detectMultiScale (parameters) - minSize()/maxSize()
        cars_cars = cars_cascade.detectMultiScale(gray, 1.5, 4)  
        
        for (x, y, w, h) in cars_cars:
            #note - cv2.rectangle (parameters) - https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9
            cv2.rectangle(frame, (x,y), (x+2, y+h), (0, 0, 255), 2)
        cv2.imshow('Cars in Traffic', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
            
detect_cars(car_video)   


# In[ ]:




