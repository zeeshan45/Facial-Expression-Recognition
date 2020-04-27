#!/usr/bin/env python
# coding: utf-8

import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img
import cv2
import time




face_cascade = cv2.CascadeClassifier('static/haarcascades/haarcascade_frontalface_default.xml')


from keras.models import model_from_json

# emotions = ('angry','fear','happy','disgust','sad','neutral','surprise')
emotions = ('angry','disgust','fear','happy','sad','surprise','neutral')

from keras import backend as K
K.clear_session()

model = model_from_json(open('26-01-2020_model_structure.json').read())
model.load_weights('26-01-2020_model_weights.h5',by_name=True)

def detect(image):
    path = 'emo_photos/'+image
    image_load = cv2.imread(path)
    # image_load = cv2.imread('static/emo_photos/zee.jpg')
    gray = cv2.cvtColor(image_load,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    pred_list = []

    
    for (x,y,w,h) in faces:
        cv2.rectangle(image_load,(x,y),(x+w,y+h),(255,0,0),2)
        
        detect_face = image_load[int(y):int(y+h),int(x):int(x+w)]
        detect_face = cv2.cvtColor(detect_face,cv2.COLOR_BGR2GRAY)

        detect_face = cv2.resize(detect_face,(48,48))
        
        img_pixels = image.img_to_array(detect_face)
        
        img_pixels = np.expand_dims(img_pixels,axis=0)
        
        img_pixels /= 255
        model._make_predict_function()
        prediction = model.predict(img_pixels)
        
        max_index = np.argmax(prediction[0])
        
        emotion = emotions[max_index]
        
        percentage = round(prediction[0][max_index]*100,2)
        
        pred_list = [emotion,percentage]
        
        imag = cv2.putText(image_load,emotion,(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.imwrite("emo_photos/image.png",imag)

    # cv2.imshow('Image Prediction',image_load)
#
#     cv2.waitKey(0)
# #     Songs.songs_scrapping(emotion)
#     cv2.destroyAllWindows()
    return pred_list
detect('zee.jpg')
# detect()