# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:36:49 2020

@author: Vivek
"""

from tensorflow.keras.models import load_model 
from tensorflow.keras.optimizers import Adam
from tensorflow import math
import os
import cv2
import numpy as np

def define_model():
    
    global model
    model=load_model("Final_model.h5",custom_objects={"math":math})
    
    count=0
    for i in model.layers:
        
        if count not in [16,7,6]:
            
            i.trainable=False
        count+=1
        
    opt=Adam(learning_rate=0.00001,beta_1=0.9,beta_2=0.999,amsgrad=True)
        
    model.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])
    
    
def max_roi(var): #Key to get the detected face with largest area
    
    return var[2]*var[3]

   
def crop_face(img): # To detect the face and crop it
    
    facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
           
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces=facecascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5,minSize=(50,50))
    l=len(faces)
           
    if l==1:     
        
        for x,y,w,h in faces:
            
            img=img[y:y+h+10,x:x+w,:]
            
    elif l>1:
        
        x,y,w,h=max(faces,key=max_roi)
        img=img[y:y+h+10,x:x+w,:]
           

   
    return img
                         

def load_images():
    
    path=os.getcwd()+"//"+"Webcam_Database_day_1"
    images=os.listdir(path) 
    
    a=cv2.imread(path+"//"+images[10])
    a=crop_face(a)
    a=cv2.resize(a,(100,100),cv2.INTER_NEAREST)
   
    
    b=cv2.imread(path+"//"+images[10])
    b=crop_face(b)
    b=cv2.resize(b,(100,100),cv2.INTER_NEAREST) 
    
    cv2.imshow("a",np.concatenate((a,b),axis=1))
    
    b=np.reshape(b,(1,100,100,3))
    a=np.reshape(a,(1,100,100,3))
  
    b=b/255
    a=a/255    
    model.predict([a,b])
    
    

