# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 22:08:51 2020

@author: VIvek
"""

import numpy as np
import cv2
import os
from copy import deepcopy

images=[]

def sort_by_folder_number(var):
    
    return int(var[6:])
    
def data():
    
    folders=os.listdir(os.getcwd())
    global images
    to_keep=[]
    
    for i in range(0,len(folders)):
        if folders[i][:6]=="Folder":
            to_keep.append(i)
    
    folders=np.array(folders)
    folders=folders[to_keep]
    
    folders=list(folders)
    folders.sort(key=sort_by_folder_number)
  
    for i in range(0,len(folders)):
        
        img=os.listdir(os.getcwd()+"//"+folders[i])
        a=[]
        
        for j in range(0,len(img)):
            
            a.append(cv2.imread(os.getcwd()+"//"+folders[i]+"//"+img[j]))
    
        images.append(a)
    
    

def max_roi(var): #Key to get the detected face with largest area
    
    return var[2]*var[3]

   
def crop_face(): # To detect the face and crop it
   facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

   to_remove=[]
   global images,l1
   l1=len(images)
   
   img=[]
   
   for i in range(0,l1):
       
        a=[]
        
        for j in range(0,len(images[i])):
           
           gray=cv2.cvtColor(images[i][j], cv2.COLOR_BGR2GRAY)
           faces=facecascade.detectMultiScale(gray,scaleFactor=1.03,minNeighbors=5,minSize=(50,50))
           l=len(faces)
           
           if l==1:     
               
               for x,y,w,h in faces:
                   
                   images[i][j]=images[i][j][y:y+h+10,x:x+w,:]
                   a.append(images[i][j])
                   
           elif l>1:
               
               x,y,w,h=max(faces,key=max_roi)
               images[i][j]=images[i][j][y:y+h+10,x:x+w,:]
               a.append(images[i][j])
        
        img.append(a)
           
   del images
   
   
   return img
                         
    
def standard_size(img): #TO standardize the image dimensions to 100*100
    
    img=cv2.resize(img, (100,100),cv2.INTER_NEAREST) # USed Inter_Nearest as it is the fastest thus used in real time
    
    return img

def rotate(img):
    
    shape=np.shape(images[0][0])[:-1]
    
    rot_matrix_15=cv2.getRotationMatrix2D((shape[0]/2,shape[1]/2),15,0.95)
    rot_matrix_07=cv2.getRotationMatrix2D((shape[0]/2,shape[1]/2),7,0.95)
    rot_matrix_neg_15=cv2.getRotationMatrix2D((shape[0]/2,shape[1]/2),-15,0.95)
    rot_matrix_neg_07=cv2.getRotationMatrix2D((shape[0]/2,shape[1]/2),-7,0.95)       
            
    a=cv2.warpAffine(img,rot_matrix_15,dsize=None)
    b=cv2.warpAffine(img,rot_matrix_neg_15,dsize=None)
    c=cv2.warpAffine(img,rot_matrix_07,dsize=None)
    d=cv2.warpAffine(img,rot_matrix_neg_07,dsize=None)
    
    return a,b,c,d
      


def save_processed_images():
    
    global images
    Processed_folder=os.path.dirname(os.getcwd())+"//Processed_Images"  #os.path.dirname gives the name of the parent directory
    normal_folder=Processed_folder+"//Normal"
    rotate_pos_15=Processed_folder+"//Rotate+15"
    rotate_neg_15=Processed_folder+"//Rotate-15"
    rotate_pos_07=Processed_folder+"//Rotate+7"
    rotate_neg_07=Processed_folder+"//Rotate-7"
    
    try:        
        os.mkdir(Processed_folder)         
        os.mkdir(normal_folder)        
        os.mkdir(rotate_pos_15)    
        os.mkdir(rotate_neg_15)       
        os.mkdir(rotate_pos_07)       
        os.mkdir(rotate_neg_07)
        
    except:
        print("Folder already Present")
        
    for i in range(0,len(images)):
        
        image_folder=normal_folder+"//Folder"+str(i+1)
        pos_15=rotate_pos_15+"//Folder"+str(i+1)
        neg_15=rotate_neg_15+"//Folder"+str(i+1)
        pos_07=rotate_pos_07+"//Folder"+str(i+1)
        neg_07=rotate_neg_07+"//Folder"+str(i+1)
        
        try:            
            os.mkdir(image_folder)
            
        except:
            print("Folder_already_Present")     
            
        try:          
            os.mkdir(pos_15)
        except:
            print("Folder_already_Present")     
        
        try:            
            os.mkdir(neg_15)
        except:
            print("Folder_already_Present")     
            
        try:           
            os.mkdir(pos_07)
        except:
            print("Folder_already_Present")     
            
        try:            
            os.mkdir(neg_07)
        except:
            print("Folder_already_Present")     
        
            
        for j in range(0,len(images[i])):
            
            images[i][j]=standard_size(images[i][j]) # Standardize the image size
            
            #blur function if needed
            # flip if needed
            
            r1,r2,r3,r4=rotate(images[i][j])
            
            cv2.imwrite(image_folder+"//Image"+str(j+1)+".jpg",images[i][j])
            cv2.imwrite(pos_15+"//Image"+str(j+1)+".jpg",r1)
            cv2.imwrite(neg_15+"//Image"+str(j+1)+".jpg",r2)
            cv2.imwrite(pos_07+"//Image"+str(j+1)+".jpg",r3)
            cv2.imwrite(neg_07+"//Image"+str(j+1)+".jpg",r4)
            
        
def main():
    
    data()
    global images
    images=crop_face()
    save_processed_images()

main()
    