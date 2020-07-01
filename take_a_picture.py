# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 00:19:21 2020

@author: VIvek
"""

import os
import cv2
import numpy as np
#from ffpyplayer.player import MediaPlayer
from winsound import Beep


try:
    folder='Webcam_Database_day_1'
    os.mkdir(folder)
except:
    print("The Directory Webcam_Database_day_1 already exists")



def smiley():  
    smiley_path=os.getcwd()+"/Interface"
    
    smile=cv2.imread(smiley_path+"//"+os.listdir(smiley_path)[0])
    
    return smile



    
def last_screen():
    
    smile=smiley()[25:-25,25:-25,:]
    
    last_screen=np.zeros((400,600,3),"uint8")
    
    cv2.line(last_screen,(30,40),(30,350),(255,255,0),5)
    cv2.line(last_screen,(570,40),(570,350),(255,255,0),5)
    
    roi=last_screen[200:375,213:388,:]
    
    last_screen[200:375,213:388,:]=cv2.bitwise_or(roi,smile)
    
    cv2.putText(last_screen,"Thanks For Participating !!",(70,75),cv2.FONT_HERSHEY_COMPLEX,1,(70,230,230),2,)
    cv2.putText(last_screen,"Have A Nice Day Ahead !!",(95,125),cv2.FONT_HERSHEY_PLAIN,2,(70,230,230),2)
    cv2.namedWindow("Last Screen")
    cv2.imshow("Last Screen",last_screen)



    
def be_still():
    
    first_screen=np.zeros((400,600,3),"uint8")
    
    cv2.line(first_screen,(30,40),(30,350),(255,255,0),5)
    cv2.line(first_screen,(570,40),(570,350),(255,255,0),5)
    
    cv2.putText(first_screen,"Please Be Still !!",(175,50),cv2.FONT_HERSHEY_PLAIN,2,(70,230,230),2)
    cv2.namedWindow("Still")
    cv2.imshow("Still",first_screen)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
    
def backwards_count():
    
    #path_audio=os.getcwd()+"/Backward_Count_Audio"
    #aud=os.listdir(path_audio)[0]
    
    #winsound.PlaySound(path_audio+"//"+aud,winsound.SND_FILENAME)
    #player = MediaPlayer(path_audio+"//"+aud)
    
    path_vid=os.getcwd()+"/Backward_Count_Video"
    
    video=os.listdir(path_vid)[0]
    vid=cv2.VideoCapture(path_vid+"//"+video)
    
    
    while(vid.isOpened()):
        ret, frame = vid.read()
        
        if ret==False:
            break
        
        
        frame=cv2.resize(frame,(600,400),interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(30) & 0xFF == 27:
            break
        
    vid.release()
    cv2.destroyAllWindows()
    
    
def name():
    
    first_screen=np.zeros((400,600,3),"uint8")
    
    cv2.line(first_screen,(30,40),(30,350),(255,255,0),5)
    cv2.line(first_screen,(570,40),(570,350),(255,255,0),5)
    
    cv2.putText(first_screen,"Please Enter Your Name in Command Prompt !!",(105,50),cv2.FONT_HERSHEY_PLAIN,1,(70,230,230),1)
    cv2.namedWindow("Still")
    cv2.imshow("Still",first_screen)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    Beep(frequency=3000,duration=1500)
    print("Enter Your Name")
    name=input()
    
    return name
    
    
    
def snap():
    
    be_still()
    
    vid=cv2.VideoCapture(0)
    
    images=[]
    l_image=0
    
    facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cv2.destroyAllWindows()
    
    while(vid.isOpened()):
        
        if l_image==20: #Number of images to be captured for one person
            break
        
        ret, frame = vid.read()
        
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img=frame.copy()
        
        faces=facecascade.detectMultiScale(gray,scaleFactor=1.03,minNeighbors=5,minSize=(50,50))
        
        if len(faces)>=1:
            images.append(img)
            l_image+=1
            
        
        for x,y,w,h in faces:
                   
            cv2.rectangle(frame,(x,y),(x+w,y+h+10),(0,255,0),1)
            cv2.namedWindow("Face")
            cv2.imshow("Face",frame)
            ch=cv2.waitKey(1)
            if ch==27:
                break
    
    cv2.destroyAllWindows()
    images=np.stack(images)
    temp=np.mean(images,axis=0)
    final_image=np.array(temp,"uint8")
    
    screen=np.zeros((400,600,3),"uint8")
    cv2.line(screen,(30,40),(30,350),(255,255,0),5)
    cv2.line(screen,(570,40),(570,350),(255,255,0),5)

    cv2.putText(screen,"Have a look at your Image !!",(65,80),cv2.FONT_HERSHEY_PLAIN,2,(70,230,230),2)
    cv2.putText(screen,"* Press 1 to again take your image.",(75,125),cv2.FONT_HERSHEY_PLAIN,1,(70,230,230),1)
    cv2.putText(screen,"* Press 2 to end the process.",(75,150),cv2.FONT_HERSHEY_PLAIN,1,(70,230,230),1)

    cv2.imshow("Image",screen)
    ch=cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    cv2.namedWindow("Image")
    cv2.imshow("Image",final_image)  
    ch="asd"
    while(True):
        
        if ch==2:
            break
        ch=cv2.waitKey(1)
        
        if ch==ord("1"):
        
            cv2.destroyAllWindows()
            final_image=snap()            
            ch=cv2.waitKey(1)

            
        elif(ch==ord("2")):      
            cv2.destroyAllWindows()
            return final_image
        
            break
        
        if cv2.getWindowProperty('Image', 0)== -1.0:
            
            break
        
    return final_image
    
def starting_interface():
    
    first_screen=np.zeros((400,600,3),"uint8")
    
    cv2.line(first_screen,(30,40),(30,350),(255,255,0),5)
    cv2.line(first_screen,(570,40),(570,350),(255,255,0),5)
    
    cv2.putText(first_screen,"Take A Snap !!",(175,50),cv2.FONT_HERSHEY_PLAIN,2,(70,230,230),2)
    cv2.putText(first_screen,"* PRESS ENTER TO START THE PROCESS.",(75,125),cv2.FONT_HERSHEY_PLAIN,1,(70,230,230),1)
    cv2.putText(first_screen,"* PRESS ESCAPE TO EXIT THE PROCESS.",(75,150),cv2.FONT_HERSHEY_PLAIN,1,(70,230,230),1)
    
    cv2.namedWindow("First Screen")
    cv2.imshow("First Screen",first_screen)
    
    while(True):
        
        ch=cv2.waitKey(1)
        
        if ch==27:
            cv2.destroyWindow("First Screen")
            break
        elif ch==10 or ch==13:
            cv2.destroyWindow("First Screen")
            person_name=name().lower() 
            
            backwards_count()            
            img=snap()
            cv2.imwrite(os.getcwd()+"//"+folder+"//"+person_name+".jpg",img)
            last_screen()
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            break
        
    
    
        
def main():
    
    starting_interface()
    
main()

