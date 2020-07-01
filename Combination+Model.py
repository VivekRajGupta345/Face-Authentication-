# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 22:43:13 2020

@author: VIvek
"""

import os
import numpy as np
import cv2
from random import seed
from random import sample
from random import choice
import pandas as pd
from tensorflow.keras.layers import Conv2D,Dense,MaxPool2D,Input,Flatten,BatchNormalization,Lambda,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow import math
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import confusion_matrix,roc_curve


seed(13)

def positive_intra_folder():
    
    path_dir=os.getcwd()+"//Processed_Images"
    
    images_folders=os.listdir(path_dir)
    
    positive_pairs_ind=[]
    
    for i in images_folders:
        
        path_images=path_dir+"//"+i
        images_subfolder=os.listdir(path_images)
        
        for j in images_subfolder:
            
            images=os.listdir(path_dir+"//"+i+"//"+j)
            
            img=[]
            
            len_img=0
            
            for k in images:
                
                img.append(cv2.imread(path_dir+"//"+i+"//"+j+"//"+k))
                len_img+=1
                
            
            for k in range(0,len_img-1):
                
                for q in range(k+1,len_img):
                    
                    positive_pairs_ind.append([img[k],img[q],1])
                    
    positive_pairs_ind= sample(positive_pairs_ind,800)
    return positive_pairs_ind
    
def positive_inter_folder():
    
    path_dir=os.getcwd()+"//Processed_Images"
    
    images_folders=os.listdir(path_dir)
    
    positive_pairs_inter=[]
    
    len_images_folder=len(images_folders)
    for i in range(0,len_images_folder-1):
        
        folder_path_1=path_dir+"//"+images_folders[i]
        
        
        list_folder=os.listdir(folder_path_1)

        for j in range(i+1,len_images_folder):
            
            folder_path_2=path_dir+"//"+images_folders[j]
            
            for k in list_folder:
                
                images=os.listdir(folder_path_2+"//"+k)
                l_images=len(images)
                
                for q in range(0,l_images-1):
                    
                    for s in range(q,l_images):
                        
                        positive_pairs_inter.append([cv2.imread(folder_path_1+"//"+k+"//"+images[q]),cv2.imread(folder_path_2+"//"+k+"//"+images[s]),1])
    
    
    positive_pairs_inter= sample(positive_pairs_inter,800)
    return positive_pairs_inter
    

def positive_pair_data():
    
    p1=positive_intra_folder()
    p2=positive_inter_folder()
    p_final=p1+p2
    
    return p_final
    


def negative_intra():
    
    path_dir=os.getcwd()+"//Processed_Images"
    
    images_folders=os.listdir(path_dir)    
    len_image_folder=len(images_folders)
    negative_pairs_intra=[]
    
    for i in range(0,len_image_folder):
        
        folder=os.listdir(path_dir+"//"+images_folders[i])
        len_folder=len(folder)
        
        for j in range(0,len_folder-1):
            
            images_f1=os.listdir(path_dir+"//"+images_folders[i]+"//"+folder[j])
            images_f2=os.listdir(path_dir+"//"+images_folders[i]+"//"+folder[j+1])
            
            for k in images_f1:
                
                for m in images_f2:
                    
                    negative_pairs_intra.append([cv2.imread(path_dir+"//"+images_folders[i]+"//"+folder[j]+"//"+k),cv2.imread(path_dir+"//"+images_folders[i]+"//"+folder[j+1]+"//"+m),0])
    
    negative_pairs_intra= sample(negative_pairs_intra,1000)
    
    return negative_pairs_intra


def negative_pairs_inter():
    
    path_dir=os.getcwd()+"//Processed_Images"    
    images_folders=os.listdir(path_dir)
    len_image_folders=len(images_folders)
    
    negative_pairs=[]
    
    integers=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    
    into=[]
    for i in range(0,len_image_folders-1):
        
        folder_list_1=os.listdir(path_dir+"//"+images_folders[i])
        
        
        for j in range(i+1,len_image_folders):
            
            folder_list_2=os.listdir(path_dir+"//"+images_folders[j])
            
            for k in folder_list_1:
                
                for l in folder_list_2:
                    
                    if k==l:
                        pass
                    else:
                        
                        images_1=os.listdir(path_dir+"//"+images_folders[i]+"//"+k)
                        images_2=os.listdir(path_dir+"//"+images_folders[j]+"//"+l)
                        
                        for m in images_1:
                            
                            for n in images_2:
                                
                                draw=choice(integers)
                                
                                if draw in [1,3] :
                                
                                    into.append(draw)
                                    img1_path=path_dir+"//"+images_folders[i]+"//"+k+"//"+m
                                    img2_path=path_dir+"//"+images_folders[j]+"//"+l+"//"+n                                
                                    negative_pairs.append([cv2.imread(img1_path),cv2.imread(img2_path),0])
    
    negative_pairs=sample(negative_pairs,1000)
    return negative_pairs
                            
def negative_pair_data():
    
    n1=negative_intra()
    n2=negative_pairs_inter()
    
    return n1+n2  
    


def dataset():
    
    positive=positive_pair_data()
    negative=negative_pair_data()
    
    global X,y
    dataset=pd.DataFrame(positive+negative)    
    dataset=dataset.sample(frac=1)
    dataset=np.array(dataset)
    X=dataset[:,:-1] /255
    y=dataset[:,-1:].astype(int)
    
    
    

def view(pair):
    
    for i in range(0,len(pair)):
        print(i)
        a=np.concatenate((pair[i][0],pair[i][1]),axis=1)
        b=0
        cv2.imshow("pair_img",a)
        print(y[i])
        while True:
            if cv2.waitKey(1)== ord("q"):
               
                break
            if cv2.getWindowProperty("pair_img",0)!=0:
                b=1
                break
        if b==1:
            break
 

 

def define_model():
    
    
    ##################################Image-1########################################################
    
    input_1=Input(shape=np.shape(X[0][0]),batch_size=None,name="Image_1") 
    
    conv_1_1=Conv2D(filters=5,kernel_size=(13,13),strides=1,padding="same",activation="relu",use_bias=True,kernel_initializer="glorot_uniform",kernel_regularizer=l2(0.01))(input_1)
    
    max_pool_1_1=MaxPool2D(pool_size=(7,7),strides=1,padding="valid")(conv_1_1)
    
    conv_2_1=Conv2D(filters=4,kernel_size=(9,9),strides=1,padding="valid",activation="relu",use_bias=True,kernel_initializer="glorot_uniform",kernel_regularizer=l2(0.01))(max_pool_1_1)
    
    max_pool_2_1=MaxPool2D(pool_size=(5,5),strides=1,padding="valid")(conv_2_1)

    norm_1_1=BatchNormalization()(max_pool_2_1)
    
    conv_3_1=Conv2D(filters=5,kernel_size=(3,3),strides=1,padding="valid",activation="relu",use_bias=True,kernel_initializer="glorot_uniform",kernel_regularizer=l2(0.001))(norm_1_1)
        
    max_pool_3_1=MaxPool2D(pool_size=(3,3),strides=1,padding="valid")(conv_3_1)
    
    ################################Image-2######################################################
    
    input_2=Input(shape=np.shape(X[0][0],),batch_size=None,name="Image_2") 
    
    conv_1_2=Conv2D(filters=5,kernel_size=(13,13),strides=1,padding="same",activation="relu",use_bias=True,kernel_initializer="glorot_uniform",kernel_regularizer=l2(0.01))(input_2)
    
    max_pool_1_2=MaxPool2D(pool_size=(7,7),strides=1,padding="valid")(conv_1_2)
    
    conv_2_2=Conv2D(filters=4,kernel_size=(9,9),strides=1,padding="valid",activation="relu",use_bias=True,kernel_initializer="glorot_uniform",kernel_regularizer=l2(0.01))(max_pool_1_2)
    
    max_pool_2_2=MaxPool2D(pool_size=(5,5),strides=1,padding="valid")(conv_2_2)

    norm_1_2=BatchNormalization()(max_pool_2_2)
    
    conv_3_2=Conv2D(filters=5,kernel_size=(3,3),strides=1,padding="valid",activation="relu",use_bias=True,kernel_initializer="glorot_uniform",kernel_regularizer=l2(0.001))(norm_1_2)
        
    max_pool_3_2=MaxPool2D(pool_size=(3,3),strides=1,padding="valid")(conv_3_2)
    
    ##########################################################################################
    
    
    lambda_func=Lambda(lambda x:math.abs(x[0]-x[1]))([max_pool_3_1,max_pool_3_2]) #([norm_1_1,norm_1_2])#
    
    ##############################################################################
    ##########################Full-Connection#####################################
    
    flat=Flatten()(lambda_func)
    
    dense_1=Dense(units=256,activation="relu",use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="glorot_uniform",kernel_regularizer=l2(0.02),activity_regularizer=l2(0.02))(flat)
    
    drop1=Dropout(0.4)(dense_1)
    
    dense_2=Dense(units=128,activation="relu",use_bias=False,kernel_initializer="glorot_uniform",bias_initializer="glorot_uniform",kernel_regularizer=l2(0.02))(drop1) #(flat)
    
    drop2=Dropout(0.4)(dense_2)
    
    dense_3=Dense(units=1,activation="sigmoid",use_bias=False,kernel_initializer="glorot_uniform",bias_initializer="glorot_uniform",kernel_regularizer=l2(0.02))(drop2)
    
    model=Model(inputs=[input_1,input_2],outputs=dense_3)
    #sgd = tensorflow.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001)
    opt=Adam(learning_rate=0.0001,beta_1=0.9,beta_2=0.999,amsgrad=True)
    
    model.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])
    
    return model

    

def K_fold_evaluation():
    
    
    skf=StratifiedKFold(n_splits=10)
    global roc
    roc=[]
    global accuracy
    global precision
    global recall
    accuracy=[]
    sensitivity=[]
    specificity=[]
    precision=[]
    recall=[]
   
    global acc
    for train_index,test_index in skf.split(X_train[:,0],y_train):
   
        model=define_model()
        
        model.fit([np.stack(X_train[:,0][train_index],axis=0),np.stack(X_train[:,1][train_index],axis=0)],y=y_train[train_index],batch_size=16,epochs=25,verbose=1)
        
        y_score=model.predict([np.stack(X_train[:,0][test_index],axis=0),np.stack(X_train[:,1][test_index],axis=0)])
        
        fpr,tpr,_=roc_curve(y_train[test_index],y_score)
        
        roc.append([fpr,tpr])
        
        for i in range(0,len(y_score)):
            if y_score[i]>=0.5:
                y_score[i]=1
            else:
                y_score[i]=0
        
        cm1=confusion_matrix(y_train[test_index],y_score)
        
        cm1=confusion_matrix(y_train[test_index],y_score)
        acc = (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
        spec= (cm1[0,0])/(cm1[0,0]+cm1[0,1])
        sens = (cm1[1,1])/(cm1[1,0]+cm1[1,1])
        prec=cm1[1,1]/(cm1[1,1]+cm1[0,1])
        rec=cm1[1,1]/(cm1[1,1]+cm1[1,0])
        
        sensitivity.append(sens)
        specificity.append(spec)
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)   
        
        
    global stats
    stats=[]
    stats.append(['Accuracy',accuracy])
    stats.append(['Specificity',specificity])
    stats.append(['Sensitivity',sensitivity])
    stats.append(['Precision',precision])
    stats.append(['Recall',recall])
    
    print("Mean Accuracy is :",np.mean(accuracy))
    print("Mean Precision is :",np.mean(precision))
    print("Mean Recall is :",np.mean(recall))
    
def data_split():
    
    global X_train,X_test,y_train,y_test
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=250,shuffle=True,random_state=13)
    #X_test and y_test will be used for final testing and before that will reamin untouched

def final_evaluation():
    
    global model
    model=define_model()
    
    model.fit([np.stack(X_train[:,0],axis=0),np.stack(X_train[:,1],axis=0)],y=y_train,batch_size=16,epochs=30,verbose=1,use_multiprocessing=True)
    
    model.save("Final_model.h5")
    
    y_score=model.predict([np.stack(X_test[:,0],axis=0),np.stack(X_test[:,1],axis=0)])
    
    for i in range(0,len(y_score)):
        if y_score[i]>=0.5:
            y_score[i]=1
        else:
            y_score[i]=0
    
    cm1=confusion_matrix(y_test,y_score)
    
    cm1=confusion_matrix(y_test,y_score)
    acc = (cm1[0,0]+cm1[1,1])/(cm1[0,0]+cm1[0,1]+cm1[1,0]+cm1[1,1])
    spec= (cm1[0,0])/(cm1[0,0]+cm1[0,1])
    sens = (cm1[1,1])/(cm1[1,0]+cm1[1,1])
    prec=cm1[1,1]/(cm1[1,1]+cm1[0,1])
    rec=cm1[1,1]/(cm1[1,1]+cm1[1,0])
    
    print("Final Accuracy on test set is :",acc)
    print("Final Precison on test set is :",prec)
    print("Final recall on test set is :",rec)
    print("Final Sensitivity on test set is :",sens)
    print("Final Specificity on test set is :",spec)
    
        
        
def main():
    dataset()
    data_split()
    K_fold_evaluation()
    #final_evaluation()
    
main()

"""a=np.stack(X[:,0],axis=0)
np.shape(X[0][0])
np.shape(a[0])

a[1]==X[:,0][1]

B=a[2,:,:,:]
cv2.imshow("img",B)

cv2.imshow("img2",X[:,0][2])"""

