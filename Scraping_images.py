# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 23:09:01 2020

@author: VIvek
"""
import pandas as pd
import os
import urllib.request as req
from selenium import webdriver
import numpy as np


def celeb_list():
    global celebs
    celebs=pd.read_csv("Celebrities.txt",header=None)
    celebs=np.array(celebs)
    
    

def scrape():
    browser=webdriver.Chrome("chromedriver")
    for i in range(0,len(celebs)):
        try:
            os.mkdir(os.getcwd()+"\Folder"+str(i+1))   
        except:
            print("Folder Already Exist")
        
        browser.get("https://images.google.com/")
        search = browser.find_element_by_name("q")
        
        search.send_keys(celebs[i][0]+" face")
        search.submit()
        
        images=browser.find_elements_by_css_selector(".rg_ic")
            
        
        for j in range(0,10):
            try:                
                req.urlretrieve(str(images[j].get_attribute("src")),os.getcwd()+"\Folder"+str(i+1)+"\img"+str(j+1)+".jpg")
            except:
                print("error")
    browser.quit()

def main():
    celeb_list()
    
    scrape()
main()