# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 01:00:22 2020

@author: VIvek
"""

from selenium import webdriver
from selenium.webdriver import ActionChains
import pandas as pd
import numpy as np
import urllib.request as req
import os

os.getcwd()
os.mkdir(os.getcwd()+"\Folder1")

celebs=pd.read_csv("Celebrities.txt",header=None)

browser=webdriver.Chrome("chromedriver")

browser.get("https://images.google.com/")
search = browser.find_element_by_name("q")
search.send_keys("Salman Khan Face") # to type
search.submit() ##to submit a query in a search box given that element

images=browser.find_elements_by_css_selector(".rg_i")
link=images[0].get_attribute("src")

req.urlretrieve(link,os.getcwd()+"\Folder1"+"\img.jpg")





"""action = webdriver.ActionChains(browser)
action.context_click(images[1]).perform()"""





"""search=browser.find_element_by_css_selector(".BwoPOe").click()# to click"""



