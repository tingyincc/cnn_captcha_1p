#!/usr/bin/env python
# coding: utf-8


import cv2
from matplotlib import pyplot as plt
import numpy as np
import csv
import os
from utils import segmentDigit, segmentDigit_binary


digit_image = []
savepath = "./train_data/"
with open('captcha_train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if(row[0] == "verify_100"):
            break
        if(row[0] == "verify_90"):
            savepath = "./valid_data/"
        digit_image = segmentDigit_binary("./verify_img/"+row[0]+".png")
        for c in range(4):
            folderpath = savepath + row[1][c]
            try:
                os.makedirs(folderpath)
            except FileExistsError:
                pass
                #print("File Already Exists.")

            cv2.imwrite(folderpath+"/"+row[0]+".jpg", digit_image[c])
