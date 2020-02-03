# -*- coding: utf-8 -*-  
#来自https://blog.csdn.net/rookie_wei/article/details/83659595

import pandas as pd
import numpy as np
import scipy.misc as sm
import os
 
emotions = {
    '0':'anger', #生气
    '1':'disgust', #厌恶
    '2':'fear', #恐惧
    '3':'happy', #开心
    '4':'sad', #伤心
    '5':'surprised', #惊讶
    '6':'normal', #中性
}
 
#创建文件夹 #Create folder
def createDir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)
 
def saveImageFromFer2013(file):
 
 
    #读取csv文件 #Read csv file
    faces_data = pd.read_csv(file)
    imageCount = 0
    #遍历csv文件内容，并将图片数据按分类保存 #Traverse csv file, classify images by expression
    for index in range(len(faces_data)):
        #解析每一行csv文件内容 #Parse each line of the csv file
        emotion_data = faces_data.loc[index][0]
        image_data = faces_data.loc[index][1]
        usage_data = faces_data.loc[index][2]
        #将图片数据转换成48*48 #Convert the images into 48*48 arrays
        data_array = list(map(float, image_data.split()))
        data_array = np.asarray(data_array)
        image = data_array.reshape(48, 48)
        dirName = usage_data
        emotionName = emotions[str(emotion_data)]
 
        #图片要保存的文件夹 #Where the images will be saved
        imagePath = os.path.join(dirName, emotionName)
 
        # 创建“用途文件夹”和“表情”文件夹 #Create classification folders by train/test and facial expression
        createDir(dirName)
        createDir(imagePath)
 
        #图片文件名 #File name of images
        imageName = os.path.join(imagePath, str(index) + '.jpg')
 
        sm.toimage(image).save(imageName)
        imageCount = index
    print('总共有' + str(imageCount) + '张图片')
 
 
if __name__ == '__main__':
    saveImageFromFer2013('fer2013.csv')