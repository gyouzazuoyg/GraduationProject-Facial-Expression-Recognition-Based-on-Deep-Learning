# -*- coding: utf-8 -*-  

#python 3.7
#林　正浩

from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from faceStandardization import normalize

#工作目录
path_WorkSpace = 'C:/GYOUZA/work/FormalProject_Workspace/GraduationProject/FacialExpressionRecognition'
#输出表情表情用的字体
path_Font = 'C:/Windows/Fonts/simhei.ttf'
#人脸识别器模型路径
path_Haar_Cascade_Model_path = '%s/haarCascades_models/haarcascade_frontalface_alt.xml' %path_WorkSpace
#path_Haar_Cascade_Model_path = '%s/haarCascades_models/haarcascade_frontalface_default.xml' %path_WorkSpace
#表情识别模型路径
#path_FacialExpressionRecognition_model = '%s/trained_models/CNN_model.h5' %path_WorkSpace
path_FacialExpressionRecognition_model = '%s/trained_models/simple_CNN.530-0.65.hdf5' %path_WorkSpace


face_classifier = cv2.CascadeClassifier(path_Haar_Cascade_Model_path)#加载opencv人脸识别器
img1 = cv2.imread('%s/temp_images/28779.jpg' %path_WorkSpace)#not empty
img2 = cv2.imread('%s/temp_images/nothing.png' %path_WorkSpace)#empty
print('type=',type(img1))
print('shape=',img1.shape)
print('element=',img1[0][0])
print('type of num=',type(img1[0][0][0]))
print('content=',img1)
forPrint = face_classifier.detectMultiScale(
        image=img1, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
#若获取图像成功，获取了x张人脸，返回值就是2维数组，每个元素是人脸的坐标四个值
#若获取图像失败，范围是个空的tuple: “()”
print("return value is :", forPrint)
if (len(forPrint) == 0):
    print("is empty")
else:
    print("not empty")
