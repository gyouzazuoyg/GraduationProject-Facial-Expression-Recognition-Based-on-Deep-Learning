# -*- coding: utf-8 -*-  

from tensorflow.keras.models import load_model
import cv2
import numpy as np
path_WorkSpace = 'C:/GYOUZA/work/FormalProject_Workspace/GraduationProject/NeuralNetworkModel'
#path_FaceRecognition_model = '%s/saved_models/simple_CNN.530-0.65.hdf5' %path_WorkSpace
path_Gender_model = '%s/saved_models/gender_mini_XCEPTION.21-0.95.hdf5' %path_WorkSpace

#载入模型
#emotion_classifier = load_model(path_FaceRecognition_model)
gender_classfier = load_model(path_Gender_model)
face_classifier = cv2.CascadeClassifier(path_Haar_Cascade_Model_path)#加载opencv人脸识别器

#输出配置
#print(gender_classfier.get_config())

#输出大体模型结构
#print(gender_classfier.summary())

#测试输出
filename = '824.jpg'
img = cv2.imread('%s/temp_images/%s' %(path_WorkSpace, filename))
#cv2.imshow('111',img)#输出图片
#cv2.waitKey(0)#
#cv2.destroyAllWindows()#
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#彩色转灰度
gray_face = cv2.resize(src=gray, dsize=(64,64))#调整图片尺寸
print('type=',type(gray))
print(gray.shape)
gray_face = cv2.equalizeHist(gray_face)#直方图均衡化
gray_face = gray_face.reshape(1,64,64,1)#调成模型所需格式

prediction = np.argmax(gender_classfier.predict(gray_face))#输出最大值
print(prediction)
