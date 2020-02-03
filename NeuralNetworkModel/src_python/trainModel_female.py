import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
from tensorflow.keras import utils as np_utils#新版tensorflow.keras得这样引入np_utils

np.random.seed(20151306)#林正浩

path_workspace = 'C:/GYOUZA/work/FormalProject_Workspace/GraduationProject/NeuralNetworkModel'
path_Gender_model = '%s/saved_models/gender_mini_XCEPTION.21-0.95.hdf5' %path_workspace

def load_data_fromCSV_male():
    #dataset=pd.read_csv('./datasets/fer2013/fer2013.csv')#取整个数据集
    dataset=pd.read_csv('./datasets/fer2013/fer2013_minimized.csv')#取整个数据集
    dataset = dataset.values
    labels = dataset[:,0]
    datas = dataset[:,1]
    flags = dataset[:,2]
    
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]

    gender_classfier = load_model(path_Gender_model)#载入性别模型

    for i in range(datas.shape[0]):
        pixels=np.array(datas[i].split(' '))#空格分隔符

        pixels = pixels.reshape(48,48)
        gray_face = np.resize(pixels,(64,64))#调整图片尺寸
        gray_face = gray_face.reshape(1,64,64,1)#调成模型所需格式
        np.argmax(gender_classfier.predict(gray_face))#0-female, 1-male

        if(flags[i]=='Training'):
            if(np.argmax(gender_classfier.predict(gray_face)) == 0):#只接收女性样本
                train_x.append(pixels)
                train_y.append(labels[i])
        else:
            if(np.argmax(gender_classfier.predict(gray_face)) == 0):
                test_x.append(pixels)
                test_y.append(labels[i])
    train_x=np.array(train_x).astype(float)/255.0
    train_y=np.array(train_y).astype(int)
    test_x=np.array(test_x).astype(float)/255.0
    test_y=np.array(test_y).astype(int)

    train_x=train_x.reshape(-1,48,48,1)
    test_x=test_x.reshape(-1,48,48,1)

    return train_x,train_y,test_x,test_y #X=图片，Y=表情标签

def load_data_fromCSV_female():
    #dataset=pd.read_csv('./datasets/fer2013/fer2013.csv')#取整个数据集
    dataset=pd.read_csv('./datasets/fer2013/fer2013_minimized.csv')#取整个数据集
    dataset = dataset.values
    labels = dataset[:,0]
    datas = dataset[:,1]
    flags = dataset[:,2]
    
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]

    gender_classfier = load_model(path_Gender_model)#载入性别模型

    for i in range(datas.shape[0]):
        pixels=np.array(datas[i].split(' '))#空格分隔符

        pixels = pixels.reshape(48,48)
        gray_face = np.resize(pixels,(64,64))#调整图片尺寸
        gray_face = gray_face.reshape(1,64,64,1)#调成模型所需格式
        np.argmax(gender_classfier.predict(gray_face))#0-female, 1-male

        if(flags[i]=='Training'):
            if(np.argmax(gender_classfier.predict(gray_face)) == 0):
                train_x.append(pixels)
                train_y.append(labels[i])
        else:
            if(np.argmax(gender_classfier.predict(gray_face)) == 0):
                test_x.append(pixels)
                test_y.append(labels[i])
    train_x=np.array(train_x).astype(float)/255.0
    train_y=np.array(train_y).astype(int)
    test_x=np.array(test_x).astype(float)/255.0
    test_y=np.array(test_y).astype(int)

    train_x=train_x.reshape(-1,48,48,1)
    test_x=test_x.reshape(-1,48,48,1)

    return train_x,train_y,test_x,test_y #X=图片，Y=表情标签


#简单CNN结构
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', input_shape=[48, 48, 1]))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])
    return model

############################################
#-------------------MAIN-------------------#
############################################



#设置训练用参数
batch_size = 10000
num_classes = 7
num_epoch = 1
IsTrain=True
#IsTrain=False


#加载数据
train_x,train_y,test_x,test_y = load_data_fromCSV_female()
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(train_y, num_classes)
Y_test = np_utils.to_categorical(test_y, num_classes)

if(IsTrain):

    model = build_model()
    model.fit(train_x, Y_train,#开始训练
                batch_size=batch_size, nb_epoch=num_epoch,
                validation_split=0.2,
                verbose=1)
    model.save('%s/saved_models/CNN_model_female.h5' %path_workspace)
else:
    model = keras.models.load_model('%s/saved_models/CNN_model.h5' %path_workspace)

score = model.evaluate(train_x, Y_train, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
