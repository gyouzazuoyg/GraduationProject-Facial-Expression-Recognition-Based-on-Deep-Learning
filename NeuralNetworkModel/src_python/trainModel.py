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

np.random.seed(20151306)#林正浩 #Lin, ZhengHao

path_workspace = 'C:/GYOUZA/work/FormalProject_Workspace/GraduationProject/NeuralNetworkModel'

#csv读取数据函数 #Load data from CSV function
def load_data_fromCSV():
    #dataset=pd.read_csv('./datasets/fer2013/fer2013.csv')#取整个数据集 #Read the whole dataset
    dataset=pd.read_csv('./datasets/fer2013/fer2013_minimized.csv')#取部分数据集 #Read part of the dataset
    dataset = dataset.values
    labels = dataset[:,0]
    datas = dataset[:,1]
    flags = dataset[:,2]
    
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]

    for i in range(datas.shape[0]):
        pixels=np.array(datas[i].split(' '))#空格分隔符

        if(flags[i]=='Training'):
            train_x.append(pixels)
            train_y.append(labels[i])
        else:
            test_x.append(pixels)
            test_y.append(labels[i])
    train_x=np.array(train_x).astype(float)/255.0
    train_y=np.array(train_y).astype(int)
    test_x=np.array(test_x).astype(float)/255.0
    test_y=np.array(test_y).astype(int)

    train_x=train_x.reshape(-1,48,48,1)
    test_x=test_x.reshape(-1,48,48,1)

    return train_x,train_y,test_x,test_y #X=图片，Y=表情标签

#简单CNN结构 #simple CNN
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



#设置训练用参数 #Set parameter for training
batch_size = 1
num_classes = 7
num_epoch = 2
#IsTrain=True
IsTrain=False


#加载数据
train_x,train_y,train_z,test_x,test_y,test_z=load_data_fromCSV()
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(train_y, num_classes)
Y_test = np_utils.to_categorical(test_y, num_classes)
print('test=', train_z[0])

if(IsTrain):

    model = build_model()
    model.fit(train_x, Y_train,#开始训练
                batch_size=batch_size, nb_epoch=num_epoch,
                validation_split=0.2,
                verbose=1)
    model.save('%s/saved_models/CNN_model.h5' %path_workspace)
else:
    model = keras.models.load_model('%s/saved_models/CNN_model.h5' %path_workspace)

score = model.evaluate(train_x, Y_train, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
