import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, Dropout, MaxPool2D
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
from tensorflow.keras import utils as np_utils#新版tensorflow.keras得这样引入np_utils
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import pickle
import time
from tensorflow.keras import regularizers

np.random.seed(20151306)#林正浩

path_workspace = 'C:/Users/win10/Desktop/DEMO_HAYASHI/CNN_simpliest_1x3x3'
path_Haar_Cascade_Model_path = '%s/saved_models/haarcascade_frontalface_alt.xml' %path_workspace

model_name = 'improved CNN-averagepool'

#显示训练过程、正确率与损失
def show_train_history(train_history):
    plt.plot(train_history.history['acc'], 'r')#画出曲线
    plt.plot(train_history.history['val_acc'], 'g')#画出曲线
    plt.plot(train_history.history['loss'], 'b')#画出曲线
    plt.plot(train_history.history['val_loss'], 'k')#画出曲线
    plt.title(model_name)#设置本plot的标题
    plt.ylabel('acc/loss')#设置y轴标记
    plt.xlabel('epoch')#设置x轴标记
    plt.legend(['train_acc', 'val_acc', 'train_loss', 'val_loss'], loc='upper left')#设置曲线注释在左上角
    plt.show()

#load 数据，并转给可以训练的格式
def load_data():
    dataset=pd.read_csv('../datasets/fer2013.csv')
    dataset=dataset.values
    labels=dataset[:,0]
    datas=dataset[:,1]
    flags=dataset[:,2]
    
    train_x=[]
    train_y=[]
    test_x=[]
    test_y=[]

    for i in range(datas.shape[0]):
        pixels=np.array(datas[i].split(' '))
        if(flags[i]=='Training'):
            train_x.append(pixels)
            train_y.append(labels[i])
        else:
            test_x.append(pixels)
            test_y.append(labels[i])
    train_x=np.array(train_x).astype(float)/255.0#归一化
    train_y=np.array(train_y).astype(int)
    test_x=np.array(test_x).astype(float)/255.0
    test_y=np.array(test_y).astype(int)

    train_x=train_x.reshape(-1,48,48,1)
    test_x=test_x.reshape(-1,48,48,1)

    return train_x,train_y,test_x,test_y

def load_data_IfFront():
    face_classifier = cv2.CascadeClassifier(path_Haar_Cascade_Model_path)#加载opencv人脸识别器
    dataset=pd.read_csv('../datasets/fer2013.csv')#取整个数据集
    #dataset=pd.read_csv('../datasets/fer2013_minimized.csv')#取整个数据集
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

        pixels = pixels.reshape(48,48)
        gray_face = np.resize(pixels,(48,48))#调整图片尺寸
        gray_face = gray_face.reshape(48,48,1).astype(np.uint8)#调成模型所需格式
        gray_face = cv2.cvtColor(np.asarray(gray_face), cv2.COLOR_GRAY2BGR)
        
        gray_face = cv2.copyMakeBorder(gray_face,40,40,40,40,cv2.BORDER_ISOLATED)#扩充图像边缘
        #判断是否为正脸-begin
        bFrontFace = False
        retFace = face_classifier.detectMultiScale(
            image = gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        if (len(retFace) == 0):
            #print("is empty")
            bFrontFace = False
        else:
            #print("not empty")
            bFrontFace = True
        #判断是否为正脸-end

        #想要正脸还是非正脸
        #bWantFrontOrNot = True#正脸
        bWantFrontOrNot = False#非正脸

        if(flags[i]=='Training'):
            if(bFrontFace == bWantFrontOrNot):
            #if(True):
                train_x.append(pixels)
                train_y.append(labels[i])
        else:
            if(bFrontFace == bWantFrontOrNot):
            #if(True):
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
def simpliest_CNN():
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', input_shape=[48, 48, 1]))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    return model

#更深的简单CNN结构
def deeper_simpliest_CNN():
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', input_shape=[48, 48, 1]))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model

#超级深的简单CNN结构
def super_deeper_simpliest_CNN():
    model = Sequential()
    model.add(Conv2D(16, (5,5), activation='relu', input_shape=[48, 48, 1]))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model

#MaxPool
#GAP
#dense
def CNN1():

    model = Sequential()
    model.add(Conv2D(16, (7,7), activation='relu', input_shape=[48, 48, 1], padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (7,7), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model

def github(input_shape=(48,48,1), num_classes=7):

    model = Sequential()
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                            name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(
        filters=num_classes, kernel_size=(3, 3), padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax', name='predictions'))
    return model

#AveragePooling2D
#GAP
def CNN2():

    model = Sequential()
    model.add(Conv2D(16, (7,7), activation='relu', input_shape=[48, 48, 1], padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (7,7), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (5,5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(7, (3, 3), activation='relu', padding='same'))
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax', name='predictions'))

    return model

#设置训练用参数
batch_size = 256
num_classes = 7
num_epoch = 1000
IsTrain=True
#IsTrain=False

checkpointer = ModelCheckpoint(filepath="../checkpoints/%s_checkpoint-{epoch:02d}e-val_acc_{val_acc:.4f}.h5" %model_name, monitor='val_acc',
    save_best_only=True, verbose=1, period=1)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_acc', patience=100, verbose=2)


#model = simpliest_CNN()
#model = deeper_simpliest_CNN()
#model = super_deeper_simpliest_CNN()
#model = CNN1()
model = CNN2()
#model = github((48,48,1),7)
model.compile(optimizer='adam', loss='categorical_crossentropy',
    metrics=['accuracy'])

if(IsTrain):
    #加载数据
    #train_x,train_y,test_x,test_y = load_data_IfFront()
    train_x,train_y,test_x,test_y = load_data()
    # print('train_x shape =', train_x.shape)
    # print('train_y shape =', train_y.shape)

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(train_y, num_classes)
    Y_test = np_utils.to_categorical(test_y, num_classes)

    start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    second_start = time.time()
    history = model.fit(train_x, Y_train,#开始训练
                batch_size=batch_size, nb_epoch=num_epoch,
                verbose=1,
                validation_data=(test_x, Y_test),
                callbacks=[checkpointer, learning_rate_reduction, early_stopping]
                )
    second_finish = time.time()
    finish = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    time_used = second_finish-second_start
    print('training duration(seconds):',time_used)
    print('start at:', start)
    print('finish at', finish)

    score = model.evaluate(test_x, Y_test, verbose=1)
    print('Evaluation of:' ,model_name)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print(model.summary())

    show_train_history(history)
else:
    train_x,train_y,test_x,test_y = load_data_IfFront()
    #train_x,train_y,test_x,test_y = load_data()
    Y_train = np_utils.to_categorical(train_y, num_classes)
    Y_test = np_utils.to_categorical(test_y, num_classes)
    model=load_model('%s/checkpoints/github_checkpoint-91e-val_acc_0.6531.h5' %path_workspace)
    score = model.evaluate(test_x, Y_test, verbose=1)
    print('Evaluation of:' ,model_name)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    #model.summary()


