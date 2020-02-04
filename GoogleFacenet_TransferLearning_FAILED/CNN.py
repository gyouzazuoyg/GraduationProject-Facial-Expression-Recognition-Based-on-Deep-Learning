#加载keras模块 #Load models
import numpy as np
import keras
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.utils import to_categorical
import pandas as pd
import pickle

np.random.seed(1337)  # for reproducibility

#LossHistory类，保存loss和acc #LossHistory class used to save loss and acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

#load 数据，并转给可以训练的格式 #load data, convert it to the format needed for training
def load_data():
    dataset=pd.read_csv('./dataset/fer2013.csv')
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
    train_x=np.array(train_x).astype(float)/255.0#归一化 #normalization to 0-1
    train_y=np.array(train_y).astype(int)
    test_x=np.array(test_x).astype(float)/255.0
    test_y=np.array(test_y).astype(int)

    train_x=train_x.reshape(-1,48,48,1)
    test_x=test_x.reshape(-1,48,48,1)

    return train_x,train_y,test_x,test_y

def save_logs(history):
    with open('./logs/history.data','wb') as f:
        f.write( pickle.dumps(history))

#build 模型 #build the model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', input_shape=[48, 48, 1]))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    model.summary()
    model.compile(loss=categorical_crossentropy,
                optimizer=Adadelta(),
                metrics=['accuracy'])
    return model

train_x,train_y,test_x,test_y=load_data()

batch_size = 128 
nb_classes = 7
nb_epoch = 20

# convert class vectors to binary class matrices
Y_train = to_categorical(train_y, nb_classes)
Y_test = to_categorical(test_y, nb_classes)

model = build_model()

history = LossHistory()

#训练 #train
model.fit(train_x, Y_train,
            batch_size=batch_size, nb_epoch=nb_epoch,
            verbose=1, 
            validation_data=(test_x, Y_test),
            callbacks=[history])

#保存模型 #save model
model.save('./saved_model/CNN_model.h5')
save_logs(history)
score = model.evaluate(test_x, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

history.loss_plot('epoch')
