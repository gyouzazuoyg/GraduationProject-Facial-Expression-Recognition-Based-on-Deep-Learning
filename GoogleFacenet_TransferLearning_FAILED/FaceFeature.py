import sys
import dlib
import numpy as np
import pandas as pd
import cv2
from sklearn.svm import SVC
from sklearn.externals import joblib

train_data_file='./data/train.data.npy'
test_data_file='./data/test.data.npy'
model_file='./saved_model/face_based.model'

IsTrain=False

#读取数据 #load data
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
    data_len=datas.shape[0]
    for i in range(data_len):
        pixels=np.array(datas[i].split(' '))
        sys.stdout.write("\r>>{0}/{1}".format(i,data_len))
        sys.stdout.flush()
        if(flags[i]=='Training'):
            train_x.append(pixels)
            train_y.append(labels[i])
        else:
            test_x.append(pixels)
            test_y.append(labels[i])
            
    train_x=np.array(train_x).astype(np.uint8)
    train_y=np.array(train_y).astype(int)
    test_x=np.array(test_x).astype(np.uint8)
    test_y=np.array(test_y).astype(int)
    sys.stdout.write('\n')
    sys.stdout.flush()

    return train_x,train_y,test_x,test_y

#灰度图转为RGB图 #convert gray to rgb
def gray2rgb(img):
    img=img.reshape(48,48)
    img_rgb=np.zeros((48,48,3),np.uint8)
    img_rgb[:,:,0]=img
    img_rgb[:,:,1]=img
    img_rgb[:,:,2]=img
    return img_rgb

#提取训练集或者测试集的feature #extract feature from training or testing
def extract_features(X,Y,out_file):
    predictor = dlib.shape_predictor("./saved_model/shape_predictor_68_face_landmarks.dat")
    face_rec_model = dlib.face_recognition_model_v1("./saved_model/dlib_face_recognition_resnet_model_v1.dat")
    # detector = dlib.get_frontal_face_detector()
    dlib_rect=dlib.rectangle(0,0,48,48)

    features=[]
    data_len=X.shape[0]
    for i in range(data_len):
        img_rgb=gray2rgb(X[i])
        shape = predictor(img_rgb, dlib_rect)
        encoding=face_rec_model.compute_face_descriptor(img_rgb, shape)#这个输出就是神经网络模型128维的特征 #the output is the 128 dimensional feature or the model
        face_encoding_label= np.zeros(129)#吧128维的特征加上label，一共129放在这里面 #128 dimensions + 1 label = 129 dimensions
        face_encoding_label[0:-1]=encoding
        face_encoding_label[-1]=Y[i]
        features.append(face_encoding_label)
        sys.stdout.write("\r>>{0}/{1}".format(i,data_len))
        sys.stdout.flush()

    features=np.array(features)
    np.save(out_file,features)#把提取的特征都保存下载【128+1】 #save all features extracted
    sys.stdout.write('\n')
    sys.stdout.flush()
    print('save file ',out_file)

#提取训练和测试图片的feature #extract feature for training and testing images
def extract_features_train_test():
    train_x,train_y,test_x,test_y=load_data()
    extract_features(train_x,train_y,train_data_file)
    extract_features(test_x,test_y,test_data_file)

#load 或者重新提取feature #load or re-extract features
def load_train_test_data(is_extract_features=False):
    if(is_extract_features):#如果已经提取过了，就不用再搞了 #if extracted, skip
        extract_features_train_test()
    print('load train and test data......')
    train_data_label=np.load(train_data_file)
    test_data_label=np.load(test_data_file)
    train_x=train_data_label[:,:-1]
    train_y=train_data_label[:,-1]
    test_x=test_data_label[:,:-1]
    test_y=test_data_label[:,-1]
    print('load data done.')
    return train_x,train_y,test_x,test_y

#保存模型 #save model
def save_model(model):
    joblib.dump(model,model_file)

#load模型 #load model
def load_model():
    clf=joblib.load(model_file)
    return clf
#训练模型 #train model
def train_model(train_x,train_y):
    clf=SVC(kernel='rbf',gamma='auto')

    print('training......')
    clf.fit(train_x,train_y)
    print('train done.')

    save_model(clf)
    return clf

#测试 #test model
def test_model():
    train_x,train_y,test_x,test_y=load_train_test_data()
    
    clf=None
    if(IsTrain):
        clf=train_model(train_x,train_y)#训练模型 #train model
    else:
        clf=load_model()#装载模型 #load model
    
    score = clf.score(test_x,test_y)
    print(score)
    
if __name__ == "__main__":
    test_model()

