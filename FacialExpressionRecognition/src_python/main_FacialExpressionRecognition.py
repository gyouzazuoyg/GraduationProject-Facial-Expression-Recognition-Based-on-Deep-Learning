# -*- coding: utf-8 -*-  

#python 3.7
#林　正浩
#Lin ZhengHao

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
path_FacialExpressionRecognition_model = '%s/trained_models/improved CNN-average on Fer+ Front_checkpoint-89e-val_acc_0.8256.h5' %path_WorkSpace

#函数
#将文字加到图片上
#返回带字的图片
def AddTextToImage(image, text, left, top, textColor=(0, 0, 0), textSize=20):#用于在图片image上加文字的函数，
    if (isinstance(image, np.ndarray)):  #判断是否是ndarray矩阵
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))#BGR色彩→RGB色彩
    draw = ImageDraw.Draw(image)#创建被绘制对象
    ttf = path_Font
    fontText = ImageFont.truetype(#配置字体，保存到fontText
        font=ttf,
        size=textSize,
        encoding="utf-8")
    draw.text(#在指定坐标绘制文本
        xy=(left,top),
        text=text,
        fill=textColor,
        font=fontText)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)#PIL图片→ndarray矩阵

#函数
#将识别出的表情加到图片上，将图片中识别出的人脸框起来
#返回带字和框的图片
def facialExpressionRecognition(image, facialExpression_classifier, face_classifier):
    #emotion_labels = { #创建emotion_labels和对应文字的dict，FER2013专用
    #    0: 'Anger',
    #    1: 'Disgust',
    #    2: 'Fear',
    #    3: 'Happy',
    #    4: 'Sad',
    #    5: 'Surprised',
    #    6: 'Normal'
    #}
    emotion_labels = { #创建emotion_labels和对应文字的dict，FER+专用
        0: 'neutral',
        1: 'happiness',
        2: 'surprise',
        3: 'sadness',
        4: 'anger',
        5: 'disgust',
        6: 'fear'
    }
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#将image转换成灰度图gray
    faces = face_classifier.detectMultiScale(
        image=gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))#scaleFactor表示检测矩形的缩小比例，minNeighbors是说图片至少要被检测到几次
    #detectMultiScale的返回值是一组(x, y, w, h)格式的元祖序列
    for (x, y, w, h) in faces:
        #将单张脸临时存储到gray_face
        gray_face = gray[(y):(y + h), (x):(x + w)]#截取脸部
        gray_face = cv2.resize(src=gray_face, dsize=(48,48))#将尺寸缩放为和fer2013相同的48x48
        gray_face = cv2.equalizeHist(gray_face)#直方图均衡化
        gray_face = gray_face / 255.0#归一化，将0-255的8位灰度值变成0-1浮点数分布
        gray_face = np.expand_dims(a=gray_face, axis=0)#在gray_face前面插入一维占位维度变成（1,48,48）
        gray_face = np.expand_dims(a=gray_face, axis=-1)#-1表示最后一维,变成(1,48,48,1)维
        gray_face = gray_face.reshape(-1,48,48,1)

        predict_result = facialExpression_classifier.predict(gray_face)
        #手动微调概率
        #predict_result[0][0] *= 1 #anger
        #print('anger = %1.3f' %predict_result[0][0], end=' ')
        #predict_result[0][1] *= 1.1 #disgust
        #print('disgust = %1.3f' %predict_result[0][1], end=' ')
        #predict_result[0][2] *= 1.2 #fear
        #print('fear = %1.3f' %predict_result[0][2], end=' ')
        #predict_result[0][3] *= 1 #happy
        #print('happy = %1.3f' %predict_result[0][3], end=' ')
        #predict_result[0][4] *= 1.1 #sad
        #print('sad = %1.3f' %predict_result[0][4], end=' ')
        #predict_result[0][5] *= 1 #surprised
        #print('surprised = %1.3f' %predict_result[0][5], end=' ')
        #predict_result[0][6] *= 1.1 #normal
        #print('normal = %1.3f' %predict_result[0][6])

        emotion_label_arg = np.argmax(predict_result)#取概率最大的表情,得出图片对应的label，从二值LIST的形式转换成0-9的数字
        emotion = emotion_labels[emotion_label_arg]#label序号换成单词
        cv2.rectangle(image, (x, y), (x+w, y+h),(255, 0, 0), 2)#把脸框出来
        image = AddTextToImage(#把字加到图片上
            image=image,
            text=emotion,
            left=x+w*0.1,
            top=y+h*0.1,
            textColor=(0, 0, 255), 
            textSize=int(h/6))
    return image

#加载模型
facialExpression_classifier = load_model(path_FacialExpressionRecognition_model)
face_classifier = cv2.CascadeClassifier(path_Haar_Cascade_Model_path)#加载opencv人脸识别器

#功能选择
functionSelect = 'camera'
#functionSelect = 'imageFile'

#识别图像文件
filename = '1234.png'
if(functionSelect == 'imageFile'):
    img = normalization('%s/temp_images/%s' %(path_WorkSpace, filename), '%s/temp_images/after/%s' %(path_WorkSpace, filename))
    img = cv2.imread('%s/temp_images/%s' %(path_WorkSpace, filename))
    img = facialExpressionRecognition(#加上表情文字和框
        image=img,
        facialExpression_classifier=facialExpression_classifier,
        face_classifier=face_classifier)
    cv2.imshow("Image",img)#显示一帧
    key=cv2.waitKey(0)
    cv2.destroyAllWindows()

#开始摄像头识别
if(functionSelect == 'camera'):
    cap = cv2.VideoCapture(0)#设置读取默认摄像头
    while(True):
        ret, frame = cap.read()#取一帧
        if(ret == True):#若成功取值
            frame = facialExpressionRecognition(#加上表情文字和框
                image=frame,
                facialExpression_classifier=facialExpression_classifier,
                face_classifier=face_classifier)
            cv2.imshow("Camera",frame)#显示一帧
            key=cv2.waitKey(1)
            if(key==ord('q') or key==27):#按Q或ESC退出
                break
        else:
            print('Fail to read image from camera!')
            break

    #释放摄像头资源，关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()