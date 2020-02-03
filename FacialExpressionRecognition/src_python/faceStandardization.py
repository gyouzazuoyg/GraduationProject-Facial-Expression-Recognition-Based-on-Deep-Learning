# -*- coding: utf-8 -*-

import face_recognition
from PIL import Image
import cv2
import numpy as np

def normalize(input,output):
    path =input
    out_path = output

    # 读取图片并识别人脸
    img = face_recognition.load_image_file(path)
    face_locations = tuple(list(face_recognition.face_locations(img)[0]))

    # 重新确定切割位置并切割
    top = face_locations[0]
    right = face_locations[1]
    bottom = face_locations[2]
    left = face_locations[3]
    cutting_position = (left, top, right, bottom)
    # 切割出人脸
    im = Image.open(path)

    region = im.crop(cutting_position)

    # 人脸缩放
    a = 50  # 人脸方格大小
    if region.size[0] >= a or region.size[1] >= a:
        region.thumbnail((a, a), Image.ANTIALIAS)
    else:
        region = region.resize((a, a), Image.ANTIALIAS)
    # 人脸旋转
    theta =trait_angle(path)
    #region = region.rotate(theta)
    # 保存人脸
    region.save(out_path)
