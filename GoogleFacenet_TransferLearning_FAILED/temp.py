import dlib
import cv2
import numpy as np
predictor = dlib.shape_predictor("./saved_model/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("./saved_model/dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()
frame = cv2.imread(r'./images/person.JPG')
face_rect = detector(frame, 1)[0]
shape = predictor(frame, face_rect)
for pt in shape.parts():
    pt_pos = (pt.x, pt.y)
    cv2.circle(frame, pt_pos, 2, (0, 255, 0), 1)
cv2.imshow("image", frame)
encoding=face_rec_model.compute_face_descriptor(frame, shape)
encoding=np.array(encoding)
print(encoding)
cv2.waitKey()
