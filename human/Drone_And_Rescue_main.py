import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
from YOLOv12 import LostMemeberDetector
import tkinter as tk
from PIL import Image, ImageTk


cap = cv2.VideoCapture(0)

imgBackground = cv2.imread('Resources/backgroud.png')
# import modes
modes_folder = 'Resources/Modes'
mode_path_list = os.listdir(modes_folder)
mode_list = []
for path in mode_path_list:
    mode_list.append(cv2.imread(os.path.join(modes_folder, path)))

# load encoded faces
print("Loading encoded faces ...")
file = open('EncodeFile.p', 'rb')
encoded_list_known_w_ids = pickle.load(file)
file.close()
encoded_list_known, ids = encoded_list_known_w_ids
# print(ids)
print("Encoded faces loaded")


while True:
    success, img = cap.read()

    img = cv2.resize(img, (800, 550))
    img_s = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    face_curr = face_recognition.face_locations(img_s)
    encode_curr = face_recognition.face_encodings(img_s, face_curr)

    imgBackground[300:300 + 550, 70:70 + 800] = img
    imgBackground[400:400 + mode_list[1].shape[0], 1050:1050 + mode_list[1].shape[1]] = mode_list[1]

    for encoded_face, face_loc in zip(encode_curr, face_curr):
        matches = face_recognition.compare_faces(encoded_list_known, encoded_face)
        face_dist = face_recognition.face_distance(encoded_list_known, encoded_face)

        match_index = np.argmin(face_dist)
        if matches[match_index]:
            # print("Detected!")
            # print(ids[match_index])
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = 70+x1, 300+y1, x2-x1, y2-y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
            imgBackground[400:400 + mode_list[0].shape[0], 1050:1050 + mode_list[0].shape[1]] = mode_list[0]
            print(ids[match_index])

    cv2.imshow("Drone & Rescue", imgBackground)
    cv2.waitKey(1)

    if cv2.waitKey(1) & (cv2.getWindowProperty("Drone & Rescue", cv2.WND_PROP_VISIBLE) < 1):
        break


cap.release()
cv2.destroyAllWindows()