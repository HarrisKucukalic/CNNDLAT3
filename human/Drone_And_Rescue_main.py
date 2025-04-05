import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
import dlib



from YOLOv12 import LostMemeberDetector
import tkinter as tk
from PIL import Image, ImageTk


cap = cv2.VideoCapture(0)

# imgBackground = cv2.imread('Resources/backgroud.png')
# # import modes
# modes_folder = 'Resources/Modes'
# mode_path_list = os.listdir(modes_folder)
# mode_list = []
# for path in mode_path_list:
#     mode_list.append(cv2.imread(os.path.join(modes_folder, path)))

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
    if not success:
        break

    img_s = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

    face_curr = face_recognition.face_locations(img_s)
    encode_curr = face_recognition.face_encodings(img_s, face_curr)

    for encoded_face, face_loc in zip(encode_curr, face_curr):
        matches = face_recognition.compare_faces(encoded_list_known, encoded_face)
        face_dist = face_recognition.face_distance(encoded_list_known, encoded_face)

        match_index = np.argmin(face_dist)
        if matches[match_index]:
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = [v * 4 for v in (y1, x2, y2, x1)]

            # Draw bounding box
            bbox = (x1, y1), (x2, y2)
            cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 1)
            cv2.putText(img, ids[match_index], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) & (cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1):
        break


cap.release()
cv2.destroyAllWindows()