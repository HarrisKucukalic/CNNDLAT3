import cv2
import pickle
import face_recognition
import numpy as np

# This class is for human face detection only. Human body is dealt with in the LiveObjectDetector class
class FaceDetector:
    def __init__(self, camera=cv2.VideoCapture(0)):
        self.camera = camera
        print("Loading encoded faces ...")
        file = open('EncodeFile.p', 'rb')
        # Loads in known faces to use to compare to the new images fed to the app.
        self.encoded_list_known_w_ids = pickle.load(file)
        file.close()
        self.encoded_list_known, self.ids = self.encoded_list_known_w_ids
        # print(ids)
        print("Encoded faces loaded")
    def process_face(self):
        # Re-loads in the list of known people from the pickle file, as people may have been added between the initialisation of the object
        file = open('EncodeFile.p', 'rb')
        self.encoded_list_known_w_ids = pickle.load(file)
        file.close()
        self.encoded_list_known, self.ids = self.encoded_list_known_w_ids
        # reads cv2 camera footage from the device used. iPhone streams were sent to the device in testing through different mobile apps.
        success, img = self.camera.read()
        if not success:
            return None
        # resizes the images to 1/4 on each dimension, making the image 1/16 of the original for easier processing.
        img_s = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
        # feed through the face_recognition library to identify the faces, uses HOG.
        face_curr = face_recognition.face_locations(img_s)
        encode_curr = face_recognition.face_encodings(img_s, face_curr)
        # goes through each detected face in the image to see if it's in our pickle file.
        for encoded_face, face_loc in zip(encode_curr, face_curr):
            matches = face_recognition.compare_faces(self.encoded_list_known, encoded_face)
            face_dist = face_recognition.face_distance(self.encoded_list_known, encoded_face)
            # If there is a match, a green box is drawn around the face with associated ID in the pickle file.
            match_index = np.argmin(face_dist)
            if matches[match_index]:
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = [v * 4 for v in (y1, x2, y2, x1)]
                # Draw bounding box
                bbox = (x1, y1), (x2, y2)
                cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 1)
                cv2.putText(img, self.ids[match_index], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # return the processed image.
        return img

    def process_img(self, img):
        # Similar to the above, but is used for front end display of the identified person in static images.
        file = open('EncodeFile.p', 'rb')
        self.encoded_list_known_w_ids = pickle.load(file)
        file.close()
        self.encoded_list_known, self.ids = self.encoded_list_known_w_ids
        img_s = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)

        face_curr = face_recognition.face_locations(img_s)
        encode_curr = face_recognition.face_encodings(img_s, face_curr)

        for encoded_face, face_loc in zip(encode_curr, face_curr):
            matches = face_recognition.compare_faces(self.encoded_list_known, encoded_face)
            face_dist = face_recognition.face_distance(self.encoded_list_known, encoded_face)

            match_index = np.argmin(face_dist)
            if matches[match_index]:
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = [v * 4 for v in (y1, x2, y2, x1)]
                # Draw bounding box
                bbox = (x1, y1), (x2, y2)
                cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 1)
                cv2.putText(img, self.ids[match_index], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # return the processed image.
        return img

    def return_face(self):
        return self.process_face()

    def __del__(self):
        self.camera.release()