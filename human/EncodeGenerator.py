import cv2
import os
import face_recognition
import pickle

# import people
image_folder = 'Images'
images_path_list = os.listdir(image_folder)
image_list = []
ids = []
for path in images_path_list:
    image_list.append(cv2.imread(os.path.join(image_folder, path)))
    ids.append(os.path.splitext(path)[0])

print(ids)


def find_encodings(images_list):
    encode_list = []
    for image in images_list:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encode_list.append(encode)

    return encode_list

print("Encoding Started ...")

encoded_list_known = find_encodings(image_list)
encoded_list_known_w_ids = [encoded_list_known, ids]
print("Encoding Complete")

# puts ids into file with encoding
file = open("EncodeFile.p", 'wb')
pickle.dump(encoded_list_known_w_ids, file)
file.close()
print("File Saved")