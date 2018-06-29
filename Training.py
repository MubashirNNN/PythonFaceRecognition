import os
import numpy as np
from PIL import Image
import cv2
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "Images")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
current_id = 0
label_ids = {}
y_labels = []
x_train = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            #print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            _id = label_ids[label]
           # print(label_ids)
            pil_image = Image.open(path).convert("L") #L represent GrayScale
            s = (500, 500)
            pil_images = pil_image.resize(s, Image.ANTIALIAS)
            image_array = np.array(pil_images, "uint8")
            print(image_array)
            print(label_ids[label])
            faces = face_cascade.detectMultiScale(image_array, 1.5 , 5)
            for (x, y, w, h) in faces:
                roi = image_array[y: y+h, x:x+w]

                x_train.append(roi)
                y_labels.append(_id)
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)
print(type(x_train))
recognizer.train(x_train, np.array(y_labels))
recognizer.save("Trained Data.yml")
