from os import path
import glob, pandas 
import numpy as np 
import os,dlib,json
import scipy.spatial.distance as distance

if not path.exists("./data/"):
    os.mkdir("./data")
if not path.exists("./data/set"):
    os.mkdir("./data/set")
if not path.exists("./data/trained_models"):
    os.mkdir("./data/trained_models")



data = pandas.read_csv("./fer2013.csv")
labels = data["emotion"]
images = data["pixels"]

face_detector = dlib.get_frontal_face_detector()
landmark_detector  = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

extracted_instances = [

]

extracted_labels = [

]

for index,img in enumerate(images):
    reshaped_img = np.asarray(img.split()).astype('uint8').reshape(48,48)
    rects = face_detector(reshaped_img,1)
    if len(rects) == 1:
        landmarks = landmark_detector(reshaped_img,rects[0]).parts()
        center_point = landmarks[30]
        point_29  = landmarks[29]
        scale_ratio = 5 / distance.euclidean( #here we scale keypoints's distance 5 times
            (center_point.x,center_point.y),
            (point_29.x,point_29.y)
        )
        features = []
        for point in landmarks:
            point_x = (point.x - center_point.x) * scale_ratio 
            point_y = (point.y - center_point.y) * scale_ratio
            features.append(point_x)
            features.append(point_y)
        extracted_instances.append(
            features
        )
        extracted_labels.append(int(labels[index]))

with open('./data/set/dataset.json','w') as output_file:
    json.dump({
        "labels": extracted_labels,
        "instances": extracted_instances
    },output_file)
       
