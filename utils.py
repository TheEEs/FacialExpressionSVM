import numpy as np 
import scipy.spatial.distance as distance
import dlib

face_detector = dlib.get_frontal_face_detector()
landmark_detector  = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def turn_image_to_feature_vector(img):
    rects = face_detector(img,1)
    if len(rects) == 1:
        landmarks = landmark_detector(img,rects[0]).parts()
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
        return np.asarray(features)