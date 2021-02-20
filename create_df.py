import cv2
import numpy as np
import dlib
import pandas as pd
import os


def image_to_dict(detector, predictor, path, num_face_landmarks):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    file_name = path.split(sep='/')[-1]
    info = file_name.split(sep='_')
    dict_arry = []
    for face in faces:
        landmarks = predictor(gray, face)
        dict = {'point_' + str(n) + '_x': (landmarks.part(n).x - face.left()) / (face.right() - face.left()) for n
               in range(0, num_face_landmarks)}
        dict.update(
           {'point_' + str(n) + '_y': (landmarks.part(n).y - face.top()) / (face.bottom() - face.top()) for n in
            range(0, num_face_landmarks)})
        dict['Age'] = info[1]
        dict['is_female'] = info[0]
        dict_arry.append(dict)
    return dict_arry


def create_DataFrame(folder, num_face_landmarks):
    folder_paths = './' + folder
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_" + str(num_face_landmarks) + "_face_landmarks.dat")
    df = pd.DataFrame()
    images_paths = os.listdir(folder_paths)
    for path in images_paths:
        print(path)
        path = folder_paths + '/' + path
        dict_arry = image_to_dict(detector, predictor, path, num_face_landmarks)
        for dict in dict_arry:
            df = df.append(dict, ignore_index=True)
    df.to_csv("new_data.csv")
    return df
