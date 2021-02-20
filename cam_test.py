import cv2
import dlib
from urllib.request import urlopen
import numpy as np


def cam():
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            landmarks = predictor(gray, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image


def image_test(filename):
    # link = "https://image.shutterstock.com/image-photo/handsome-unshaven-young-darkskinned-male-260nw-640011838.jpg"
    # image = url_to_image(link)
    points_arr = [(0, 1), (0, 17), (0, 36)
        , (1, 2), (1, 36), (1, 41)
        , (2, 3), (2, 31), (2, 41)
        , (3, 4), (3, 31), (3, 48)
        , (4, 5), (4, 48)
        , (5, 6), (5, 48)
        , (6, 7), (6, 48), (6, 59)
        , (7, 8), (7, 58), (7, 59)
        , (8, 9), (8, 56), (8, 57), (8, 58)
        , (9, 10), (9, 55), (9, 56)
        , (10, 11), (10, 54), (10, 55)
        , (11, 12), (11, 54)
        , (12, 13), (12, 54)
        , (13, 14), (13, 35), (13, 54)
        , (14, 15), (14, 35), (14, 46)
        , (15, 16), (15, 45), (15, 46)
        , (16, 26), (16, 45)
        , (17, 18), (17, 36)
        , (18, 19), (18, 36), (18, 37)
        , (19, 20), (19, 37), (19, 38)
        , (20, 21), (20, 23), (20, 38), (20, 39)
        , (21, 22), (21, 23), (21, 27), (21, 39)
        , (22, 23), (22, 27), (22, 42)
        , (23, 24), (23, 42), (23, 43)
        , (24, 25), (24, 43), (24, 44)
        , (25, 26), (25, 44), (25, 45)
        , (26, 45)
        , (27, 28), (27, 39), (27, 42)
        , (28, 29), (28, 39), (28, 42)
        , (29, 30), (29, 31), (29, 35), (29, 39), (29, 40), (29, 42), (29, 47)
        , (30, 31), (30, 32), (30, 33), (30, 34), (30, 35)
        , (31, 32), (31, 40), (31, 41), (31, 48), (31, 49), (31, 50)
        , (32, 33), (32, 50), (32, 51)
        , (33, 34), (33, 51)
        , (34, 35), (34, 51), (34, 52)
        , (35, 46), (35, 47), (35, 52), (35, 53), (35, 54)
        , (36, 37), (36, 41)
        , (37, 38), (37, 40), (37, 41)
        , (38, 39), (38, 40)
        , (39, 40)
        , (40, 41)
        , (42, 43), (42, 47)
        , (43, 44), (43, 47)
        , (44, 45), (44, 46),(44,47)
        , (45, 46)
        , (46, 47)
        , (48, 49), (48, 59), (48, 60)
        , (49, 50), (49, 60), (49, 61)
        , (50, 51), (50, 61), (50, 62)
        , (51, 52), (51, 62)
        , (52, 53), (52, 62), (52, 63)
        , (53, 54), (53, 63), (53, 64)
        , (54, 55), (54, 64)
        , (55, 56), (55, 64), (55, 65)
        , (56, 57), (56, 65), (56, 66)
        , (57, 58), (57, 66)
        , (58, 59), (58, 66), (58, 67)
        , (59, 60), (59, 67)
        , (60, 61), (60, 67)
        , (61, 62), (61, 66), (61, 67)
        , (62, 63), (62, 66)
        , (63, 64), (63, 65), (63, 66)
        , (64, 65)
        , (65, 66)
        , (66, 67)]
    image = cv2.imread(filename)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # _, cap = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()+10
        y2 = face.bottom()+10
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
        for n in points_arr:
            cv2.line(image, (landmarks.part(n[0]).x, landmarks.part(n[0]).y), (landmarks.part(n[1]).x, landmarks.part(n[1]).y), (255, 0, 0), 1)
    cv2.imwrite(filename+'_1', image)
    cv2.imshow("Frame", image)
    cv2.waitKey(360)
