import numpy as np

import io
import json
import cv2
import os
import const                #const.py


def detect_face(binaryimg):
    data = {"success": False}
    if binaryimg is None:
        return data

    # convert the binary image to image
    image = read_cv2_image(binaryimg)

    # convert the image to grayscale
    imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # load the face cascade detector,
    facecascade = cv2.CascadeClassifier(const.PATH_FACE_DETECTOR)

    # detect faces in the image
    facedetects = facecascade.detectMultiScale(imagegray, scaleFactor=1.1, minNeighbors=5,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # construct a list of bounding boxes from the detection
    facerect = [(int(fx), int(fy), int(fx + fw), int(fy + fh)) for (fx, fy, fw, fh) in facedetects]

    # update the data dictionary with the faces detected
    data.update({"num_faces": len(facerect), "faces": facerect, "success": True})
    print(json.dumps(data))

    # return the data dictionary as a JSON response
    return data


def read_cv2_image(binaryimg):

    stream = io.BytesIO(binaryimg)

    image = np.asarray(bytearray(stream.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

