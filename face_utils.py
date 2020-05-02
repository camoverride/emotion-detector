"""
This module contains functions that process image data (i.e. `decode_image`) as well as functions
that make calls to external machine learning API's (i.e. `crop_face` and `get_emotions`). When additional
functions are created that make calls to external API's, they should all be placed here.
"""

import io
import base64
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


def decode_image(image_string):
    """
    Takes an image sent as a string from the client and returns an image object.

    Parameters
    ----------
    image_string: str
        An string sent from the client that can be decoded into an image.

    Returns
    -------
    numpy array
        A numpy array representing all the data from the webcam frame.
    """
    image_data = base64.b64decode(str(image_string.split(',')[1]))
    frame = Image.open(io.BytesIO(image_data))
    frame = np.array(frame)

    return frame
    

def crop_face(frame):
    """
    Sends the image to the object detection model to locate faces. After the faces are located, only one
    is selected and its coordinates are returned.

    Parameters
    ----------
    frame: Image
        A numpy array representing all the data from the webcam frame.

    Returns
    -------
    numpy array
        A numpy array that has been cropped to the face.
    """
    # TODO: actually implement this as a web API.
    face_detection_model = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    
    # Get the coordinates for all the faces from the model.
    faces = face_detection_model.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE)

    # Crop the image to the coordinates of the face.
    face_coords = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = face_coords
    cropped_face = frame[fY:fY + fH, fX:fX + fW]

    return cropped_face



def get_emotions(cropped_face):
    """
    Accepts a frame from a videostream and sends it to the tensorflow server. Returns a
    softmax over predicted categories. This requires additional smoothing and munging,
    but this process is lightweight and can be done on the Flask server.

    Parameters
    ----------
    cropped_face: a numpy array.
        A numpy array representing an image cropped to a specific face.

    Returns
    -------
    str
        The most likely emotion, taken from the softmax returned by the model.
    """
    face_debug = cropped_face # Used for local debugging only.

    # TODO: actually implement this as a web API.
    emotion_model = load_model("models/XCEPTION.72.model", compile=False)

    # Resize, black-and-white, reshape, and normalize  face.
    cropped_face = cv2.resize(cropped_face, (48, 48))
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
    cropped_face = cropped_face.reshape(48, 48, 1)
    cropped_face = cropped_face / 255

    # Get the predictions (a vector of length 7) and map it to the appropriate emotion.
    preds = emotion_model.predict(np.array([cropped_face, cropped_face]))[0]
    emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    max_emotion = np.argmax(preds)
    prediction = emotions[max_emotion]

    # Local debug. TODO: remove this and make it a test.
    gray_face = cv2.cvtColor(face_debug, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", gray_face)
    print(prediction)

    return prediction
