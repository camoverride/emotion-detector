"""
This module contains functions that process image data (i.e. `decode_image`) as well as functions
that make calls to external machine learning API's (i.e. `crop_face` and `get_emotions`). When additional
functions are created that make calls to external API's, they should all be placed here.
"""

import io
import base64
import json
import requests
import numpy as np
import cv2
from PIL import Image
from skimage.transform import resize


# MODEL_SERVER_URL = "34.83.242.28"
MODEL_SERVER_URL = "localhost"


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
    image_data = base64.b64decode(str(image_string.split(",")[1]))
    frame = Image.open(io.BytesIO(image_data))
    frame = np.array(frame)

    return frame


def get_faces(frame):
    """
    This function uses the haarcascade model to identify faces in an image.

    TODO: implement this in tensorflow and add it to the model server.

    Parameters
    ----------
    frame: Image
        A numpy array representing all the data from a webcam page.

    output_dimensions
        A tuple that represents the desired (width, height) of the output image. Usually width = height.

    Returns
    -------
    numpy array
        A list of lists containing the coordinates of all the faces in the image.
    """
    face_detection_model = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    
    # Get the coordinates for all the faces from the model.
    faces = face_detection_model.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48), flags=cv2.CASCADE_SCALE_IMAGE)

    return faces


def crop_face(frame):
    """
    Takes a list of faces and crops the image to a face.

    Parameters
    ----------
    frame: Image
        A numpy array representing all the data from the webcam frame.

    Returns
    -------
    numpy array
        A numpy array that has been cropped to the face and resized to width x height. If
    
    list
        A list containing the 4 coordinates of the face: x, y, height, width.
    """
    faces = get_faces(frame)

    # Crop the image to the coordinates of the first face.
    face_coords = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = face_coords
    cropped_face = frame[fY:fY + fH, fX:fX + fW]

    # Convert to grayscale. https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert
    cropped_face = np.dot(cropped_face[...,:3], [0.2989, 0.5870, 0.1140])

    # Regularize.
    cropped_face = cropped_face / 255

    # Resize to the desired dimensions.
    cropped_face = resize(cropped_face, (48, 48))

    # Reshape to fit model.
    cropped_face = cropped_face.reshape(1, 48, 48, 1)

    return cropped_face, face_coords


def crop_face_large(frame):
    cropped_face = resize(frame, (224, 224))

    # Reshape to fit model.
    cropped_face = cropped_face.reshape(1, 224, 224, 3).astype('float32')

    # this is needed because data seems to be rescaled to (0, 1) interval, whereas model expects values from 0-255
    # TODO: check this!
    cropped_face = cropped_face * 100

    return cropped_face


def get_emotions(cropped_face):
    """
    Accepts a frame from a videostream and sends it to the tensorflow server which returns a
    softmax over predicted categories. This argmax from this softmax is then returned as the
    predicted emotion.

    TODO: the model possibly over-predicts neutral and happy and under-predicts dsigust and
    scared, so some post-processing should be done on the softmax in order to return a better
    prediction.

    Parameters
    ----------
    cropped_face: a numpy array.
        A numpy array representing an image cropped to a specific face. The shape is 1x48x48x1,
        representing 1 face of 48x48 pixels and 1 color channel.

    Returns
    -------
    str
        The most likely emotion, taken from the softmax returned by the model.
    """
    # Create the request object.
    data = json.dumps({"signature_name": "serving_default", "instances": cropped_face.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(f"http://{MODEL_SERVER_URL}:8080/v1/models/emotion_model:predict", data=data, headers=headers)

    # Get the prediction.
    predictions = np.array(json.loads(json_response.text)["predictions"])
    emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    max_emotion = np.argmax(predictions[0])
    prediction = emotions[max_emotion]

    return prediction


def get_gender(cropped_face):
    """
    Accepts a frame from a videostream and sends it to the tensorflow server which returns a
    softmax over predicted categories. This argmax from this softmax is then returned as the
    predicted gender.

    Parameters
    ----------
    cropped_face: a numpy array.
        A numpy array representing an image cropped to a specific face. The shape is 1x48x48x1,
        representing 1 face of 48x48 pixels and 1 color channel.

    Returns
    -------
    str
        The most likely gender, taken from the softmax returned by the model.
    """
    # Create the request object.

    data = json.dumps({"signature_name": "serving_default", "instances": cropped_face.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(f"http://{MODEL_SERVER_URL}:8080/v1/models/gender_model:predict", data=data, headers=headers)

    # Get the prediction.
    predictions = np.array(json.loads(json_response.text)["predictions"])
    genders = ["female", "male"]
    max_gender= np.argmax(predictions[0])
    prediction = genders[max_gender]

    return prediction


def get_age(cropped_face):
    """
    Accepts a frame from a videostream and sends it to the tensorflow server which returns a
    softmax over predicted age categories. This argmax from this softmax is then returned as the
    predicted age.

    Parameters
    ----------
    cropped_face: a numpy array.
        A numpy array representing an image cropped to a specific face. The shape is 1x48x48x1,
        representing 1 face of 48x48 pixels and 1 color channel.

    Returns
    -------
    str
        The most likely gender, taken from the softmax returned by the model.
    """
    # Create the request object.

    data = json.dumps({"signature_name": "serving_default", "instances": cropped_face.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(f"http://{MODEL_SERVER_URL}:8080/v1/models/age_model:predict", data=data, headers=headers)

    # Get the prediction.
    predictions = np.array(json.loads(json_response.text)["predictions"])
    pred = np.argmax(predictions[0])

    return pred
