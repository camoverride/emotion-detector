"""
This module contains functions that process image data (i.e. `decode_image`) as well as functions
that make calls to external machine learning API's (i.e. `crop_face` and `get_model_pred`).
When additional functions are created that make calls to external API's, they should all
be placed here.
"""

import io
import base64
import json
import requests
import numpy as np
import cv2
from PIL import Image
from skimage.transform import resize


FACE_DETECTION_MODEL = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")


def decode_image(image_string: str) -> np.ndarray:
    """
    Takes an image sent as a string from the client and returns an image object.

    Parameters
    ----------
    image_string: str
        A string sent from the client that can be decoded into an image.

    Returns
    -------
    numpy ndarray
        A numpy array representing all the data from the webcam frame.
        The shape of this image is (height, width, 3)
    """
    image_data = base64.b64decode(str(image_string.split(",")[1]))
    frame = Image.open(io.BytesIO(image_data))
    frame = np.array(frame)

    return frame


def crop_face(frame: np.ndarray, width=48, height=48, scaling_factor=1, grayscale=True) -> np.ndarray:
    """
    Takes an image and crops it to an individual face. If there are multiple
    faces in the image, only one is chosen. The location of the chosen face
    is indicated by a bounding box.
    This function returns two things: an numpy array which is the cropped face,
    and a list of coordinates for drawing the bounding box.

    Parameters
    ----------
    frame: numpy ndarray
        A numpy array representing all the data from the webcam frame.
        The shape will be (height, width, 3)

    Returns
    -------
    numpy ndarray
        A numpy array that has been cropped to the face and resized to width x height.
        The shape of this array is (1, width, height, 1).

    list
        A list containing the 4 coordinates of the face: x, y, diff_height, diff_width.
        Where diff_height/width represent the distance down and across from (x, y)
    """
    # Get the coordinates for all the faces from the model.
    faces = FACE_DETECTION_MODEL.detectMultiScale(frame,
                scaleFactor=1.1, minNeighbors=5, minSize=(width, height), flags=cv2.CASCADE_SCALE_IMAGE)

    # Crop the image to the coordinates of the first face.
    face_coords = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (x_coord, y_coord, diff_width, diff_height) = face_coords
    cropped_face = frame[y_coord:y_coord + diff_height, x_coord:x_coord + diff_width]

    if grayscale:
        # Convert to grayscale.
        # https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert
        cropped_face = np.dot(cropped_face[...,:3], [0.2989, 0.5870, 0.1140])

        # Regularize.
        cropped_face = cropped_face / 255

        # Resize to the desired dimensions.
        cropped_face = resize(cropped_face, (width, height))

        # Reshape to fit model.
        cropped_face = cropped_face.reshape(1, width, height, 1)

    else:
        # Regularize.
        cropped_face = cropped_face / 255

        # Resize to the desired dimensions.
        cropped_face = resize(cropped_face, (width, height))

        # Reshape to fit model.
        cropped_face = cropped_face.reshape(1, width, height, 3)

    # Some models require that the data is rescaled
    cropped_face = cropped_face * scaling_factor

    return cropped_face, face_coords


def get_model_pred(cropped_face: np.ndarray, model_server_url: str,
            model_server_port: str, model_version: str, model_name: str) -> dict:
    """
    Accepts a frame from a videostream and sends it to the tensorflow server which returns a
    softmax over predicted categories. This argmax from this softmax is then chosen as the
    predicted category. A dict is returned where there is a prediction key and then a list
    of all the predictions, which may have names. Decoding is model-specific.

    Parameters
    ----------
    cropped_face: numpy ndarray.
        A numpy array representing an image cropped to a specific face. The shape is
        (1, width, height, 1), representing 1 face of width/height pixels and 1 color
        channel. Different models will require different width/heights.

    Returns
    -------
    dict
        A dict object with the key "prediction" that is a list of predictions (usually just)
        a single element. This can be iterated over to find the predicted categories.
    """
    # Create the request object.
    data = json.dumps({"signature_name": "serving_default", "instances": cropped_face.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(f"http://{model_server_url}:{model_server_port}/v{model_version}/models/{model_name}:predict",
                data=data, headers=headers)

    # Get the prediction.
    if model_name == "emotion_model":
        emotion_categories = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

        predictions = np.array(json.loads(json_response.text)["predictions"])
        max_emotion = np.argmax(predictions[0])
        prediction = emotion_categories[max_emotion]

        return {"prediction": prediction}

    elif model_name == "age_gender_model":
        gender_categories = ["female", "male"]
        age_categories = list(range(101))

        predictions = json.loads(json_response.text)["predictions"][0]
        max_gender = gender_categories[np.argmax(predictions["dense"])]
        max_age = age_categories[np.argmax(predictions["dense_1"])]

        return {"prediction": {
                    "gender": max_gender,
                    "age": max_age
                    }
                }
