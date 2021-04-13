"""
WARNING: These functions all require that a model server is up and running.
"""
import json
import unittest
import requests
import numpy as np
from face_utils import decode_image, crop_face, get_model_pred


# Configuration for server and model information.
MODEL_SERVER_URL = "localhost"
MODEL_SERVER_PORT = "8080"
EMOTION_MODEL_VERSION = "1"


class TestModelFunctions(unittest.TestCase):
    """
    Tests for all the functions in `face_utils.py`
    """
    def test_get_model_pred_dummy_data(self):
        """
        This uses dummy data to test the model inputs and outputs. It directly calls the model using
        the `URL` and `PORT` variables defined above.
        """
        image_data = np.random.rand(3, 48, 48, 1)
        data = json.dumps({"signature_name": "serving_default",
                   "instances": image_data.tolist()})

        headers = {"content-type": "application/json"}
        json_response = requests.post(f"http://{MODEL_SERVER_URL}:{MODEL_SERVER_PORT}/v{EMOTION_MODEL_VERSION}/models/emotion_model:predict",
                                      data=data, headers=headers)

        predictions = np.array(json.loads(json_response.text)["predictions"])

        # Three prediction vectors should be return
        assert len(predictions) == 3

        # Each prediction vector has seven elements
        assert all(len(pred) == 7 for pred in predictions)

        # Each prediction vector is a positive float
        assert all([emotion > 0 for pred in predictions for emotion in pred])


    def test_decode_image(self):
        """
        Take an image string - which is the data returned by the client (web browser) and make sure
        that is can be properly decoded into a numpy array.
        """
        with open ("tests/image_string.txt") as file:
            image_string = file.read()

        webcam_frame = decode_image(image_string)

        # This is the shape of the frame captured by the particular webcam
        self.assertEqual(webcam_frame.shape, (480, 640, 3))


    def test_crop_face(self):
        """
        Take a frame from the webcam that's already been decoded and saved as a
        numpy array and confirm that it can be cropped.
        """
        webcam_image = np.load("tests/decoded_frame.npy")
        cropped_face, _ = crop_face(webcam_image)

        # This is the shape of the cropped face array, which is a single color channel.
        self.assertEqual(cropped_face.shape, (1, 48, 48, 1))

        # Each pixel value should be in the inteval [0, 1]
        cropf = cropped_face[0]
        assert all([pixel > 0 and pixel < 1 for row in cropf for pixel in row])


    def test_get_model_pred(self):
        """
        Take a cropped face and run the `get_model_pred` function to confirm that the
        correct emotion is being returned.
        """
        EMOTION_CATEGORIES = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

        cropped_face = np.load("tests/cropped_face.npy")

        # The expression on the face is neutral.
        self.assertEqual(get_model_pred(cropped_face, model_server_url=MODEL_SERVER_URL,
                    model_server_port=MODEL_SERVER_PORT, model_version=EMOTION_MODEL_VERSION,
                    model_name="emotion_model", categories=EMOTION_CATEGORIES), "neutral")
