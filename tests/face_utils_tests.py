"""
WARNING: These functions all require that a model server is up and running.
"""
import json
import numpy
import requests
import unittest
import numpy as np
from face_utils import decode_image, crop_face, get_emotions


URL = "localhost"
PORT = "8080"
EMOTION_MODEL_VERSION = "1"


class TestModelFunctions(unittest.TestCase):
    def test_get_emotion_model_dummy_data(self):
        """
        This uses dummy data to test the model inputs and outputs. It directly calls the model using
        the `URL` and `PORT` variables defined above.
        """
        image_data = numpy.random.rand(3, 48, 48, 1)
        data = json.dumps({"signature_name": "serving_default",
                   "instances": image_data.tolist()})

        headers = {"content-type": "application/json"}
        json_response = requests.post(f"http://{URL}:{PORT}/v{EMOTION_MODEL_VERSION}/models/emotion_model:predict", data=data, headers=headers)

        predictions = numpy.array(json.loads(json_response.text)["predictions"])

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
        with open ("tests/image_string.txt") as f:
            image_string = f.read()
        
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
        cf = cropped_face[0]
        assert all([pixel > 0 and pixel < 1 for row in cf for pixel in row])
        

    def test_get_emotions(self):
        """
        Take a cropped face and run the `get_emotions` function to confirm that the
        correct emotion is being returned.
        """
        cropped_face = np.load("tests/cropped_face.npy")
        
        # The expression on the face is neutral.
        self.assertEqual(get_emotions(cropped_face), "neutral")
