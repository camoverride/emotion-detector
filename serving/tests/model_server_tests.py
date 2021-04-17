"""
Initialize the model server before running these tests.
"""

import json
import numpy
import requests
import unittest


# Model server config
URL = "localhost"
PORT = "8080"


class TestModelFunctions(unittest.TestCase):
    """
    Tests for all the functions in `face_utils.py`
    """
    def test_emotion_model(self):
        """
        Sends some random data to the emotion model to ensure that the response
        looks correct.
        """
        # Data for the emotion model is (48, 48, 1) - one color channel.
        image_data = numpy.random.rand(1, 48, 48, 1) 

        data = json.dumps({"signature_name": "serving_default",
                        "instances": image_data.tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post(f"http://{URL}:{PORT}/v1/models/emotion_model:predict", data=data, headers=headers)

        predictions = numpy.array(json.loads(json_response.text)["predictions"])

        # There are seven categories predicted.
        assert predictions.shape == (1, 7)


    def test_emotion_model(self):
        """
        Sends some random data to the age-gender model to ensure that the response
        looks correct.
        """
        # Data for the emotion model is (64, 64, 3) - 3 color channels.
        image_data = numpy.random.rand(1, 64, 64, 3) 

        data = json.dumps({"signature_name": "serving_default",
                        "instances": image_data.tolist()})
        headers = {"content-type": "application/json"}
        json_response = requests.post(f"http://{URL}:{PORT}/v1/models/age_gender_model:predict", data=data, headers=headers)

        predictions = numpy.array(json.loads(json_response.text)["predictions"])

        # There are two gender categories.
        assert len(predictions[0]["dense"]) == 2

        # There are 101 age cetegories.
        assert len(predictions[0]["dense_1"]) == 101
