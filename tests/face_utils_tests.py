from face_utils import decode_image, crop_face, get_emotions

import unittest
import numpy as np


class TestSum(unittest.TestCase):
    def test_get_emotions(self):
        # Dummy data that mimics the shape of the input.
        cropped_face = np.ones([1, 48, 48, 1]) # TODO: actually load local data.
        self.assertEqual(get_emotions(cropped_face), "neutral")


    # def test_crop_face():
    #     # Dummy data that mimics the frame.
    #     frame = 0 # TODO: actually load local data.
    #     cropped_face = cropped_face(frame)
    #     print(cropped_face)


    # def test_decode_image():
    #     # Dummy data that mimics the image string.
    #     image_string = 0 # TODO: actually load local data.
    #     frame = decode_image(image_string)
    #     print(cropped_face)
