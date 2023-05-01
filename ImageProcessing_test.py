
"""
run: pytest
"""

from main import Engine, Image
import pytest, cv2

class TestClass:
    def test_rotate(self):
        img = cv2.imread("TestImg/img0.jpg")
        size = img.shape
        rotated_size = Image.rotate(img).shape
        assert size[0] == rotated_size[1] and size[1] == rotated_size[0]

    def test_resize(self):
        img = cv2.imread("TestImg/img0.jpg")
        size = img.shape
        resized_size = Image.resize(img, 0.25).shape
        assert size[0]/4 == resized_size[0] and size[1]/4 == resized_size[1]
