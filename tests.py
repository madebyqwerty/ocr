
from main import Engine
import unittest

class TestStringMethods(unittest.TestCase):
    """
    Tohle je příklad z dokumentace, jen abych nezapomněl, jak se to dělá, než to použiju
    """
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

def image_test():
    Engine.process(f"TestImg/img0.jpg")
    Engine.process(f"TestImg/img1.jpg")
    Engine.process(f"TestImg/img2.jpg")
    Engine.process(f"TestImg/img3.jpg")
    Engine.process(f"TestImg/img4.jpg")
    Engine.process(f"TestImg/img5.jpg")

if __name__ == '__main__':
    while True:
        what = input("\n1) Unit tests\n2) Image tests\n--> ")
        if what == "1": unittest.main()
        elif what == "2": image_test()