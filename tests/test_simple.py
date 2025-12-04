import unittest

class TestSimple(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)

if __name__ == '__main__':
    with open("simple_test_results.txt", "w") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        unittest.main(testRunner=runner, exit=False)
