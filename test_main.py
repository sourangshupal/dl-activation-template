import unittest
import math
from main import sigmoid, relu, tanh_activation, softmax


class TestActivations(unittest.TestCase):

    def test_sigmoid_zero(self):
        self.assertAlmostEqual(sigmoid(0), 0.5, places=5)

    def test_sigmoid_positive(self):
        self.assertAlmostEqual(sigmoid(2), 1 / (1 + math.exp(-2)), places=5)

    def test_relu_negative(self):
        self.assertEqual(relu(-3), 0)

    def test_relu_positive(self):
        self.assertEqual(relu(5), 5)

    def test_tanh(self):
        self.assertAlmostEqual(tanh_activation(1.0), math.tanh(1.0), places=5)

    def test_softmax_sums_to_one(self):
        result = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(result), 1.0, places=5)

    def test_softmax_length(self):
        result = softmax([0.5, 1.5, 2.5])
        self.assertEqual(len(result), 3)

    def test_softmax_max_index(self):
        result = softmax([1.0, 2.0, 3.0])
        self.assertEqual(result.index(max(result)), 2)


if __name__ == "__main__":
    unittest.main()
